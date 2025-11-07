# ============================================================
# Neural PDE Automation (Allen–Cahn + Adaptive JNet + SCCT)
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 1. PDE: 1D Allen–Cahn + 分段常数控制 c(t; θ)
#   u_t = ε u_xx + u - u^3 + c(t; θ),  x ∈ (0,1), t ∈ (0,T]
#   u(t,0) = u(t,1) = 0
#   u(0,x) = u0(x)
# ============================================================

T_final = 1.0
epsilon = 0.01        # diffusion
K_ctrl = 4            # 控制维度: K 段时间片
time_segments = torch.linspace(0.0, T_final, K_ctrl + 1)  # t0,...,tK
lambda_theta = 1e-2   # 控制正则

def control_c(t, theta):
    """
    分段常数控制 c(t; θ), θ ∈ R^K_ctrl
    t: (N,) 实数时间
    theta: (K_ctrl,)
    返回: (N,)
    """
    t = t.unsqueeze(-1)  # (N,1)
    seg_left = time_segments[:-1].to(t.device)   # (K,)
    seg_right = time_segments[1:].to(t.device)   # (K,)

    mask = (t >= seg_left) & (t < seg_right)
    mask |= (t == T_final) & (seg_right == T_final)
    theta = theta.to(t.device)
    c_val = (mask.float() * theta).sum(dim=-1)
    return c_val

def init_u0(x):
    """初值: 基本模态 + 小扰动"""
    return torch.sin(math.pi * x) + 0.1 * torch.sin(2 * math.pi * x)

# ============================================================
# 2. PINN Teacher: u_φ(t,x;θ)
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64, depth=4, activation=nn.Tanh):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden] * (depth - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class PINN_AllenCahn(nn.Module):
    def __init__(self, K_ctrl):
        super().__init__()
        # 输入: (t, x, θ1,...,θK) → u
        self.net = MLP(in_dim=2 + K_ctrl, out_dim=1, hidden=64, depth=4)
    def forward(self, t, x, theta):
        """
        t, x: (N,1)
        theta: (K_ctrl,) or (N,K_ctrl)
        """
        if theta.dim() == 1:
            theta_expand = theta.unsqueeze(0).expand(t.shape[0], -1)
        else:
            theta_expand = theta
        inp = torch.cat([t, x, theta_expand], dim=-1)
        return self.net(inp)

def pinn_loss(model, theta,
              n_int=1024, n_bc=128, n_ic=128,
              T_final=1.0, epsilon=0.01,
              w_r=1.0, w_b=10.0, w_ic=10.0):
    """
    对给定 θ 计算 PINN 损失:
      L_PINN = w_r E|r|² + w_b E|u_bc|² + w_ic E|u_ic - u0|²
    """
    model.train()
    theta = theta.to(device)

    # Interior points
    t_int = torch.rand(n_int, 1, device=device) * T_final
    x_int = torch.rand(n_int, 1, device=device)
    t_int.requires_grad_(True)
    x_int.requires_grad_(True)

    u_int = model(t_int, x_int, theta)

    # u_t, u_xx
    u_t = torch.autograd.grad(
        u_int, t_int,
        grad_outputs=torch.ones_like(u_int),
        retain_graph=True,
        create_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u_int, x_int,
        grad_outputs=torch.ones_like(u_int),
        retain_graph=True,
        create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x_int,
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True
    )[0]

    c_int = control_c(t_int.squeeze(-1), theta).unsqueeze(-1)

    # PDE residual: u_t - ε u_xx - u + u^3 - c = 0
    r_int = u_t - epsilon * u_xx - u_int + u_int**3 - c_int
    loss_int = (r_int**2).mean()

    # Boundary: u(t,0)=u(t,1)=0
    t_bc = torch.rand(n_bc, 1, device=device) * T_final
    x_bc0 = torch.zeros(n_bc, 1, device=device)
    x_bc1 = torch.ones(n_bc, 1, device=device)
    u_bc0 = model(t_bc, x_bc0, theta)
    u_bc1 = model(t_bc, x_bc1, theta)
    loss_bc = (u_bc0**2).mean() + (u_bc1**2).mean()

    # Initial: u(0,x)=u0(x)
    x_ic = torch.rand(n_ic, 1, device=device)
    t_ic = torch.zeros_like(x_ic)
    u_ic = model(t_ic, x_ic, theta)
    u0 = init_u0(x_ic)
    loss_ic = ((u_ic - u0)**2).mean()

    loss = w_r*loss_int + w_b*loss_bc + w_ic*loss_ic
    return loss, {
        "loss": loss.item(),
        "loss_int": loss_int.item(),
        "loss_bc": loss_bc.item(),
        "loss_ic": loss_ic.item()
    }

def train_pinn_for_theta(theta,
                         n_steps=1500,
                         lr=1e-3,
                         verbose_every=200):
    """
    针对给定 θ 训练 PINN teacher.
    """
    model = PINN_AllenCahn(K_ctrl=K_ctrl).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    theta_t = theta.to(device)
    log_hist = []
    pbar = tqdm(range(1, n_steps+1),
                desc=f"Train PINN θ={theta.cpu().numpy()}")
    for it in pbar:
        optimizer.zero_grad()
        loss, loss_dict = pinn_loss(model, theta_t)
        loss.backward()
        optimizer.step()
        log_hist.append(loss_dict)
        if it % verbose_every == 0 or it == 1:
            pbar.set_postfix(loss=loss_dict["loss"],
                             lint=loss_dict["loss_int"],
                             lbc=loss_dict["loss_bc"],
                             lic=loss_dict["loss_ic"])
    return model, log_hist

# ============================================================
# 3. 控制泛函 J(θ) via PINN
#   J(θ) ≈ ∫∫ 0.5 u² dx dt + 0.5 λ θ² T
# ============================================================

def estimate_J_pinn(theta, model,
                    n_samples=5000,
                    T_final=1.0):
    theta = theta.to(device)
    model.eval()
    t = torch.rand(n_samples, 1, device=device) * T_final
    x = torch.rand(n_samples, 1, device=device)
    with torch.no_grad():
        u = model(t, x, theta)
    state_cost = 0.5 * (u**2)
    state_term = state_cost.mean() * T_final

    theta_norm_sq = (theta**2).sum()
    control_term = 0.5 * lambda_theta * theta_norm_sq * T_final
    J = state_term + control_term
    return J.item(), {
        "state_term": state_term.item(),
        "control_term": control_term.item()
    }

# ============================================================
# 4. SCCT 诊断: Φ², H, K_eff
# 这里用简单离散版:
#   Φ² = Var(u)
#   H ≈ sum (u_i - u_j)² / N²  或 用近邻差分近似
# 为省事用 mean-field: H = 2 N² Var(u) → K_eff ~ 2
# ============================================================

def scct_stats_from_model(model, theta,
                          n_samples=2048,
                          T_eval=None):
    """
    简单版本: 在某个时间截面 (或时间混合) 上采样 u，用 mean-field 公式:
      Φ² = Var(u), H = 2 N² Φ², K_eff = 2
    主要目的是提供 Φ² 与 H 的数量级，后面可扩展更精细定义。
    """
    model.eval()
    theta = theta.to(device)
    if T_eval is None:
        # 在 [0,T_final] 内随机采样时间
        t = torch.rand(n_samples, 1, device=device) * T_final
    else:
        t = torch.full((n_samples, 1), float(T_eval), device=device)
    x = torch.rand(n_samples, 1, device=device)
    with torch.no_grad():
        u = model(t, x, theta)  # (N,1)
    u_flat = u.view(-1)
    phi2 = torch.var(u_flat, unbiased=False)
    N = u_flat.shape[0]
    H = 2 * (N**2) * phi2
    K_eff = 2.0  # 在这个简化定义下简单等于 2
    return {
        "Phi2": phi2.item(),
        "H": H.item(),
        "K_eff": K_eff
    }

# ============================================================
# 5. 构建 Teacher 数据集: { (θ_i, J_pinn(θ_i)) }
# ============================================================

def sample_theta_uniform(n_samples, low=-1.0, high=1.0):
    return (low + (high - low) * torch.rand(n_samples, K_ctrl))

def build_teacher_dataset(n_theta=8):
    thetas = sample_theta_uniform(n_theta)
    J_values = []
    teacher_models = []
    scct_info = []

    for i in range(n_theta):
        theta_i = thetas[i]
        model_i, log_hist = train_pinn_for_theta(theta_i,
                                                 n_steps=1500,
                                                 lr=1e-3,
                                                 verbose_every=300)
        teacher_models.append(model_i)
        J_i, parts = estimate_J_pinn(theta_i, model_i)
        scct_i = scct_stats_from_model(model_i, theta_i)
        print(f"[Teacher] i={i}, θ={theta_i.numpy()}, "
              f"J_pinn={J_i:.6e}, parts={parts}, SCCT={scct_i}")
        J_values.append(J_i)
        scct_info.append(scct_i)

    thetas = thetas
    J_values = torch.tensor(J_values).unsqueeze(-1)  # (n_theta,1)
    return thetas, J_values, teacher_models, scct_info

# ⚠️ 注意：实际跑的时候先用小 n_theta，比如 4 或 6，避免太慢。
# thetas_train, J_train, teacher_models, scct_info = build_teacher_dataset(n_theta=4)

# ============================================================
# 6. JNet / log-JNet Surrogates
# ============================================================

class JNet(nn.Module):
    def __init__(self, K_ctrl, hidden=64, depth=3):
        super().__init__()
        self.net = MLP(in_dim=K_ctrl, out_dim=1, hidden=hidden, depth=depth)
    def forward(self, theta):
        return self.net(theta)

def train_jnet(thetas, J_values,
               n_steps=2000, lr=1e-3, log_prefix="[JNet]"):
    model = JNet(K_ctrl).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    thetas_t = thetas.to(device)
    J_t = J_values.to(device)

    pbar = tqdm(range(1, n_steps+1), desc=log_prefix)
    for it in pbar:
        opt.zero_grad()
        pred = model(thetas_t)
        loss = ((pred - J_t)**2).mean()
        loss.backward()
        opt.step()
        if it % 200 == 0 or it == 1:
            pbar.set_postfix(loss=loss.item())
    return model

def train_logjnet(thetas, J_values,
                  n_steps=2000, lr=1e-3, eps=1e-6,
                  log_prefix="[log-JNet]"):
    model = JNet(K_ctrl).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    thetas_t = thetas.to(device)
    J_t = J_values.to(device)
    logJ = torch.log(J_t + eps)

    pbar = tqdm(range(1, n_steps+1), desc=log_prefix)
    for it in pbar:
        opt.zero_grad()
        log_pred = model(thetas_t)
        loss = ((log_pred - logJ)**2).mean()
        loss.backward()
        opt.step()
        if it % 200 == 0 or it == 1:
            pbar.set_postfix(loss=loss.item())
    return model

def eval_surrogates(theta_test, jnet, logjnet, eps=1e-6):
    theta_t = theta_test.to(device)
    with torch.no_grad():
        J_pred = jnet(theta_t.unsqueeze(0)).squeeze(0).item()
        logJ_pred = logjnet(theta_t.unsqueeze(0)).squeeze(0).item()
        J_pred_log = math.exp(logJ_pred)
    return {
        "Jnet": J_pred,
        "Jnet_log": J_pred_log
    }

# ============================================================
# 7. 自适应采样: 发现 surrogate 误差大 → 补点 & 重训
# ============================================================

def adaptive_refine(
    thetas_train, J_train, teacher_models,
    jnet, logjnet,
    n_new_candidates=5,
    err_tol=5e-3,   # surrogate 误差阈值
    max_refine=3
):
    """
    简单版本：
    - 采样若干新的 θ_cand
    - 用 jnet 预测 J_cand
    - 选几个代表 (这里可以直接全部)
    - 为这些 θ 训练新的 PINN, 得到 J_pinn
    - 若 |J_pinn - Jnet| > err_tol，则加入训练集
    - 重训 jnet/logjnet
    """
    thetas_cur = thetas_train.clone()
    J_cur = J_train.clone()
    teachers_cur = list(teacher_models)

    for r in range(max_refine):
        print(f"\n[Adaptive] refine round {r+1}/{max_refine}")
        theta_cands = sample_theta_uniform(n_new_candidates)
        added = 0
        for i in range(n_new_candidates):
            theta_i = theta_cands[i]
            # surrogate 预测
            sur = eval_surrogates(theta_i, jnet, logjnet)
            J_est = sur["Jnet_log"]  # 用 log-surrogate 的指数值
            print(f"  cand i={i}, θ={theta_i.numpy()}, J_est≈{J_est:.6e}")

            # 训练真正的 PINN teacher, 计算 J_pinn
            model_i, log_hist = train_pinn_for_theta(theta_i,
                                                     n_steps=1500,
                                                     lr=1e-3,
                                                     verbose_every=300)
            J_i, parts = estimate_J_pinn(theta_i, model_i)
            err = abs(J_i - J_est)
            print(f"   → Teacher J_pinn={J_i:.6e}, err={err:.3e}")
            if err > err_tol:
                print("   → add to dataset.")
                thetas_cur = torch.cat([thetas_cur, theta_i.unsqueeze(0)], dim=0)
                J_cur = torch.cat([J_cur, torch.tensor([[J_i]])], dim=0)
                teachers_cur.append(model_i)
                added += 1
            else:
                print("   → error below tol, skip.")

        if added == 0:
            print("[Adaptive] no new points added; stop.")
            break
        else:
            print(f"[Adaptive] added {added} new points, retrain surrogates.")
            jnet = train_jnet(thetas_cur, J_cur, n_steps=1500, lr=1e-3,
                              log_prefix=f"[JNet refine {r+1}]")
            logjnet = train_logjnet(thetas_cur, J_cur, n_steps=1500, lr=1e-3,
                                    log_prefix=f"[log-JNet refine {r+1}]")
    return thetas_cur, J_cur, teachers_cur, jnet, logjnet

# ============================================================
# 8. Surrogate-based 控制优化 (θ)
# ============================================================

def optimize_theta_with_surrogate(
    jnet_or_logjnet, use_log=True,
    theta_init=None,
    n_steps=200,
    lr=5e-2,
    print_every=20
):
    """
    使用 Adam 在 surrogate 上做 θ 优化。
    use_log=True: 模型返回 logJ, 我们优化 J = exp(logJ)
    """
    if theta_init is None:
        theta = torch.zeros(K_ctrl, device=device)
    else:
        theta = theta_init.to(device)
    theta = theta.clone().detach().requires_grad_(True)

    opt = optim.Adam([theta], lr=lr)

    hist = []
    for it in range(1, n_steps+1):
        opt.zero_grad()
        if use_log:
            logJ = jnet_or_logjnet(theta.unsqueeze(0)).squeeze(0)
            J_val = torch.exp(logJ)
        else:
            J_val = jnet_or_logjnet(theta.unsqueeze(0)).squeeze(0)
        J_val.backward()
        opt.step()

        hist.append((it, theta.detach().cpu().numpy().copy(), J_val.item()))
        if it % print_every == 0 or it == 1:
            print(f"[Control] step {it:04d}, θ={theta.detach().cpu().numpy()}, J≈{J_val.item():.6e}")
    theta_star = theta.detach().cpu()
    return theta_star, hist

# ============================================================
# 9. 一键运行示例（谨慎：会比较慢，建议先调小 n_theta）
# ============================================================

if __name__ == "__main__":
    # 1) 构建初始 Teacher 数据集
    n_theta_init = 4    # 可以先 4，没问题再升到 8、12...
    thetas_train, J_train, teacher_models, scct_info = build_teacher_dataset(n_theta=n_theta_init)

    # 2) 训练初始 JNet / log-JNet
    jnet = train_jnet(thetas_train, J_train, n_steps=1500, lr=1e-3,
                      log_prefix="[JNet init]")
    logjnet = train_logjnet(thetas_train, J_train, n_steps=1500, lr=1e-3,
                            log_prefix="[log-JNet init]")

    # 3) 自适应 refine
    thetas_ref, J_ref, teacher_ref, jnet_ref, logjnet_ref = adaptive_refine(
        thetas_train, J_train, teacher_models,
        jnet, logjnet,
        n_new_candidates=3,
        err_tol=5e-3,
        max_refine=2
    )

    # 4) 用 logJNet surrogate 做 θ 控制优化
    theta_init = torch.zeros(K_ctrl)  # 起点
    theta_star, ctrl_hist = optimize_theta_with_surrogate(
        logjnet_ref, use_log=True,
        theta_init=theta_init,
        n_steps=100,
        lr=5e-2,
        print_every=20
    )
    print("\n[Result] θ* (surrogate control) =", theta_star.numpy())

    # 5) 在 θ* 上真实训练一个 PINN, 评估 J_pinn 和 SCCT
    model_star, _ = train_pinn_for_theta(theta_star,
                                         n_steps=1500,
                                         lr=1e-3,
                                         verbose_every=300)
    J_star, parts_star = estimate_J_pinn(theta_star, model_star)
    scct_star = scct_stats_from_model(model_star, theta_star)
    print(f"[Result] At θ*, J_pinn≈{J_star:.6e}, parts={parts_star}, SCCT={scct_star}")
