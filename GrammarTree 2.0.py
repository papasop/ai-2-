# ============================================================
# GrammarTree 2.0: 多基函数语法树版 L7 AutoPDE under SCCT
#  - Teacher PDE : u_t = 0.82 u_xx + 0.5 u - u^3
#  - Core PDE    : u_t = 0.80 u_xx + 0.5 u - u^3
#  - Hidden residual: 0.02 u_xx
#  - GrammarTree 2.0: r(u,u_x,u_xx) = Σ w_i φ_i(u,u_x,u_xx)
#    φ_i 由一个简单“语法生成器”自动构造
#  - Loss: 终时刻轨道 + SCCT(T) + meta window 结构 + L1 稀疏
#  - Math-auto 0.5: SymPy 拼出完整 PDE，并和老师方程做符号对比
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sympy import symbols, simplify

# 这里假设你已经有这些全局对象（来自 v1.1）：
# device, dx, dt, u0, teacher_traj, uT_true, phi2_true_traj, H_true_traj
# laplace_x_dirichlet, grad_x_centered, scct_stats_torch

print("Using device:", device)
print(f"[Check] u0 device: {u0.device}, teacher_traj shape: {teacher_traj.shape}")

# 把 teacher 终时刻结构量转回 torch
phi2_true_T = torch.tensor(phi2_true_traj[-1], device=device)
H_true_T    = torch.tensor(H_true_traj[-1],    device=device)

# -----------------------------
# 1. 语法基函数生成器 φ_i(u,u_x,u_xx)
# -----------------------------
GRAMMAR_TERMS = [
    "u",          # φ0
    "u_x",        # φ1
    "u_xx",       # φ2  （物理正确项的主要候选）
    "u**2",       # φ3
    "u**3",       # φ4
    "u*u_x",      # φ5
    "u*u_xx",     # φ6
    "u_x*u_xx",   # φ7
]

def grammar_features(u):
    """
    给定 u(x)，自动生成一组 φ_i(u,u_x,u_xx) 特征。
    返回 [Nx, n_terms] 的张量。
    """
    u_x  = grad_x_centered(u, dx)
    u_xx = laplace_x_dirichlet(u, dx)

    feats = []
    # φ0 = u
    feats.append(u)
    # φ1 = u_x
    feats.append(u_x)
    # φ2 = u_xx
    feats.append(u_xx)
    # φ3 = u**2
    feats.append(u**2)
    # φ4 = u**3
    feats.append(u**3)
    # φ5 = u*u_x
    feats.append(u * u_x)
    # φ6 = u*u_xx
    feats.append(u * u_xx)
    # φ7 = u_x*u_xx
    feats.append(u_x * u_xx)

    return torch.stack(feats, dim=-1)   # [Nx, n_terms]

n_terms = len(GRAMMAR_TERMS)
print(f"[GrammarTree 2.0] n_terms = {n_terms}, terms = {GRAMMAR_TERMS}")

# -----------------------------
# 2. GrammarTree 2.0 PDE 模型
# -----------------------------
class GrammarTree20PDE(nn.Module):
    def __init__(self):
        super().__init__()
        # 残差系数：对每个 φ_i 有一个 w_i
        self.w = nn.Parameter(torch.zeros(n_terms))
        # 全局缩放 γ（类似 v1.1）
        self.gamma = nn.Parameter(torch.tensor(0.05))

    def residual(self, u):
        """
        r(u) = Σ w_i φ_i(u,u_x,u_xx)
        """
        feats = grammar_features(u)              # [Nx, n_terms]
        return feats @ self.w                   # [Nx]

    def pde_rhs(self, u):
        u_xx = laplace_x_dirichlet(u, dx)
        core = 0.8 * u_xx + 0.5 * u - u**3
        r = self.residual(u)
        return core + self.gamma * r

    def step(self, u):
        u_new = u + dt * self.pde_rhs(u)
        u_new[0] = 0.0
        u_new[-1] = 0.0
        return u_new

    def simulate_final(self, u0, Nt):
        u = u0.clone()
        for _ in range(Nt):
            u = self.step(u)
        phi2, H = scct_stats_torch(u)
        return u, phi2, H

    def simulate_traj_scct(self, u0, Nt):
        u = u0.clone()
        phi2s, Hs = [], []
        for _ in range(Nt):
            u = self.step(u)
            phi2, H = scct_stats_torch(u)
            phi2s.append(phi2)
            Hs.append(H)
        return torch.stack(phi2s), torch.stack(Hs)

# -----------------------------
# 3. 轨道误差工具
# -----------------------------
def rollout_and_error(model, steps):
    u_teacher = u0.clone()
    u_model = u0.clone()
    errs = []
    with torch.no_grad():
        for _ in range(steps):
            # 老师一步
            u_teacher = teacher_step(u_teacher)
            # 模型一步
            u_model = model.step(u_model)
            errs.append(torch.mean((u_teacher - u_model)**2).item())
    return np.array(errs)

# -----------------------------
# 4. Training: SCCT + meta window + 稀疏
# -----------------------------
def train_grammar_tree20_meta(epochs=150, print_every=10):
    model = GrammarTree20PDE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    lambda_scct  = 5e-2   # 终时刻 structural constraint
    lambda_time  = 1e-2   # 早期时间窗结构
    lambda_gamma = 1e-4   # γ 正则
    lambda_w     = 1e-4   # w 稀疏（促使一些基函数自动“关掉”）

    meta_window = 400
    phi2_true_mean_meta = torch.tensor(phi2_true_traj[:meta_window].mean(), device=device)
    H_true_mean_meta    = torch.tensor(H_true_traj[:meta_window].mean(),    device=device)

    for ep in range(1, epochs+1):
        opt.zero_grad()

        # 终时刻轨道 + 结构
        u_pred_T, phi2_pred_T, H_pred_T = model.simulate_final(u0, Nt_teacher)
        misfit = F.mse_loss(u_pred_T, uT_true)

        loss_scct_T = torch.abs(phi2_pred_T - phi2_true_T) \
                    + torch.abs(H_pred_T - H_true_T)

        # 早期时间结构 [0, 0.04]
        phi2_t, H_t = model.simulate_traj_scct(u0, meta_window)
        loss_time = (phi2_t.mean() - phi2_true_mean_meta)**2 \
                  + (H_t.mean()   - H_true_mean_meta)**2

        g = model.gamma
        w = model.w

        loss = misfit \
             + lambda_scct  * loss_scct_T \
             + lambda_time  * loss_time \
             + lambda_gamma * torch.sum(torch.abs(g)) \
             + lambda_w     * torch.sum(torch.abs(w))

        loss.backward()
        opt.step()

        if ep % print_every == 0 or ep == 1:
            eff = (g * w).detach().cpu().numpy()
            print(f"[GrammarTree 2.0] Epoch {ep:03d} | "
                  f"loss={loss.item():.3e}, misfit={misfit.item():.3e}, "
                  f"Φ²(T)={phi2_pred_T.item():.3e}, H(T)={H_pred_T.item():.3e}, "
                  f"γ={g.item():+.3e}")
            print("   w (raw):", ", ".join([f"{wi:+.2e}" for wi in w.detach().cpu().numpy()]))
            print("   eff=γ*w:", ", ".join([f"{ei:+.2e}" for ei in eff]))

    return model

print("\n[Training GrammarTree 2.0 L7(meta)...]")
model_gt20 = train_grammar_tree20_meta()

# -----------------------------
# 5. 数值自检：轨道 + 结构
# -----------------------------
steps_err = 400
err_gt20 = rollout_and_error(model_gt20, steps_err)
t_err = np.arange(steps_err) * dt

with torch.no_grad():
    phi2_gt20, H_gt20 = model_gt20.simulate_traj_scct(u0, steps_err)

phi2_teacher_early = phi2_true_traj[:steps_err]

print(f"\n[Mean L2 orbit error over [0,{steps_err*dt:.3f}]]: {err_gt20.mean():.3e}")
print(f"[Early-time Φ²] teacher={phi2_teacher_early.mean():.3e}, "
      f"GrammarTree2.0={phi2_gt20.mean().item():.3e}, "
      f"ratio≈{phi2_gt20.mean().item()/phi2_teacher_early.mean():.3f}")

# 打印最终系数
g_final = model_gt20.gamma.detach().cpu().item()
w_final = model_gt20.w.detach().cpu().numpy()
eff_final = g_final * w_final

print("\n[GrammarTree 2.0] Final coefficients:")
for i, (name, wi, ei) in enumerate(zip(GRAMMAR_TERMS, w_final, eff_final)):
    print(f"  φ{i}(u) = {name:8s} : w={wi:+.6e}, γ*w={ei:+.6e}")

# 尤其关注 φ2 = u_xx 的总系数
eff_uxx = eff_final[2]
print(f"\n[Effective coeff on u_xx] γ*w_u_xx = {eff_uxx:+.6e} (target ≈ +2.000e-02)")

# -----------------------------
# 6. SymPy 符号自检：拼出完整 PDE 并和老师对比
# -----------------------------
u_sym, u_x_sym, u_xx_sym = symbols("u u_x u_xx")

# 构造 SymPy 版本的 φ_i
phi_sym = [
    u_sym,                    # "u"
    u_x_sym,                  # "u_x"
    u_xx_sym,                 # "u_xx"
    u_sym**2,                 # "u**2"
    u_sym**3,                 # "u**3"
    u_sym*u_x_sym,            # "u*u_x"
    u_sym*u_xx_sym,           # "u*u_xx"
    u_x_sym*u_xx_sym,         # "u_x*u_xx"
]

# core & teacher
core_rhs_sym    = -u_sym**3 + 0.5*u_sym + 0.8*u_xx_sym
teacher_rhs_sym = -u_sym**3 + 0.5*u_sym + 0.82*u_xx_sym

# GrammarTree 2.0 学到的残差
residual_sym = 0
for wi, phi_i in zip(eff_final, phi_sym):
    residual_sym += wi * phi_i

discovered_rhs_sym = core_rhs_sym + residual_sym

print("\n[SymPy] Core PDE RHS(u):")
print("  ", core_rhs_sym)
print("[SymPy] Teacher PDE RHS(u):")
print("  ", teacher_rhs_sym)
print("[SymPy] Discovered PDE RHS(u) from GrammarTree 2.0:")
print("  ", discovered_rhs_sym)

# 符号对比差异
diff_sym = simplify(discovered_rhs_sym - teacher_rhs_sym)
print("\n[SymPy] Discovered RHS(u) - Teacher RHS(u) =")
print("  ", diff_sym)

print("\n[Summary - GrammarTree 2.0]")
print(f"  - Target hidden residual: 0.02 * u_xx")
print(f"  - Effective coeff on u_xx from GrammarTree 2.0: {eff_uxx:+.6e}")
print("  - 若其他 φ_i（如 u, u*u_xx 等）的 γ*w 接近 0，则说明语法树自动“关掉”了非物理项，")
print("    把主要质量集中在物理正确的 u_xx 上。")
