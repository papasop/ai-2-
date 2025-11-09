# ============================================================
# GrammarTree 3.0: 自我精简 & 回写版 L7 AutoPDE under SCCT
#  - Stage 1: 训练 GrammarTree 2.0 (多基函数语法树残差)
#  - Stage 2: 基于 |γ w_i| 自动选主结构 → 构造精简版 GrammarTree 3.0
#  - Stage 3: 再训练 & 用 SymPy 对比老师 PDE
#  - Extra: 导出 residual 数据集给 PySR 使用
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sympy import symbols, simplify

print("Using device:", device)
print(f"[Check] u0 device: {u0.device}, teacher_traj shape: {teacher_traj.shape}")

# -----------------------------
# 0. 教师结构量（终时刻）
# -----------------------------
phi2_true_T = torch.tensor(phi2_true_traj[-1], device=device)
H_true_T    = torch.tensor(H_true_traj[-1],    device=device)

# ============================================================
# 1. GrammarTree 2.0: 多基函数语法树残差
# ============================================================

GRAMMAR_TERMS = [
    "u",          # φ0
    "u_x",        # φ1
    "u_xx",       # φ2  （物理正确项候选）
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
    feats.append(u)             # φ0 = u
    feats.append(u_x)           # φ1 = u_x
    feats.append(u_xx)          # φ2 = u_xx
    feats.append(u**2)          # φ3 = u**2
    feats.append(u**3)          # φ4 = u**3
    feats.append(u * u_x)       # φ5 = u*u_x
    feats.append(u * u_xx)      # φ6 = u*u_xx
    feats.append(u_x * u_xx)    # φ7 = u_x*u_xx

    return torch.stack(feats, dim=-1)   # [Nx, n_terms]

n_terms = len(GRAMMAR_TERMS)
print(f"[GrammarTree 2.0] n_terms = {n_terms}, terms = {GRAMMAR_TERMS}")

class GrammarTree20PDE(nn.Module):
    """
    Stage 1: 原始 GrammarTree 2.0
    u_t = 0.8 u_xx + 0.5 u - u^3 + gamma * Σ w_i φ_i(u)
    """
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_terms))
        self.gamma = nn.Parameter(torch.tensor(0.05))

    def residual(self, u):
        feats = grammar_features(u)          # [Nx, n_terms]
        return feats @ self.w                # [Nx]

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

# ============================================================
# 2. Stage 1 训练：GrammarTree 2.0 (SCCT + meta window)
# ============================================================

def train_grammar_tree20_meta(epochs=150, print_every=10):
    model = GrammarTree20PDE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    lambda_scct  = 5e-2   # 终时刻 structural constraint
    lambda_time  = 1e-2   # 早期时间窗结构
    lambda_gamma = 1e-4   # γ 正则
    lambda_w     = 1e-4   # w 稀疏

    meta_window = 400
    phi2_true_mean_meta = torch.tensor(phi2_true_traj[:meta_window].mean(), device=device)
    H_true_mean_meta    = torch.tensor(H_true_traj[:meta_window].mean(),    device=device)

    for ep in range(1, epochs+1):
        opt.zero_grad()

        # 终时刻轨道 + 结构
        u_pred_T, phi2_pred_T, H_pred_T = model.simulate_final(u0, Nt_teacher)
        misfit = F.mse_loss(u_pred_T, uT_true)

        loss_scct_T = torch.abs(phi2_pred_T - phi2_true_T) \
                    + torch.abs(H_pred_T    - H_true_T)

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

print("\n[Stage 1] Training GrammarTree 2.0 L7(meta)...")
model_gt20 = train_grammar_tree20_meta()

# ============================================================
# 3. 自动“符号精简”：根据 |γ w_i| 选主结构
# ============================================================

with torch.no_grad():
    g20  = model_gt20.gamma.detach().cpu().item()
    w20  = model_gt20.w.detach().cpu().numpy()
    eff20 = g20 * w20

print("\n[Stage 2] GrammarTree 2.0 learned coefficients:")
for i, (name, wi, ei) in enumerate(zip(GRAMMAR_TERMS, w20, eff20)):
    print(f"  φ{i}(u) = {name:8s} : w={wi:+.6e}, γ*w={ei:+.6e}")

# 设一个自动剪枝阈值：保留 |γ w_i| 最大的若干项
# 这里简单用绝对值阈值，你可以手动调整：
threshold = 5e-4   # 小于这个就视为“几乎 0”，自动关掉
keep_mask = np.abs(eff20) > threshold
keep_indices = np.where(keep_mask)[0]

print("\n[Stage 2] Automatic structural selection:")
print("  threshold on |γ w_i| =", threshold)
print("  kept indices:", keep_indices.tolist())
print("  kept terms  :", [GRAMMAR_TERMS[i] for i in keep_indices])

# ============================================================
# 4. GrammarTree 3.0: 精简版 PDE （只保留主结构 φᵢ）
# ============================================================

class GrammarTree30PDE(nn.Module):
    """
    Stage 2+3: 精简版 GrammarTree 3.0
    只保留 keep_indices 对应的 φ_i
    并从 Stage 1 的解初始化参数，再训练一次
    """
    def __init__(self, keep_indices, model_stage1):
        super().__init__()
        self.keep_indices = list(keep_indices)

        # 用 Stage 1 的 γ 初始化
        self.gamma = nn.Parameter(model_stage1.gamma.detach().clone())

        # 只为保留的 φ_i 分配 w 参数，并用 Stage 1 的值初始化
        w_stage1 = model_stage1.w.detach().clone()
        w_init = w_stage1[self.keep_indices]
        self.w = nn.Parameter(w_init)

    def residual(self, u):
        feats_all = grammar_features(u)                      # [Nx, n_terms]
        feats_keep = feats_all[:, self.keep_indices]         # [Nx, n_keep]
        return feats_keep @ self.w                           # [Nx]

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

def train_grammar_tree30_meta(model_stage1,
                              keep_indices,
                              epochs=100,
                              print_every=10):
    model = GrammarTree30PDE(keep_indices, model_stage1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    lambda_scct  = 5e-2
    lambda_time  = 1e-2
    lambda_gamma = 1e-4
    lambda_w     = 1e-4

    meta_window = 400
    phi2_true_mean_meta = torch.tensor(phi2_true_traj[:meta_window].mean(), device=device)
    H_true_mean_meta    = torch.tensor(H_true_traj[:meta_window].mean(),    device=device)

    for ep in range(1, epochs+1):
        opt.zero_grad()

        u_pred_T, phi2_pred_T, H_pred_T = model.simulate_final(u0, Nt_teacher)
        misfit = F.mse_loss(u_pred_T, uT_true)

        loss_scct_T = torch.abs(phi2_pred_T - phi2_true_T) \
                    + torch.abs(H_pred_T    - H_true_T)

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
            print(f"[GrammarTree 3.0] Epoch {ep:03d} | "
                  f"loss={loss.item():.3e}, misfit={misfit.item():.3e}, "
                  f"Φ²(T)={phi2_pred_T.item():.3e}, H(T)={H_pred_T.item():.3e}, "
                  f"γ={g.item():+.3e}")
            print("   w_keep (raw):", ", ".join([f"{wi:+.2e}" for wi in w.detach().cpu().numpy()]))
            print("   eff=γ*w_keep :", ", ".join([f"{ei:+.2e}" for ei in eff]))

    return model

print("\n[Stage 3] Training GrammarTree 3.0 (pruned)...")
model_gt30 = train_grammar_tree30_meta(model_gt20, keep_indices)

# ============================================================
# 5. 符号对比：用 SymPy 拼出 3.0 PDE 并对比老师
# ============================================================

with torch.no_grad():
    g30  = model_gt30.gamma.detach().cpu().item()
    w30  = model_gt30.w.detach().cpu().numpy()
    eff30 = g30 * w30

print("\n[GrammarTree 3.0] Final kept coefficients:")
for idx, wi, ei in zip(keep_indices, w30, eff30):
    print(f"  φ{idx}(u) = {GRAMMAR_TERMS[idx]:8s} : w={wi:+.6e}, γ*w={ei:+.6e}")

# SymPy 符号：
u_sym, u_x_sym, u_xx_sym = symbols("u u_x u_xx")

phi_sym_all = [
    u_sym,                    # "u"
    u_x_sym,                  # "u_x"
    u_xx_sym,                 # "u_xx"
    u_sym**2,                 # "u**2"
    u_sym**3,                 # "u**3"
    u_sym*u_x_sym,            # "u*u_x"
    u_sym*u_xx_sym,           # "u*u_xx"
    u_x_sym*u_xx_sym,         # "u_x*u_xx"
]

core_rhs_sym    = -u_sym**3 + 0.5*u_sym + 0.8*u_xx_sym
teacher_rhs_sym = -u_sym**3 + 0.5*u_sym + 0.82*u_xx_sym

# GrammarTree 3.0 残差
residual_sym_30 = 0
for idx, coeff in zip(keep_indices, eff30):
    residual_sym_30 += coeff * phi_sym_all[idx]

discovered_rhs_sym_30 = core_rhs_sym + residual_sym_30
diff_sym_30 = simplify(discovered_rhs_sym_30 - teacher_rhs_sym)

print("\n[SymPy] Core PDE RHS(u):")
print("  ", core_rhs_sym)
print("[SymPy] Teacher PDE RHS(u):")
print("  ", teacher_rhs_sym)
print("[SymPy] Discovered PDE RHS(u) from GrammarTree 3.0:")
print("  ", discovered_rhs_sym_30)
print("\n[SymPy] Discovered RHS(u) - Teacher RHS(u) =")
print("  ", diff_sym_30)

# 尤其关注 u_xx 的总系数
# 老师: 0.8 + 0.02 = 0.82
# 我们: 0.8 + sum(eff30 * 对应 φ_i 中的 u_xx 贡献), 这里只是看 Γ*w_φ2 主项
eff_uxx_30 = 0.0
for idx, coeff in zip(keep_indices, eff30):
    if GRAMMAR_TERMS[idx] == "u_xx":
        eff_uxx_30 += coeff
print(f"\n[Summary - GrammarTree 3.0]")
print(f"  - Effective residual coeff on u_xx (from kept φ_i): {eff_uxx_30:+.6e}")
print(f"  - Target hidden residual: 0.02 * u_xx")

# ============================================================
# 6. Extra: 导出 residual 数据集给 PySR (可选)
# ============================================================

def export_residual_dataset_for_pysr(model, n_time=400):
    """
    把 GrammarTree 模型的残差数据导出成 numpy，
    方便你在另一个单元里用 PySR/AI Feynman 做外部符号回归。
    特征使用 [u, u_x, u_xx]，目标是 r = RHS_model - RHS_core.
    """
    u = u0.clone().to(device)
    U_list, Ux_list, Uxx_list, R_list = [], [], [], []

    for _ in range(n_time):
        # 一步模型演化
        u = model.step(u)

        u_x  = grad_x_centered(u, dx)
        u_xx = laplace_x_dirichlet(u, dx)

        core = 0.8 * u_xx + 0.5 * u - u**3
        rhs_model = model.pde_rhs(u)
        r = rhs_model - core    # 应该等于 gamma * residual(u)

        U_list.append(u.detach().cpu().numpy())
        Ux_list.append(u_x.detach().cpu().numpy())
        Uxx_list.append(u_xx.detach().cpu().numpy())
        R_list.append(r.detach().cpu().numpy())

    U   = np.array(U_list).reshape(-1)
    Ux  = np.array(Ux_list).reshape(-1)
    Uxx = np.array(Uxx_list).reshape(-1)
    R   = np.array(R_list).reshape(-1)

    X = np.stack([U, Ux, Uxx], axis=1)  # [N, 3]

    np.savez("residual_dataset_pysr.npz", X=X, R=R)
    print("\n[Export] Saved residual_dataset_pysr.npz")
    print("  - X shape:", X.shape, " (columns: [u, u_x, u_xx])")
    print("  - R shape:", R.shape, " (target residual)")

print("\n[Stage 4] Export residual dataset for PySR from GrammarTree 3.0...")
export_residual_dataset_for_pysr(model_gt30)
