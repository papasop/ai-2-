# ============================================
# GrammarTree 5.0 - One-Click Colab Prototype
#  - 1D scalar PDE with hidden diffusion increment
#  - Dynamic SCCT weights
#  - Adaptive pruning threshold (median-based)
#  - PySR symbolic audit on residual
# ============================================

# -------------------------
# 0. 安装依赖
# -------------------------
!pip install -q numpy sympy matplotlib pysr torch

# ---- 环境变量（可选）----
import os
os.environ["JULIA_NUM_THREADS"] = "4"
os.environ["PYTHONHASHSEED"] = "0"

# -------------------------
# 1. 导入库（顺序：先 PySR / juliacall，再 torch）
# -------------------------
from pysr import PySRRegressor

try:
    import juliacall
    from juliacall import Main as jl
    print("[Init] juliacall imported, Julia available.")
except Exception as e:
    print("[Init] juliacall not explicitly available (PySR will manage Julia).")
    print("       detail:", repr(e))

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams["figure.figsize"] = (5, 3)
plt.rcParams["figure.dpi"] = 120

# -------------------------
# 2. 设备 & 随机种子
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Init] Using device:", device)

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DTYPE = torch.float32
def to_device(x):
    return x.to(device=device, dtype=DTYPE)

print("[Init] Random seed set to", SEED)
print("[Init] GrammarTree 5.0 header OK.\n")

# -------------------------
# 3. PDE 宇宙：网格 & 差分算子
# -------------------------
Nx = 64
x = torch.linspace(0.0, 1.0, Nx, dtype=DTYPE)
dx = float(x[1] - x[0])
dt = 1e-4
Nt_teacher = 2000    # T = 0.2
meta_window = 400    # 早期窗口 [0, 0.04]

print(f"[Grid] Nx={Nx}, dx={dx:.6f}, dt={dt:.1e}, Nt={Nt_teacher}, T={Nt_teacher*dt:.3f}")

def grad_x_centered(U, dx):
    """
    中心差分梯度，Dirichlet 边界 0
    U: [Nx]
    返回: [Nx]
    """
    G = torch.zeros_like(U)
    G[1:-1] = (U[2:] - U[:-2]) / (2.0 * dx)
    # 边界默认 0
    return G

def laplace_x_dirichlet(U, dx):
    """
    二阶中心差分 Laplacian，Dirichlet 边界 0
    U: [Nx]
    返回: [Nx]
    """
    L = torch.zeros_like(U)
    L[1:-1] = (U[2:] - 2.0 * U[1:-1] + U[:-2]) / (dx * dx)
    # 边界默认 0
    return L

# -------------------------
# 4. SCCT 结构统计 (标量场版)
# -------------------------
def scct_stats_torch(u):
    """
    u: [Nx]
    返回:
      - phi2: <u^2>
      - H   : histogram entropy of |u|
    """
    u_flat = u.view(-1)
    phi2 = torch.mean(u_flat * u_flat)

    eps = 1e-12
    absu = torch.abs(u_flat)
    vmax = torch.max(absu)

    if vmax < eps:
        return phi2, torch.tensor(0.0, device=u.device, dtype=u.dtype)

    normed = absu / vmax
    hist = torch.histc(normed, bins=64, min=0.0, max=1.0)
    probs = hist / (torch.sum(hist) + eps)
    H = -torch.sum(probs * torch.log(probs + eps))
    return phi2, H

# -------------------------
# 5. Teacher & Core PDE 定义（标量）
#    Teacher: 0.82 u_xx + 0.5 u - u^3
#    Core   : 0.80 u_xx + 0.5 u - u^3
#    Hidden : +0.02 u_xx
# -------------------------
def teacher_rhs(u):
    u_xx = laplace_x_dirichlet(u, dx)
    return 0.82 * u_xx + 0.5 * u - u**3

def core_rhs(u):
    u_xx = laplace_x_dirichlet(u, dx)
    return 0.80 * u_xx + 0.5 * u - u**3

def teacher_step(u):
    u_new = u + dt * teacher_rhs(u)
    u_new[0]  = 0.0
    u_new[-1] = 0.0
    return u_new

# -------------------------
# 6. 构造 Teacher 轨道 & SCCT
# -------------------------
# 初始化：中等幅度 + 小噪声（可视为 4.4-A 的标量简化版）
u0_np = 0.6 * np.sin(np.pi * x.numpy()) + 0.2 * np.sin(2.0 * np.pi * x.numpy()) \
        + 0.05 * np.random.randn(Nx)
u0 = torch.tensor(u0_np, dtype=DTYPE)
u0 = to_device(u0)

print("[Teacher] u0 shape:", u0.shape, "device:", u0.device)

def simulate_teacher(u0, Nt):
    u = u0.clone()
    traj = torch.zeros((Nt, Nx), dtype=DTYPE, device=device)
    phi2_list = torch.zeros(Nt, dtype=DTYPE, device=device)
    H_list    = torch.zeros(Nt, dtype=DTYPE, device=device)
    for n in range(Nt):
        u = teacher_step(u)
        traj[n] = u
        phi2, H = scct_stats_torch(u)
        phi2_list[n] = phi2
        H_list[n]    = H
    return traj, phi2_list, H_list

print("[Teacher] Simulating teacher trajectory...")
teacher_traj, phi2_true_traj, H_true_traj = simulate_teacher(u0, Nt_teacher)

u_T_true    = teacher_traj[-1]                  # [Nx]
phi2_true_T = phi2_true_traj[-1]
H_true_T    = H_true_traj[-1]
phi2_true_meta_mean = phi2_true_traj[:meta_window].mean()

print(f"[Teacher] Final Φ²(T)   = {phi2_true_T.item():.3e}, H(T) = {H_true_T.item():.3f}")
print(f"[Teacher] <Φ²>_meta     = {phi2_true_meta_mean.item():.3e}\n")

# -------------------------
# 7. Grammar 5.0: Grammar terms for scalar u
# -------------------------
GRAMMAR_TERMS = [
    "u",       # 0
    "u_x",     # 1
    "u_xx",    # 2 (target hidden 0.02)
    "u^2",     # 3
    "u^3",     # 4
    "u*u_x",   # 5
    "u*u_xx",  # 6
    "u_x^2",   # 7
    "u_x*u_xx" # 8
]
n_terms = len(GRAMMAR_TERMS)
print(f"[GrammarTree 5.0] n_terms = {n_terms}, terms = {GRAMMAR_TERMS}\n")

def grammar_features_scalar(u):
    """
    u: [Nx] -> feats: [Nx, n_terms]
    """
    u_x  = grad_x_centered(u, dx)
    u_xx = laplace_x_dirichlet(u, dx)

    t0 = u
    t1 = u_x
    t2 = u_xx
    t3 = u**2
    t4 = u**3
    t5 = u * u_x
    t6 = u * u_xx
    t7 = u_x * u_x
    t8 = u_x * u_xx

    feats = torch.stack(
        [t0, t1, t2, t3, t4, t5, t6, t7, t8],
        dim=-1
    )  # [Nx, n_terms]
    return feats

# -------------------------
# 8. GrammarTree 5.0 PDE 模型
# -------------------------
class GrammarTree50PDE(nn.Module):
    def __init__(self, mask=None):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_terms, dtype=DTYPE))
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=DTYPE))
        if mask is None:
            mask = torch.ones(n_terms, dtype=DTYPE)
        self.register_buffer("mask", mask)

    def residual(self, u):
        feats = grammar_features_scalar(u)    # [Nx, n_terms]
        eff_w = self.gamma * self.w * self.mask
        r = feats @ eff_w
        return r

    def pde_rhs(self, u):
        core = core_rhs(u)
        r = self.residual(u)
        return core + r

    def step(self, u):
        u_new = u + dt * self.pde_rhs(u)
        u_new[0]  = 0.0
        u_new[-1] = 0.0
        return u_new

    def simulate_final(self, u0, Nt):
        u = u0.clone()
        for _ in range(Nt):
            u = self.step(u)
        phi2, H = scct_stats_torch(u)
        return u, phi2, H

    def simulate_traj_phi2(self, u0, Nt):
        u = u0.clone()
        phi2s = []
        for _ in range(Nt):
            u = self.step(u)
            phi2, _ = scct_stats_torch(u)
            phi2s.append(phi2)
        return torch.stack(phi2s)

# -------------------------
# 9. 轨道误差评估工具
# -------------------------
def rollout_and_error(model, steps):
    u_teacher = u0.clone()
    u_model   = u0.clone()
    errs = []
    with torch.no_grad():
        for _ in range(steps):
            u_teacher = teacher_step(u_teacher)
            u_model   = model.step(u_model)
            errs.append(torch.mean((u_teacher - u_model)**2).item())
    return np.array(errs)

# -------------------------
# 10. Stage 1: 动态 SCCT 训练 (Dense)
# -------------------------
def train_stage1(epochs=150):
    model = GrammarTree50PDE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    # base 超参数
    lambda_scct_base  = 5e-2
    lambda_meta_base  = 1e-2
    lambda_gamma = 1e-5
    lambda_w     = 1e-5

    print("[Stage 1] Training GrammarTree 5.0 (dense, dynamic SCCT)...")
    for ep in range(1, epochs + 1):
        opt.zero_grad()

        # 动态权重：前期更重视 misfit，后期逐渐抬高 SCCT/meta 比例
        t = ep / epochs
        lambda_scct = lambda_scct_base * (0.5 + 0.5 * t)
        lambda_meta = lambda_meta_base * (0.5 + 0.5 * t)

        # 终时刻 & SCCT
        u_pred_T, phi2_pred_T, H_pred_T = model.simulate_final(u0, Nt_teacher)
        misfit = F.mse_loss(u_pred_T, u_T_true)

        loss_scct_T = torch.abs(phi2_pred_T - phi2_true_T) \
                    + torch.abs(H_pred_T    - H_true_T)

        # 早期窗口 Φ²
        phi2_t = model.simulate_traj_phi2(u0, meta_window)
        loss_meta = (phi2_t.mean() - phi2_true_meta_mean)**2

        g = model.gamma
        w = model.w

        loss = misfit \
             + lambda_scct * loss_scct_T \
             + lambda_meta * loss_meta \
             + lambda_gamma * torch.sum(torch.abs(g)) \
             + lambda_w     * torch.sum(torch.abs(w))

        loss.backward()
        opt.step()

    # 训练结束 summary
    with torch.no_grad():
        u_pred_T, phi2_pred_T, H_pred_T = model.simulate_final(u0, Nt_teacher)
        misfit_final = F.mse_loss(u_pred_T, u_T_true).item()
        phi2_meta_model = model.simulate_traj_phi2(u0, meta_window).mean().item()
        eff = (model.gamma * model.w * model.mask).detach().cpu().numpy()

    print("[Stage 1][Summary] misfit≈{:.3e}, Φ²(T)={:.3e}, H(T)={:.3f}".format(
        misfit_final, phi2_pred_T.item(), H_pred_T.item()))
    print("  <Φ²>_meta(model) = {:.3e}".format(phi2_meta_model))
    print("  γ(Stage 1) = {:+.3e}".format(model.gamma.item()))
    print("  eff(u_xx idx 2) = {:+.6e}".format(eff[2]))
    print()

    return model

model_stage1 = train_stage1()

# -------------------------
# 11. Stage 2: 自适应剪枝 + 细化
# -------------------------
def adaptive_pruning_mask(model, floor_threshold=5e-4, factor=0.4):
    with torch.no_grad():
        eff = (model.gamma * model.w * model.mask).detach().cpu().numpy()
        abs_eff = np.abs(eff)
        med = np.median(abs_eff)
        if med < floor_threshold:
            tau = floor_threshold
        else:
            tau = max(floor_threshold, factor * med)

        keep = abs_eff >= tau
    return torch.tensor(keep.astype(np.float32)), tau, eff

mask_stage2, tau_prune, eff_stage1 = adaptive_pruning_mask(model_stage1)
print("[Stage 2] Adaptive pruning threshold τ = {:.3e}".format(tau_prune))
kept_indices = np.where(mask_stage2.cpu().numpy() > 0.5)[0].tolist()
kept_terms = [GRAMMAR_TERMS[i] for i in kept_indices]
lambda_k = len(kept_indices) / n_terms
print("  kept indices =", kept_indices)
print("  kept terms   =", kept_terms)
print("  λ_k = {:.3f} ({}/{})".format(lambda_k, len(kept_indices), n_terms))
print()

def train_stage2(mask, epochs=80):
    model = GrammarTree50PDE(mask=mask).to(device)
    # 初始化参数为 Stage1 的结果（便于收敛）
    with torch.no_grad():
        model.w.copy_(model_stage1.w.detach())
        model.gamma.copy_(model_stage1.gamma.detach())

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    lambda_scct = 5e-2
    lambda_meta = 1e-2
    lambda_gamma = 1e-5
    lambda_w     = 1e-5

    print("[Stage 2] Refining pruned GrammarTree 5.0 ...")
    for ep in range(1, epochs + 1):
        opt.zero_grad()

        u_pred_T, phi2_pred_T, H_pred_T = model.simulate_final(u0, Nt_teacher)
        misfit = F.mse_loss(u_pred_T, u_T_true)

        loss_scct_T = torch.abs(phi2_pred_T - phi2_true_T) \
                    + torch.abs(H_pred_T    - H_true_T)

        phi2_t = model.simulate_traj_phi2(u0, meta_window)
        loss_meta = (phi2_t.mean() - phi2_true_meta_mean)**2

        g = model.gamma
        w = model.w

        loss = misfit \
             + lambda_scct * loss_scct_T \
             + lambda_meta * loss_meta \
             + lambda_gamma * torch.sum(torch.abs(g)) \
             + lambda_w     * torch.sum(torch.abs(w * model.mask))

        loss.backward()
        opt.step()

    # Summary
    with torch.no_grad():
        u_pred_T, phi2_pred_T, H_pred_T = model.simulate_final(u0, Nt_teacher)
        misfit_final = F.mse_loss(u_pred_T, u_T_true).item()
        phi2_meta_model = model.simulate_traj_phi2(u0, meta_window).mean().item()
        eff = (model.gamma * model.w * model.mask).detach().cpu().numpy()

    print("[Stage 2][Summary] misfit≈{:.3e}, Φ²(T)={:.3e}, H(T)={:.3f}".format(
        misfit_final, phi2_pred_T.item(), H_pred_T.item()))
    print("  <Φ²>_meta(model) = {:.3e}".format(phi2_meta_model))
    print("  γ(Stage 2) = {:+.3e}".format(model.gamma.item()))
    print("  eff(u_xx idx 2) = {:+.6e}".format(eff[2]))
    print("  target hidden: +2.000000e-02 * u_xx")
    abs_err = abs(eff[2] - 0.02)
    rel_err = abs_err / 0.02 * 100.0
    print("  |Δc| = {:.3e}, relative error = {:.3f} %".format(abs_err, rel_err))
    print()
    return model, eff

model_stage2, eff_stage2 = train_stage2(mask_stage2)

# -------------------------
# 12. 数值自检：轨道 + 结构
# -------------------------
steps_err = 400
err_50 = rollout_and_error(model_stage2, steps_err)
t_err = np.arange(steps_err) * dt

with torch.no_grad():
    phi2_50 = model_stage2.simulate_traj_phi2(u0, steps_err)

phi2_teacher_early = phi2_true_traj[:steps_err]

print("[Eval] Mean L2 orbit error over [0,{:.3f}] = {:.3e}".format(
    steps_err*dt, err_50.mean()))
print("[Eval] Early-time Φ² teacher={:.3e}, GT5.0={:.3e}, ratio≈{:.3f}".format(
    phi2_teacher_early.mean().item(),
    phi2_50.mean().item(),
    phi2_50.mean().item() / phi2_teacher_early.mean().item()))
print()

print("[GrammarTree 5.0] Effective hidden residual (u-equation):")
print("  target:  +2.000000e-02 * u_xx")
print("  Stage1:  {:+.6e} * u_xx".format(eff_stage1[2]))
print("  Stage2:  {:+.6e} * u_xx".format(eff_stage2[2]))
print()

# -------------------------
# 13. Stage 3: PySR 符号回归审计 (residual)
# -------------------------
print("[Stage 3] Building residual dataset for PySR (u-equation)...")

# 采样前 N_time_samp 个时间步 & 空间内点
N_time_samp = 400
time_indices = range(N_time_samp)

X_list = []
R_list = []

with torch.no_grad():
    for n in time_indices:
        u_t = teacher_traj[n]       # [Nx]
        u_t = u_t.to(device)

        u_x  = grad_x_centered(u_t, dx)
        u_xx = laplace_x_dirichlet(u_t, dx)

        rhs_teacher = teacher_rhs(u_t)
        rhs_core    = core_rhs(u_t)
        resid = rhs_teacher - rhs_core   # 完全来自 hidden term (理论上 ~0.02 u_xx)

        # 只用内部点
        for i in range(1, Nx-1):
            X_list.append([
                float(u_t[i].item()),
                float(u_x[i].item()),
                float(u_xx[i].item())
            ])
            R_list.append(float(resid[i].item()))

X = np.array(X_list, dtype=np.float64)
R = np.array(R_list, dtype=np.float64)
print(f"[Stage 3] Dataset shapes: X={X.shape}, R={R.shape}")

print("\n[Stage 3] Running PySR symbolic regression (residual ≈ f(u, u_x, u_xx))...")

model_pysr = PySRRegressor(
    niterations=1200,
    populations=20,
    population_size=50,
    binary_operators=["+", "-", "*"],
    unary_operators=[],
    maxsize=15,
    model_selection="best",
    elementwise_loss="L2DistLoss()",
    parallelism="serial",
    random_state=0,
)

model_pysr.fit(X, R, variable_names=["u", "u_x", "u_xx"])

eqs = model_pysr.equations_
best_row = eqs.sort_values("score", ascending=False).iloc[0]
row_idx = int(best_row.name)

print("\n[Stage 3] PySR best equation:")
print("  equation  :", best_row["equation"])
print(f"  loss      = {best_row['loss']:.3e}")
print(f"  complexity= {int(best_row['complexity'])}")

sym_list = model_pysr.sympy()
sym_best = sym_list[row_idx]

x0, x1, x2 = sp.symbols("x0 x1 x2")   # 对应 [u, u_x, u_xx]

coef_uxx = float(sym_best.subs({x0: 0.0, x1: 0.0, x2: 1.0}))
true_coeff = 0.02
abs_err_pysr = abs(coef_uxx - true_coeff)
rel_err_pysr = abs_err_pysr / true_coeff * 100.0

print("\n[Stage 3] Interpreted as residual ≈ c * u_xx")
print(f"  c (PySR)     = {coef_uxx:.9f}")
print(f"  c (target)   = {true_coeff:.9f}")
print(f"  |Δc|         = {abs_err_pysr:.3e}")
print(f"  relative err = {rel_err_pysr:.3f} %")

print("\n[Summary] GrammarTree 5.0 + dynamic SCCT + adaptive pruning + PySR audit finished.")
