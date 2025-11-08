# ============================================================
# L7-SCCT-NGT (Paper Edition, Colab Version)
#   轨道 + 结构(Φ², H, K=Φ²/H) + 律 (PySR 符号回归)
#   Teacher PDE: u_t = 0.82 u_xx + 0.5 u - u^3
#   Core PDE  : u_t = 0.80 u_xx + 0.5 u - u^3
#   → 隐藏残差: r_true = 0.02 u_xx
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pysr import PySRRegressor
from sympy import symbols

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 1. Grid & Initial condition
# -----------------------------
Lx = 1.0
Nx = 64
dx = Lx / (Nx - 1)
dt = 1e-4
Nt_teacher = 2000
T = Nt_teacher * dt

x = torch.linspace(0.0, Lx, Nx, device=device)

# Slight random perturbation (to break symmetry)
torch.manual_seed(0)
u0 = torch.sin(np.pi * x) + 0.05 * torch.randn_like(x)

print(f"[Grid] Nx={Nx}, dx={dx:.3e}, dt={dt:.3e}, Nt={Nt_teacher}, T={T:.3f}")

# -----------------------------
# 2. SCCT stats: Φ² & H
# -----------------------------
def scct_stats_torch(u, num_bins=64):
    """
    Φ² = mean(u^2)
    H  = -∑ p log p, p 来自 |u| 归一化后的直方图
    """
    u_flat = u.view(-1)
    phi2 = torch.mean(u_flat**2)

    v = torch.abs(u_flat)
    vmax = torch.max(v)
    if vmax < 1e-12:
        return phi2, torch.tensor(0.0, device=u.device)

    v_norm = v / vmax
    hist = torch.histc(v_norm, bins=num_bins, min=0.0, max=1.0)
    p = hist / (hist.sum() + 1e-12)
    H = -torch.sum(p * torch.log(p + 1e-12))
    return phi2, H

# -----------------------------
# 3. Teacher PDE
#     u_t = 0.82 u_xx + 0.5 u - u^3
# -----------------------------
def laplace_x_dirichlet(u, dx):
    l = torch.zeros_like(u)
    l[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
    l[0] = 0.0   # Dirichlet boundary, u=0 → u_xx ≈ 0
    l[-1] = 0.0
    return l

def grad_x_centered(u, dx):
    g = torch.zeros_like(u)
    g[1:-1] = (u[2:] - u[:-2]) / (2*dx)
    g[0] = 0.0
    g[-1] = 0.0
    return g

def teacher_rhs(u):
    u_xx = laplace_x_dirichlet(u, dx)
    rhs = 0.82 * u_xx + 0.5 * u - u**3
    return rhs

def teacher_step(u):
    u_new = u + dt * teacher_rhs(u)
    u_new[0] = 0.0
    u_new[-1] = 0.0
    return u_new

def simulate_teacher(u0, Nt):
    u = u0.clone()
    traj, phi2s, Hs = [], [], []
    for _ in range(Nt):
        u = teacher_step(u)
        phi2, H = scct_stats_torch(u)
        traj.append(u.clone())
        phi2s.append(phi2.item())
        Hs.append(H.item())
    return torch.stack(traj), np.array(phi2s), np.array(Hs)

with torch.no_grad():
    teacher_traj, phi2_true_traj, H_true_traj = simulate_teacher(u0, Nt_teacher)
    uT_true = teacher_traj[-1]
    phi2_true_T = phi2_true_traj[-1]
    H_true_T = H_true_traj[-1]

print(f"[Teacher] Φ²_true(T)={phi2_true_T:.3e}, H_true(T)={H_true_T:.3e}")

meta_window = 400
phi2_true_mean_meta = phi2_true_traj[:meta_window].mean()
H_true_mean_meta = H_true_traj[:meta_window].mean()

# -----------------------------
# 4. Neural Grammar Tree
# -----------------------------
class NeuralGrammarTree(nn.Module):
    def __init__(self, feat_dim=3, hidden=16):
        super().__init__()
        self.node1 = nn.Linear(feat_dim, hidden)
        self.node2 = nn.Linear(feat_dim, hidden)
        self.combine = nn.Linear(2*hidden, 1)

    def forward(self, u):
        u_xx = laplace_x_dirichlet(u, dx)
        u_x = grad_x_centered(u, dx)
        feat = torch.stack([u, u_x, u_xx], dim=-1)  # [Nx, 3]
        h1 = torch.tanh(self.node1(feat))
        h2 = torch.tanh(self.node2(feat))
        h = torch.cat([h1, h2], dim=-1)
        return self.combine(h).squeeze(-1)  # [Nx]

# -----------------------------
# 5. SCCT-NGT model
# -----------------------------
class SCCT_NGT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tree = NeuralGrammarTree()
        self.gamma = nn.Parameter(torch.tensor(0.05))  # nonzero init

    def pde_rhs(self, u):
        u_xx = laplace_x_dirichlet(u, dx)
        core = 0.8 * u_xx + 0.5 * u - u**3   # core PDE
        f_ngt = self.tree(u)                # learned residual structure
        return core + self.gamma * f_ngt

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

    def simulate_traj_full(self, u0, Nt):
        u = u0.clone()
        traj = []
        for _ in range(Nt):
            u = self.step(u)
            traj.append(u.clone())
        return torch.stack(traj)

# -----------------------------
# 6. Training (pair/mask/meta)
# -----------------------------
def train_scct_ngt_mode(mode, epochs=80, print_every=10):
    model = SCCT_NGT().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    λ_scct_dict = {"pair": 0.0, "mask": 1e-2, "meta": 5e-2}
    λ_γ = 1e-3
    λ_time = 1e-2

    for ep in range(1, epochs+1):
        opt.zero_grad()
        u_pred_T, φ2_pred, H_pred = model.simulate_final(u0, Nt_teacher)
        misfit = F.mse_loss(u_pred_T, uT_true)
        loss = misfit

        λ_scct = λ_scct_dict[mode]

        # 终时刻结构约束
        if λ_scct > 0:
            loss_scct_T = torch.abs(φ2_pred - phi2_true_T) + torch.abs(H_pred - H_true_T)
            loss = loss + λ_scct * loss_scct_T

        # 全时窗结构约束（meta 模式）
        if mode == "meta":
            φ2_t, H_t = model.simulate_traj_scct(u0, meta_window)
            loss_time = (φ2_t.mean() - phi2_true_mean_meta)**2 + (H_t.mean() - H_true_mean_meta)**2
            loss = loss + λ_time * loss_time

        # γ 正则：鼓励残差强度不要乱飙
        loss = loss + λ_γ * torch.abs(model.gamma)

        loss.backward()
        opt.step()

        if ep % print_every == 0 or ep == 1:
            print(
                f"[{mode:5s}] Epoch {ep:03d} | "
                f"loss={loss.item():.3e}, misfit={misfit.item():.3e}, "
                f"Φ²(T)={φ2_pred.item():.3e}, H(T)={H_pred.item():.3e}, "
                f"γ={model.gamma.item():+.3e}"
            )
    return model

print("\n[Training L7-SCCT-NGT modes...]")
model_pair = train_scct_ngt_mode("pair")
model_mask = train_scct_ngt_mode("mask")
model_meta = train_scct_ngt_mode("meta")

# -----------------------------
# 7. L7 Visualizer: 结构 (Φ², H, K=Φ²/H)
# -----------------------------
def run_dynamics(model, steps=400):
    u = u0.clone()
    φ2s, Hs, Ks = [], [], []
    with torch.no_grad():
        for _ in range(steps):
            u = model.step(u)
            φ2, H = scct_stats_torch(u)
            φ2s.append(φ2.item())
            Hs.append(H.item())
            Ks.append(φ2.item() / (H.item() + 1e-12))
    return np.array(φ2s), np.array(Hs), np.array(Ks)

print("\n[Running L7 Visualizer simulations...]")
steps_vis = 400
results = {}
for name, model in zip(["pair", "mask", "meta"], [model_pair, model_mask, model_meta]):
    φ2s, Hs, Ks = run_dynamics(model, steps_vis)
    results[name] = (φ2s, Hs, Ks)
    print(f"  [{name}] Φ²_mean={φ2s.mean():.3e}, H_mean={Hs.mean():.3e}, K_mean≈{Ks.mean():.3e}")

t_vis = np.arange(steps_vis) * dt

plt.figure(figsize=(14,4))
for i, (title, idx) in enumerate(zip([r"$\Phi^2(t)$", "H(t)", r"$K(t)=\Phi^2/H$"], [0, 1, 2])):
    plt.subplot(1, 3, i+1)
    for md, ls in zip(["pair", "mask", "meta"], ["-", "--", ":"]):
        plt.plot(t_vis, results[md][idx], ls, label=md)
    plt.title(title)
    plt.xlabel("t")
    plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 8. 轨道：Teacher vs L7 误差
# -----------------------------
def rollout_and_error(model, steps):
    u_teacher = u0.clone()
    u_model = u0.clone()
    errs = []
    with torch.no_grad():
        for _ in range(steps):
            u_teacher = teacher_step(u_teacher)
            u_model = model.step(u_model)
            errs.append(torch.mean((u_teacher - u_model)**2).item())
    return np.array(errs)

steps_err = 400
err_pair = rollout_and_error(model_pair, steps_err)
err_mask = rollout_and_error(model_mask, steps_err)
err_meta = rollout_and_error(model_meta, steps_err)
t_err = np.arange(steps_err) * dt

plt.figure(figsize=(6,4))
plt.semilogy(t_err, err_pair, label="pair")
plt.semilogy(t_err, err_mask, label="mask")
plt.semilogy(t_err, err_meta, label="meta")
plt.xlabel("t")
plt.ylabel(r"Mean $L^2$ error")
plt.title("Orbit misfit vs Teacher")
plt.legend()
plt.tight_layout()
plt.show()

print(f"\n[Mean L2 orbit error over [0,{steps_err*dt:.3f}]]:")
for nm, e in zip(["pair", "mask", "meta"], [err_pair, err_mask, err_meta]):
    print(f"  {nm:5s}: {e.mean():.3e}")

# -----------------------------
# 9. 律：Teacher & L7(meta) 残差，用 PySR 拟合
# -----------------------------
print("\n[Law Discovery] Building residual datasets for Teacher & L7(meta)...")

# (a) Teacher 残差数据集
with torch.no_grad():
    Nt_samples = min(Nt_teacher, 200)
    us = teacher_traj[:Nt_samples].to(device)   # [Nt, Nx]
    X_list, y_list = [], []
    for u in us:
        u_x = grad_x_centered(u, dx)
        u_xx = laplace_x_dirichlet(u, dx)
        # 真正的残差 = (0.82 - 0.8) * u_xx = 0.02 u_xx
        r = (0.82 - 0.8) * u_xx
        X_list.append(torch.stack([u, u_x, u_xx], dim=-1))
        y_list.append(r)
    X_true = torch.cat(X_list, dim=0).cpu().numpy()
    y_true = torch.cat(y_list, dim=0).cpu().numpy()

print(f"[Teacher Residual Dataset] X: {X_true.shape}, y: {y_true.shape}")

model_true = PySRRegressor(
    niterations=2400,
    binary_operators=["+", "-", "*"],
    unary_operators=[],
    populations=15,
    progress=True,
    maxsize=10,
    loss="loss(x, y) = (x - y)^2",
)
model_true.fit(X_true, y_true)

print("\n[Teacher Residual - PySR Model]")
print(model_true)
print(model_true.equations_)

u_sym, ux_sym, uxx_sym = symbols("u u_x u_xx")
best_idx_true = int(model_true.equations_["loss"].idxmin())
expr_true = model_true.sympy()[best_idx_true]
expr_true = expr_true.subs({
    symbols("x0"): u_sym,
    symbols("x1"): ux_sym,
    symbols("x2"): uxx_sym,
})

print("\n[True Residual Law (SymPy)]")
print("r_true(u, u_x, u_xx) ≈", expr_true)

print("\n[True Residual Law LaTeX]")
print(model_true.latex()[best_idx_true])

# (b) L7(meta) 残差数据集：r_L7 = γ * f_ngt(u, u_x, u_xx)
with torch.no_grad():
    Nt_samples_meta = Nt_samples
    u_meta_traj = model_meta.simulate_traj_full(u0, Nt_samples_meta)  # [Nt, Nx]
    X_list, y_list = [], []
    for u in u_meta_traj:
        u_x = grad_x_centered(u, dx)
        u_xx = laplace_x_dirichlet(u, dx)
        f_ngt_val = model_meta.tree(u)
        r_l7 = model_meta.gamma.detach() * f_ngt_val
        X_list.append(torch.stack([u, u_x, u_xx], dim=-1))
        y_list.append(r_l7)
    X_l7 = torch.cat(X_list, dim=0).cpu().numpy()
    y_l7 = torch.cat(y_list, dim=0).cpu().numpy()

print(f"\n[L7(meta) Residual Dataset] X: {X_l7.shape}, y: {y_l7.shape}")
print(f"[L7(meta)] γ ≈ {model_meta.gamma.item():.3e}")

model_l7 = PySRRegressor(
    niterations=2400,
    binary_operators=["+", "-", "*"],
    unary_operators=[],
    populations=15,
    progress=True,
    maxsize=10,
    loss="loss(x, y) = (x - y)^2",
)
model_l7.fit(X_l7, y_l7)

print("\n[L7(meta) Residual - PySR Model]")
print(model_l7)
print(model_l7.equations_)

best_idx_l7 = int(model_l7.equations_["loss"].idxmin())
expr_l7 = model_l7.sympy()[best_idx_l7]
expr_l7 = expr_l7.subs({
    symbols("x0"): u_sym,
    symbols("x1"): ux_sym,
    symbols("x2"): uxx_sym,
})

print("\n[L7 Learned Residual Law (SymPy)]")
print("r_L7(u, u_x, u_xx) ≈", expr_l7)

print("\n[L7 Learned Residual Law LaTeX]")
print(model_l7.latex()[best_idx_l7])

print("\nDone. 现在你有：")
print("  - 轨道：Teacher vs L7 rollout + 误差（log 轴）")
print("  - 结构：Φ²(t), H(t), K(t)=Φ²/H 三种模式对比")
print("  - 律：Teacher 残差律 & L7(meta) 残差律的符号表达式（SymPy + LaTeX）")

