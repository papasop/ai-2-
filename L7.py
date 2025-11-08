#@title L7 Auto-PDE 一键版：轨道 + 律 + 结构

# ============================================================
# 0. 安装 PySR（符号回归）并优先导入，避免 torch 先导入的警告
# ============================================================
!pip install pysr sympy --quiet

from pysr import PySRRegressor
import sympy as sp

# ============================================================
# 1. 常用库
# ============================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 2. 网格 + 真实 Teacher PDE（隐藏 u_x 项）
# ============================================================
Lx = 1.0
Nx = 64
dx = Lx / (Nx - 1)
dt = 1e-4
Nt_teacher = 3000
T = Nt_teacher * dt
x = torch.linspace(0, Lx, Nx, device=device)

torch.manual_seed(1)
u0 = torch.sin(2 * np.pi * x) + 0.05 * torch.randn_like(x)

def laplace_x_dirichlet(u, dx):
    l = torch.zeros_like(u)
    l[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
    return l

def teacher_rhs(u):
    """真实 PDE 的 RHS（包含隐藏 u_x 项）"""
    u_xx = laplace_x_dirichlet(u, dx)
    u_x = torch.zeros_like(u)
    u_x[1:-1] = (u[2:] - u[:-2]) / (2*dx)
    rhs = 0.85 * u_xx + 0.4 * u - u**3 - 0.15 * u_x  # 真正 PDE
    return rhs, u_x, u_xx

def real_teacher_step(u):
    rhs, _, _ = teacher_rhs(u)
    u_new = u + dt * rhs
    u_new[0] = u_new[-1] = 0.0
    return u_new

# 生成 Teacher 轨道（用于终态对比 + PySR 数据）
u = u0.clone()
teacher_traj = []
for _ in range(Nt_teacher):
    teacher_traj.append(u.clone())
    u = real_teacher_step(u)
uT_true = u.clone()
teacher_traj = torch.stack(teacher_traj, dim=0)  # [Nt_teacher, Nx]
print(f"[Real Teacher] Generated u(T={T:.3f})")

# ============================================================
# 3. 结构量：SCCT φ² & H
# ============================================================
def scct_stats_torch(u, num_bins=64):
    """φ² = mean(u²), H = histogram 熵"""
    u_flat = u.view(-1)
    phi2 = torch.mean(u_flat**2)
    v = torch.abs(u_flat)
    vmax = torch.max(v)
    if vmax < 1e-12:
        return phi2, torch.tensor(0.0, device=device)
    v_norm = v / vmax
    hist = torch.histc(v_norm, bins=num_bins, min=0.0, max=1.0)
    p = hist / (hist.sum() + 1e-12)
    H = -torch.sum(p * torch.log(p + 1e-12))
    return phi2, H

φ2_true_T, H_true_T = scct_stats_torch(uT_true)
print(f"[Teacher end] φ2_true={φ2_true_T:.3e}, H_true={H_true_T:.3f}")

# ============================================================
# 4. L7 模型：core + γ·f_ngt(u, u_x, u_xx)
# ============================================================
class NeuralGrammarTree(nn.Module):
    def __init__(self, feat_dim=3, hidden=16):
        super().__init__()
        self.node1 = nn.Linear(feat_dim, hidden)
        self.node2 = nn.Linear(feat_dim, hidden)
        self.combine = nn.Linear(2*hidden, 1)

    def forward(self, u):
        # 构造特征 [u, u_x, u_xx]
        u_xx = laplace_x_dirichlet(u, dx)
        u_x = torch.zeros_like(u)
        u_x[1:-1] = (u[2:] - u[:-2]) / (2*dx)
        feat = torch.stack([u, u_x, u_xx], dim=-1)  # [Nx, 3]
        h1 = torch.tanh(self.node1(feat))
        h2 = torch.tanh(self.node2(feat))
        h = torch.cat([h1, h2], dim=-1)
        return self.combine(h).squeeze(-1)  # [Nx]

class SCCT_NGT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tree = NeuralGrammarTree()
        self.gamma = nn.Parameter(torch.tensor(0.05))  # 可学习 γ

    def pde_rhs(self, u):
        u_xx = laplace_x_dirichlet(u, dx)
        core = 0.8 * u_xx + 0.4 * u - u**3  # 已知 core
        return core + self.gamma * self.tree(u)

    def step(self, u):
        u_new = u + dt * self.pde_rhs(u)
        u_new[0] = u_new[-1] = 0.0
        return u_new

    def simulate_final(self, u0, Nt):
        u = u0.clone()
        for _ in range(Nt):
            u = self.step(u)
        phi2, H = scct_stats_torch(u)
        return u, phi2, H

# ============================================================
# 5. 三模式训练：pair / mask / meta
# ============================================================
def train_mode(mode, epochs=100):
    model = SCCT_NGT().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    λ_scct = {"pair": 0.0, "mask": 1e-2, "meta": 5e-2}[mode]

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        u_pred, φ2, H = model.simulate_final(u0, Nt_teacher)
        loss = F.mse_loss(u_pred, uT_true)
        # 结构约束（只对 mask/meta 为正）
        if λ_scct > 0:
            loss += λ_scct * (torch.abs(φ2 - φ2_true_T) + torch.abs(H - H_true_T))
        # γ 正则，鼓励找到“必要的”残差律
        loss += 1e-3 * torch.abs(model.gamma)
        loss.backward()
        opt.step()

        if ep % 30 == 0 or ep <= 3:
            print(f"[{mode}] Ep {ep:3d} | loss={loss.item():.2e} | γ={model.gamma.item():+.3f}")
    return model

print("\n[Training L7 modes...]")
model_pair = train_mode("pair", 80)
model_mask = train_mode("mask", 80)
model_meta = train_mode("meta", 120)

# ============================================================
# 6. 训练结果（meta）+ 结构终态
# ============================================================
with torch.no_grad():
    u_final, φ2_final, H_final = model_meta.simulate_final(u0, Nt_teacher)
    misfit = F.mse_loss(u_final, uT_true).item()

print(f"\n[L7 Final/meta] misfit={misfit:.2e}, "
      f"φ2={φ2_final:.3e}, H={H_final:.3f}, γ={model_meta.gamma.item():+.3f}")
print(f"[Teacher  end ] φ2_true={φ2_true_T:.3e}, H_true={H_true_T:.3f}")

print("\n[Learned f_ngt weights (meta) - parameter means]")
for name, p in model_meta.tree.named_parameters():
    print(f"  {name:20s}: mean={p.data.mean():+.3e}")

# ============================================================
# 7. 结构视角 & 轨道：Teacher vs meta-L7 rollout
# ============================================================
def rollout_teacher(u0, steps):
    u = u0.clone()
    traj = []
    φ2s, Hs = [], []
    with torch.no_grad():
        for _ in range(steps):
            traj.append(u.detach().cpu().numpy())
            φ2, H = scct_stats_torch(u)
            φ2s.append(φ2.item())
            Hs.append(H.item())
            u = real_teacher_step(u)
    return np.array(traj), np.array(φ2s), np.array(Hs)

def rollout_model(model, u0, steps):
    u = u0.clone()
    traj = []
    φ2s, Hs = [], []
    with torch.no_grad():
        for _ in range(steps):
            traj.append(u.detach().cpu().numpy())
            φ2, H = scct_stats_torch(u)
            φ2s.append(φ2.item())
            Hs.append(H.item())
            u = model.step(u)
    return np.array(traj), np.array(φ2s), np.array(Hs)

Nt_vis = 1000
print("\n[Visualizer] Running rollout: Teacher vs meta-L7 ...")
teacher_traj_vis, φ2_teacher, H_teacher = rollout_teacher(u0, Nt_vis)
meta_traj_vis,    φ2_meta_v, H_meta_v   = rollout_model(model_meta, u0, Nt_vis)

err_t = np.mean((meta_traj_vis - teacher_traj_vis)**2, axis=1)
t_vis = np.arange(Nt_vis) * dt
K_t = φ2_meta_v / (φ2_teacher + 1e-12)

print(f"[Visualizer/meta] mean L2 error over time = {err_t.mean():.2e}")
print(f"[Visualizer/meta] final L2 error          = {err_t[-1]:.2e}")
print(f"[Visualizer/meta] K(t) mean               = {K_t.mean():.3f}")

# ---- 图 1: Φ²(t) 对比 ----
plt.figure(figsize=(10,4))
plt.plot(t_vis, φ2_teacher, 'k',  label='Teacher Φ²(t)')
plt.plot(t_vis, φ2_meta_v, 'b--', label='meta Φ²(t)')
plt.xlabel("t"); plt.ylabel("Φ²")
plt.title("Teacher vs meta-L7: Φ²(t) rollout")
plt.legend(); plt.tight_layout(); plt.show()

# ---- 图 2: H(t) 对比 ----
plt.figure(figsize=(10,4))
plt.plot(t_vis, H_teacher, 'k',  label='Teacher H(t)')
plt.plot(t_vis, H_meta_v, 'b--', label='meta H(t)')
plt.xlabel("t"); plt.ylabel("H")
plt.title("Teacher vs meta-L7: H(t) rollout")
plt.legend(); plt.tight_layout(); plt.show()

# ---- 图 3: K(t) & L2 error(t) ----
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(t_vis, K_t, 'g')
plt.axhline(1.0, color='k', linestyle='--')
plt.xlabel("t"); plt.ylabel("K(t)=Φ²_L7/Φ²_Teacher")
plt.title("K(t) structure ratio")

plt.subplot(1,2,2)
plt.plot(t_vis, err_t, 'r')
plt.xlabel("t"); plt.ylabel("L2 error")
plt.title("L2(u_L7 - u_teacher)^2 over time")
plt.tight_layout(); plt.show()

# ---- 图 4: 若干时间截面 u(x,t) 对比 ----
snap_ids = [0, Nt_vis//3, 2*Nt_vis//3, Nt_vis-1]
plt.figure(figsize=(12,8))
xx = x.cpu().numpy()
for i, sid in enumerate(snap_ids, 1):
    plt.subplot(2, 2, i)
    plt.plot(xx, teacher_traj_vis[sid], 'k',  label='Teacher')
    plt.plot(xx, meta_traj_vis[sid],    'b--', label='meta')
    plt.title(f"u(x, t={sid*dt:.3f})")
    if i == 1:
        plt.legend()
plt.tight_layout(); plt.show()

# ============================================================
# 8. 律 Part 1：Teacher 残差律（true_rhs - core）
# ============================================================
print("\n[Law Discovery] Building dataset from Teacher (residual: true_rhs - core) ...")

Nt_data = min(800, Nt_teacher)  # 只取前面一段，避免全躺平
us = teacher_traj[:Nt_data]     # [Nt_data, Nx]

feat_list = []
resid_list = []

with torch.no_grad():
    for u in us:
        u = u.to(device)
        rhs_true, u_x, u_xx = teacher_rhs(u)
        core = 0.8 * u_xx + 0.4 * u - u**3
        resid = rhs_true - core  # 真残差 = 0.05*u_xx - 0.15*u_x

        feat = torch.stack([u, u_x, u_xx], dim=-1)  # [Nx, 3]
        feat_list.append(feat.view(-1, 3))
        resid_list.append(resid.view(-1))

features_true = torch.cat(feat_list, dim=0).cpu().numpy()   # [N,3]
residual_true = torch.cat(resid_list, dim=0).cpu().numpy()  # [N]
print("Teacher residual dataset:", features_true.shape, residual_true.shape)

variable_names = ["u", "u_x", "u_xx"]

true_resid_model = PySRRegressor(
    niterations=120,
    binary_operators=["+", "-", "*"],
    unary_operators=[],
    maxsize=10,
    populations=20,
    procs=0,
    model_selection="best",
    progress=True,
)

true_resid_model.fit(features_true, residual_true, variable_names=variable_names)

print("\n[True Residual Law from Teacher (SymPy)]")
true_resid_sym = true_resid_model.sympy()
print("residual_true(u, u_x, u_xx) ≈", true_resid_sym)
print("\n[True Residual Law LaTeX]")
print(true_resid_model.latex())

# ============================================================
# 9. 律 Part 2：L7 meta 学到的残差律（γ · f_ngt）
# ============================================================
print("\n[Law Discovery] Building dataset from L7 meta (γ * f_ngt) ...")

feat_list_ngt = []
ngt_list = []

with torch.no_grad():
    for u in us:
        u = u.to(device)
        f_ngt = model_meta.gamma * model_meta.tree(u)  # [Nx]

        u_xx = laplace_x_dirichlet(u, dx)
        u_x = torch.zeros_like(u)
        u_x[1:-1] = (u[2:] - u[:-2]) / (2*dx)
        feat = torch.stack([u, u_x, u_xx], dim=-1)

        feat_list_ngt.append(feat.view(-1, 3))
        ngt_list.append(f_ngt.view(-1))

features_ngt = torch.cat(feat_list_ngt, dim=0).cpu().numpy()
rhs_ngt = torch.cat(ngt_list, dim=0).cpu().numpy()
print("L7 residual dataset:", features_ngt.shape, rhs_ngt.shape)

ngt_resid_model = PySRRegressor(
    niterations=120,
    binary_operators=["+", "-", "*"],
    unary_operators=[],
    maxsize=10,
    populations=20,
    procs=0,
    model_selection="best",
    progress=True,
)

ngt_resid_model.fit(features_ngt, rhs_ngt, variable_names=variable_names)

print("\n[L7 Learned Residual Law (SymPy)]")
ngt_resid_sym = ngt_resid_model.sympy()
print("residual_L7(u, u_x, u_xx) ≈", ngt_resid_sym)
print("\n[L7 Learned Residual Law LaTeX]")
print(ngt_resid_model.latex())

print("\nDone. 现在你有：")
print("  - 轨道：Teacher vs meta-L7 rollout + 误差 + 截面图")
print("  - 结构：Φ²(t), H(t), K(t) 对比")
print("  - 律：Teacher 残差律 & L7 学到的残差律（符号形式）")
