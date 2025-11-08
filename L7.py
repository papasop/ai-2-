# ============================================================
# L7-SCCT-NGT (Paper Edition)
#   Structural Compression Complexity Theory - Neural Grammar Tree
#   时间结构守恒曲线（Φ², H, K=Φ²/H）
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
# 2. SCCT stats
# -----------------------------
def scct_stats_torch(u, num_bins=64):
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
# 3. Teacher PDE (slightly mismatched)
#     u_t = 0.82 u_xx + 0.5 u - u^3
# -----------------------------
def laplace_x_dirichlet(u, dx):
    l = torch.zeros_like(u)
    l[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
    l[0] = 0.0
    l[-1] = 0.0
    return l

def teacher_step(u):
    u_xx = laplace_x_dirichlet(u, dx)
    rhs = 0.82 * u_xx + 0.5 * u - u**3
    u_new = u + dt * rhs
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

print(f"[Teacher] Φ²_true={phi2_true_T:.3e}, H_true={H_true_T:.3e}")

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
        u_x = torch.zeros_like(u)
        u_x[1:-1] = (u[2:] - u[:-2]) / (2*dx)
        feat = torch.stack([u, u_x, u_xx], dim=-1)
        h1 = torch.tanh(self.node1(feat))
        h2 = torch.tanh(self.node2(feat))
        h = torch.cat([h1, h2], dim=-1)
        return self.combine(h).squeeze(-1)

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
        core = 0.8 * u_xx + 0.5 * u - u**3
        f_ngt = self.tree(u)
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

# -----------------------------
# 6. Training (pair/mask/meta)
# -----------------------------
def train_scct_ngt_mode(mode, epochs=80, print_every=10):
    model = SCCT_NGT().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    λ_pair, λ_mask, λ_meta = 0.0, 1e-2, 5e-2
    λ_γ, λ_time = 1e-3, 1e-2

    for ep in range(1, epochs+1):
        opt.zero_grad()
        u_pred_T, φ2_pred, H_pred = model.simulate_final(u0, Nt_teacher)
        misfit = F.mse_loss(u_pred_T, uT_true)
        loss = misfit

        if mode == "pair":
            lam_scct = λ_pair
        elif mode == "mask":
            lam_scct = λ_mask
        else:
            lam_scct = λ_meta

        if lam_scct > 0:
            loss_scct_T = torch.abs(φ2_pred - phi2_true_T) + torch.abs(H_pred - H_true_T)
            loss += lam_scct * loss_scct_T

        if mode == "meta":
            φ2_t, H_t = model.simulate_traj_scct(u0, meta_window)
            loss_time = (φ2_t.mean() - phi2_true_mean_meta)**2 + (H_t.mean() - H_true_mean_meta)**2
            loss += λ_time * loss_time

        loss += λ_γ * torch.abs(model.gamma)
        loss.backward()
        opt.step()

        if ep % print_every == 0 or ep == 1:
            print(f"[{mode:5s}] Epoch {ep:03d} | loss={loss.item():.3e}, misfit={misfit.item():.3e}, "
                  f"φ2={φ2_pred.item():.3e}, H={H_pred.item():.3e}, γ={model.gamma.item():+.3e}")
    return model

print("\n[Training L7 modes...]")
model_pair = train_scct_ngt_mode("pair")
model_mask = train_scct_ngt_mode("mask")
model_meta = train_scct_ngt_mode("meta")

# -----------------------------
# 7. L7 Visualizer
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
    print(f"  [{name}] Φ²_mean={φ2s.mean():.3e}, H_mean={Hs.mean():.3e}, K≈{Ks.mean():.3e}")

t_vis = np.arange(steps_vis) * dt

plt.figure(figsize=(12,4))
for i, (title, idx) in enumerate(zip([r"$\Phi^2(t)$", "H(t)", "K(t)=Φ²/H"], [0,1,2])):
    plt.subplot(1,3,i+1)
    for md, c in zip(["pair","mask","meta"], ["r","g","b"]):
        plt.plot(t_vis, results[md][idx], c, label=md)
    plt.title(title)
    plt.xlabel("t")
    plt.legend()
plt.tight_layout()
plt.show()

