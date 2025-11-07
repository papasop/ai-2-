# ============================================================
# L3-style PDE model discovery with:
#  - Teacher PDE solver
#  - log-JNet surrogate with global log-sup error
#  - L1 sparsity (term selection) in coefficient space
#  - Soft "mask" for operator presence (continuous relaxation)
#  - SCCT-style structural stats (Φ², H)
# ============================================================

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------
# 0. Device & seeds
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(0)
np.random.seed(0)

# --------------------------
# 1. PDE config
#    1D reaction–diffusion:
#    u_t = a0 u_xx
#        + a1 u
#        + a2 u^2
#        + a3 u^3
#        + a4 u^5
#    Dirichlet: u(t,0)=u(t,1)=0, t∈[0,T]
# --------------------------

Nx = 50         # spatial grid points
Lx = 1.0
dx = Lx / (Nx - 1)
T  = 0.05       # final time (short to keep runtime)
dt = 1e-4       # time step
Nt = int(T / dt)

x = torch.linspace(0.0, Lx, Nx, device=device)
t_grid = torch.linspace(0.0, T, Nt+1, device=device)

print(f"[PDE] Nx={Nx}, Nt={Nt}, dx={dx:.4e}, dt={dt:.4e}")

# 数值稳定性：限制扩散系数范围
max_diffusion = 1.0
print(f"[PDE] dt * max_diffusion / dx^2 ≈ {dt * max_diffusion / dx**2:.3f}")

# --------------------------
# 2. Operator library & PDE solver
#    α = (a0, a1, a2, a3, a4)
#    term0: u_xx
#    term1: u
#    term2: u^2
#    term3: u^3
#    term4: u^5
# --------------------------

NUM_TERMS = 5  # a0,...,a4

def pde_rhs(u, alpha):
    """
    u: (Nx,) at one time, Dirichlet boundaries.
    alpha: (5,) => coefficients a0,...,a4
    returns: rhs(u) = u_t at interior points (Nx,)
    """
    a0, a1, a2, a3, a4 = alpha
    # u_xx with central difference, boundaries 0
    u_xx = torch.zeros_like(u)
    u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2

    # reaction terms
    u1 = u
    u2 = u**2
    u3 = u**3
    u5 = u**5

    rhs = a0 * u_xx + a1 * u1 + a2 * u2 + a3 * u3 + a4 * u5
    # Enforce Dirichlet BCs
    rhs[0] = 0.0
    rhs[-1] = 0.0
    return rhs

def solve_pde(alpha, u0, Nt=Nt, dt=dt):
    """
    Explicit Euler PDE solver.
    alpha: (5,) tensor on device
    u0: (Nx,) initial condition tensor
    returns: u(t_n, x_j) with shape (Nt+1, Nx)
    """
    alpha = alpha.to(device)
    u = torch.zeros((Nt+1, Nx), device=device)
    u[0] = u0.clone()
    for n in range(Nt):
        rhs = pde_rhs(u[n], alpha)
        u[n+1] = u[n] + dt * rhs
        # Dirichlet BC
        u[n+1, 0] = 0.0
        u[n+1, -1] = 0.0
    return u

# --------------------------
# 3. SCCT-style structural stats
# --------------------------

def scct_stats(u_snapshot, num_bins=64):
    """
    Very simple SCCT proxy:
      - Φ² = average u^2
      - H  = entropy of |u| histogram (as a proxy for structure entropy)
      - K_eff fixed to 2.0 (just a placeholder)
    u_snapshot: (Nx,) tensor
    """
    u_flat = u_snapshot.detach().cpu().numpy()
    phi2 = float(np.mean(u_flat**2))

    # histogram of |u|
    absu = np.abs(u_flat)
    hist, bin_edges = np.histogram(absu, bins=num_bins, density=True)
    hist = hist + 1e-12
    hist = hist / hist.sum()
    H = float(-np.sum(hist * np.log(hist)))
    K_eff = 2.0
    return {"Phi2": phi2, "H": H, "K_eff": K_eff}

# --------------------------
# 4. Ground-truth PDE & data
# --------------------------

# 真 PDE 参数（你可以改成任何你喜欢的「真」方程）
alpha_true = torch.tensor([0.8, 0.5, 0.0, -1.0, 0.0], device=device)
print("[Truth] α_true =", alpha_true.cpu().numpy())

# 初值
u0 = torch.sin(math.pi * x)  # smooth, compatible with BC
u_data = solve_pde(alpha_true, u0)  # (Nt+1, Nx)
print("[Truth] Data shape (t,x) =", tuple(u_data.shape))

# 真 PDE 的 SCCT
scct_true = scct_stats(u_data[-1])
print("[SCCT] Ground-truth final snapshot:", scct_true)

# --------------------------
# 5. Teacher functional J(α)
#    misfit + λ_L1 * ||α||_1 + λ_SCCT * |Φ² - Φ²_true|
# --------------------------

lambda_misfit = 1.0
lambda_L1     = 1e-3
lambda_scct   = 1e-2

def teacher_J(alpha, return_parts=False):
    """
    alpha: (5,) tensor (no grad required)
    returns: scalar J(alpha)
    """
    with torch.no_grad():
        u_pred = solve_pde(alpha, u0)  # (Nt+1, Nx)
        # Misfit vs data (space-time L2)
        diff = u_pred - u_data
        misfit = lambda_misfit * torch.mean(diff**2)

        # L1 on coefficients
        reg_ctrl = lambda_L1 * torch.sum(torch.abs(alpha))

        # SCCT penalty on final snapshot
        stats = scct_stats(u_pred[-1])
        phi2_diff = abs(stats["Phi2"] - scct_true["Phi2"])
        scct_pen = lambda_scct * phi2_diff

        J = misfit + reg_ctrl + scct_pen

    if return_parts:
        return J.item(), {
            "misfit": float(misfit.cpu()),
            "reg_ctrl": float(reg_ctrl.cpu()),
            "scct_pen": float(scct_pen),
            "scct": stats,
        }
    else:
        return J.item()

J_true_star, parts_star = teacher_J(alpha_true, return_parts=True)
print(f"[Teacher] J(α_true)≈{J_true_star:.4e}, parts={parts_star}")

# --------------------------
# 6. 构建 Teacher 数据集：α → J(α)
# --------------------------

N_train = 200  # teacher dataset size

def sample_alpha_batch(n):
    """
    Sample random coefficients in some box.
    a0: diffusion in [0.1, 1.0]
    a1..a4: in [-1, 1]
    """
    a0 = np.random.uniform(0.1, 1.0, size=(n,1))
    a_rest = np.random.uniform(-1.0, 1.0, size=(n, NUM_TERMS-1))
    alpha = np.concatenate([a0, a_rest], axis=1)
    return alpha

print("[Dataset] Building teacher dataset with N=", N_train, " ...")
theta_list = []
J_list = []
start_time = time.time()
for i, alpha_np in enumerate(sample_alpha_batch(N_train)):
    alpha_t = torch.tensor(alpha_np, device=device, dtype=torch.float32)
    Jval = teacher_J(alpha_t)
    theta_list.append(alpha_np)
    J_list.append([Jval])
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{N_train} done...")
end_time = time.time()
theta_train = torch.tensor(np.array(theta_list), device=device, dtype=torch.float32)
J_train = torch.tensor(np.array(J_list), device=device, dtype=torch.float32)

print(f"[Dataset] Done in {end_time-start_time:.1f}s")
print("[Dataset] theta_train shape:", theta_train.shape, "J_train shape:", J_train.shape)
print("[Dataset] J range: [{:.3e}, {:.3e}]".format(J_train.min().item(), J_train.max().item()))

# --------------------------
# 7. log-JNet surrogate: α → log J(α)
# --------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64, depth=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.Tanh())
        for _ in range(depth-1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

logJ_net = MLP(NUM_TERMS, 1, hidden=64, depth=3).to(device)

optimizer = optim.Adam(logJ_net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

logJ_target = torch.log(J_train + 1e-8)

print("\n[log-JNet init] Training surrogate logJ_net(α) ...")
n_epochs = 2000
logJ_net.train()
t0 = time.time()
for epoch in range(1, n_epochs+1):
    optimizer.zero_grad()
    pred = logJ_net(theta_train)
    loss = loss_fn(pred, logJ_target)
    loss.backward()
    optimizer.step()

    if epoch % 400 == 0 or epoch == 1:
        print(f"[log-JNet init] Epoch {epoch:4d}/{n_epochs}, loss={loss.item():.4e}")
t1 = time.time()
print(f"[log-JNet init] Done in {t1-t0:.1f}s")

# --------------------------
# 8. 经验 log-sup 误差 bound
# --------------------------

def eval_log_sup_error(num_samples=200):
    logJ_net.eval()
    with torch.no_grad():
        alphas_np = sample_alpha_batch(num_samples)
        alphas = torch.tensor(alphas_np, device=device, dtype=torch.float32)
        J_true_vals = []
        for i in range(num_samples):
            J_true_vals.append(teacher_J(alphas[i]))
        J_true_vals = torch.tensor(J_true_vals, device=device).view(-1,1)
        logJ_true = torch.log(J_true_vals + 1e-8)
        logJ_pred = logJ_net(alphas)
        errors = torch.abs(logJ_pred - logJ_true)
        eps = errors.max().item()
    return eps, {
        "alphas": alphas,
        "J_true": J_true_vals,
        "logJ_true": logJ_true,
        "logJ_pred": logJ_pred,
        "errors": errors,
    }

eps_global, _info = eval_log_sup_error(num_samples=200)
print("\n[Bound] Global log-sup error ε_global ≈ {:.3f}".format(eps_global))
print("        ⇒ multiplicative error factor ≤ exp(ε_global) ≈ {:.3f}".format(math.exp(eps_global)))

# --------------------------
# 9. L3-style Sparse Structural Control:
#    - Continuous θ (coeffs) + soft mask m ∈ (0,1)
#    - α = m ⊙ θ
#    - Optimize surrogate J_hat(α) + λθ ||θ||_1 + λm ||m||_1
# --------------------------

logJ_net.eval()

lambda_theta_ctrl = 1e-3
lambda_mask_ctrl  = 1e-3
n_steps_ctrl = 200

# 初始化参数：从 α_true 附近出发
theta = (alpha_true + 0.2*torch.randn_like(alpha_true)).clone().detach().to(device)
theta.requires_grad_(True)

# mask logits, 初始接近 1
logits_m = torch.ones(NUM_TERMS, device=device) * 2.0
logits_m.requires_grad_(True)

optimizer_ctrl = optim.Adam([theta, logits_m], lr=5e-2)

print("\n[Control] surrogate descent in (θ, m)-space (soft mask)")
for step in range(1, n_steps_ctrl+1):
    optimizer_ctrl.zero_grad()

    m_soft = torch.sigmoid(logits_m)          # ∈(0,1)
    alpha_eff = m_soft * theta                # α = m ⊙ θ

    # surrogate J_hat(α)
    logJ_hat = logJ_net(alpha_eff.view(1,-1))
    J_hat = torch.exp(logJ_hat)[0,0]

    # L1 稀疏
    l1_theta = torch.sum(torch.abs(theta))
    l1_mask = torch.sum(torch.abs(m_soft))

    loss_ctrl = J_hat + lambda_theta_ctrl*l1_theta + lambda_mask_ctrl*l1_mask

    loss_ctrl.backward()
    optimizer_ctrl.step()

    if step % 20 == 1 or step == n_steps_ctrl:
        print(f"[Control] step {step:4d}, "
              f"J_hat≈{J_hat.item():.4e}, "
              f"||θ||₁={l1_theta.item():.3e}, "
              f"m={m_soft.detach().cpu().numpy()}")

# --------------------------
# 10. 硬化 mask，得到最终选中的 PDE 结构
# --------------------------

with torch.no_grad():
    m_soft = torch.sigmoid(logits_m)
    m_hard = (m_soft > 0.5).float()
    alpha_star = m_hard * theta
    alpha_star_np = alpha_star.cpu().numpy()
    m_hard_np = m_hard.cpu().numpy()

J_star, parts_star = teacher_J(alpha_star, return_parts=True)

print("\n[Result] Learned sparse structure:")
print("  θ (continuous)   =", theta.detach().cpu().numpy())
print("  m_soft           =", m_soft.cpu().numpy())
print("  m_hard (0/1)     =", m_hard_np)
print("  α* = m_hard ⊙ θ  =", alpha_star_np)
print(f"  True J(α*) ≈ {J_star:.4e}, parts={parts_star}")
print(f"  ||α* - α_true||₂ ≈ {torch.norm(alpha_star - alpha_true).item():.3e}")

# 对比 SCCT 结构
scct_learned = parts_star["scct"]
print("\n[SCCT] Ground-truth final snapshot:", scct_true)
print("[SCCT] Learned-model final snapshot:", scct_learned)

print("\nDone. What you now have:")
print("  - Teacher PDE J(α) with SCCT penalty.")
print("  - log-JNet surrogate with global log-sup error bound.")
print("  - L1-sparse control over coefficients θ and soft masks m.")
print("  - Automatic 'operator selection' via m_hard, plus SCCT comparison.")

