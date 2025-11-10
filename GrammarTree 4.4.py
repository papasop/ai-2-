# ============================================
# GrammarTree 4.4-A (AMP=3.0, weaker regularization) - One-Click Colab
# ============================================

!pip install -q numpy sympy matplotlib pysr torch

import os
os.environ["JULIA_NUM_THREADS"] = "4"
os.environ["PYTHONHASHSEED"] = "0"

from pysr import PySRRegressor

try:
    import juliacall
    from juliacall import Main as jl
    print("[Init] juliacall imported, Julia available.")
except Exception as e:
    print("[Init] Could not import juliacall explicitly, PySR will manage. detail:", repr(e))

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams["figure.figsize"] = (5, 3)
plt.rcParams["figure.dpi"] = 120

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
print("[Init] GrammarTree 4.4-A header OK.\n")

# ========== Grid ==========
Nx = 64
x = torch.linspace(0.0, 1.0, Nx, dtype=DTYPE)
x_np = x.numpy()
dx = float(x[1] - x[0])
dt = 1e-4
Nt_teacher = 2000
meta_window = 400

print(f"[Grid] Nx={Nx}, dx={dx:.6f}, dt={dt:.1e}, Nt={Nt_teacher}, T={Nt_teacher*dt:.3f}")

def grad_x_centered(U, dx):
    G = torch.zeros_like(U)
    G[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / (2.0 * dx)
    return G

def laplace_x_dirichlet(U, dx):
    L = torch.zeros_like(U)
    L[:, 1:-1] = (U[:, 2:] - 2.0 * U[:, 1:-1] + U[:, :-2]) / (dx * dx)
    return L

# ========== SCCT + invariants ==========
def scct_invariants_torch(U):
    C, N = U.shape
    U_flat = U.view(-1)
    phi2 = torch.mean(U_flat * U_flat)

    eps = 1e-12
    absU = torch.abs(U_flat)
    vmax = torch.max(absU)
    if vmax < eps:
        H = torch.tensor(0.0, device=U.device, dtype=U.dtype)
    else:
        normed = absU / vmax
        hist = torch.histc(normed, bins=64, min=0.0, max=1.0)
        probs = hist / (torch.sum(hist) + eps)
        H = -torch.sum(probs * torch.log(probs + eps))

    x_grid = x.to(U.device)
    dx_local = dx

    u = U[0, :]
    v = U[1, :]
    w = U[2, :]

    P_u = torch.sum(u) * dx_local
    P_v = torch.sum(v) * dx_local
    P_w = torch.sum(w) * dx_local

    L_u = torch.sum(x_grid * u) * dx_local
    L_v = torch.sum(x_grid * v) * dx_local
    L_w = torch.sum(x_grid * w) * dx_local

    P = torch.stack([P_u, P_v, P_w], dim=0)
    L = torch.stack([L_u, L_v, L_w], dim=0)

    return phi2, H, P, L

# ========== Teacher & Core PDE ==========
def teacher_rhs(U):
    u = U[0:1, :]
    v = U[1:2, :]
    w = U[2:3, :]

    u_x  = grad_x_centered(u, dx)
    u_xx = laplace_x_dirichlet(u, dx)
    v_x  = grad_x_centered(v, dx)
    v_xx = laplace_x_dirichlet(v, dx)
    w_x  = grad_x_centered(w, dx)
    w_xx = laplace_x_dirichlet(w, dx)

    rhs_u = 0.82 * u_xx + 0.5 * u - u**3
    rhs_v = 0.75 * v_xx - 0.3 * v**3 + 0.10 * u_x
    rhs_w = 0.65 * w_xx - 0.25 * w**3 + 0.05 * v_x
    return torch.cat([rhs_u, rhs_v, rhs_w], dim=0)

def core_rhs(U):
    u = U[0:1, :]
    v = U[1:2, :]
    w = U[2:3, :]

    u_x  = grad_x_centered(u, dx)
    u_xx = laplace_x_dirichlet(u, dx)
    v_x  = grad_x_centered(v, dx)
    v_xx = laplace_x_dirichlet(v, dx)
    w_x  = grad_x_centered(w, dx)
    w_xx = laplace_x_dirichlet(w, dx)

    rhs_u = 0.80 * u_xx + 0.5 * u - u**3
    rhs_v = 0.75 * v_xx - 0.3 * v**3 + 0.10 * u_x
    rhs_w = 0.65 * w_xx - 0.25 * w**3 + 0.05 * v_x
    return torch.cat([rhs_u, rhs_v, rhs_w], dim=0)

def teacher_step(U):
    U_new = U + dt * teacher_rhs(U)
    U_new[:, 0]  = 0.0
    U_new[:, -1] = 0.0
    return U_new

# ========== Teacher trajectory ==========
AMP = 3.0
u0_np = AMP * (np.sin(np.pi * x_np) + 0.2 * np.sin(2 * np.pi * x_np))
v0_np = AMP * (0.5 * np.sin(2 * np.pi * x_np))
w0_np = AMP * (0.3 * np.sin(3 * np.pi * x_np))

U0 = torch.stack([
    torch.tensor(u0_np, dtype=DTYPE),
    torch.tensor(v0_np, dtype=DTYPE),
    torch.tensor(w0_np, dtype=DTYPE),
], dim=0)
U0 = to_device(U0)

print("[Teacher] U0 shape:", U0.shape, "device:", U0.device)

def simulate_teacher(U0, Nt):
    U = U0.clone()
    phi2_list = torch.zeros(Nt, dtype=DTYPE, device=device)
    H_list    = torch.zeros(Nt, dtype=DTYPE, device=device)
    P_list    = torch.zeros((Nt, 3), dtype=DTYPE, device=device)
    L_list    = torch.zeros((Nt, 3), dtype=DTYPE, device=device)
    for n in range(Nt):
        U = teacher_step(U)
        phi2, H, P, L = scct_invariants_torch(U)
        phi2_list[n] = phi2
        H_list[n]    = H
        P_list[n]    = P
        L_list[n]    = L
    return phi2_list, H_list, P_list, L_list, U

print("[Teacher] Simulating teacher trajectory...")
phi2_true_traj, H_true_traj, P_true_traj, L_true_traj, U_T_true = simulate_teacher(U0, Nt_teacher)

phi2_true_T   = phi2_true_traj[-1]
H_true_T      = H_true_traj[-1]
P_true_T      = P_true_traj[-1]
L_true_T      = L_true_traj[-1]
phi2_true_meta_mean = phi2_true_traj[:meta_window].mean()

print(f"[Teacher] Final Φ²(T)   = {phi2_true_T.item():.3e}, H(T) = {H_true_T.item():.3f}")
print(f"[Teacher] Final P(T)     = {P_true_T.cpu().numpy()}")
print(f"[Teacher] Final L(T)     = {L_true_T.cpu().numpy()}")
print(f"[Teacher] <Φ²>_meta     = {phi2_true_meta_mean.item():.3e}\n")

# ========== Grammar ==========
GRAMMAR_TERMS = [
    "u", "u_x", "u_xx", "v", "v_x", "w", "w_x",
    "u*u_x", "u*v_x", "u*w_x", "u^2", "v^2",
]
n_terms = len(GRAMMAR_TERMS)
print(f"[GrammarTree 4.4-A] n_terms = {n_terms}, terms = {GRAMMAR_TERMS}\n")

def grammar_features_vector(U):
    u = U[0:1, :]
    v = U[1:2, :]
    w = U[2:3, :]

    u_x  = grad_x_centered(u, dx)
    u_xx = laplace_x_dirichlet(u, dx)
    v_x  = grad_x_centered(v, dx)
    w_x  = grad_x_centered(w, dx)

    t0  = u[0]
    t1  = u_x[0]
    t2  = u_xx[0]
    t3  = v[0]
    t4  = v_x[0]
    t5  = w[0]
    t6  = w_x[0]
    t7  = u[0] * u_x[0]
    t8  = u[0] * v_x[0]
    t9  = u[0] * w_x[0]
    t10 = u[0] ** 2
    t11 = v[0] ** 2

    feats = torch.stack(
        [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11],
        dim=-1
    )
    return feats

# ========== GrammarTree model ==========
class GrammarTree44A(nn.Module):
    def __init__(self, mask=None):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(3, n_terms, dtype=DTYPE))
        self.gamma = nn.Parameter(torch.tensor(0.05, dtype=DTYPE))
        if mask is None:
            mask_np = np.ones((3, n_terms), dtype=np.float32)
        else:
            mask_np = mask.astype(np.float32)
        self.register_buffer("mask", torch.tensor(mask_np, dtype=DTYPE))

    def residual(self, U):
        feats = grammar_features_vector(U)         # [Nx, n_terms]
        w_eff = self.w * self.mask                # [3, n_terms]
        r = torch.matmul(w_eff, feats.transpose(0, 1))  # [3, Nx]
        return self.gamma * r

    def pde_rhs(self, U):
        core = core_rhs(U)
        r = self.residual(U)
        return core + r

    def step(self, U):
        U_new = U + dt * self.pde_rhs(U)
        U_new[:, 0]  = 0.0
        U_new[:, -1] = 0.0
        return U_new

    def simulate_final(self, U0, Nt):
        U = U0.clone()
        for _ in range(Nt):
            U = self.step(U)
        phi2, H, P, L = scct_invariants_torch(U)
        return U, phi2, H, P, L

    def simulate_traj_phi2(self, U0, Nt):
        U = U0.clone()
        phi2s = []
        for _ in range(Nt):
            U = self.step(U)
            phi2, _, _, _ = scct_invariants_torch(U)
            phi2s.append(phi2)
        return torch.stack(phi2s)

def rollout_and_error(model, steps):
    U_teacher = U0.clone()
    U_model   = U0.clone()
    errs = []
    with torch.no_grad():
        for _ in range(steps):
            U_teacher = teacher_step(U_teacher)
            U_model   = model.step(U_model)
            errs.append(torch.mean((U_teacher - U_model)**2).item())
    return np.array(errs)

# ========== Stage 1 ==========
def train_stage1(epochs=200):
    model = GrammarTree44A(mask=None).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    lambda_scct  = 5e-2
    lambda_time  = 1e-2
    lambda_gamma = 1e-5    # weaker
    lambda_w     = 1e-5    # weaker

    for ep in range(1, epochs + 1):
        opt.zero_grad()

        U_pred_T, phi2_pred_T, H_pred_T, P_pred_T, L_pred_T = model.simulate_final(U0, Nt_teacher)
        misfit = F.mse_loss(U_pred_T, U_T_true)

        loss_scct_T = torch.abs(phi2_pred_T - phi2_true_T) \
                    + torch.abs(H_pred_T    - H_true_T)     \
                    + torch.mean((P_pred_T - P_true_T)**2) \
                    + torch.mean((L_pred_T - L_true_T)**2)

        phi2_t = model.simulate_traj_phi2(U0, meta_window)
        loss_time = (phi2_t.mean() - phi2_true_meta_mean)**2

        g = model.gamma
        w = model.w

        loss = misfit \
             + lambda_scct  * loss_scct_T \
             + lambda_time  * loss_time   \
             + lambda_gamma * torch.sum(torch.abs(g)) \
             + lambda_w     * torch.sum(torch.abs(w))

        loss.backward()
        opt.step()

    with torch.no_grad():
        U_pred_T, phi2_pred_T, H_pred_T, P_pred_T, L_pred_T = model.simulate_final(U0, Nt_teacher)
        phi2_t = model.simulate_traj_phi2(U0, meta_window)
        phi2_meta_model = phi2_t.mean()
        eff = (model.gamma * model.w).detach().cpu().numpy()
        eff_u_xx = eff[0, 2]

    print("[Stage 1] Training GrammarTree 4.4-A (dense, SCCT+P+L+meta) ...")
    print(f"[Stage 1][Summary] misfit≈{misfit.item():.3e}, Φ²(T)={phi2_pred_T.item():.3e}, H(T)={H_pred_T.item():.3f}")
    print(f"  P(T)={P_pred_T.cpu().numpy()}, L(T)={L_pred_T.cpu().numpy()}")
    print(f"  <Φ²>_meta(model) = {phi2_meta_model.item():.3e}")
    print(f"  γ(Stage 1) = {model.gamma.item():+.3e}")
    print(f"  eff(u_eq, u_xx idx 2) = {eff_u_xx:+.6e}\n")

    return model

model_stage1 = train_stage1()

# ========== Stage 2: pruning + refine ==========
threshold = 5e-4
with torch.no_grad():
    g1 = model_stage1.gamma.item()
    w1 = model_stage1.w.detach().cpu().numpy()
eff1 = g1 * w1

keep_masks = []
for c_idx, ch_name in enumerate(["u_eq", "v_eq", "w_eq"]):
    eff_c = eff1[c_idx]
    keep_mask_c = np.abs(eff_c) > threshold
    keep_masks.append(keep_mask_c)
    kept_indices = np.where(keep_mask_c)[0]
    kept_terms   = [GRAMMAR_TERMS[i] for i in kept_indices]
    frac = len(kept_indices) / n_terms
    print(f"[Stage 2] {ch_name}: kept indices = {list(kept_indices)}, kept terms = {kept_terms}")
    print(f"  [Stage 2] λ_k({ch_name}) = {frac:.3f} ({len(kept_indices)}/{n_terms})")
keep_masks = np.stack(keep_masks, axis=0)

print()
model_stage2 = GrammarTree44A(mask=keep_masks).to(device)
with torch.no_grad():
    model_stage2.gamma.copy_(model_stage1.gamma)
    model_stage2.w.copy_(model_stage1.w)

opt2 = torch.optim.Adam(model_stage2.parameters(), lr=3e-3)
lambda_scct2  = 5e-2
lambda_time2  = 1e-2
lambda_gamma2 = 1e-5
lambda_w2     = 0.0

epochs_stage2 = 80
for ep in range(1, epochs_stage2 + 1):
    opt2.zero_grad()
    U_pred_T, phi2_pred_T, H_pred_T, P_pred_T, L_pred_T = model_stage2.simulate_final(U0, Nt_teacher)
    misfit2 = F.mse_loss(U_pred_T, U_T_true)
    loss_scct_T2 = torch.abs(phi2_pred_T - phi2_true_T) \
                 + torch.abs(H_pred_T    - H_true_T)     \
                 + torch.mean((P_pred_T - P_true_T)**2) \
                 + torch.mean((L_pred_T - L_true_T)**2)
    phi2_t2 = model_stage2.simulate_traj_phi2(U0, meta_window)
    loss_time2 = (phi2_t2.mean() - phi2_true_meta_mean)**2
    g2 = model_stage2.gamma
    w2 = model_stage2.w
    loss2 = misfit2 \
          + lambda_scct2  * loss_scct_T2 \
          + lambda_time2  * loss_time2   \
          + lambda_gamma2 * torch.sum(torch.abs(g2)) \
          + lambda_w2     * torch.sum(torch.abs(w2))
    loss2.backward()
    opt2.step()

with torch.no_grad():
    U_pred_T2, phi2_pred_T2, H_pred_T2, P_pred_T2, L_pred_T2 = model_stage2.simulate_final(U0, Nt_teacher)
    phi2_t2 = model_stage2.simulate_traj_phi2(U0, meta_window)
    phi2_meta_model2 = phi2_t2.mean()
    eff2 = (model_stage2.gamma * model_stage2.w).detach().cpu().numpy()
    eff2_u_xx = eff2[0, 2]

true_hidden = 0.02
abs_err2 = abs(eff2_u_xx - true_hidden)
rel_err2 = abs_err2 / true_hidden * 100.0

print("[Stage 2] Refining pruned GrammarTree 4.4-A ...")
print(f"[Stage 2][Summary] misfit≈{misfit2.item():.3e}, Φ²(T)={phi2_pred_T2.item():.3e}, H(T)={H_pred_T2.item():.3f}")
print(f"  P(T)={P_pred_T2.cpu().numpy()}, L(T)={L_pred_T2.cpu().numpy()}")
print(f"  <Φ²>_meta(model) = {phi2_meta_model2.item():.3e}")
print(f"  γ(Stage 2) = {model_stage2.gamma.item():+.3e}")
print(f"  eff(u_eq, u_xx idx 2) = {eff2_u_xx:+.6e}")
print(f"  target hidden: +{true_hidden:.6f} * u_xx")
print(f"  |Δc| = {abs_err2:.3e}, relative error = {rel_err2:.3f} %\n")

# ========== Eval ==========
steps_err = 400
err_gt = rollout_and_error(model_stage2, steps_err)

with torch.no_grad():
    phi2_gt_traj = model_stage2.simulate_traj_phi2(U0, steps_err)
phi2_teacher_early = phi2_true_traj[:steps_err]

print(f"[Eval] Mean L2 orbit error over [0,{steps_err*dt:.3f}] = {err_gt.mean():.3e}")
print(f"[Eval] Early-time Φ² teacher={phi2_teacher_early.mean().item():.3e}, "
      f"GT4.4-A={phi2_gt_traj.mean().item():.3e}, "
      f"ratio≈{phi2_gt_traj.mean().item()/phi2_teacher_early.mean().item():.3f}\n")

print("[GrammarTree 4.4-A] Effective hidden residual (u-equation):")
print(f"  target:  +{true_hidden:.6f} * u_xx")
print(f"  learned: {eff2_u_xx:+.6e} * u_xx (index 2)\n")

# ========== Stage 4: PySR ==========
print("[Stage 4] Building residual dataset for PySR (u-equation)...")
N_time_samp = 400
time_indices = range(N_time_samp)

X_list = []
R_list = []

with torch.no_grad():
    U = U0.clone()
    for n in range(N_time_samp):
        U = teacher_step(U)
        u = U[0:1, :]
        u_x  = grad_x_centered(u, dx)
        u_xx = laplace_x_dirichlet(u, dx)
        core_t   = core_rhs(U)
        rhs_gt   = model_stage2.pde_rhs(U)
        resid_u  = rhs_gt[0] - core_t[0]
        for i in range(1, Nx-1):
            X_list.append([float(u[0, i].item()),
                           float(u_x[0, i].item()),
                           float(u_xx[0, i].item())])
            R_list.append(float(resid_u[i].item()))

X = np.array(X_list, dtype=np.float64)
R = np.array(R_list, dtype=np.float64)
print(f"[Stage 4] Dataset shapes: X={X.shape}, R={R.shape}\n")

np.savez("gt44A_residual_dataset_pysr.npz", X=X, R=R)

print("[Stage 4] Running PySR symbolic regression (u-equation residual)...")
model_pysr = PySRRegressor(
    niterations=1200,
    populations=20,
    population_size=50,
    binary_operators=["+", "-", "*"],
    unary_operators=[],
    maxsize=20,
    model_selection="best",
    elementwise_loss="L2DistLoss()",
    parallelism="serial",
    deterministic=True,
    random_state=0,
)

model_pysr.fit(X, R, variable_names=["u", "u_x", "u_xx"])

eqs = model_pysr.equations_
best_row = eqs.sort_values("score", ascending=False).iloc[0]
row_idx = int(best_row.name)

print("\n[Stage 4] PySR best equation:")
print("  equation  :", best_row["equation"])
print(f"  loss      = {best_row['loss']:.3e}")
print(f"  complexity= {int(best_row['complexity'])}")

sym_list = model_pysr.sympy()
sym_best = sym_list[row_idx]
x0, x1, x2 = sp.symbols("x0 x1 x2")
coef_uxx = float(sym_best.subs({x0: 0.0, x1: 0.0, x2: 1.0}))

abs_err_pysr = abs(coef_uxx - true_hidden)
rel_err_pysr = abs_err_pysr / true_hidden * 100.0

print("\n[Stage 4] Interpreted as residual ≈ c * u_xx (from PySR)")
print(f"  c (PySR)     = {coef_uxx:.9f}")
print(f"  c (target)   = {true_hidden:.9f}")
print(f"  |Δc|         = {abs_err_pysr:.3e}")
print(f"  relative err = {rel_err_pysr:.3f} %")

print("\n[Done] GrammarTree 4.4-A (AMP=3.0, weaker reg) + PySR pipeline finished.")
