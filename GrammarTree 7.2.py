# ============================================
# GrammarTree 7.2 - 2D Vector Burgers Prototype
#  - 2D vector Burgers teacher PDE
#  - Zero-prior residual library: [v, lap_v, Q_S v, Q_O v]
#  - Multi-IC + SCCT (Φ², entropy, mass, Φ⁴)
#  - Stage 1 dense + Stage 2 pruning
#  - L1 weights weakened 10x (esp. lap_v)
#  - DO_PYSR = True  => symbolic audit on residual_x
# ============================================

# -------------------------
# 0. Install deps
# -------------------------
!pip install -q numpy sympy matplotlib pysr torch

# -------------------------
# 1. Imports & config
# -------------------------
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["JULIA_NUM_THREADS"] = "4"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from pysr import PySRRegressor

%matplotlib inline
plt.rcParams["figure.figsize"] = (4, 4)
plt.rcParams["figure.dpi"] = 120

DO_PYSR = True                # <<< 按你的要求，打开 PySR 审计
CORE_MODE = "core+residual"   # core-only / residual-only / core+residual

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Init] Using device:", device)

DTYPE = torch.float32
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("[Init] Random seed set to", SEED)
print("[Init] GrammarTree 7.2 header OK.\n")


# -------------------------
# 2. Grid & basic ops (2D periodic)
# -------------------------
Nx, Ny = 32, 32
x = torch.linspace(0.0, 1.0, Nx, dtype=DTYPE)
y = torch.linspace(0.0, 1.0, Ny, dtype=DTYPE)
dx = float(x[1] - x[0])
dy = float(y[1] - y[0])
dt = 1e-4
Nt = 2000   # T = 0.2
T = Nt * dt

print(f"[Grid] Nx={Nx}, Ny={Ny}, dx={dx:.6f}, dy={dy:.6f}, dt={dt:.1e}, Nt={Nt}, T={T:.3f}")

X, Y = torch.meshgrid(x, y, indexing="ij")

def to_dev(t):
    return t.to(device=device, dtype=DTYPE)

X = to_dev(X)
Y = to_dev(Y)

def roll2d(u, shift_x, shift_y):
    # u: [..., Nx, Ny]
    return torch.roll(torch.roll(u, shifts=shift_x, dims=-2),
                      shifts=shift_y, dims=-1)

def grad2d(u, dx, dy):
    # u: [..., Nx, Ny] (periodic)
    dudx = (roll2d(u, -1, 0) - roll2d(u, +1, 0)) / (2.0 * dx)
    dudy = (roll2d(u, 0, -1) - roll2d(u, 0, +1)) / (2.0 * dy)
    return dudx, dudy

def laplace2d(u, dx, dy):
    # u: [..., Nx, Ny]
    u_xx = (roll2d(u, -1, 0) - 2.0 * u + roll2d(u, +1, 0)) / (dx * dx)
    u_yy = (roll2d(u, 0, -1) - 2.0 * u + roll2d(u, 0, +1)) / (dy * dy)
    return u_xx + u_yy

# -------------------------
# 3. SCCT-style stats for vector field v(x,y,2)
# -------------------------
def scct_stats_vec(v):
    """
    v: [2, Nx, Ny]
    返回:
      Φ² = <|v|²>,
      H  = entropy of |v|,
      M  = mean(v_x),
      Φ⁴ = <|v|⁴>
    """
    vx = v[0]
    vy = v[1]
    mag = torch.sqrt(vx * vx + vy * vy)
    flat = mag.reshape(-1)

    phi2 = torch.mean(flat * flat)
    phi4 = torch.mean((flat * flat) ** 2)
    M = torch.mean(vx)

    eps = 1e-12
    vmax = torch.max(flat.abs())
    if vmax < eps:
        H = torch.tensor(0.0, dtype=DTYPE, device=device)
    else:
        normed = flat.abs() / (vmax + eps)
        hist = torch.histc(normed, bins=64, min=0.0, max=1.0)
        probs = hist / (torch.sum(hist) + eps)
        H = -torch.sum(probs * torch.log(probs + eps))

    return phi2, H, M, phi4


# -------------------------
# 4. Teacher PDE: 2D vector Burgers
#    v_t + (v · ∇)v = ν Δ v
# -------------------------
nu_core = 0.10
nu_hidden = 0.02
nu_teacher = nu_core + nu_hidden   # 真正 teacher 的粘性

def teacher_rhs(v):
    """
    v: [2, Nx, Ny]
    """
    vx = v[0]
    vy = v[1]

    dvx_dx, dvx_dy = grad2d(vx, dx, dy)
    dvy_dx, dvy_dy = grad2d(vy, dx, dy)

    adv_x = vx * dvx_dx + vy * dvx_dy
    adv_y = vx * dvy_dx + vy * dvy_dy

    lap_vx = laplace2d(vx, dx, dy)
    lap_vy = laplace2d(vy, dx, dy)

    rhs_x = -adv_x + nu_teacher * lap_vx
    rhs_y = -adv_y + nu_teacher * lap_vy
    return torch.stack([rhs_x, rhs_y], dim=0)

def core_rhs(v):
    # 同样结构，只是 ν = nu_core
    vx = v[0]
    vy = v[1]

    dvx_dx, dvx_dy = grad2d(vx, dx, dy)
    dvy_dx, dvy_dy = grad2d(vy, dx, dy)

    adv_x = vx * dvx_dx + vy * dvx_dy
    adv_y = vx * dvy_dx + vy * dvy_dy

    lap_vx = laplace2d(vx, dx, dy)
    lap_vy = laplace2d(vy, dx, dy)

    rhs_x = -adv_x + nu_core * lap_vx
    rhs_y = -adv_y + nu_core * lap_vy
    return torch.stack([rhs_x, rhs_y], dim=0)

def euler_step(v, rhs_fun):
    return v + dt * rhs_fun(v)

def simulate_teacher(v0, Nt):
    v = v0.clone()
    traj = torch.zeros((Nt, 2, Nx, Ny), dtype=DTYPE, device=device)
    phi2s = torch.zeros(Nt, dtype=DTYPE, device=device)
    Hs    = torch.zeros(Nt, dtype=DTYPE, device=device)
    Ms    = torch.zeros(Nt, dtype=DTYPE, device=device)
    phi4s = torch.zeros(Nt, dtype=DTYPE, device=device)

    for n in range(Nt):
        v = euler_step(v, teacher_rhs)
        traj[n] = v
        phi2, H, M, phi4 = scct_stats_vec(v)
        phi2s[n] = phi2
        Hs[n]    = H
        Ms[n]    = M
        phi4s[n] = phi4

    return traj, phi2s, Hs, Ms, phi4s


# -------------------------
# 5. Generate multi-IC teacher trajectories
# -------------------------
def make_ic(seed_offset=0):
    # 几个随机的涡旋 + 噪声
    rng = np.random.RandomState(SEED + seed_offset)
    k1 = 2.0 * np.pi
    k2 = 4.0 * np.pi
    phase1 = rng.rand() * 2.0 * np.pi
    phase2 = rng.rand() * 2.0 * np.pi

    vx0 = 0.2 * torch.sin(k1 * X + phase1) * torch.cos(k2 * Y + phase2)
    vy0 = -0.2 * torch.cos(k1 * X + phase2) * torch.sin(k2 * Y + phase1)
    noise_x = 0.02 * torch.randn_like(vx0)
    noise_y = 0.02 * torch.randn_like(vy0)
    return torch.stack([vx0 + noise_x, vy0 + noise_y], dim=0)

n_ic = 3
teacher_trajs = []
teacher_stats = []  # list of dicts

print("[Teacher] Generating multi-IC teacher trajectories...")
for i in range(n_ic):
    v0 = make_ic(seed_offset=10 * i)
    v0 = to_dev(v0)
    traj, phi2s, Hs, Ms, phi4s = simulate_teacher(v0, Nt)
    teacher_trajs.append(traj)
    stats = {
        "v0": v0,
        "traj": traj,
        "phi2": phi2s,
        "H": Hs,
        "M": Ms,
        "phi4": phi4s,
    }
    teacher_stats.append(stats)
    print(f"[Teacher] IC #{i}: Φ²(T)={phi2s[-1].item():.3e}, "
          f"H(T)={Hs[-1].item():.3f}, M(T)={Ms[-1].item():+.3e}, "
          f"<Φ²>_meta={phi2s[:400].mean().item():.3e}")
print()


# -------------------------
# 6. Grammar 7.2: tensor operator library
# -------------------------
GRAMMAR_TERMS = [
    "v",        # 0
    "lap_v",    # 1
    "Q_S * v",  # 2
    "Q_O * v",  # 3
]
n_terms = len(GRAMMAR_TERMS)
print(f"[GrammarTree 7.2] n_terms = {n_terms}, terms = {GRAMMAR_TERMS}\n")

def tensor_features(v):
    """
    v: [2, Nx, Ny]
    返回 features: [n_terms, 2, Nx, Ny]
    """
    vx = v[0]
    vy = v[1]

    dvx_dx, dvx_dy = grad2d(vx, dx, dy)
    dvy_dx, dvy_dy = grad2d(vy, dx, dy)

    # 对称部分 S，反对称部分 O
    Sxx = dvx_dx
    Syy = dvy_dy
    Sxy = 0.5 * (dvx_dy + dvy_dx)
    Syx = Sxy

    Oxy = 0.5 * (dvx_dy - dvy_dx)
    Oyx = -Oxy

    # S², O² 的 Frobenius norm^2 作为标量场
    S2 = Sxx * Sxx + 2.0 * Sxy * Syx + Syy * Syy
    O2 = 2.0 * Oxy * Oyx

    lap_vx = laplace2d(vx, dx, dy)
    lap_vy = laplace2d(vy, dx, dy)
    lap_v = torch.stack([lap_vx, lap_vy], dim=0)

    QS_v = torch.stack([S2 * vx, S2 * vy], dim=0)
    QO_v = torch.stack([O2 * vx, O2 * vy], dim=0)

    feats = torch.stack([
        v,
        lap_v,
        QS_v,
        QO_v
    ], dim=0)   # [n_terms, 2, Nx, Ny]
    return feats


# -------------------------
# 7. GrammarTree 7.2 PDE model
# -------------------------
class GrammarTree72PDE(nn.Module):
    def __init__(self, mask=None):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_terms, dtype=DTYPE))
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=DTYPE))
        if mask is None:
            mask = torch.ones(n_terms, dtype=DTYPE)
        self.register_buffer("mask", mask)

    def residual(self, v):
        feats = tensor_features(v)       # [n_terms, 2, Nx, Ny]
        eff = self.gamma * self.w * self.mask   # [n_terms]
        eff_reshaped = eff.view(n_terms, 1, 1, 1)
        r = torch.sum(eff_reshaped * feats, dim=0)  # [2, Nx, Ny]
        return r

    def pde_rhs(self, v):
        if CORE_MODE == "core-only":
            return core_rhs(v)
        elif CORE_MODE == "residual-only":
            return self.residual(v)
        elif CORE_MODE == "core+residual":
            return core_rhs(v) + self.residual(v)
        else:
            raise ValueError(f"Unknown CORE_MODE={CORE_MODE}")

    def step(self, v):
        return v + dt * self.pde_rhs(v)

    def simulate(self, v0, Nt):
        v = v0.clone()
        for _ in range(Nt):
            v = self.step(v)
        phi2, H, M, phi4 = scct_stats_vec(v)
        return v, phi2, H, M, phi4

    def simulate_phi2_traj(self, v0, steps):
        v = v0.clone()
        phi2s = []
        for _ in range(steps):
            v = self.step(v)
            phi2, _, _, _ = scct_stats_vec(v)
            phi2s.append(phi2)
        return torch.stack(phi2s, dim=0)


# -------------------------
# 8. Rollout error util
# -------------------------
def rollout_error(model, v0, steps):
    v_t = v0.clone()
    v_m = v0.clone()
    errs = []
    with torch.no_grad():
        for _ in range(steps):
            v_t = euler_step(v_t, teacher_rhs)
            v_m = model.step(v_m)
            errs.append(torch.mean((v_t - v_m) ** 2).item())
    return np.array(errs)


# -------------------------
# 9. Stage 1 training (multi-IC, SCCT)
#    这里减弱 L1 权重：lambda_w_base = [5e-5, 1e-5, 5e-5, 5e-5]
# -------------------------
def train_stage1(epochs=120):
    model = GrammarTree72PDE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    lambda_phi2_base = 5e-2
    lambda_H_base    = 5e-2
    lambda_M_base    = 1e-2
    lambda_phi4_base = 1e-2

    lambda_gamma = 1e-5
    # <<< 关键修改：L1 正则整体减弱 10 倍，lap_v 的权重特别小
    lambda_w_base = torch.tensor([5e-5, 1e-5, 5e-5, 5e-5],
                                 dtype=DTYPE, device=device)

    print("[Stage 1] Training GrammarTree 7.2 (dense, multi-IC, dynamic SCCT+M+Phi4, CORE_MODE=core+residual)...")
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        t = ep / epochs

        lam_phi2 = lambda_phi2_base * (0.5 + 0.5 * t)
        lam_H    = lambda_H_base    * (0.5 + 0.5 * t)
        lam_M    = lambda_M_base    * (0.5 + 0.5 * t)
        lam_phi4 = lambda_phi4_base * (0.5 + 0.5 * t)

        loss_total = 0.0
        for ic in range(n_ic):
            stats = teacher_stats[ic]
            v0 = stats["v0"]
            traj_true = stats["traj"]
            phi2_true = stats["phi2"]
            H_true    = stats["H"]
            M_true    = stats["M"]
            phi4_true = stats["phi4"]

            vT_true = traj_true[-1]
            phi2_true_T = phi2_true[-1]
            H_true_T    = H_true[-1]
            M_true_T    = M_true[-1]
            phi4_true_T = phi4_true[-1]

            vT_pred, phi2_pred_T, H_pred_T, M_pred_T, phi4_pred_T = model.simulate(v0, Nt)

            misfit = F.mse_loss(vT_pred, vT_true)

            loss_scct = torch.abs(phi2_pred_T - phi2_true_T) \
                      + torch.abs(H_pred_T    - H_true_T)    \
                      + torch.abs(M_pred_T    - M_true_T)    \
                      + torch.abs(phi4_pred_T - phi4_true_T)

            phi2_pred_traj = model.simulate_phi2_traj(v0, 400)
            phi2_meta_true = phi2_true[:400].mean()
            phi2_meta_pred = phi2_pred_traj.mean()
            loss_meta = (phi2_meta_pred - phi2_meta_true) ** 2

            loss_ic = misfit \
                    + lam_phi2 * torch.abs(phi2_pred_T - phi2_true_T) \
                    + lam_H    * torch.abs(H_pred_T    - H_true_T)    \
                    + lam_M    * torch.abs(M_pred_T    - M_true_T)    \
                    + lam_phi4 * torch.abs(phi4_pred_T - phi4_true_T) \
                    + 1e-1 * loss_meta   # meta 权重适中

            loss_total = loss_total + loss_ic

        g = model.gamma
        w = model.w

        # term-wise L1
        eff_w = w * model.mask
        l1_w = torch.sum(lambda_w_base * torch.abs(eff_w))
        l1_gamma = lambda_gamma * torch.abs(g)

        loss = loss_total + l1_w + l1_gamma
        loss.backward()
        opt.step()

    # Summary
    with torch.no_grad():
        stats0 = teacher_stats[0]
        v0 = stats0["v0"]
        traj_true = stats0["traj"]
        phi2_true = stats0["phi2"]
        H_true    = stats0["H"]

        vT_true = traj_true[-1]
        vT_pred, phi2_pred_T, H_pred_T, M_pred_T, phi4_pred_T = model.simulate(v0, Nt)
        misfit_final = F.mse_loss(vT_pred, vT_true).item()
        phi2_meta_model = model.simulate_phi2_traj(v0, 400).mean().item()

        eff = (model.gamma * model.w * model.mask).detach().cpu().numpy()

    print("[Stage 1][Summary] (IC #0) misfit≈{:.3e}, Φ²(T)={:.3e}, H(T)={:.3f}".format(
        misfit_final, phi2_pred_T.item(), H_pred_T.item()))
    print("  <Φ²>_meta(model, IC #0) = {:.3e}".format(phi2_meta_model))
    print("  γ(Stage 1) = {:+.3e}".format(model.gamma.item()))
    for i, name in enumerate(GRAMMAR_TERMS):
        print(f"  eff({name:8s} idx {i}) = {eff[i]:+.6e}")
    print()

    return model

model_stage1 = train_stage1()


# -------------------------
# 10. Adaptive pruning (floor_threshold 放宽到 1e-4)
# -------------------------
def adaptive_pruning_mask(model, floor_threshold=1e-4, factor=0.4):
    with torch.no_grad():
        eff = (model.gamma * model.w * model.mask).detach().cpu().numpy()
        abs_eff = np.abs(eff)
        med = np.median(abs_eff)
        if med < floor_threshold:
            tau = floor_threshold
        else:
            tau = max(floor_threshold, factor * med)
        keep = abs_eff >= tau
    return torch.tensor(keep.astype(np.float32), device=device), tau, eff

mask_stage2, tau_prune, eff_stage1 = adaptive_pruning_mask(model_stage1)
kept_indices = np.where(mask_stage2.detach().cpu().numpy() > 0.5)[0].tolist()
kept_terms = [GRAMMAR_TERMS[i] for i in kept_indices]
lambda_k = len(kept_indices) / n_terms

print("[Stage 2] Adaptive pruning threshold τ = {:.3e}".format(tau_prune))
print("  kept indices =", kept_indices)
print("  kept terms   =", kept_terms)
print("  λ_k = {:.3f} ({}/{})".format(lambda_k, len(kept_indices), n_terms))
print()


# -------------------------
# 11. Stage 2 training (pruned)
#     同样使用减弱后的 L1 权重
# -------------------------
def train_stage2(mask, epochs=80):
    model = GrammarTree72PDE(mask=mask).to(device)
    with torch.no_grad():
        model.w.copy_(model_stage1.w.detach())
        model.gamma.copy_(model_stage1.gamma.detach())

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    lambda_phi2 = 5e-2
    lambda_H    = 5e-2
    lambda_M    = 1e-2
    lambda_phi4 = 1e-2

    lambda_gamma = 1e-5
    lambda_w_base = torch.tensor([5e-5, 1e-5, 5e-5, 5e-5],
                                 dtype=DTYPE, device=device)

    print("[Stage 2] Refining pruned GrammarTree 7.2 (multi-IC, CORE_MODE=core+residual)...")
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss_total = 0.0
        for ic in range(n_ic):
            stats = teacher_stats[ic]
            v0 = stats["v0"]
            traj_true = stats["traj"]
            phi2_true = stats["phi2"]
            H_true    = stats["H"]
            M_true    = stats["M"]
            phi4_true = stats["phi4"]

            vT_true = traj_true[-1]
            phi2_true_T = phi2_true[-1]
            H_true_T    = H_true[-1]
            M_true_T    = M_true[-1]
            phi4_true_T = phi4_true[-1]

            vT_pred, phi2_pred_T, H_pred_T, M_pred_T, phi4_pred_T = model.simulate(v0, Nt)

            misfit = F.mse_loss(vT_pred, vT_true)

            loss_ic = misfit \
                    + lambda_phi2 * torch.abs(phi2_pred_T - phi2_true_T) \
                    + lambda_H    * torch.abs(H_pred_T    - H_true_T)    \
                    + lambda_M    * torch.abs(M_pred_T    - M_true_T)    \
                    + lambda_phi4 * torch.abs(phi4_pred_T - phi4_true_T)

            loss_total = loss_total + loss_ic

        g = model.gamma
        w = model.w

        eff_w = w * model.mask
        l1_w = torch.sum(lambda_w_base * torch.abs(eff_w))
        l1_gamma = lambda_gamma * torch.abs(g)

        loss = loss_total + l1_w + l1_gamma
        loss.backward()
        opt.step()

    with torch.no_grad():
        stats0 = teacher_stats[0]
        v0 = stats0["v0"]
        traj_true = stats0["traj"]
        phi2_true = stats0["phi2"]
        H_true    = stats0["H"]

        vT_true = traj_true[-1]
        vT_pred, phi2_pred_T, H_pred_T, M_pred_T, phi4_pred_T = model.simulate(v0, Nt)
        misfit_final = F.mse_loss(vT_pred, vT_true).item()
        phi2_meta_model = model.simulate_phi2_traj(v0, 400).mean().item()

        eff = (model.gamma * model.w * model.mask).detach().cpu().numpy()

    print("[Stage 2][Summary] (IC #0) misfit≈{:.3e}, Φ²(T)={:.3e}, H(T)={:.3f}".format(
        misfit_final, phi2_pred_T.item(), H_pred_T.item()))
    print("  <Φ²>_meta(model, IC #0) = {:.3e}".format(phi2_meta_model))
    print("  γ(Stage 2) = {:+.3e}".format(model.gamma.item()))
    for i, name in enumerate(GRAMMAR_TERMS):
        print(f"  eff({name:8s} idx {i}) = {eff[i]:+.6e}")
    print()

    return model, eff

model_stage2, eff_stage2 = train_stage2(mask_stage2)

# -------------------------
# 12. Evaluation
# -------------------------
err_arr = rollout_error(model_stage2, teacher_stats[0]["v0"], steps=400)
phi2_teacher_early = teacher_stats[0]["phi2"][:400]
phi2_model_early   = model_stage2.simulate_phi2_traj(teacher_stats[0]["v0"], 400)

print("[Eval] Mean L2 orbit error over [0,{:.3f}] = {:.3e}".format(
    400 * dt, err_arr.mean()))
print("[Eval] Early-time Φ² teacher≈{:.3e}, GT7.2≈{:.3e}, ratio≈{:.3f}".format(
    phi2_teacher_early.mean().item(),
    phi2_model_early.mean().item(),
    phi2_model_early.mean().item() / (phi2_teacher_early.mean().item() + 1e-12)))
print()

print("[GrammarTree 7.2] Effective hidden residual (vector Burgers prototype):")
for i, name in enumerate(GRAMMAR_TERMS):
    print(f"  eff({name:8s} idx {i}) Stage1 = {eff_stage1[i]:+.6e}, Stage2 = {eff_stage2[i]:+.6e}")
print()


# -------------------------
# 13. Stage 3: PySR audit (optional)
#     residual_x ≈ f(vx, lap_vx, |S|²)
# -------------------------
if DO_PYSR:
    print("[Stage 3] Building dataset for PySR: target = teacher_rhs(v) - core_rhs(v) ...")
    X_list = []
    R_list = []

    with torch.no_grad():
        for ic in range(n_ic):
            traj = teacher_stats[ic]["traj"]
            for n in range(0, 400):  # 前 400 步
                v = traj[n]      # [2, Nx, Ny]
                rhs_t = teacher_rhs(v)
                rhs_c = core_rhs(v)
                resid = rhs_t - rhs_c    # 理论上 ~ (nu_hidden * lap_v)

                feats = tensor_features(v)   # [n_terms, 2, Nx, Ny]
                v_field   = feats[0, 0]      # vx
                lap_vx    = feats[1, 0]      # lap_vx
                QS_vx     = feats[2, 0]      # Q_S v_x

                for i in range(Nx):
                    for j in range(Ny):
                        X_list.append([
                            float(v_field[i, j].item()),
                            float(lap_vx[i, j].item()),
                            float(QS_vx[i, j].item()),
                        ])
                        R_list.append(float(resid[0, i, j].item()))

    X = np.array(X_list, dtype=np.float64)
    R = np.array(R_list, dtype=np.float64)
    print(f"[Stage 3] Dataset shapes: X={X.shape}, R={R.shape}")

    print("\n[Stage 3] Running PySR symbolic regression (residual_x ≈ f(vx, lap_vx, QS_vx))...")
    model_pysr = PySRRegressor(
        niterations=800,
        populations=20,
        population_size=40,
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        maxsize=15,
        model_selection="best",
        elementwise_loss="L2DistLoss()",
        parallelism="serial",
        random_state=0,
    )
    model_pysr.fit(X, R, variable_names=["vx", "lap_vx", "QS_vx"])
    eqs = model_pysr.equations_
    best_row = eqs.sort_values("score", ascending=False).iloc[0]
    row_idx = int(best_row.name)

    print("\n[Stage 3] PySR best equation:")
    print("  equation  :", best_row["equation"])
    print(f"  loss      = {best_row['loss']:.3e}")
    print(f"  complexity= {int(best_row['complexity'])}")

    sym_list = model_pysr.sympy()
    sym_best = sym_list[row_idx]
    vx_s, lapvx_s, QSvx_s = sp.symbols("vx lapvx QSvx")

    # 只看 lap_vx 的系数： f(0,1,0) - f(0,0,0)
    c_lap = float(sym_best.subs({vx_s: 0.0, lapvx_s: 1.0, QSvx_s: 0.0})
                  - sym_best.subs({vx_s: 0.0, lapvx_s: 0.0, QSvx_s: 0.0}))

    true_c = nu_hidden
    abs_err = abs(c_lap - true_c)
    rel_err = abs_err / (true_c + 1e-12) * 100.0

    print("\n[Stage 3] Interpreted as residual_x ≈ c * lap_vx + ...")
    print(f"  c (PySR)     = {c_lap:.6f}")
    print(f"  c (target)   = {true_c:.6f}")
    print(f"  |Δc|         = {abs_err:.3e}")
    print(f"  relative err = {rel_err:.3f} %")

print("\n[Summary] GrammarTree 7.2 (tensor PDE, L1 weakened + PySR audit) finished.")
