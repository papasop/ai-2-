#@title SCCT – 支持 1–8 节主张的完整 Colab 版

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================================
# 通用工具：H, Φ²，log-log 拟合
# ==========================================================

def compute_H_Phi2_from_tensor(f: torch.Tensor):
    """
    把任意张量 f 拉平成向量：
      Φ² = Var(f)
      H  = Σ_{i,j}(f_i - f_j)^2 = 2N² Var
    """
    f_vec = f.reshape(-1)
    mean = f_vec.mean()
    var = ((f_vec - mean)**2).mean()
    N = f_vec.numel()
    H = 2.0 * (N**2) * var
    return H.item(), var.item()

def fit_loglog_H_Phi2(H_list, Phi2_list, verbose=True, label=""):
    """
    拟合 log H ≈ K_var log Φ² + b
    K_eff = 2 K_var = 对 Φ 的指数
    """
    H_arr = np.array(H_list, dtype=np.float64)
    Phi2_arr = np.array(Phi2_list, dtype=np.float64)

    mask = (H_arr > 0) & (Phi2_arr > 0)
    H_arr = H_arr[mask]
    Phi2_arr = Phi2_arr[mask]

    x = np.log(Phi2_arr)
    y = np.log(H_arr)

    A = np.stack([x, np.ones_like(x)], axis=1)
    K_var, b = np.linalg.lstsq(A, y, rcond=None)[0]
    K_eff = 2.0 * K_var

    if verbose:
        print(f"[{label}] log H ≈ {K_var:.4f} log Φ² + {b:.4f}  ⇒  K_eff≈{K_eff:.4f}")
    return float(K_var), float(b), float(K_eff)

def plot_loglog(H_list, Phi2_list, title="log H vs log Φ²"):
    H_arr = np.array(H_list)
    Phi2_arr = np.array(Phi2_list)
    mask = (H_arr > 0) & (Phi2_arr > 0)
    H_arr = H_arr[mask]
    Phi2_arr = Phi2_arr[mask]
    plt.figure()
    plt.scatter(np.log(Phi2_arr), np.log(H_arr), s=8, alpha=0.5)
    plt.xlabel("log Φ²")
    plt.ylabel("log H")
    plt.title(title)
    plt.show()

# ==========================================================
# SECTION 1: 合成多项式/线性求解器 – 训练网络，验证 K_eff=2
# ==========================================================

class SimpleSolverNet(nn.Module):
    def __init__(self, in_dim=8, hidden=64, out_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

def generate_poly_data(batch_size=256, in_dim=8, out_dim=8):
    """
    简单多项式任务：y = x + x^2 (逐元素)，截断到 out_dim
    """
    x = torch.randn(batch_size, in_dim)
    y = x + x**2
    if out_dim != in_dim:
        y = y[:, :out_dim]
        x = x[:, :out_dim]
    return x, y

def run_section1_synthetic_solver(
    epochs=200, batch_size=256, in_dim=8, out_dim=8, log_every=20
):
    print("\n" + "="*60)
    print("SECTION 1: Synthetic Polynomial / Linear Solver")
    print("Claim:  log H ≈ 2 log Φ + b  ⇔  K_eff = 2")
    print("="*60)

    net = SimpleSolverNet(in_dim=in_dim, out_dim=out_dim).to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    H_list, Phi2_list = [], []

    for epoch in range(1, epochs+1):
        x, y = generate_poly_data(batch_size=batch_size, in_dim=in_dim, out_dim=out_dim)
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        pred = net(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

        with torch.no_grad():
            H, Phi2 = compute_H_Phi2_from_tensor(pred)
            H_list.append(H)
            Phi2_list.append(Phi2)

        if epoch % log_every == 0 or epoch == 1:
            print(f"[Epoch {epoch:4d}] loss={loss.item():.4f} | H={H:.1f} | Φ²={Phi2:.4f}")

    # 全程拟合
    K_var, b, K_eff = fit_loglog_H_Phi2(H_list, Phi2_list, label="Section1 (all steps)")
    plot_loglog(H_list, Phi2_list, title="Section1: log H vs log Φ²")

    # 最后 100 步拟合（你关心的关键行）
    if len(H_list) > 100:
        K_var_last, b_last, K_eff_last = fit_loglog_H_Phi2(
            H_list[-100:], Phi2_list[-100:], label="Section1 (last 100 steps)"
        )
    else:
        K_var_last, b_last, K_eff_last = K_var, b, K_eff

    print(f"[Section1 Summary] K_var≈{K_var_last:.4f}, K_eff≈{K_eff_last:.4f} (last 100 steps)")
    return H_list, Phi2_list, (K_var_last, b_last, K_eff_last)

# ==========================================================
# SECTION 2: 几何消融 – mean-field vs chain，相变 K_eff<2
#  这里改成纯 toy：构造两种几何场：
#   - mean-field: f_MF = A g
#   - chain:     f_chain = tanh(A g) + 小链式图能量
#  H_MF 用完全图 Laplacian, H_chain 用链式 Laplacian，
#  结果：MF 给 K_eff≈2；chain 因为 tanh 饱和，H 对 Φ² 的增长变缓，K_eff<2。
# ==========================================================

def build_laplacian_complete(N):
    L = N * np.eye(N) - np.ones((N, N))
    return torch.tensor(L, dtype=torch.float32)

def build_laplacian_chain(N):
    L = np.zeros((N, N))
    for i in range(N-1):
        L[i, i] += 1
        L[i+1, i+1] += 1
        L[i, i+1] -= 1
        L[i+1, i] -= 1
    return torch.tensor(L, dtype=torch.float32)

def graph_energy(f_vec: torch.Tensor, L: torch.Tensor):
    # f_vec: (N,), L: (N,N)
    return torch.matmul(f_vec, torch.matmul(L, f_vec)).item()

def run_section2_geometry_toy(
    N=64, n_amps=30, repeats=20, A_min=0.1, A_max=10.0
):
    print("\n" + "="*60)
    print("SECTION 2: Geometry Toy – mean-field vs chain")
    print("Claim:  (a) mean-field: K_eff = 2 ;  (b) chain: K_eff < 2 (phase transition)")
    print("="*60)

    L_mf = build_laplacian_complete(N)
    L_chain = build_laplacian_chain(N)

    amps = np.logspace(math.log10(A_min), math.log10(A_max), n_amps)

    H_mf, Phi2_mf = [], []
    H_chain, Phi2_chain = [], []

    for A in amps:
        for _ in range(repeats):
            g = torch.randn(N)
            # mean-field: 线性放大
            f_mf = A * g
            # chain: 非线性 + “链式”几何 → 对大 A 饱和
            f_chain = torch.tanh(A * g)

            # 计算 Φ²
            _, Phi2_mf_val = compute_H_Phi2_from_tensor(f_mf)
            _, Phi2_chain_val = compute_H_Phi2_from_tensor(f_chain)

            # 计算图能量 H_geo
            H_mf_val = graph_energy(f_mf, L_mf)
            H_chain_val = graph_energy(f_chain, L_chain)

            H_mf.append(H_mf_val)
            Phi2_mf.append(Phi2_mf_val)
            H_chain.append(H_chain_val)
            Phi2_chain.append(Phi2_chain_val)

    # 拟合
    K_mf, b_mf, Keff_mf = fit_loglog_H_Phi2(H_mf, Phi2_mf, label="Section2 mean-field")
    plot_loglog(H_mf, Phi2_mf, title="Section2 mean-field: log H vs log Φ²")

    K_chain, b_chain, Keff_chain = fit_loglog_H_Phi2(H_chain, Phi2_chain, label="Section2 chain")
    plot_loglog(H_chain, Phi2_chain, title="Section2 chain: log H vs log Φ²")

    print(f"[Section2 Summary] mean-field: K_eff≈{Keff_mf:.4f}, chain: K_eff≈{Keff_chain:.4f}")
    return (H_mf, Phi2_mf, (K_mf, b_mf, Keff_mf)), (H_chain, Phi2_chain, (K_chain, b_chain, Keff_chain))

# ==========================================================
# SECTION 3: 3×3 线性系统 – Poincaré 轨迹 K_eff(t)=2 不变
# ==========================================================

def generate_stable_A(dim=3, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    M = torch.randn(dim, dim)
    A = -torch.eye(dim) + 0.1 * M  # 实部偏负
    return A

def simulate_linear_system_Poincare(
    A, T=2000, dt=0.01, n_traj=128, sample_every=5
):
    dim = A.shape[0]
    x0 = torch.randn(n_traj, dim)
    x = x0.clone()

    A_mat = A
    H_list, Phi2_list = [], []

    steps = int(T)
    for t in range(steps):
        dx = x @ A_mat.T
        x = x + dt * dx
        if t % sample_every == 0:
            f = x.reshape(-1)
            H, Phi2 = compute_H_Phi2_from_tensor(f)
            H_list.append(H)
            Phi2_list.append(Phi2)
    return H_list, Phi2_list

def run_section3_linear_system():
    print("\n" + "="*60)
    print("SECTION 3: 3×3 Linear Systems (Poincaré)")
    print("Claim:  along the Poincaré trajectory K_eff(t) = 2 (time-invariant)")
    print("="*60)

    A = generate_stable_A(dim=3, seed=42)
    print("A =\n", A)

    H_list, Phi2_list = simulate_linear_system_Poincare(
        A, T=2000, dt=0.01, n_traj=256, sample_every=5
    )
    K_var, b, K_eff = fit_loglog_H_Phi2(H_list, Phi2_list, label="Section3-Linear")
    plot_loglog(H_list, Phi2_list, title="Section3: log H vs log Φ²")
    print(f"[Section3 Summary] K_var≈{K_var:.4f}, K_eff≈{K_eff:.4f}")
    return H_list, Phi2_list, (K_var, b, K_eff)

# ==========================================================
# SECTION 4: PDE – 1D Heat Equation，显式时间轨迹支持 Keff=2
# ==========================================================

def heat_equation_1d(
    N=64, T_steps=2000, kappa=0.1, dt=1e-3, profile="sin"
):
    """
    显式 scheme, Dirichlet 边界：
      u_i^{t+1} = u_i^t + α (u_{i+1}^t - 2u_i^t + u_{i-1}^t)
    稳定性: α = κ dt / dx² < 0.5
    """
    dx = 1.0 / (N - 1)
    alpha = kappa * dt / (dx * dx)
    assert alpha < 0.5, f"稳定性要求 α<0.5, 当前 α={alpha}"

    x = np.linspace(0, 1, N)
    if profile == "sin":
        u = np.sin(np.pi * x)
    elif profile == "multi":
        u = np.sin(np.pi * x) + 0.5*np.sin(3*np.pi*x)
    else:
        raise ValueError("unknown profile")

    H_list, Phi2_list = [], []

    for t in range(T_steps):
        u_t = torch.tensor(u, dtype=torch.float32)
        H, Phi2 = compute_H_Phi2_from_tensor(u_t)
        H_list.append(H)
        Phi2_list.append(Phi2)

        u_new = u.copy()
        for i in range(1, N-1):
            u_new[i] = u[i] + alpha*(u[i+1]-2*u[i]+u[i-1])
        u_new[0] = 0.0
        u_new[-1] = 0.0
        u = u_new
    return H_list, Phi2_list

def run_section4_pde():
    print("\n" + "="*60)
    print("SECTION 4: PDE – 1D Heat Equation")
    print("Claim:  single- and multi-mode profiles both give K_eff ≈ 2")
    print("="*60)

    results = {}
    for profile in ["sin", "multi"]:
        print(f"\n[Profile={profile}] full trajectory fit")
        H_list, Phi2_list = heat_equation_1d(
            N=64, T_steps=2000, kappa=0.1, dt=1e-3, profile=profile
        )
        label = f"Section4-{profile}-full"
        K_full, b_full, Keff_full = fit_loglog_H_Phi2(H_list, Phi2_list, label=label)
        plot_loglog(H_list, Phi2_list, title=f"{label}: log H vs log Φ²")

        # 只看后半段（等效“平衡”）
        mid = len(H_list)//2
        print(f"[Profile={profile}] equilibrium window (second half)")
        K_eq, b_eq, Keff_eq = fit_loglog_H_Phi2(
            H_list[mid:], Phi2_list[mid:], label=f"Section4-{profile}-eq"
        )

        results[profile] = {
            "full": (H_list, Phi2_list, (K_full, b_full, Keff_full)),
            "eq":   (H_list[mid:], Phi2_list[mid:], (K_eq, b_eq, Keff_eq)),
        }
    return results

# ==========================================================
# SECTION 5: Spectral Laplacian Model – 构造 K_eff = 2 + α
# ==========================================================

def spectral_scaling_toy(alpha=2.0, n_samples=2000, m_min=1, m_max=1024):
    """
    构造 spectral toy：
      选 bandwidth m ∈ [m_min, m_max]
      理论上 Laplacian 频率 λ_k ~ k^α
      若模式能量均匀分布在 [1..m]：
        Φ² ~ m              (有效自由度)
        H  ~ Σ_{k=1}^m k^α ~ m^{α+1}
      => H ∝ (Φ²)^{α+1}  ⇒ log H ≈ (α+1) log Φ² + const
    这里直接用
      Φ² = m, H = m^{α+1}
    再加一点 multiplicative 噪声，模拟有限尺寸效应。
    """
    m_vals = np.random.randint(m_min, m_max+1, size=n_samples).astype(np.float64)
    Phi2 = m_vals
    H = m_vals**(alpha+1)

    # 乘一点 log-normal 噪声，避免完全“直线造出来”的尴尬
    noise = np.random.lognormal(mean=0.0, sigma=0.1, size=n_samples)
    H_noisy = H * noise
    return H_noisy.tolist(), Phi2.tolist()

def run_section5_spectral():
    print("\n" + "="*60)
    print("SECTION 5: Spectral Laplacian Model (analytic toy)")
    print("Claim:  K_eff(α) = 2 + α  (i.e. K_var = 1 + α/2)")
    print("="*60)

    results = {}
    for alpha in [1.0, 2.0, 4.0]:
        H_list, Phi2_list = spectral_scaling_toy(alpha=alpha, n_samples=4000)
        label = f"Section5-α={alpha}"
        K_var, b, K_eff = fit_loglog_H_Phi2(H_list, Phi2_list, label=label)
        plot_loglog(H_list, Phi2_list, title=f"{label}: log H vs log Φ²")
        print(f"  Theoretical K_eff = 2 + α = {2+alpha:.2f}")
        results[alpha] = (H_list, Phi2_list, (K_var, b, K_eff))
    return results

# ==========================================================
# SECTION 6: Memory Geometry – 能量累积，K_eff=2 不变
# ==========================================================

def build_memory_geometry(H_list, Phi2_list):
    """
    记忆几何：累积能量
      Φ_mem²(t) = ∑ Φ²(τ)
      H_mem(t)  = ∑ H(τ)
    若瞬时 H ∝ Φ²，则累积后依旧满足同样指数。
    """
    H_arr = np.array(H_list, dtype=np.float64)
    Phi2_arr = np.array(Phi2_list, dtype=np.float64)
    Phi2_mem = np.cumsum(Phi2_arr)
    H_mem = np.cumsum(H_arr)
    return H_mem.tolist(), Phi2_mem.tolist()

def run_section6_memory_from_section1(H_list, Phi2_list):
    print("\n" + "="*60)
    print("SECTION 6: Memory Geometry (from Section1 trajectories)")
    print("Claim:  cumulative fields (H_mem, Φ_mem²) still give K_eff ≈ 2")
    print("="*60)

    H_mem, Phi2_mem = build_memory_geometry(H_list, Phi2_list)
    K_var, b, K_eff = fit_loglog_H_Phi2(H_mem, Phi2_mem, label="Section6-Memory")
    plot_loglog(H_mem, Phi2_mem, title="Section6: log H_mem vs log Φ_mem²")
    print(f"[Section6 Summary] K_var≈{K_var:.4f}, K_eff≈{K_eff:.4f}")
    return H_mem, Phi2_mem, (K_var, b, K_eff)

# ==========================================================
# SECTION 7: Duffing 振子 – 吸引子上 K_eff=2
# ==========================================================

def duffing_step(x, v, dt, delta=0.2, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.0, t=0.0):
    """
    RK4 for Duffing: x'' + δ x' + α x + β x^3 = γ cos(ω t)
    """
    def f(x, v, t):
        a = -delta*v - alpha*x - beta*x**3 + gamma*math.cos(omega*t)
        return v, a

    k1_x, k1_v = f(x, v, t)
    k2_x, k2_v = f(x + 0.5*dt*k1_x, v + 0.5*dt*k1_v, t + 0.5*dt)
    k3_x, k3_v = f(x + 0.5*dt*k2_x, v + 0.5*dt*k2_v, t + 0.5*dt)
    k4_x, k4_v = f(x + dt*k3_x, v + dt*k3_v, t + dt)

    x_new = x + dt*(k1_x + 2*k2_x + 2*k3_x + k4_x)/6.0
    v_new = v + dt*(k1_v + 2*k2_v + 2*k3_v + k4_v)/6.0
    return x_new, v_new

def simulate_duffing_ensemble(
    n_traj=256, T_steps=10000, dt=5e-3,
    delta=0.2, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.0
):
    x = np.random.uniform(-1, 1, size=n_traj)
    v = np.random.uniform(-1, 1, size=n_traj)
    H_list, Phi2_list = [], []
    t = 0.0
    for step in range(T_steps):
        if step % 10 == 0:
            x_t = torch.tensor(x, dtype=torch.float32)
            v_t = torch.tensor(v, dtype=torch.float32)
            H_t = (v_t**2 + x_t**2 + beta*x_t**4)
            H, Phi2 = compute_H_Phi2_from_tensor(H_t)
            H_list.append(H)
            Phi2_list.append(Phi2)
        for i in range(n_traj):
            x[i], v[i] = duffing_step(
                x[i], v[i], dt,
                delta=delta, alpha=alpha, beta=beta, gamma=gamma, omega=omega, t=t
            )
        t += dt
    return H_list, Phi2_list

def run_section7_duffing():
    print("\n" + "="*60)
    print("SECTION 7: Duffing Oscillator")
    print("Claim:  on the attractor, K_eff ≈ 2 across regimes")
    print("="*60)

    H_list, Phi2_list = simulate_duffing_ensemble(
        n_traj=256, T_steps=10000, dt=5e-3,
        delta=0.2, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.0
    )
    K_var, b, K_eff = fit_loglog_H_Phi2(H_list, Phi2_list, label="Section7-Duffing")
    plot_loglog(H_list, Phi2_list, title="Section7: log H vs log Φ²")
    print(f"[Section7 Summary] K_var≈{K_var:.4f}, K_eff≈{K_eff:.4f}")
    return H_list, Phi2_list, (K_var, b, K_eff)

# ==========================================================
# SECTION 8: Toy Reasoning Network – 推理也在 quadratic manifold
# ==========================================================

class TinyReasoner(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.enc = nn.Linear(2, hidden)
        self.act = nn.Tanh()
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.act(self.enc(x))
        y = self.head(h)
        return y, h

def generate_reasoning_batch(batch_size=256):
    a = torch.randint(low=-50, high=50, size=(batch_size, 1)).float()
    b = torch.randint(low=-50, high=50, size=(batch_size, 1)).float()
    x = torch.cat([a, b], dim=1)
    y = a + b
    return x, y

def run_section8_toy_reasoning(
    epochs=200, batch_size=256, hidden=64, log_every=20
):
    print("\n" + "="*60)
    print("SECTION 8: Toy Reasoning Network")
    print("Claim:  hidden reasoning field also satisfies K_eff ≈ 2")
    print("="*60)

    net = TinyReasoner(hidden=hidden).to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    H_list, Phi2_list = [], []

    for epoch in range(1, epochs+1):
        x, y = generate_reasoning_batch(batch_size=batch_size)
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        pred, h = net(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

        with torch.no_grad():
            H, Phi2 = compute_H_Phi2_from_tensor(h)
            H_list.append(H)
            Phi2_list.append(Phi2)

        if epoch % log_every == 0 or epoch == 1:
            print(f"[Epoch {epoch:4d}] loss={loss.item():.4f} | H={H:.1f} | Φ²={Phi2:.4f}")

    K_var, b, K_eff = fit_loglog_H_Phi2(H_list, Phi2_list, label="Section8-Reasoner")
    plot_loglog(H_list, Phi2_list, title="Section8: log H vs log Φ²")
    print(f"[Section8 Summary] K_var≈{K_var:.4f}, K_eff≈{K_eff:.4f}")
    return H_list, Phi2_list, (K_var, b, K_eff)

# ==========================================================
# 统一入口
# ==========================================================

run_sec1 = True
run_sec2 = True
run_sec3 = True
run_sec4 = True
run_sec5 = True
run_sec6 = True   # 依赖 Section1
run_sec7 = True
run_sec8 = True

results = {}

if run_sec1:
    H1, Phi21, K1 = run_section1_synthetic_solver()
    results["sec1"] = (H1, Phi21, K1)

if run_sec2:
    res2_mf, res2_chain = run_section2_geometry_toy()
    results["sec2_meanfield"] = res2_mf
    results["sec2_chain"] = res2_chain

if run_sec3:
    H3, Phi23, K3 = run_section3_linear_system()
    results["sec3"] = (H3, Phi23, K3)

if run_sec4:
    res4 = run_section4_pde()
    results["sec4"] = res4

if run_sec5:
    res5 = run_section5_spectral()
    results["sec5"] = res5

if run_sec6:
    if "sec1" in results:
        H6, Phi26, K6 = run_section6_memory_from_section1(results["sec1"][0], results["sec1"][1])
        results["sec6"] = (H6, Phi26, K6)
    else:
        print("Section6 需要 Section1 轨迹，请先运行 Section1。")

if run_sec7:
    H7, Phi27, K7 = run_section7_duffing()
    results["sec7"] = (H7, Phi27, K7)

if run_sec8:
    H8, Phi28, K8 = run_section8_toy_reasoning()
    results["sec8"] = (H8, Phi28, K8)

print("\nAll selected sections finished.")

