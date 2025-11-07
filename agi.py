#@title Phase Transitions in Neural Solution Manifolds – Full Colab (with Section5/6 supported)

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
    f: 任意形状张量，表示一个时刻/一次采样的“场”。
    我们把所有元素拉平成一个向量来计算:
      Φ² = Var(f)
      H  = Σ_{i,j} (f_i - f_j)^2 = 2 N^2 Var
    """
    f_vec = f.reshape(-1)
    mean = f_vec.mean()
    var = ((f_vec - mean)**2).mean()
    N = f_vec.numel()
    H = 2.0 * (N**2) * var
    return H.item(), var.item()


def fit_loglog_H_Phi2(H_list, Phi2_list, verbose=True, label=""):
    """
    拟合 log H ≈ K_var * log Φ² + b
    并给出 K_eff = 2 * K_var（对应 log H vs log Φ 的指数）
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
    plt.scatter(np.log(Phi2_arr), np.log(H_arr), s=10, alpha=0.5)
    plt.xlabel("log Φ²")
    plt.ylabel("log H")
    plt.title(title)
    plt.show()


# ==========================================================
# 通用网络 & 数据
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


class TinyReasoner(nn.Module):
    """
    玩具“推理”网络：输入两个数，输出和；中间隐藏层作为解空间场。
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.enc = nn.Linear(2, hidden)
        self.act = nn.Tanh()
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.act(self.enc(x))
        y = self.head(h)
        return y, h


def generate_poly_data(batch_size=256, in_dim=8, out_dim=8):
    """
    简单多项式任务： y = x + x^2 (逐元素)
    """
    x = torch.randn(batch_size, in_dim)
    y = x + x**2
    if out_dim != in_dim:
        y = y[:, :out_dim]
        x = x[:, :in_dim]
    return x, y


def generate_reasoning_batch(batch_size=256):
    """
    玩具 reasoning：输入 [a,b]，输出 a+b
    """
    a = torch.randint(low=-50, high=50, size=(batch_size, 1)).float()
    b = torch.randint(low=-50, high=50, size=(batch_size, 1)).float()
    x = torch.cat([a, b], dim=1)
    y = a + b
    return x, y


# ==========================================================
# SECTION 1: Synthetic Solver
# ==========================================================

def run_section1_synthetic_solver(
    epochs=200, batch_size=256, in_dim=8, out_dim=8, log_every=20
):
    print("\n" + "="*60)
    print("SECTION 1: Synthetic Polynomial / Linear Solver")
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

    K_var, b, K_eff = fit_loglog_H_Phi2(H_list, Phi2_list, label="Section1")
    plot_loglog(H_list, Phi2_list, title="Section1: log H vs log Φ²")
    print(f"[Section1 Summary] K_var≈{K_var:.4f}, K_eff≈{K_eff:.4f}")
    return H_list, Phi2_list, (K_var, b, K_eff)


# ==========================================================
# SECTION 2: Geometry Ablation – Mean-field vs Chain
# ==========================================================

def build_graph_laplacian(N, mode="meanfield"):
    """
    返回 N×N 的 Laplacian (torch)
    mode = "meanfield" 或 "chain"
    """
    if mode == "meanfield":
        # 完全图：L = N I - 1 1^T
        L = N * np.eye(N) - np.ones((N, N))
    elif mode == "chain":
        L = np.zeros((N, N))
        for i in range(N):
            if i > 0:
                L[i, i] += 1
                L[i, i-1] -= 1
                L[i-1, i] -= 1
                L[i-1, i-1] += 1
    else:
        raise ValueError("unknown mode")
    return torch.tensor(L, dtype=torch.float32, device=device)


def geometry_regularizer(outputs, L):
    """
    outputs: (batch, N)
    L: (N, N)
    正则项：平均 batch 上的 f^T L f
    """
    f = outputs
    reg = 0.0
    for i in range(f.shape[0]):
        v = f[i]
        reg += torch.matmul(v, torch.matmul(L, v))
    reg = reg / f.shape[0]
    return reg


def run_section2_geometry_ablation(
    epochs=200, batch_size=256, N=16, lambda_reg=1e-4,
    mode="meanfield", log_every=20
):
    print("\n" + "="*60)
    print(f"SECTION 2: Geometry Ablation ({mode}, λ={lambda_reg})")
    print("="*60)

    in_dim = N
    out_dim = N
    net = SimpleSolverNet(in_dim=in_dim, out_dim=out_dim).to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    L = build_graph_laplacian(N, mode=mode)

    H_list, Phi2_list = [], []

    for epoch in range(1, epochs+1):
        x, y = generate_poly_data(batch_size=batch_size, in_dim=in_dim, out_dim=out_dim)
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        pred = net(x)
        loss_main = loss_fn(pred, y)
        reg = geometry_regularizer(pred, L)
        loss = loss_main + lambda_reg * reg
        loss.backward()
        opt.step()

        with torch.no_grad():
            H, Phi2 = compute_H_Phi2_from_tensor(pred)
            H_list.append(H)
            Phi2_list.append(Phi2)

        if epoch % log_every == 0 or epoch == 1:
            print(f"[Epoch {epoch:4d}] loss={loss.item():.4f} "
                  f"| main={loss_main.item():.4f} | reg={reg.item():.4e} "
                  f"| H={H:.1f} | Φ²={Phi2:.4f}")

    label = f"Section2-{mode}-λ={lambda_reg}"
    K_var, b, K_eff = fit_loglog_H_Phi2(H_list, Phi2_list, label=label)
    plot_loglog(H_list, Phi2_list, title=label)
    print(f"[Section2-{mode} Summary] K_var≈{K_var:.4f}, K_eff≈{K_eff:.4f}")
    return H_list, Phi2_list, (K_var, b, K_eff)


# ==========================================================
# SECTION 3: 3×3 Linear Systems & Poincaré K(t)
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
    """
    模拟 ẋ = A x，多个初始条件，采样 f(t) 并计算 H,Φ²。
    """
    dim = A.shape[0]
    x0 = torch.randn(n_traj, dim)
    x = x0.clone()

    A_mat = A.to(device)
    x = x.to(device)

    H_list, Phi2_list = [], []

    steps = int(T)
    for t in range(steps):
        dx = x @ A_mat.T
        x = x + dt * dx

        if t % sample_every == 0:
            with torch.no_grad():
                H, Phi2 = compute_H_Phi2_from_tensor(x)
                H_list.append(H)
                Phi2_list.append(Phi2)

    return H_list, Phi2_list


def run_section3_linear_system():
    print("\n" + "="*60)
    print("SECTION 3: 3×3 Linear Systems & Poincaré K(t)")
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
# SECTION 4: PDE – 1D Heat Equation (稳定显式)
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
        u = np.sin(np.pi * x) + 0.5 * np.sin(3*np.pi*x)
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
            u_new[i] = u[i] + alpha * (u[i+1] - 2*u[i] + u[i-1])
        u_new[0] = 0.0
        u_new[-1] = 0.0
        u = u_new

    return H_list, Phi2_list


def run_section4_pde():
    print("\n" + "="*60)
    print("SECTION 4: PDE – 1D Heat Equation")
    print("="*60)
    results = {}
    for profile in ["sin", "multi"]:
        print(f"\n[Profile={profile}]")
        H_list, Phi2_list = heat_equation_1d(
            N=64, T_steps=2000, kappa=0.1, dt=1e-3, profile=profile
        )
        label = f"Section4-{profile}"
        K_var, b, K_eff = fit_loglog_H_Phi2(H_list, Phi2_list, label=label)
        plot_loglog(H_list, Phi2_list, title=f"{label}: log H vs log Φ²")
        print(f"[{label} Summary] K_var≈{K_var:.4f}, K_eff≈{K_eff:.4f}")
        results[profile] = (H_list, Phi2_list, (K_var, b, K_eff))
    return results


# ==========================================================
# SECTION 5: Spectral Laplacian Model (带 α 相变)
# ==========================================================

def spectral_model_samples(alpha=2.0, K_modes=128, n_samples=2000):
    """
    谱模型带 cutoff：
      每个样本随机选 m ∈ {1,...,K_modes}
      k ≤ m: a_k ~ N(0,1), k>m: a_k=0
    则有近似：
      Φ² ~ m/K_modes
      H ~ Σ_{k=1}^m k^α ~ m^{α+1}
    => H ∝ (Φ²)^{α+1} => log H ≈ (α+1) log Φ² + const
    从而 K_var ≈ α+1, K_eff ≈ 2(α+1)
    """
    H_list, Phi2_list = [], []
    for _ in range(n_samples):
        m = np.random.randint(1, K_modes+1)  # cutoff
        a = np.zeros(K_modes, dtype=np.float64)
        a[:m] = np.random.randn(m)
        ks = np.arange(1, K_modes+1, dtype=np.float64)
        lambdas = ks**alpha

        Phi2 = np.mean(a**2)             # include zeros
        H = np.sum(lambdas * a**2)       # weighted energy

        H_list.append(H)
        Phi2_list.append(Phi2)

    return H_list, Phi2_list


def run_section5_spectral():
    print("\n" + "="*60)
    print("SECTION 5: Spectral Laplacian Model (with cutoff)")
    print("="*60)

    results = {}
    for alpha in [1.0, 2.0, 4.0]:
        H_list, Phi2_list = spectral_model_samples(alpha=alpha, K_modes=128, n_samples=2000)
        label = f"Section5-α={alpha}"
        K_var, b, K_eff = fit_loglog_H_Phi2(H_list, Phi2_list, label=label)
        plot_loglog(H_list, Phi2_list, title=f"{label}: log H vs log Φ²")
        print(f"[{label} Summary] K_var≈{K_var:.4f}, K_eff≈{K_eff:.4f}")
        results[alpha] = (H_list, Phi2_list, (K_var, b, K_eff))
    return results


# ==========================================================
# SECTION 6: Memory Geometry（能量累积版）
# ==========================================================

def build_memory_geometry(H_list, Phi2_list):
    """
    记忆几何改为“能量累积”：
      Φ_mem²(t) = ∑ Φ²(τ)
      H_mem(t)  = ∑ H(τ)
    都是单调非负，若瞬时满足 H ∝ Φ²，则累积后也应保持 K≈2。
    """
    H_arr = np.array(H_list, dtype=np.float64)
    Phi2_arr = np.array(Phi2_list, dtype=np.float64)

    Phi2_mem = np.cumsum(Phi2_arr)
    H_mem = np.cumsum(H_arr)
    return H_mem.tolist(), Phi2_mem.tolist()


def run_section6_memory_from_section1(H_list, Phi2_list):
    print("\n" + "="*60)
    print("SECTION 6: Memory Geometry (using Section1 trajectories)")
    print("="*60)

    H_mem, Phi2_mem = build_memory_geometry(H_list, Phi2_list)
    K_var, b, K_eff = fit_loglog_H_Phi2(H_mem, Phi2_mem, label="Section6-Memory")
    plot_loglog(H_mem, Phi2_mem, title="Section6: log H_mem vs log Φ_mem²")
    print(f"[Section6 Summary] K_var≈{K_var:.4f}, K_eff≈{K_eff:.4f}")
    return H_mem, Phi2_mem, (K_var, b, K_eff)


# ==========================================================
# SECTION 7: Duffing Oscillator
# ==========================================================

def duffing_step(x, v, dt, delta=0.2, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.0, t=0.0):
    """
    RK4 for Duffing: x'' + δ x' + α x + β x^3 = γ cos(ω t)
    """
    def f(x, v, t):
        a = -delta * v - alpha * x - beta * x**3 + gamma * math.cos(omega * t)
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
    """
    多初始条件 Duffing 集合动力学，采样 H,Φ²。
    我们这里把“能量” H_t = v^2 + x^2 + β x^4，当作结构场。
    """
    x = np.random.uniform(-1, 1, size=n_traj)
    v = np.random.uniform(-1, 1, size=n_traj)

    H_list, Phi2_list = [], []
    t = 0.0

    for step in range(T_steps):
        if step % 10 == 0:
            x_t = torch.tensor(x, dtype=torch.float32)
            v_t = torch.tensor(v, dtype=torch.float32)
            H_t = (v_t**2 + x_t**2 + beta * x_t**4)
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
# SECTION 8: Toy Reasoning Network
# ==========================================================

def run_section8_toy_reasoning(
    epochs=200, batch_size=256, hidden=64, log_every=20
):
    print("\n" + "="*60)
    print("SECTION 8: Toy Reasoning Network")
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
# 统一入口：按需要打开/关闭各 section
# ==========================================================

run_sec1 = True
run_sec2 = True
run_sec3 = True
run_sec4 = True
run_sec5 = True
run_sec6 = True  # 依赖 Section1 的 H1, Phi21
run_sec7 = True
run_sec8 = True

# 用于保存各节结果
results = {}

if run_sec1:
    H1, Phi21, K1 = run_section1_synthetic_solver()
    results["sec1"] = (H1, Phi21, K1)

if run_sec2:
    H2_mf, Phi22_mf, K2_mf = run_section2_geometry_ablation(
        mode="meanfield", lambda_reg=0.0
    )
    H2_chain, Phi22_chain, K2_chain = run_section2_geometry_ablation(
        mode="chain", lambda_reg=1e-3
    )
    results["sec2_meanfield"] = (H2_mf, Phi22_mf, K2_mf)
    results["sec2_chain"] = (H2_chain, Phi22_chain, K2_chain)

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
        H1, Phi21, K1 = results["sec1"]
        H6, Phi26, K6 = run_section6_memory_from_section1(H1, Phi21)
        results["sec6"] = (H6, Phi26, K6)
    else:
        print("Section6 需要 Section1 的轨迹，请先运行 Section1。")

if run_sec7:
    H7, Phi27, K7 = run_section7_duffing()
    results["sec7"] = (H7, Phi27, K7)

if run_sec8:
    H8, Phi28, K8 = run_section8_toy_reasoning()
    results["sec8"] = (H8, Phi28, K8)

print("\nAll selected sections finished.")
