# ================================
# 后半段：J_pinn, JNet, log-JNet, 地形 + ControlNet
# ================================
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

pi = np.pi
T_final = 1.0
theta1_min, theta1_max = 0.5, 2.0
theta2_min, theta2_max = -1.0, 1.0

lambda1, lambda2 = 0.1, 0.1  # 和前面保持一致

# ===== 真解 & 真 J =====
def u_true_np(t, x, theta1, theta2):
    return np.exp((-(pi**2)*theta1 + theta2) * t) * np.sin(pi * x)

def J_true_np(theta1, theta2, lambda1=lambda1, lambda2=lambda2,
              Nt=200, Nx=200):
    t_vals = np.linspace(0.0, T_final, Nt)
    x_vals = np.linspace(0.0, 1.0, Nx)
    tt, xx = np.meshgrid(t_vals, x_vals, indexing="ij")
    u = u_true_np(tt, xx, theta1, theta2)
    L = 0.5 * (u**2 + lambda1 * theta1**2 + lambda2 * theta2**2)
    Lx = np.trapezoid(L, x_vals, axis=1)
    J = np.trapezoid(Lx, t_vals)
    return J

# ===== 用 PINN 做 J_pinn(θ1,θ2) 积分 =====
def J_pinn_theta(theta_t, num_samples=5000,
                 lambda1=lambda1, lambda2=lambda2):
    """
    theta_t: [B,2] -> (θ1,θ2)
    返回 [B,1]: J_pinn(θ1,θ2) ≈ ∬ 0.5*(u^2 + λ1 a1^2 + λ2 a2^2) dx dt
    依赖已经训练好的 u_net, a_net
    """
    u_net.eval()
    a_net.eval()
    B = theta_t.shape[0]

    theta1 = theta_t[:, 0:1]  # [B,1]
    theta2 = theta_t[:, 1:2]  # [B,1]

    t = torch.rand(B * num_samples, 1, device=device) * T_final
    x = torch.rand(B * num_samples, 1, device=device)

    theta1_rep = theta1.repeat_interleave(num_samples, dim=0)
    theta2_rep = theta2.repeat_interleave(num_samples, dim=0)

    inp = torch.cat([t, x, theta1_rep, theta2_rep], dim=1)  # [B*num_samples,4]
    with torch.no_grad():
        u_pred = u_net(inp)      # [B*num_samples,1]
        a_pred = a_net(inp)      # [B*num_samples,2]
    a1 = a_pred[:, 0:1]
    a2 = a_pred[:, 1:2]

    L = 0.5 * (u_pred**2 + lambda1 * a1**2 + lambda2 * a2**2)
    L = L.view(B, num_samples, 1)
    J = L.mean(dim=1)
    return J  # [B,1]

print("[测试 J_pinn 与 J_true 差距]")
test_thetas_2d = [
    (0.5, -0.5),
    (1.0,  0.0),
    (1.5,  0.3),
    (2.0,  0.8),
]
for t1, t2 in test_thetas_2d:
    θ_t = torch.tensor([[t1, t2]], dtype=torch.float32, device=device)
    Jp = J_pinn_theta(θ_t).item()
    Jt = float(J_true_np(t1, t2))
    print(f"θ=({t1:.2f},{t2:.2f})  J_pinn={Jp:.6f}  J_true={Jt:.6f}  diff={Jp-Jt:+.2e}")


# ===== JNet 定义 =====
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64, depth=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class JNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP(2, 1, hidden=64, depth=3)
    def forward(self, theta):
        return self.net(theta)

# ===== 生成 JNet 训练数据 =====
print("\n[JNet] 生成训练数据 (用 PINN+变分积分 做老师)...")
n1, n2 = 20, 20
theta1_vals = torch.linspace(theta1_min, theta1_max, n1, device=device)
theta2_vals = torch.linspace(theta2_min, theta2_max, n2, device=device)
Θ1_grid, Θ2_grid = torch.meshgrid(theta1_vals, theta2_vals, indexing="ij")
theta_train = torch.stack([Θ1_grid.reshape(-1), Θ2_grid.reshape(-1)], dim=1)  # [n1*n2,2]

with torch.no_grad():
    J_train = J_pinn_theta(theta_train, num_samples=3000)

print("  theta_train shape:", theta_train.shape, " J_train shape:", J_train.shape)

# ===== 训练 JNet(θ1,θ2) =====
j_net = JNet2D().to(device)
optimizer_j = torch.optim.Adam(j_net.parameters(), lr=1e-3)
mse = nn.MSELoss()

print("\n[JNet 架构]\n", j_net)
print("\n========== 开始训练 JNet(θ1,θ2) ==========")

for epoch in range(1, 2001):
    j_net.train()
    pred = j_net(theta_train)
    loss_j = mse(pred, J_train)

    optimizer_j.zero_grad()
    loss_j.backward()
    optimizer_j.step()

    if epoch == 1 or epoch % 400 == 0:
        with torch.no_grad():
            for (t1, t2) in [(0.5, -0.5), (1.0, 0.0), (2.0, 0.8)]:
                θ_t = torch.tensor([[t1, t2]], dtype=torch.float32, device=device)
                Jp = J_pinn_theta(θ_t).item()
                Jn = j_net(θ_t).item()
                Jt = float(J_true_np(t1, t2))
                print(f"[JNet] Epoch {epoch:4d}/2000 | θ=({t1:.1f},{t2:.1f}) | "
                      f"J_net={Jn:+.6f} J_pinn={Jp:+.6f} J_true≈{Jt:+.6f} (net-pinn={Jn-Jp:+.2e})")
        print("-" * 60)

print("\nJNet(θ1,θ2) 训练完成 ✅\n")

# ===== （可选）log-JNet =====
use_log_J = True
if use_log_J:
    print("[log-JNet] 开始训练 log-JNet(θ1,θ2)")
    j_net_log = JNet2D().to(device)
    opt_log = torch.optim.Adam(j_net_log.parameters(), lr=1e-3)
    eps = 1e-6
    logJ_train = torch.log(J_train + eps)

    for epoch in range(1, 1201):
        j_net_log.train()
        pred_log = j_net_log(theta_train)
        loss_log = mse(pred_log, logJ_train)

        opt_log.zero_grad()
        loss_log.backward()
        opt_log.step()

        if epoch == 1 or epoch % 300 == 0:
            with torch.no_grad():
                for (t1, t2) in [(0.5, -0.5), (1.0, 0.0), (2.0, 0.8)]:
                    θ_t = torch.tensor([[t1, t2]], dtype=torch.float32, device=device)
                    Jp = J_pinn_theta(θ_t).item()
                    Jn_log = torch.exp(j_net_log(θ_t)).item()
                    Jt = float(J_true_np(t1, t2))
                    print(f"[logJNet] Epoch {epoch:4d}/1200 | θ=({t1:.1f},{t2:.1f}) | "
                          f"exp(J_net_log)={Jn_log:+.6f} J_pinn={Jp:+.6f} J_true≈{Jt:+.6f} "
                          f"(logNet-pinn={Jn_log-Jp:+.2e})")
            print("-" * 60)
    print("\nlog-JNet(θ1,θ2) 训练完成 ✅\n")

# ===== θ-平面地形: J_true / J_net / exp(log-J_net) =====
print("[构建 θ-平面上的 J(θ1,θ2) 地形]")

theta1_plot = np.linspace(theta1_min, theta1_max, 60)
theta2_plot = np.linspace(theta2_min, theta2_max, 60)
Θ1_plot, Θ2_plot = np.meshgrid(theta1_plot, theta2_plot, indexing="ij")

J_true_grid = np.zeros_like(Θ1_plot)
for i in range(Θ1_plot.shape[0]):
    for j in range(Θ1_plot.shape[1]):
        J_true_grid[i, j] = J_true_np(Θ1_plot[i, j], Θ2_plot[i, j])

theta_flat = np.stack([Θ1_plot.reshape(-1), Θ2_plot.reshape(-1)], axis=1)
theta_flat_torch = torch.tensor(theta_flat, dtype=torch.float32, device=device)
with torch.no_grad():
    J_net_flat = j_net(theta_flat_torch).cpu().numpy().reshape(Θ1_plot.shape)
    if use_log_J:
        J_lognet_flat = torch.exp(j_net_log(theta_flat_torch)).cpu().numpy().reshape(Θ1_plot.shape)

# 真 J 的谷底
idx_min_true = np.argmin(J_true_grid)
i_min, j_min = np.unravel_index(idx_min_true, J_true_grid.shape)
theta1_star_true = Θ1_plot[i_min, j_min]
theta2_star_true = Θ2_plot[i_min, j_min]
J_star_true = J_true_grid[i_min, j_min]
print(f"  J_true 最小值大约在 θ*≈({theta1_star_true:.4f}, {theta2_star_true:.4f}), "
      f"J_true≈{J_star_true:.6f}")

fig, axs = plt.subplots(1, 2, figsize=(11, 4))

cs1 = axs[0].contourf(Θ1_plot, Θ2_plot, J_true_grid, levels=30)
axs[0].plot(theta1_star_true, theta2_star_true, 'r*', markersize=10, label='true min')
axs[0].set_title("J_true(θ1,θ2)")
axs[0].set_xlabel("θ1")
axs[0].set_ylabel("θ2")
axs[0].legend()
fig.colorbar(cs1, ax=axs[0])

cs2 = axs[1].contourf(Θ1_plot, Θ2_plot, J_net_flat, levels=30)
axs[1].plot(theta1_star_true, theta2_star_true, 'r*', markersize=10, label='true min')
axs[1].set_title("J_net(θ1,θ2)")
axs[1].set_xlabel("θ1")
axs[1].set_ylabel("θ2")
axs[1].legend()
fig.colorbar(cs2, ax=axs[1])

plt.tight_layout()
plt.show()

if use_log_J:
    plt.figure(figsize=(5,4))
    cs = plt.contourf(Θ1_plot, Θ2_plot, J_lognet_flat, levels=30)
    plt.plot(theta1_star_true, theta2_star_true, 'r*', markersize=10, label='true min')
    plt.title("exp(log-JNet)(θ1,θ2)")
    plt.xlabel("θ1")
    plt.ylabel("θ2")
    plt.legend()
    plt.colorbar(cs)
    plt.tight_layout()
    plt.show()

# ===== ControlNet: 在 J_net(θ1,θ2) 上跑梯度 =====
print("\n[Control] 在 J_net(θ1,θ2) 上做梯度下降寻找 θ*")

theta_control = torch.tensor([1.0, 0.0], dtype=torch.float32, device=device, requires_grad=True)
opt_control = torch.optim.Adam([theta_control], lr=5e-2)
num_steps = 200

theta_path = []

for step in range(1, num_steps + 1):
    opt_control.zero_grad()
    θ_in = theta_control.view(1, 2)
    J_est = j_net(θ_in)
    J_est_scalar = J_est[0, 0]
    J_est_scalar.backward()
    opt_control.step()

    with torch.no_grad():
        theta_control.data[0].clamp_(theta1_min, theta1_max)
        theta_control.data[1].clamp_(theta2_min, theta2_max)
        theta_path.append(theta_control.detach().cpu().numpy().copy())

    if step == 1 or step % 20 == 0 or step == num_steps:
        θ1_val, θ2_val = theta_control.detach().cpu().numpy()
        with torch.no_grad():
            Jp = J_pinn_theta(theta_control.view(1, 2)).item()
        Jt = float(J_true_np(θ1_val, θ2_val))
        print(f"[step {step:03d}] θ=({θ1_val:+.4f},{θ2_val:+.4f}) | "
              f"J_net(θ)={J_est_scalar.item():.6f} | "
              f"J_pinn(θ)={Jp:.6f} | J_true(θ)≈{Jt:.6f}")

theta_path = np.array(theta_path)
theta_star_ctrl = theta_control.detach().cpu().numpy()
J_star_ctrl_true = float(J_true_np(theta_star_ctrl[0], theta_star_ctrl[1]))

print("\n[Control] 训练结束 ✅")
print(f"  学到的最优控制 θ* ≈ ({theta_star_ctrl[0]:.6f}, {theta_star_ctrl[1]:.6f})")
print(f"  对应 J_true(θ*) ≈ {J_star_ctrl_true:.6f}")

plt.figure(figsize=(6,5))
cs = plt.contourf(Θ1_plot, Θ2_plot, J_true_grid, levels=30)
plt.plot(theta1_star_true, theta2_star_true, 'y*', markersize=12, label='true min')
plt.plot(theta_path[:,0], theta_path[:,1], 'r.-', label='ControlNet path')
plt.xlabel("θ1")
plt.ylabel("θ2")
plt.title("ControlNet trajectory on J_true(θ1,θ2)")
plt.legend()
plt.colorbar(cs)
plt.tight_layout()
plt.show()
