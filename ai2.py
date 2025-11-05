# ============================================================
# Colab 一键运行：Quadratic Closure Law 验证 + 证伪测试
# ============================================================

import math, random, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 固定随机种子，方便复现
SEED = 20251106
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ------------------------------------------------------------
# 1. 构造几何：链 / 完全图 / 星形 图的 Laplacian
# ------------------------------------------------------------
def make_laplacian(N, graph_type="chain"):
    A = np.zeros((N, N), dtype=np.float32)
    if graph_type == "chain":
        for i in range(N-1):
            A[i, i+1] = 1.0
            A[i+1, i] = 1.0
    elif graph_type == "complete":
        A[:] = 1.0
        np.fill_diagonal(A, 0.0)
    elif graph_type == "star":
        # 节点 0 为中心
        for i in range(1, N):
            A[0, i] = 1.0
            A[i, 0] = 1.0
    else:
        raise ValueError("Unknown graph_type: " + str(graph_type))
    D = np.diag(A.sum(axis=1))
    L = D - A
    return torch.tensor(L, dtype=torch.float32, device=device)


# ------------------------------------------------------------
# 2. 几何能量 H_geo 与 “流形体积” Phi 的定义
#   - H_geo = 平均 f^T L f
#   - Phi^2 = Var(activations) (这里简单用所有节点的总体方差)
# ------------------------------------------------------------
def compute_H_and_Phi(h, L):
    """
    h: [B, N] 第二隐层激活
    L: [N, N] Laplacian
    """
    B, N = h.shape
    # H_geo = 平均样本的 f^T L f
    # [B,N] @ [N,N] -> [B,N] -> 每行与自身点乘
    fLf = torch.einsum("bi,ij,bj->b", h, L, h)
    H_geo = fLf.mean().item()

    # Phi^2 = activations 的总体方差
    x = h.detach().cpu().numpy().reshape(-1)
    Phi2 = float(np.var(x))
    Phi = math.sqrt(Phi2 + 1e-12)
    return H_geo, Phi


# ------------------------------------------------------------
# 3. 简单数据集：6x6 输入 → 二分类
# ------------------------------------------------------------
def make_dataset(num_samples=4096, in_dim=36):
    X = np.random.randn(num_samples, in_dim).astype(np.float32)
    # 随机线性超平面
    w = np.random.randn(in_dim).astype(np.float32)
    logits = X @ w
    y = (logits > 0).astype(np.float32)  # 0/1 标签
    X = torch.tensor(X)
    y = torch.tensor(y).view(-1, 1)
    return X, y

X, y = make_dataset()
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# ------------------------------------------------------------
# 4. 模型定义：36 → 6 → 6 → 1
# ------------------------------------------------------------
class SmallNet(nn.Module):
    def __init__(self, in_dim=36, width=6, activation="relu"):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 1)
        self.activation = activation.lower()

    def act(self, x):
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "tanh":
            return torch.tanh(x)
        elif self.activation == "linear":
            return x
        else:
            raise ValueError("Unknown activation: " + str(self.activation))

    def forward(self, x, return_h2=False):
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        out = self.fc3(h2)
        if return_h2:
            return out, h2
        return out


# ------------------------------------------------------------
# 5. 拟合 K：log H = K log Phi + b
# ------------------------------------------------------------
def fit_K(logPhi, logH):
    import numpy as np
    x = np.array(logPhi)
    y = np.array(logH)
    # 去掉异常与 NaN
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5 or np.std(x) < 1e-6:
        return float("nan"), float("nan")
    K, b = np.polyfit(x, y, 1)
    # R^2
    y_pred = K * x + b
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-18
    R2 = 1 - ss_res / ss_tot
    return float(K), float(R2)


def instantaneous_K(logPhi, logH, window=5):
    import numpy as np
    x = np.array(logPhi)
    y = np.array(logH)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    Ks = []
    for i in range(len(x) - window + 1):
        xx = x[i:i+window]
        yy = y[i:i+window]
        if np.std(xx) < 1e-6:
            continue
        k, _ = np.polyfit(xx, yy, 1)
        Ks.append(k)
    Ks = np.array(Ks)
    if len(Ks) == 0:
        return float("nan"), float("nan")
    return float(Ks.mean()), float(Ks.std())


# ------------------------------------------------------------
# 6. 单个配置的实验封装
# ------------------------------------------------------------
def run_experiment(
    name,
    graph_type="chain",
    activation="relu",
    lambda_geo=0.0,
    optimizer_type="sgd",
    epochs=200,
    width=6,
    lr=1e-2
):
    print(f"\n=== 运行: {name} ===")
    model = SmallNet(in_dim=36, width=width, activation=activation).to(device)
    if optimizer_type.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_type.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Unknown optimizer: " + optimizer_type)

    criterion = nn.BCEWithLogitsLoss()
    N = width
    L = make_laplacian(N, graph_type=graph_type)

    H_hist, Phi_hist = [], []

    X_all, y_all = X.to(device), y.to(device)

    for epoch in range(epochs+1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits, h2 = model(xb, return_h2=True)
            loss_ce = criterion(logits, yb)
            # 几何能量项
            H_geo_batch, _ = compute_H_and_Phi(h2, L)
            loss = loss_ce + lambda_geo * H_geo_batch
            loss.backward()
            opt.step()

        # 每隔若干 epoch 记录一次几何统计
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits, h2 = model(X_all, return_h2=True)
                loss_all = criterion(logits, y_all).item()
                H_geo, Phi = compute_H_and_Phi(h2, L)
            H_hist.append(H_geo + 1e-12)
            Phi_hist.append(Phi + 1e-12)
            print(f"  [Epoch {epoch:3d}] Loss: {loss_all:.4f}, H: {H_geo:.6f}, Φ: {Phi:.6f}")

    # 拟合 K
    import numpy as np
    logH = np.log(np.array(H_hist))
    logPhi = np.log(np.array(Phi_hist))

    K, R2 = fit_K(logPhi, logH)
    K_inst_mean, K_inst_std = instantaneous_K(logPhi, logH, window=5)

    print(f"  → Global: K={K:.3f}, R²={R2:.3f}")
    print(f"  → Instant: {K_inst_mean:.3f} ± {K_inst_std:.3f}")

    return {
        "name": name,
        "graph": graph_type,
        "activation": activation,
        "lambda_geo": lambda_geo,
        "optimizer": optimizer_type,
        "K": K,
        "R2": R2,
        "K_inst_mean": K_inst_mean,
        "K_inst_std": K_inst_std,
    }


# ------------------------------------------------------------
# 7. 定义若干配置：验证 + 证伪
# ------------------------------------------------------------
configs = [
    dict(name="Chain λ=0 (ReLU)",
         graph_type="chain", activation="relu",
         lambda_geo=0.0, optimizer_type="sgd"),
    dict(name="Chain λ=0.01 (ReLU)",
         graph_type="chain", activation="relu",
         lambda_geo=0.01, optimizer_type="sgd"),
    dict(name="Complete λ=0 (ReLU) [Mean-field]",
         graph_type="complete", activation="relu",
         lambda_geo=0.0, optimizer_type="sgd"),
    dict(name="Chain λ=0 (Linear) [Activation Falsify]",
         graph_type="chain", activation="linear",
         lambda_geo=0.0, optimizer_type="sgd"),
    dict(name="Chain λ=0 (ReLU, Adam) [Optimizer Falsify]",
         graph_type="chain", activation="relu",
         lambda_geo=0.0, optimizer_type="adam"),
    dict(name="Star λ=0 (ReLU) [Geometry Falsify]",
         graph_type="star", activation="relu",
         lambda_geo=0.0, optimizer_type="sgd"),
]

results = []
for cfg in configs:
    res = run_experiment(**cfg)
    results.append(res)

# ------------------------------------------------------------
# 8. 打印总表 + 自动判定
# ------------------------------------------------------------
print("\n" + "="*60)
print("                 最终结果汇总")
print("="*60)
header = f"{'Config':35s} {'K':>7s} {'R²':>7s} {'K_inst_mean':>13s} {'K_inst_std':>11s}"
print(header)
print("-"*60)
for r in results:
    name = r["name"][:33] + (".." if len(r["name"])>33 else "")
    print(f"{name:35s} {r['K']:7.3f} {r['R2']:7.3f} {r['K_inst_mean']:13.3f} {r['K_inst_std']:11.3f}")
print("="*60)

# 自动判定：谁算“验证”，谁算“证伪”
def verdict(r):
    K, R2 = r["K"], r["R2"]
    if abs(K-2.0) < 0.3 and R2 > 0.8:
        return "✅ Quadratic Closure 持续成立"
    elif R2 < 0.5 or abs(K-2.0) > 1.0:
        return "❌ 证伪条件（偏离 2 且拟合差）"
    else:
        return "⚠ 边界/部分成立"

print("\n" + "="*60)
print("                   自动判定")
print("="*60)
for r in results:
    print(f"{r['name']:40s} -> {verdict(r)}")
print("="*60)
print("提示：关注哪些配置明显偏离 K≈2，即为 Quadratic Closure 的有效边界。")
