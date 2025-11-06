# ============================================================
#  Quadratic Closure & Neural Mathematical Automation Colab
#  - Very-hard synthetic math (larger + OOD)
#  - Meanfield vs Chain + λ_geo 对比
#  - GSM8K subset 几何 QA (真实数据支撑)
# ============================================================

!pip -q install datasets scikit-learn

import math
import random
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.neighbors import NearestNeighbors
from datasets import load_dataset

# -----------------------
#  全局设置
# -----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
#  Part 1: Very-hard synthetic math (bigger + OOD)
# ============================================================

# ---------- 1.1 数据集生成 ----------
def make_very_hard_math_dataset(n_each=200, seed=0,
                                range_roots=(-10, 10),
                                range_coeff_small=(-5, 5),
                                range_coeff_big=(-10, 10),
                                x_range=(-5, 5)):
    """
    生成 very hard 数学数据集:
    - poly:  Evaluate at x = x0: a x^2 + b x + c
    - system: Solve for x,y: A x + B y = E, C x + D y = F
    - quad: Solve for x: x^2 + b x + c = 0 (with integer roots)

    参数范围可调，用于 ID / OOD。
    """
    rng = np.random.default_rng(seed)
    items = []
    idx = 0

    # ---- poly ----
    for _ in range(n_each):
        a = rng.integers(range_coeff_small[0], range_coeff_small[1] + 1)
        a = 1 if a == 0 else a
        b = rng.integers(range_coeff_big[0], range_coeff_big[1] + 1)
        c = rng.integers(range_coeff_big[0], range_coeff_big[1] + 1)
        x0 = rng.integers(x_range[0], x_range[1] + 1)
        val = int(a * x0 * x0 + b * x0 + c)
        text = f"Evaluate at x = {x0}: {a} * x^2 + {b} * x + {c}"
        items.append({
            "id": idx,
            "kind": "poly",
            "text": text,
            "answer": str(val),
            "input": [a, b, c, 0, x0, 0, 0, 0],
            "target": [val, 0],
        })
        idx += 1

    # ---- system ----
    for _ in range(n_each):
        while True:
            x = rng.integers(range_roots[0], range_roots[1] + 1)
            y = rng.integers(range_roots[0], range_roots[1] + 1)
            A = rng.integers(range_coeff_small[0], range_coeff_small[1] + 1)
            A = 1 if A == 0 else A
            B = rng.integers(range_coeff_small[0], range_coeff_small[1] + 1)
            C = rng.integers(range_coeff_small[0], range_coeff_small[1] + 1)
            D = rng.integers(range_coeff_small[0], range_coeff_small[1] + 1)
            D = 1 if D == 0 else D
            det = A * D - B * C
            if det != 0:
                E = int(A * x + B * y)
                F_ = int(C * x + D * y)
                break
        text = f"Solve for x,y: {A}*x + {B}*y = {E}, {C}*x + {D}*y = {F_}"
        items.append({
            "id": idx,
            "kind": "system",
            "text": text,
            "answer": f"{x},{y}",
            "input": [A, B, C, D, E, F_, 0, 0],
            "target": [x, y],
        })
        idx += 1

    # ---- quad ----
    for _ in range(n_each):
        r1 = rng.integers(range_roots[0], range_roots[1] + 1)
        r2 = rng.integers(range_roots[0], range_roots[1] + 1)
        b = -int(r1 + r2)
        c = int(r1 * r2)
        text = f"Solve for x: x^2 + ({b})*x + ({c}) = 0"
        roots = sorted([r1, r2])
        items.append({
            "id": idx,
            "kind": "quad",
            "text": text,
            "answer": f"{roots[0]},{roots[1]}",
            "input": [1, b, c, 0, 0, 0, 0, 0],
            "target": roots,
        })
        idx += 1

    return items


# --------- 1.2 PyTorch Dataset ----------
KIND2IDX = {"poly": 0, "system": 1, "quad": 2}

class MathDataset(Dataset):
    def __init__(self, items, S_in=10.0, S_out=20.0):
        self.items = items
        self.S_in = S_in
        self.S_out = S_out

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        kind_idx = KIND2IDX[it["kind"]]
        x = np.array(it["input"], dtype=np.float32) / self.S_in
        y = np.array(it["target"], dtype=np.float32) / self.S_out
        return {
            "kind": kind_idx,
            "x": x,
            "y": y,
        }


# --------- 1.3 几何 MathGeometricSolver ----------
class MathGeometricSolver(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.kind_emb = nn.Embedding(3, 8)
        self.proj = nn.Linear(8 + 8, dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.LayerNorm(dim)
            ) for _ in range(3)
        ])
        self.head = nn.Linear(dim, 2)

    def forward(self, kind_idx, x):
        """
        kind_idx: (B,) long
        x: (B, 8) float
        """
        k = self.kind_emb(kind_idx)  # (B,8)
        z = torch.cat([x, k], dim=-1)  # (B,16)
        h = self.proj(z)  # (B,dim)
        for block in self.blocks:
            h = h + block(h)
        y_hat = self.head(h)  # (B,2)
        return h, y_hat


# --------- 1.4 Laplacian & 几何能量 ----------
def build_chain_laplacian(n: int) -> torch.Tensor:
    """
    1D 链图 Laplacian: diag(1,2,...,2,1) with -1 on neighbors
    """
    L = torch.zeros(n, n)
    for i in range(n):
        if i > 0:
            L[i, i-1] = -1.0
        if i < n-1:
            L[i, i+1] = -1.0
        deg = 0
        if i > 0:
            deg += 1
        if i < n-1:
            deg += 1
        L[i, i] = deg
    return L


def batch_geo_stats(h, geo_mode="meanfield", L_chain=None):
    """
    h: (B, D)
    返回:
      H_batch_mean, Phi2_batch_mean
    geo_mode:
      - "meanfield": H = N^2 * Var(h)
      - "chain": H = h^T L h
    """
    B, D = h.shape
    if geo_mode == "meanfield":
        mu = h.mean(dim=1, keepdim=True)         # (B,1)
        var = (h - mu).pow(2).mean(dim=1)       # (B,)
        H = (D ** 2) * var                      # (B,)
        Phi2 = var
    elif geo_mode == "chain":
        assert L_chain is not None
        # h: (B,D), L: (D,D)
        # For each sample: H_i = h_i^T L h_i
        HL = torch.matmul(h, L_chain.to(h.device))   # (B,D)
        H = (HL * h).sum(dim=1)
        mu = h.mean(dim=1, keepdim=True)
        Phi2 = (h - mu).pow(2).mean(dim=1)
    else:
        raise ValueError(f"Unknown geo_mode={geo_mode}")
    return H.mean().item(), Phi2.mean().item(), H.detach(), Phi2.detach()


# --------- 1.5 训练函数 ----------
@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 64
    lr: float = 3e-4
    geo_mode: str = "meanfield"  # "meanfield" or "chain"
    lambda_geo: float = 0.0
    S_in: float = 10.0
    S_out: float = 20.0


def train_math_solver(train_items, cfg: TrainConfig):
    dataset = MathDataset(train_items, S_in=cfg.S_in, S_out=cfg.S_out)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = MathGeometricSolver(dim=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

    if cfg.geo_mode == "chain":
        L_chain = build_chain_laplacian(64)   # hidden dim = 64
    else:
        L_chain = None

    H_hist = []
    Phi_hist = []

    print(f"\n=== Training geo_mode={cfg.geo_mode}, λ_geo={cfg.lambda_geo} ===")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_task = 0.0
        total_geo = 0.0
        total_H = 0.0
        total_Phi2 = 0.0
        n_batches = 0

        for batch in loader:
            kind = batch["kind"].to(device)
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            h, y_hat = model(kind, x)
            task_loss = F.mse_loss(y_hat, y)

            if cfg.lambda_geo > 0.0:
                H_mean, Phi2_mean, H_all, Phi_all = batch_geo_stats(
                    h, geo_mode=cfg.geo_mode, L_chain=L_chain
                )
                geo_loss = cfg.lambda_geo * H_all.mean()
            else:
                # 仍然计算 H, Phi 用于统计，但不加损失
                H_mean, Phi2_mean, H_all, Phi_all = batch_geo_stats(
                    h, geo_mode=cfg.geo_mode, L_chain=L_chain
                )
                geo_loss = 0.0

            loss = task_loss + geo_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_task += task_loss.item()
            total_geo += geo_loss if isinstance(geo_loss, float) else geo_loss.item()
            total_H += H_mean
            total_Phi2 += Phi2_mean
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_task = total_task / n_batches
        avg_geo = total_geo / n_batches
        avg_H = total_H / n_batches
        avg_Phi2 = total_Phi2 / n_batches

        H_hist.append(avg_H)
        Phi_hist.append(avg_Phi2)

        if (epoch == 1) or (epoch % 20 == 0) or (epoch == cfg.epochs):
            print(f"[Epoch {epoch:3d}] loss={avg_loss:.4f} | task={avg_task:.4f} "
                  f"| H={avg_H:.4f} | Phi2={avg_Phi2:.4f}")

    # 拟合 K
    H_arr = np.array(H_hist)
    Phi_arr = np.array(Phi_hist)
    eps = 1e-8
    logH = np.log(H_arr + eps)
    logPhi = np.log(np.sqrt(Phi_arr) + eps)
    K_fit, b = np.polyfit(logPhi, logH, 1)
    print(f"\n[H-Φ 拟合] log H = K log Φ + b, K_fit ≈ {K_fit:.4f}")

    return model, K_fit, (H_hist, Phi_hist)


# --------- 1.6 数值 sanity check + k-navigation ----------
def sanity_check_meanfield(model, dataset, S_out=20.0, num_tests=5):
    """
    对 meanfield 模型检查 H_geo vs N^2 Phi^2.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    it = iter(loader)
    print("\n[Sanity Check] meanfield: H_direct vs N^2 Phi^2")
    with torch.no_grad():
        for i in range(num_tests):
            batch = next(it)
            kind = batch["kind"].to(device)
            x = batch["x"].to(device)
            _, y_hat = model(kind, x)  # 只用 hidden 几何，稍后重新 forward
            # 再 forward 一次拿 hidden
            h, _ = model(kind, x)
            h = h.squeeze(0)  # (D,)
            D = h.shape[0]
            mu = h.mean()
            Phi2 = ((h - mu) ** 2).mean().item()
            H_direct = 0.0
            for a in range(D):
                for b_ in range(D):
                    H_direct += float((h[a] - h[b_])**2)
            H_direct *= 0.5
            H_id = (D ** 2) * Phi2
            diff = H_direct - H_id
            print(f"test {i}: H_direct={H_direct:.4f}, N^2Phi^2={H_id:.4f}, diff={diff:.2e}")


def evaluate_k_navigation(model, items, S_in=10.0, S_out=20.0, ks=[1,2,3,4,6,8,12]):
    """
    在解空间上做 k-NN 导航:
    - from 预测解 (y_pred) → 最近的 k 个真实解 (y_true)
    - 命中同一题目则记为成功
    """
    dataset = MathDataset(items, S_in=S_in, S_out=S_out)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()

    all_pred = []
    all_true = []

    with torch.no_grad():
        for batch in loader:
            kind = batch["kind"].to(device)
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            _, y_hat = model(kind, x)
            # 还原 scale
            y_hat_unscaled = (y_hat * S_out).cpu().numpy()
            y_unscaled = (y * S_out).cpu().numpy()
            all_pred.append(y_hat_unscaled)
            all_true.append(y_unscaled)

    all_pred = np.concatenate(all_pred, axis=0)   # (N,2)
    all_true = np.concatenate(all_true, axis=0)   # (N,2)
    N = all_pred.shape[0]

    nbrs = NearestNeighbors(n_neighbors=max(ks), metric="euclidean").fit(all_true)

    print("\n=== k-导航正确率（解空间几何） ===")
    for k in ks:
        success = 0
        for i in range(N):
            dists, idx = nbrs.kneighbors(all_pred[i:i+1], n_neighbors=k)
            # 如果 i 在最近的 k 个 true index 中，就算成功
            if i in idx[0]:
                success += 1
        acc = success / N
        print(f"k = {k:2d}  accuracy = {acc:.3f}")

    # 打几个示例
    print("\nSample predictions (unscaled):")
    for i in random.sample(range(N), 5):
        print("Q:", items[i]["text"])
        print("True answer:", items[i]["answer"], "| pred approx:", all_pred[i])
        print()


# --------- 1.7 运行一个主实验 + OOD ----------
def run_synthetic_block():
    # 训练集（ID）
    print("Generating ID very hard dataset (bigger)...")
    train_items = make_very_hard_math_dataset(
        n_each=200,   # 3 * 200 = 600 题
        seed=0,
        range_roots=(-10, 10),
        range_coeff_small=(-5, 5),
        range_coeff_big=(-10, 10),
        x_range=(-5, 5),
    )
    print("Train size:", len(train_items))

    # 测试集（ID）
    test_items_id = make_very_hard_math_dataset(
        n_each=100,   # 300 题
        seed=123,
        range_roots=(-10, 10),
        range_coeff_small=(-5, 5),
        range_coeff_big=(-10, 10),
        x_range=(-5, 5),
    )
    print("Test ID size:", len(test_items_id))

    # 测试集（OOD）：更大的系数 / 根 / x
    test_items_ood = make_very_hard_math_dataset(
        n_each=100,
        seed=456,
        range_roots=(-20, 20),
        range_coeff_small=(-10, 10),
        range_coeff_big=(-20, 20),
        x_range=(-10, 10),
    )
    print("Test OOD size:", len(test_items_ood))

    # 训练一个 meanfield + λ_geo=0 的主模型
    cfg = TrainConfig(
        epochs=120,
        batch_size=64,
        lr=3e-4,
        geo_mode="meanfield",
        lambda_geo=0.0,
        S_in=10.0,
        S_out=20.0,
    )
    model, K_fit, _ = train_math_solver(train_items, cfg)

    # Sanity check mean-field
    sanity_check_meanfield(model, MathDataset(test_items_id, cfg.S_in, cfg.S_out))

    # k-navigation on ID
    print("\n[ID evaluation]")
    evaluate_k_navigation(model, test_items_id, S_in=cfg.S_in, S_out=cfg.S_out)

    # k-navigation on OOD
    print("\n[OOD evaluation]")
    evaluate_k_navigation(model, test_items_ood, S_in=cfg.S_in, S_out=cfg.S_out)

    return model, cfg, train_items, test_items_id, test_items_ood, K_fit


# ============================================================
#  Part 2: meanfield vs chain + λ_geo 系统对比
# ============================================================

def run_geo_ablation(train_items):
    configs = [
        ("meanfield", 0.0),
        ("meanfield", 1e-4),
        ("meanfield", 1e-3),
        ("chain", 0.0),
        ("chain", 1e-4),
        ("chain", 1e-3),
    ]
    results = []

    for geo_mode, lam in configs:
        cfg = TrainConfig(
            epochs=80,      # 稍微少一点，整体速度快
            batch_size=64,
            lr=3e-4,
            geo_mode=geo_mode,
            lambda_geo=lam,
            S_in=10.0,
            S_out=20.0,
        )
        model, K_fit, (H_hist, Phi_hist) = train_math_solver(train_items, cfg)

        # 用训练集上的最终 task loss 简单估计
        # （这里为了简单，不再遍历 dataset 重新算 loss）
        final_H = H_hist[-1]
        final_Phi2 = Phi_hist[-1]

        # 用训练集评估 k=2 导航
        ks = [2]
        dataset = MathDataset(train_items, cfg.S_in, cfg.S_out)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch in loader:
                kind = batch["kind"].to(device)
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                _, y_hat = model(kind, x)
                y_hat_unscaled = (y_hat * cfg.S_out).cpu().numpy()
                y_unscaled = (y * cfg.S_out).cpu().numpy()
                all_pred.append(y_hat_unscaled)
                all_true.append(y_unscaled)
        all_pred = np.concatenate(all_pred, axis=0)
        all_true = np.concatenate(all_true, axis=0)
        N = all_pred.shape[0]
        nbrs = NearestNeighbors(n_neighbors=max(ks), metric="euclidean").fit(all_true)

        acc_k2 = 0.0
        for i in range(N):
            d, idx = nbrs.kneighbors(all_pred[i:i+1], n_neighbors=2)
            if i in idx[0]:
                acc_k2 += 1
        acc_k2 /= N

        results.append({
            "geo_mode": geo_mode,
            "lambda_geo": lam,
            "K_fit": K_fit,
            "H_final": final_H,
            "Phi2_final": final_Phi2,
            "acc_k2_train": acc_k2,
        })

    print("\n================== Geo Ablation Summary ==================")
    print("geo_mode   λ_geo      K_fit      acc@k=2(train)   H_final     Phi2_final")
    for r in results:
        print(f"{r['geo_mode']:9s} {r['lambda_geo']:8.4g}  "
              f"{r['K_fit']:8.4f}    {r['acc_k2_train']:8.3f}    "
              f"{r['H_final']:9.1f}   {r['Phi2_final']:9.4f}")
    return results


# ============================================================
#  Part 3: GSM8K subset 几何 QA（真实数据支撑）
# ============================================================

# --------- 3.1 文本处理 & 模型 ----------
def build_char_vocab(texts, min_freq=1):
    from collections import Counter
    counter = Counter()
    for t in texts:
        counter.update(list(t))
    chars = [c for c, f in counter.items() if f >= min_freq]
    chars = sorted(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos

def encode_bow(text, stoi, max_len=256):
    v = np.zeros(len(stoi), dtype=np.float32)
    for ch in text[:max_len]:
        if ch in stoi:
            v[stoi[ch]] += 1.0
    if v.sum() > 0:
        v /= v.sum()
    return v

class GSM8KTextGeoEncoder(nn.Module):
    def __init__(self, in_dim, dim=64):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.LayerNorm(dim)
            ) for _ in range(2)
        ])

    def forward(self, x):
        h = self.proj(x)
        for block in self.blocks:
            h = h + block(h)
        return h  # (B,dim)

def contrastive_loss(q, a, tau=0.1):
    """
    简单 InfoNCE：q,a 共享 batch，q_i 对应 a_i
    """
    q = F.normalize(q, dim=-1)
    a = F.normalize(a, dim=-1)
    logits = torch.matmul(q, a.t()) / tau  # (B,B)
    labels = torch.arange(q.size(0), device=q.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def train_gsm8k_geo_encoder(train_ds, test_ds, train_size=512, test_size=256, epochs=10, batch_size=64):
    # 取子集
    train_subset = train_ds.select(range(min(train_size, len(train_ds))))
    test_subset = test_ds.select(range(min(test_size, len(test_ds))))

    # 构造文本（问题 + 最终答案行）
    def get_final_answer(ans_text):
        # GSM8K 格式: 最后一行是 "#### x"
        lines = ans_text.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("####"):
                return line
        return lines[-1]

    train_Q = [ex["question"] for ex in train_subset]
    train_A = [get_final_answer(ex["answer"]) for ex in train_subset]
    test_Q = [ex["question"] for ex in test_subset]
    test_A = [get_final_answer(ex["answer"]) for ex in test_subset]

    print("Train subset size:", len(train_Q))
    print("Test  subset size:", len(test_Q))
    print("Example train item:")
    print("Q:", train_Q[0])
    print("A:", train_A[0])

    # 构造 vocab
    stoi, itos = build_char_vocab(train_Q + train_A)
    V = len(stoi)
    print("Vocab size:", V)

    # 构造 BOW 向量
    Xq_train = np.stack([encode_bow(q, stoi) for q in train_Q])
    Xa_train = np.stack([encode_bow(a, stoi) for a in train_A])
    Xq_test = np.stack([encode_bow(q, stoi) for q in test_Q])
    Xa_test = np.stack([encode_bow(a, stoi) for a in test_A])

    Xq_train_t = torch.tensor(Xq_train, dtype=torch.float32)
    Xa_train_t = torch.tensor(Xa_train, dtype=torch.float32)
    Xq_test_t = torch.tensor(Xq_test, dtype=torch.float32)
    Xa_test_t = torch.tensor(Xa_test, dtype=torch.float32)

    # 模型
    encoder = GSM8KTextGeoEncoder(in_dim=V, dim=64).to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=3e-4, weight_decay=1e-5)

    # 构建简单 DataLoader (索引形式)
    train_indices = list(range(len(train_Q)))

    print("\n=== Training geometric QA solver on GSM8K subset ===")
    H_hist, Phi_hist = [], []

    for epoch in range(1, epochs + 1):
        random.shuffle(train_indices)
        total_loss = 0.0
        total_contrast = 0.0
        total_H = 0.0
        total_Phi2 = 0.0
        n_batches = 0

        encoder.train()
        for i in range(0, len(train_indices), batch_size):
            idx = train_indices[i:i+batch_size]
            xq = Xq_train_t[idx].to(device)
            xa = Xa_train_t[idx].to(device)

            hq = encoder(xq)
            ha = encoder(xa)
            loss_contrast = contrastive_loss(hq, ha, tau=0.1)

            # 这里用 meanfield 几何统计
            h_all = torch.cat([hq, ha], dim=0)  # (2B,dim)
            B2, D = h_all.shape
            mu = h_all.mean(dim=1, keepdim=True)
            var = (h_all - mu).pow(2).mean(dim=1)
            H = (D ** 2) * var
            Phi2 = var
            H_mean = H.mean().item()
            Phi2_mean = Phi2.mean().item()

            loss = loss_contrast  # 这里不加 geo reg，只观测几何

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_contrast += loss_contrast.item()
            total_H += H_mean
            total_Phi2 += Phi2_mean
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_contrast = total_contrast / n_batches
        avg_H = total_H / n_batches
        avg_Phi2 = total_Phi2 / n_batches

        H_hist.append(avg_H)
        Phi_hist.append(avg_Phi2)

        print(f"[Epoch {epoch:3d}] loss={avg_loss:.4f} | contrast={avg_contrast:.4f} "
              f"| H={avg_H:.4f} | Phi2={avg_Phi2:.4f}")

    # 拟合 K
    H_arr = np.array(H_hist)
    Phi_arr = np.array(Phi_hist)
    eps = 1e-8
    logH = np.log(H_arr + eps)
    logPhi = np.log(np.sqrt(Phi_arr) + eps)
    K_fit, b = np.polyfit(logPhi, logH, 1)
    print(f"\n[H-Φ 拟合] log H = K log Φ + b, K_fit ≈ {K_fit:.4f}")

    # 编码 test 集
    encoder.eval()
    with torch.no_grad():
        Q_test_embs = encoder(Xq_test_t.to(device)).cpu().numpy()
        A_test_embs = encoder(Xa_test_t.to(device)).cpu().numpy()

    print("Q_test_embs:", Q_test_embs.shape, "A_test_embs:", A_test_embs.shape)

    # k-导航: 从 Q_embedding 出发找 A_embedding
    ks = [1, 2, 3, 4, 6, 8, 12]
    nbrs = NearestNeighbors(n_neighbors=max(ks), metric="cosine").fit(A_test_embs)
    N = Q_test_embs.shape[0]

    print("\n=== k-导航正确率 (GSM8K test subset) ===")
    for k in ks:
        success = 0
        for i in range(N):
            d, idx = nbrs.kneighbors(Q_test_embs[i:i+1], n_neighbors=k)
            if i in idx[0]:
                success += 1
        acc = success / N
        print(f"k = {k:2d}  acc = {acc:.3f}")

    # 打几条 NN 示例
    print("\nSample Q/A nearest neighbor check:\n")
    for i in random.sample(range(N), 5):
        print("Q:", test_Q[i])
        print("True A:", test_A[i])
        d, idx = nbrs.kneighbors(Q_test_embs[i:i+1], n_neighbors=4)
        print("Nearest answers (by index):")
        for j in idx[0]:
            print("  index:", j, "|", test_A[j][:80], "...")
        print()

    return K_fit, Q_test_embs, A_test_embs, (train_subset, test_subset)


# ============================================================
#  主控：依次运行三个模块
# ============================================================

# 1) very-hard 数学：ID + OOD + K & k-nav
main_model, main_cfg, train_items, test_items_id, test_items_ood, K_main = run_synthetic_block()

# 2) meanfield vs chain + λ_geo 对比
geo_results = run_geo_ablation(train_items)

# 3) GSM8K subset 几何 QA
print("\nLoading GSM8K dataset...")
gsm8k = load_dataset("openai/gsm8k", "main")
train_ds = gsm8k["train"]
test_ds = gsm8k["test"]
K_gsm, Q_embs, A_embs, gsm_subsets = train_gsm8k_geo_encoder(train_ds, test_ds)

print("\nDone.")
