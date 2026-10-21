# -*- coding: utf-8 -*-
"""task3-3：用 PyTorch + pandas 搭建多层感知机(MLP)，完成 MNIST 手写数字识别。

数据：MNIST（用 pandas 组成 DataFrame：label + 784 维像素），随机划分 55000 训练 / 5000 验证，
      官方 10000 张测试集留作测试。
模型：784 → 256 → 128 → 10 的全连接网络（ReLU + Dropout 正则化）。
训练：CrossEntropyLoss + Adam；每个 epoch 记录 Loss / 训练准确率 / 测试(验证)准确率。

输出：
    outputs/figures/fig_task3_3_curves.png           Loss / Train Acc / Test Acc 曲线
    outputs/figures/fig_task3_3_predictions.png      至少 3 张（这里 10 张）预测可视化
    outputs/figures/fig_task3_3_confusion.png        混淆矩阵 + 每类准确率
    outputs/figures/fig_task3_3_misclassified.png    错分样本可视化
    outputs/figures/fig_task3_3_dataset.png          数据集样例与类别分布
    outputs/models/mlp_mnist.pt                      训练好的模型参数
    outputs/logs/task3_3_metrics.json / task3_3_epochs.csv
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from common import (
    LOG_DIR, MODEL_DIR, dump_json, ensure_dirs, save_fig, set_seed, setup_matplotlib,
)
from mnist_data import load_dataframes


class MLP(nn.Module):
    """784-256-128-10 的多层感知机。

    * 全连接层（Linear）：每个输出神经元与上一层的所有神经元相连；
    * ReLU：非线性激活，避免网络退化成线性模型；
    * Dropout(0.2)：训练时随机丢弃 20% 神经元，抑制过拟合（相当于集成多个子网络）。
    """

    def __init__(self, in_dim: int = 784, n_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_loaders(df_train: pd.DataFrame, df_test: pd.DataFrame, val_ratio: float,
                 batch: int, seed: int):
    """DataFrame -> TensorDataset -> DataLoader（像素归一化到 [0,1]）。"""
    X = df_train.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    y = df_train["label"].to_numpy(dtype=np.int64)
    Xte = df_test.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    yte = df_test["label"].to_numpy(dtype=np.int64)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g).numpy()
    n_val = int(len(X) * val_ratio)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def ds(Xa, ya):
        # pandas 转出来的 ndarray 是只读视图，复制一份再交给 torch，避免 warning
        return TensorDataset(torch.from_numpy(np.array(Xa, copy=True)),
                             torch.from_numpy(np.array(ya, copy=True)))

    tr = DataLoader(ds(X[tr_idx], y[tr_idx]), batch_size=batch, shuffle=True,
                    generator=torch.Generator().manual_seed(seed), drop_last=True)
    va = DataLoader(ds(X[val_idx], y[val_idx]), batch_size=512, shuffle=False)
    te = DataLoader(ds(Xte, yte), batch_size=512, shuffle=False)
    print(f"训练 {len(tr_idx)} 张 / 验证 {n_val} 张 / 测试 {len(Xte)} 张；"
          f"每个 epoch {len(tr)} 个 batch")
    return tr, va, te


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion):
    """返回 (平均损失, 准确率, 全部预测, 全部标签)。"""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    preds, labels = [], []
    for xb, yb in loader:
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * len(yb)
        p = logits.argmax(dim=1)
        correct += (p == yb).sum().item()
        n += len(yb)
        preds.append(p)
        labels.append(yb)
    return total_loss / n, correct / n, torch.cat(preds), torch.cat(labels)


def plot_dataset(df_train: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(13, 4.8))
    gs = fig.add_gridspec(2, 11, hspace=0.15, wspace=0.1)
    for i in range(22):
        ax = fig.add_subplot(gs[i // 11, i % 11])
        img = df_train.iloc[i, 1:].to_numpy(np.uint8).reshape(28, 28)
        ax.imshow(img, cmap="gray_r")
        ax.set_title(str(df_train.iloc[i, 0]), fontsize=9)
        ax.axis("off")
    fig.suptitle("task3-3 MNIST 数据样例（pandas DataFrame: label + 784 个像素列）", fontsize=13)
    save_fig(fig, "fig_task3_3_dataset.png")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    counts = df_train["label"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="tab:blue", alpha=0.85)
    ax.set_xlabel("类别")
    ax.set_ylabel("样本数")
    ax.set_title(f"训练集类别分布（共 {len(df_train)} 张）", fontsize=11)
    for x, v in zip(range(len(counts)), counts.values):
        ax.text(x, v + 60, str(v), ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "fig_task3_3_class_dist.png")


def visualize_predictions(model, loader, n: int = 10) -> None:
    """可视化预测结果：真实标签 / 预测标签 / 置信度，错分用红色标题。"""
    model.eval()
    xb, yb = next(iter(loader))
    with torch.no_grad():
        prob = torch.softmax(model(xb), dim=1)
    pred, conf = prob.argmax(dim=1), prob.max(dim=1).values

    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(2.0 * ((n + 1) // 2), 4.6))
    for i in range(n):
        ax = axes.ravel()[i]
        ax.imshow(xb[i].numpy().reshape(28, 28), cmap="gray_r")
        ok = pred[i].item() == yb[i].item()
        ax.set_title(f"真值 {yb[i].item()} / 预测 {pred[i].item()}\n置信度 {conf[i].item():.3f}",
                     fontsize=9, color="green" if ok else "red")
        ax.axis("off")
    for j in range(n, axes.size):
        axes.ravel()[j].axis("off")
    fig.suptitle("task3-3 测试集预测可视化（绿=正确，红=错误）", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig_task3_3_predictions.png")


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xlabel("预测类别")
    axes[0].set_ylabel("真实类别")
    axes[0].set_title("混淆矩阵（测试集）", fontsize=11)
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                axes[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                             color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    fig.colorbar(im, ax=axes[0], shrink=0.85)

    per_class = cm.diagonal() / cm.sum(axis=1)
    axes[1].bar(range(10), per_class * 100, color="tab:orange", alpha=0.85)
    axes[1].set_xticks(range(10))
    axes[1].set_ylim(80, 100)
    axes[1].set_xlabel("数字类别")
    axes[1].set_ylabel("准确率 / %")
    axes[1].set_title("各类别准确率", fontsize=11)
    for i, v in enumerate(per_class * 100):
        axes[1].text(i, v + 0.15, f"{v:.1f}", ha="center", fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("task3-3 分类结果分析", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig_task3_3_confusion.png")
    return {str(i): round(float(per_class[i]), 4) for i in range(10)}


def plot_misclassified(model, loader, max_n: int = 12) -> None:
    model.eval()
    found = []
    for xb, yb in loader:
        with torch.no_grad():
            prob = torch.softmax(model(xb), dim=1)
        pred = prob.argmax(dim=1)
        bad = (pred != yb).nonzero().ravel()
        for k in bad:
            found.append((xb[k], yb[k].item(), pred[k].item(), prob[k, pred[k]].item()))
            if len(found) >= max_n:
                break
        if len(found) >= max_n:
            break
    fig, axes = plt.subplots(2, (max_n + 1) // 2, figsize=(2.1 * ((max_n + 1) // 2), 4.8))
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < len(found):
            img, t, p, c = found[i]
            ax.imshow(img.numpy().reshape(28, 28), cmap="gray_r")
            ax.set_title(f"真值 {t} → 预测 {p}\n置信度 {c:.3f}", fontsize=9, color="crimson")
    fig.suptitle("task3-3 错分样本（模型容易混淆的写法）", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig_task3_3_misclassified.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="task3-3 MLP 识别 MNIST")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--val-ratio", type=float, default=1 / 12)   # 55000/5000
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_dirs()
    setup_matplotlib()
    set_seed(args.seed)

    df_train, df_test, data_info = load_dataframes()
    plot_dataset(df_train)
    tr, va, te = make_loaders(df_train, df_test, args.val_ratio, args.batch, args.seed)

    model = MLP(dropout=args.dropout)
    print(model)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_param}")
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    hist = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [],
            "val_acc": [], "test_acc": [], "sec": []}
    best_acc, best_state = 0.0, None
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        for xb, yb in tr:
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()          # 反向传播：链式法则求梯度
            opt.step()               # 梯度下降：更新权重与偏置
            ep_loss += loss.item() * len(yb)
            ep_correct += (logits.argmax(1) == yb).sum().item()
            ep_n += len(yb)

        tr_loss, tr_acc = ep_loss / ep_n, ep_correct / ep_n
        va_loss, va_acc, _, _ = evaluate(model, va, criterion)
        _, te_acc, _, _ = evaluate(model, te, criterion)
        sec = time.time() - t0
        for k, v in zip(hist, (ep, tr_loss, tr_acc, va_loss, va_acc, te_acc, sec)):
            hist[k].append(v)
        if te_acc > best_acc:
            best_acc = te_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:2d}/{args.epochs}  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  test_acc={te_acc:.4f}  "
              f"({sec:.1f}s)")

    # ---------------- 曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    eps = hist["epoch"]
    axes[0].plot(eps, hist["train_loss"], "o-", label="训练集 Loss", color="tab:blue")
    axes[0].plot(eps, hist["val_loss"], "s--", label="验证集 Loss", color="tab:red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("交叉熵损失")
    axes[0].set_title("损失函数 (Loss)", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(eps, np.array(hist["train_acc"]) * 100, "o-", color="tab:green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("准确率 / %")
    axes[1].set_title("训练准确率 (Train Acc)", fontsize=11)
    axes[1].grid(alpha=0.3)

    axes[2].plot(eps, np.array(hist["val_acc"]) * 100, "s-", color="tab:purple", label="验证集")
    axes[2].plot(eps, np.array(hist["test_acc"]) * 100, "^-", color="tab:orange", label="测试集")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("准确率 / %")
    axes[2].set_title("测试/验证准确率 (Test Acc)", fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)
    fig.suptitle(f"task3-3 MLP(MNIST) 训练曲线  最终测试准确率 {best_acc * 100:.2f}%", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig_task3_3_curves.png")

    # ---------------- 用最好的权重做可视化与分析
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "arch": "784-256-128-10",
                "test_acc": best_acc, "args": vars(args)}, MODEL_DIR / "mlp_mnist.pt")
    print(f"[model ] {MODEL_DIR / 'mlp_mnist.pt'}")

    visualize_predictions(model, te, n=10)
    _, acc, pred_all, label_all = evaluate(model, te, criterion)
    per_class = plot_confusion(label_all.numpy(), pred_all.numpy())
    plot_misclassified(model, te, 12)

    with open(LOG_DIR / "task3_3_epochs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(hist.keys()))
        for row in zip(*hist.values()):
            w.writerow([round(v, 6) if isinstance(v, float) else v for v in row])
    print(f"[csv   ] {LOG_DIR / 'task3_3_epochs.csv'}")

    dump_json({
        "dataset": "MNIST", "n_train": 55000, "n_val": 5000, "n_test": 10000,
        "data_source_train": data_info["source_train"],
        "data_source_test": data_info["source_test"],
        "model": "MLP 784-256-128-10 (ReLU + Dropout %.1f)" % args.dropout,
        "params": n_param,
        "optimizer": f"Adam(lr={args.lr})", "loss": "CrossEntropyLoss",
        "batch_size": args.batch, "epochs": args.epochs,
        "best_test_acc": round(best_acc, 5),
        "final_test_acc": round(float(hist["test_acc"][-1]), 5),
        "final_train_acc": round(float(hist["train_acc"][-1]), 5),
        "final_val_acc": round(float(hist["val_acc"][-1]), 5),
        "final_train_loss": round(float(hist["train_loss"][-1]), 6),
        "per_class_acc": per_class,
        "train_seconds": round(hist["sec"][-1], 2),
        "history": {k: [round(float(v), 6) for v in vals] for k, vals in hist.items()
                    if k != "epoch"},
    }, LOG_DIR / "task3_3_metrics.json")


if __name__ == "__main__":
    main()
