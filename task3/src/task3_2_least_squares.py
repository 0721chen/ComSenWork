# -*- coding: utf-8 -*-
"""task3-2：用 PyTorch 的自动求导 + 梯度下降(Adam) 完成"最小二乘"函数拟合。

真实函数（不含噪声）：
    y = 2·x1² + 1.5·x1·x2 + 3·sin(x1) + 0.5·cos(2·x2) + e^(-0.5·x1) + 0.8·x2 + 2026

拟合模型（待求参数 a1..a6, b）：
    ŷ = a1·x1² + a2·x1·x2 + a3·sin(x1) + a4·cos(2·x2) + e^(a5·x1) + a6·x2 + b

训练数据：x1, x2 ~ U(-3, 3)，y = 真值 + 噪声(均匀分布 U(-3, 3))。
损失：MSE；即最小二乘。优化器：Adam(lr 可调)；参数初始化：b = mean(y)，其余为 0。

输出：
    outputs/figures/fig_task3_2_surface.png   真值曲面 / 拟合曲面 / 误差热力图
    outputs/figures/fig_task3_2_slice.png     固定 x2 或 x1 的 2D 切片对比
    outputs/figures/fig_task3_2_loss.png      Loss 曲线 + 参数收敛曲线
    outputs/figures/fig_task3_2_scatter.png   预测值-真实值散点 + 残差分布
    outputs/logs/task3_2_result.json          参数、R²、RMSE
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm

from common import LOG_DIR, dump_json, ensure_dirs, save_fig, set_seed, setup_matplotlib

TRUE = {"a1": 2.0, "a2": 1.5, "a3": 3.0, "a4": 0.5, "a5": -0.5, "a6": 0.8, "b": 2026.0}
NAMES = list(TRUE.keys())
NOISE_VAR = 3.0      # 噪声 U(-3,3) 的方差 = (3-(-3))^2/12


def true_func_np(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return (2.0 * x1 ** 2 + 1.5 * x1 * x2 + 3.0 * np.sin(x1) + 0.5 * np.cos(2.0 * x2)
            + np.exp(-0.5 * x1) + 0.8 * x2 + 2026.0)


def make_dataset(n: int, rng: np.random.Generator):
    x1 = rng.uniform(-3, 3, n)
    x2 = rng.uniform(-3, 3, n)
    y = true_func_np(x1, x2) + rng.uniform(-3, 3, n)      # 题目要求的均匀噪声
    X = torch.tensor(np.stack([x1, x2], axis=1), dtype=torch.float32)
    Y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    return X, Y


class Model(nn.Module):
    """就是题目给定的参数化函数，只是每个系数都变成可学习参数。"""

    def __init__(self, y_mean: float):
        super().__init__()
        self.a1 = nn.Parameter(torch.tensor(0.0))
        self.a2 = nn.Parameter(torch.tensor(0.0))
        self.a3 = nn.Parameter(torch.tensor(0.0))
        self.a4 = nn.Parameter(torch.tensor(0.0))
        self.a5 = nn.Parameter(torch.tensor(0.0))
        self.a6 = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(float(y_mean)))   # 用 y 的均值初始化，收敛更快

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return (self.a1 * x1 ** 2
                + self.a2 * x1 * x2
                + self.a3 * torch.sin(x1)
                + self.a4 * torch.cos(2.0 * x2)
                + torch.exp(self.a5 * x1)
                + self.a6 * x2
                + self.b)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def main() -> None:
    ap = argparse.ArgumentParser(description="task3-2 梯度下降最小二乘拟合")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_dirs()
    setup_matplotlib()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    Xtr, Ytr = make_dataset(args.n_train, rng)
    Xte, Yte = make_dataset(args.n_test, rng)
    print(f"训练样本 {tuple(Xtr.shape)}，测试样本 {tuple(Xte.shape)}")
    print(f"y 的范围: [{Ytr.min().item():.2f}, {Ytr.max().item():.2f}]")

    model = Model(y_mean=Ytr.mean().item())
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_hist, param_hist = [], []
    for step in range(args.steps):
        idx = torch.randint(0, len(Xtr), (args.batch,))
        xb, yb = Xtr[idx], Ytr[idx]
        pred = model(xb)
        loss = criterion(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == args.steps - 1:
            with torch.no_grad():
                full = criterion(model(Xtr), Ytr).item()
            loss_hist.append((step, full))
            param_hist.append([p.item() for p in model.parameters()])
            if step % 500 == 0:
                print(f"step {step:5d}  train MSE = {full:8.4f}  "
                      f"a1={model.a1.item():.4f} a5={model.a5.item():.4f} b={model.b.item():.4f}")

    # ---------------- 评估
    with torch.no_grad():
        pred_tr = model(Xtr).numpy().ravel()
        pred_te = model(Xte).numpy().ravel()
    ytr, yte = Ytr.numpy().ravel(), Yte.numpy().ravel()

    fitted = {n: getattr(model, n).item() for n in NAMES}
    r2_tr, r2_te = r2_score(ytr, pred_tr), r2_score(yte, pred_te)
    rmse_tr = float(np.sqrt(np.mean((ytr - pred_tr) ** 2)))
    rmse_te = float(np.sqrt(np.mean((yte - pred_te) ** 2)))

    print("\n参数对比:")
    print(f"{'参数':>6} {'真实值':>10} {'拟合值':>12} {'绝对误差':>10} {'相对误差':>10}")
    rel = {}
    for n in NAMES:
        err = fitted[n] - TRUE[n]
        rel[n] = abs(err / TRUE[n]) * 100 if TRUE[n] != 0 else 0.0
        print(f"{n:>6} {TRUE[n]:>10.4f} {fitted[n]:>12.4f} {err:>10.4f} {rel[n]:>9.2f}%")
    print(f"\nR²  (训练集) = {r2_tr:.6f}    RMSE = {rmse_tr:.4f}")
    print(f"R²  (测试集) = {r2_te:.6f}    RMSE = {rmse_te:.4f}")

    # ---------------- 图 1：3D 曲面
    g = np.linspace(-3, 3, 60)
    G1, G2 = np.meshgrid(g, g)
    Z_true = true_func_np(G1, G2)
    with torch.no_grad():
        grid = torch.tensor(np.stack([G1.ravel(), G2.ravel()], axis=1), dtype=torch.float32)
        Z_fit = model(grid).numpy().reshape(G1.shape)
    Z_err = np.abs(Z_fit - Z_true)

    fig = plt.figure(figsize=(16, 5.4))
    for i, (Z, title) in enumerate([
        (Z_true, "原函数真实曲面 y = f(x1, x2)（无噪声）"),
        (Z_fit, f"PyTorch 拟合曲面  R²(测试集)={r2_te:.5f}"),
    ], start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        surf = ax.plot_surface(G1, G2, Z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.95)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        ax.set_title(title, fontsize=11)
        ax.view_init(elev=24, azim=-128)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.09)

    ax = fig.add_subplot(1, 3, 3)
    im = ax.contourf(G1, G2, Z_err, levels=20, cmap="magma")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"|拟合曲面 - 真实曲面|  (max={Z_err.max():.3f})", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("task3-2 真实函数 vs PyTorch 拟合函数", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig_task3_2_surface.png")

    # ---------------- 图 2：2D 切片
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    t = np.linspace(-3, 3, 400)
    for ax, (fix_name, fix_val, use_x1) in zip(axes, [("x2", 0.0, True), ("x1", 0.0, False)]):
        if use_x1:      # 固定 x2=0，画 y 随 x1 的变化
            xs = t
            Zt = true_func_np(t, np.full_like(t, fix_val))
            xt = torch.tensor(np.stack([t, np.full_like(t, fix_val)], axis=1), dtype=torch.float32)
            with torch.no_grad():
                Zf = model(xt).numpy().ravel()
            mask = np.abs(Xtr[:, 1].numpy() - fix_val) < 0.12
            sc_x, sc_y = Xtr[mask, 0].numpy(), Ytr[mask, 0].numpy()
            ax.set_xlabel("x1")
            ax.set_title(f"固定 x2 = {fix_val}", fontsize=11)
        else:           # 固定 x1=0
            xs = t
            Zt = true_func_np(np.full_like(t, fix_val), t)
            xt = torch.tensor(np.stack([np.full_like(t, fix_val), t], axis=1), dtype=torch.float32)
            with torch.no_grad():
                Zf = model(xt).numpy().ravel()
            mask = np.abs(Xtr[:, 0].numpy() - fix_val) < 0.12
            sc_x, sc_y = Xtr[mask, 1].numpy(), Ytr[mask, 0].numpy()
            ax.set_xlabel("x2")
            ax.set_title(f"固定 x1 = {fix_val}", fontsize=11)
        ax.scatter(sc_x, sc_y, s=8, color="tab:gray", alpha=0.45, label="含噪训练样本")
        ax.plot(xs, Zt, color="tab:blue", lw=2.2, label="真实函数")
        ax.plot(xs, Zf, color="tab:red", lw=2.0, ls="--", label="拟合函数")
        ax.set_ylabel("y")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("task3-2 切片对比：拟合曲线与真实曲线几乎重合", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig_task3_2_slice.png")

    # ---------------- 图 3：Loss 与参数收敛
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    steps = [s for s, _ in loss_hist]
    axes[0].plot(steps, [l for _, l in loss_hist], color="tab:blue", lw=1.6)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("迭代步数 (batch)")
    axes[0].set_ylabel("训练 MSE (log)")
    axes[0].set_title(f"损失函数下降曲线（均匀噪声 U(-3,3) 的理论下限 ≈ {3 ** 2 / 3:.2f}）", fontsize=11)
    axes[0].grid(alpha=0.3, which="both")

    ph = np.array(param_hist)
    for i, n in enumerate(NAMES):
        axes[1].plot(steps, ph[:, i], lw=1.3, label=n)
        axes[1].axhline(TRUE[n], color=axes[1].lines[-1].get_color(), ls=":", lw=1)
    axes[1].set_xlabel("迭代步数")
    axes[1].set_ylabel("参数值")
    axes[1].set_title("参数收敛过程（虚线=真实值）", fontsize=11)
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(alpha=0.3)
    fig.suptitle("task3-2 训练过程", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig_task3_2_loss.png")

    # ---------------- 图 4：预测 vs 真实 + 残差
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].scatter(yte, pred_te, s=10, alpha=0.5, color="tab:purple")
    lo, hi = min(yte.min(), pred_te.min()), max(yte.max(), pred_te.max())
    axes[0].plot([lo, hi], [lo, hi], color="black", ls="--", lw=1)
    axes[0].set_xlabel("真实 y (含噪声)")
    axes[0].set_ylabel("预测 ŷ")
    axes[0].set_title(f"测试集预测值 vs 真实值  R²={r2_te:.5f}", fontsize=11)
    axes[0].grid(alpha=0.3)

    res = yte - pred_te
    axes[1].hist(res, bins=50, color="tab:green", alpha=0.8)
    axes[1].set_xlabel("残差 y - ŷ")
    axes[1].set_ylabel("频数")
    axes[1].set_title(f"残差分布（均值 {res.mean():.3f}，标准差 {res.std():.3f}）", fontsize=11)
    axes[1].grid(alpha=0.3)
    fig.suptitle("task3-2 拟合质量评估", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig_task3_2_scatter.png")

    dump_json({
        "true_params": TRUE,
        "fitted_params": {n: round(v, 6) for n, v in fitted.items()},
        "abs_error": {n: round(fitted[n] - TRUE[n], 6) for n in NAMES},
        "rel_error_pct": {n: round(rel[n], 4) for n in NAMES},
        "r2_train": round(r2_tr, 6),
        "r2_test": round(r2_te, 6),
        # 均匀噪声 U(-3,3) 的方差 = (b-a)^2/12 = 6^2/12 = 3；由此可得 R² 的理论上限
        "r2_theoretical_max": round(1.0 - NOISE_VAR / float(np.var(yte)), 6),
        "rmse_train": round(rmse_tr, 6),
        "rmse_test": round(rmse_te, 6),
        "train_config": {"steps": args.steps, "lr": args.lr, "batch": args.batch,
                         "optimizer": "Adam", "loss": "MSE", "noise": "U(-3,3)",
                         "n_train": args.n_train, "n_test": args.n_test, "seed": args.seed},
        "loss_curve": [{"step": s, "mse": round(float(l), 6)} for s, l in loss_hist],
    }, LOG_DIR / "task3_2_result.json")


if __name__ == "__main__":
    main()
