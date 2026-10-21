# -*- coding: utf-8 -*-
"""task2-1：用 OpenCV + Pillow 打开并"显示"图片，并分离输出三个通道。

要点：
1. OpenCV 的 imread 得到的是 BGR 顺序，Pillow 得到的是 RGB 顺序 —— 必须显式转换才能正确显示；
2. 通道分离既可以 cv2.split，也可以 numpy 切片；
3. 单通道图必须用灰度(colormap=gray)显示，否则会误导成"蓝色/绿色/红色"图。

输出：
    outputs/figures/fig_task2_1_open.png      cv2 与 PIL 读取结果对比
    outputs/figures/fig_task2_1_channels.png  三通道分离 + 直方图
    outputs/images/channel_B|G|R.png          单独保存的三个通道（灰度图）
    outputs/data/task2_1_info.json            图像基本信息
"""

from __future__ import annotations

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from common import (
    DATA_DIR, FIG_DIR, IMG_DIR, bgr2rgb, dump_json, ensure_dirs, resolve_asset,
    save_fig, setup_matplotlib, show_panel,
)


def main() -> None:
    ensure_dirs(FIG_DIR, IMG_DIR, DATA_DIR)
    setup_matplotlib()
    img_path = resolve_asset("example1.jpg")

    # ---------------- 1. 两种库的读取方式
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)          # OpenCV: BGR, ndarray
    img_pil = Image.open(img_path)                                  # Pillow: RGB, Image
    img_rgb = np.array(img_pil)                                     # 转成 ndarray 便于计算

    print(f"OpenCV 读取: shape={img_bgr.shape}, dtype={img_bgr.dtype}, 通道顺序=BGR")
    print(f"Pillow 读取: mode={img_pil.mode}, size={img_pil.size}, 通道顺序=RGB")

    # 验证两库的通道顺序差异：OpenCV 第 0 通道 == Pillow 第 2 通道
    b0, g0, r0 = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
    diff_direct = np.abs(img_bgr.astype(int) - img_rgb.astype(int)).mean()
    diff_fixed = np.abs(img_bgr[:, :, ::-1].astype(int) - img_rgb.astype(int)).max()
    print(f"cv2 结果直接当 RGB 用时，与 Pillow 的平均绝对误差 = {diff_direct:.2f} (通道确实相反)")
    print(f"cv2 转 RGB 后与 Pillow 的最大差异 = {diff_fixed} (完全一致)")
    assert diff_fixed == 0

    # ---------------- 2. "显示"图片：无界面环境下用 matplotlib 渲染
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    show_panel(axes[0], img_bgr, "OpenCV 读入(BGR→RGB 正确显示)")
    axes[1].imshow(img_bgr)  # 故意不转换 -> 颜色错误，直观说明 BGR/RGB 问题
    axes[1].set_title("OpenCV 原始 BGR 直接显示(颜色错误)", fontsize=11)
    axes[1].axis("off")
    axes[2].imshow(img_rgb)
    axes[2].set_title("Pillow 读入(RGB, 与左图一致)", fontsize=11)
    axes[2].axis("off")
    fig.suptitle(f"task2-1 图像读取对比：{img_path.name}  {img_bgr.shape[1]}×{img_bgr.shape[0]}", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig_task2_1_open.png")

    # ---------------- 3. 通道分离（cv2.split 与 numpy 切片两种写法结果一致）
    b, g, r = cv2.split(img_bgr)
    assert np.array_equal(b, img_bgr[:, :, 0]) and np.array_equal(r, img_bgr[:, :, 2])

    from common import imwrite
    imwrite(IMG_DIR / "channel_B.png", b)
    imwrite(IMG_DIR / "channel_G.png", g)
    imwrite(IMG_DIR / "channel_R.png", r)

    # 单通道"着色"显示：只保留该通道，其余置零，便于肉眼看出每个通道贡献的颜色分量
    zeros = np.zeros_like(b)
    color_B = cv2.merge([b, zeros, zeros])
    color_G = cv2.merge([zeros, g, zeros])
    color_R = cv2.merge([zeros, zeros, r])

    fig = plt.figure(figsize=(13, 8.4))
    gs = fig.add_gridspec(3, 4, hspace=0.28, wspace=0.15)

    show_panel(fig.add_subplot(gs[0, 0]), img_bgr, "原图 (BGR)")
    show_panel(fig.add_subplot(gs[0, 1]), color_B, "仅 B 通道(着色)")
    show_panel(fig.add_subplot(gs[0, 2]), color_G, "仅 G 通道(着色)")
    show_panel(fig.add_subplot(gs[0, 3]), color_R, "仅 R 通道(着色)")

    show_panel(fig.add_subplot(gs[1, 0]), b, "B 通道灰度图")
    show_panel(fig.add_subplot(gs[1, 1]), g, "G 通道灰度图")
    show_panel(fig.add_subplot(gs[1, 2]), r, "R 通道灰度图")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    show_panel(fig.add_subplot(gs[1, 3]), gray, "灰度图 (0.299R+0.587G+0.114B)")

    ax = fig.add_subplot(gs[2, :])
    for name, ch, color in (("Blue", b, "tab:blue"), ("Green", g, "tab:green"), ("Red", r, "tab:red")):
        hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).ravel()
        ax.plot(hist, color=color, label=f"{name} 通道", lw=1.4)
    ax.set_title("三通道像素值直方图", fontsize=11)
    ax.set_xlabel("像素值 (0-255)")
    ax.set_ylabel("像素个数")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("task2-1 三通道分离（上：单通道着色；中：灰度图；下：直方图）", fontsize=13)
    save_fig(fig, "fig_task2_1_channels.png")

    # ---------------- 4. 数值统计
    info = {
        "file": img_path.name,
        "shape_hwc": list(img_bgr.shape),
        "dtype": str(img_bgr.dtype),
        "cv2_channel_order": "BGR",
        "pillow_channel_order": "RGB",
        "mean": {k: round(float(v.mean()), 3) for k, v in (("B", b), ("G", g), ("R", r))},
        "std": {k: round(float(v.std()), 3) for k, v in (("B", b), ("G", g), ("R", r))},
        "max_channel_value": {k: int(v.max()) for k, v in (("B", b), ("G", g), ("R", r))},
        "gray_mean": round(float(gray.mean()), 3),
    }
    dump_json(info, DATA_DIR / "task2_1_info.json")
    print("通道均值:", info["mean"])


if __name__ == "__main__":
    main()
