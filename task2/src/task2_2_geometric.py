# -*- coding: utf-8 -*-
"""task2-2：对图片做 缩放 / 旋转 / 平移 / 翻转。

* 缩放：cv2.resize（两种插值 INTER_NEAREST / INTER_LINEAR 对比）
* 旋转：cv2.getRotationMatrix2D + warpAffine（可 expand，避免裁切）
* 平移：仿射矩阵 M=[[1,0,tx],[0,1,ty]] + warpAffine
* 翻转：cv2.flip（水平/垂直/水平+垂直）
* 综合仿射：旋转+缩放+平移组合成一个矩阵

输出：outputs/figures/fig_task2_2_geometric.png 与 outputs/images/task2_2_*.png
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import FIG_DIR, IMG_DIR, ensure_dirs, imwrite, resolve_asset, save_fig, setup_matplotlib, show_panel


def rotate(img: np.ndarray, angle: float, expand: bool = True, scale: float = 1.0):
    """绕图像中心旋转 angle 度；expand=True 时放大画布保证不被裁切。"""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    if not expand:
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += new_w / 2.0 - center[0]
    M[1, 2] += new_h / 2.0 - center[1]
    # 白色填充保持与原图背景一致
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(245, 245, 245))


def translate(img: np.ndarray, tx: int, ty: int) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(245, 245, 245))


def flip(img: np.ndarray, mode: str) -> np.ndarray:
    codes = {"horizontal": 1, "vertical": 0, "both": -1}
    return cv2.flip(img, codes[mode])


def main() -> None:
    ensure_dirs(FIG_DIR, IMG_DIR)
    setup_matplotlib()
    src = resolve_asset("example1.jpg")
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    print(f"原图: {w}×{h}")

    # ---------------- 各种几何变换
    scaled_down = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)   # 缩小用 AREA 更平滑
    scaled_up = cv2.resize(img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_LINEAR)   # 放大用 LINEAR
    scaled_nearest = cv2.resize(img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_NEAREST)
    rotated = rotate(img, 45, expand=True)
    rotated_30 = rotate(img, -30, expand=False)          # 不扩画布 -> 四角会被裁掉
    translated = translate(img, tx=140, ty=90)
    flip_h = flip(img, "horizontal")
    flip_v = flip(img, "vertical")
    flip_both = flip(img, "both")

    # ---------------- 综合仿射：先绕中心旋转 25°，再整体缩小 0.7，再平移 (60, 20)
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), 25, 0.7)
    M_trans = np.float32([[1, 0, 60], [0, 1, 20]])
    M = np.vstack([M_rot, [0, 0, 1]]) @ np.vstack([M_trans, [0, 0, 1]])
    affine = cv2.warpAffine(img, M[:2], (w, h), flags=cv2.INTER_LINEAR, borderValue=(245, 245, 245))
    print("综合仿射矩阵 M =\n", np.round(M[:2], 3))

    for name, im in [
        ("task2_2_scale_0.4.jpg", scaled_down),
        ("task2_2_scale_1.6.jpg", scaled_up),
        ("task2_2_rotate_45.jpg", rotated),
        ("task2_2_translate.jpg", translated),
        ("task2_2_flip_horizontal.jpg", flip_h),
        ("task2_2_flip_vertical.jpg", flip_v),
        ("task2_2_flip_both.jpg", flip_both),
        ("task2_2_affine_combo.jpg", affine),
    ]:
        imwrite(IMG_DIR / name, im)

    # ---------------- 汇总图
    panels = [
        (img, f"原图 {w}×{h}"),
        (scaled_down, f"缩放 ×0.4 (INTER_AREA) {scaled_down.shape[1]}×{scaled_down.shape[0]}"),
        (scaled_up, f"缩放 ×1.6 (INTER_LINEAR) {scaled_up.shape[1]}×{scaled_up.shape[0]}"),
        (scaled_nearest, "缩放 ×1.6 (INTER_NEAREST, 有锯齿)"),
        (rotated, f"旋转 +45° (expand) {rotated.shape[1]}×{rotated.shape[0]}"),
        (rotated_30, "旋转 -30° (不 expand, 被裁切)"),
        (translated, "平移 tx=140, ty=90"),
        (flip_h, "水平翻转 (flipCode=1)"),
        (flip_v, "垂直翻转 (flipCode=0)"),
        (flip_both, "水平+垂直翻转 (flipCode=-1)"),
        (affine, "综合仿射: 旋转25°+缩放0.7+平移(60,20)"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (im, title) in zip(axes.ravel(), panels):
        show_panel(ax, im, title)
    fig.suptitle("task2-2 缩放 / 旋转 / 平移 / 翻转 / 仿射组合", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "fig_task2_2_geometric.png")


if __name__ == "__main__":
    main()
