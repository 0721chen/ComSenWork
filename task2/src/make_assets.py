# -*- coding: utf-8 -*-
"""生成 task-2 所需的模拟素材（example1.jpg / example2.mp4 / example3.mp4 / example4.mp4）。

作业附带的 examples 目录没有随题目仓库一起提供，为了能完整跑通并复现全部结果，
本脚本用 OpenCV 合成一套**结构等价、且带真值（ground truth）**的素材：

* example1.jpg  : 白底 + 若干几何图形（圆、矩形、三角形、五角星、圆环、椭圆），用于轮廓检测；
* example2.mp4  : 静态纹理背景 + 运动红球，包含 "运动 → 静止 → 光照突变 → 快速运动" 四个阶段，
                 用于对比 "帧差法(不做背景减除)" 与 "MOG2 背景减除" 的差异；
* example3.mp4  : 红球沿 Lissajous 曲线平滑运动，背景有干扰色块与自身阴影，用于正常追踪；
* example4.mp4  : 红球高速运动 + 被灰色立柱遮挡 + 同色系干扰目标，用于考核追踪器的鲁棒性。

同时把每个球体的真实圆心逐帧写入 assets/*_gt.csv，供 task2-5 计算追踪误差。

用法：
    python src/make_assets.py            # 缺失的文件才生成
    python src/make_assets.py --force    # 强制重新生成
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from common import ASSETS_DIR, TASK2_DIR, VideoWriter

W, H, FPS = 640, 360, 30
BALL_R = 16
BALL_BGR = (30, 60, 220)  # 红色


# ------------------------------------------------------------------ 通用绘制
def make_background(w: int, h: int, seed: int) -> np.ndarray:
    """生成一张静态纹理背景（竖直渐变 + 噪声 + 几何结构），保证每帧"相同"。"""
    rng = np.random.default_rng(seed)
    grad = np.linspace(45, 95, h).reshape(h, 1)
    bg = np.dstack([grad * 1.15, grad * 1.0, grad * 0.90])  # BGR，偏蓝灰
    bg = bg + rng.normal(0, 7, (h, w, 1))
    bg = np.repeat(bg, 3, axis=2) if bg.shape[2] == 1 else bg
    bg = np.clip(bg, 0, 255).astype(np.uint8)

    # 静态结构：地面线、柱子、干扰色块
    cv2.rectangle(bg, (0, int(h * 0.78)), (w, h), (70, 78, 70), -1)
    cv2.rectangle(bg, (40, int(h * 0.55)), (70, h), (95, 95, 100), -1)
    cv2.rectangle(bg, (w - 90, int(h * 0.45)), (w - 60, h), (95, 95, 100), -1)
    cv2.circle(bg, (int(w * 0.25), int(h * 0.25)), 12, (120, 90, 60), -1)
    cv2.rectangle(bg, (int(w * 0.62), int(h * 0.18)), (int(w * 0.70), int(h * 0.26)), (80, 130, 110), -1)
    cv2.rectangle(bg, (int(w * 0.15), int(h * 0.60)), (int(w * 0.28), int(h * 0.66)), (60, 110, 130), -1)
    return bg


def draw_ball(frame: np.ndarray, center, radius=BALL_R, color=BALL_BGR, shadow: bool = True) -> None:
    """画一个带高光与地面阴影的球体。"""
    cx, cy = int(center[0]), int(center[1])
    if shadow:
        sy = min(H - 6, cy + radius + 12)
        cv2.ellipse(frame, (cx, sy), (int(radius * 1.5), 5), 0, 0, 360, (30, 30, 30), -1)
    cv2.circle(frame, (cx, cy), radius, color, -1)
    cv2.circle(frame, (cx, cy), radius, (10, 10, 10), 1, cv2.LINE_AA)
    cv2.circle(frame, (cx - radius // 3, cy - radius // 3), max(2, radius // 5), (235, 245, 255), -1, cv2.LINE_AA)


def add_noise(frame: np.ndarray, rng: np.random.Generator, sigma: float = 3.5) -> np.ndarray:
    noise = rng.normal(0, sigma, frame.shape[:2]).astype(np.float32)
    out = frame.astype(np.float32) + noise[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def write_gt(path: Path, rows, header=("frame", "x", "y", "radius", "visible")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        w.writerows(rows)
    print(f"[gt     ] {path}")


# ------------------------------------------------------------------ example1.jpg
def make_example1(path: Path) -> None:
    """白底 + 多种几何图形，用于轮廓检测（每个图形的面积/周长为解析可验证值）。"""
    size = 800
    img = np.full((size, size, 3), 245, np.uint8)

    cv2.circle(img, (200, 200), 90, (40, 40, 200), -1)                       # 圆
    cv2.rectangle(img, (360, 120), (560, 260), (60, 160, 60), -1)            # 矩形
    triangle = np.array([[200, 560], [360, 350], [80, 350]], np.int32)       # 三角形
    cv2.fillPoly(img, [triangle], (200, 120, 40))
    star = []                                                                 # 五角星
    for i in range(10):
        r = 110 if i % 2 == 0 else 46
        a = -math.pi / 2 + i * math.pi / 5
        star.append([int(560 + r * math.cos(a)), int(480 + r * math.sin(a))])
    cv2.fillPoly(img, [np.array(star, np.int32)], (170, 60, 210))
    cv2.circle(img, (700, 330), 70, (40, 70, 160), 18)                       # 圆环(有内孔)
    cv2.ellipse(img, (400, 690), (110, 50), 25, 0, 360, (240, 140, 90), -1)  # 椭圆

    rng = np.random.default_rng(7)
    img = np.clip(img.astype(np.float32) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    print(f"[image  ] {path}  ({size}x{size})")

    # ---- 解析真值（用于验证轮廓面积/周长检测精度）
    def polygon_metrics(pts):
        pts = np.asarray(pts, np.float64)
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        perim = float(np.sum(np.hypot(*(np.roll(pts, -1, axis=0) - pts).T)))
        return float(area), perim

    tri_area, tri_perim = polygon_metrics(triangle)
    star_area, star_perim = polygon_metrics(np.array(star, np.float64))
    a_, b_ = 110.0, 50.0
    hh = ((a_ - b_) / (a_ + b_)) ** 2
    ell_perim = math.pi * (a_ + b_) * (1 + 3 * hh / (10 + math.sqrt(4 - 3 * hh)))
    meta = {
        "image": path.name,
        "shapes": [
            {"name": "circle", "centroid": [200, 200], "area_px2": round(math.pi * 90 ** 2, 1),
             "perimeter_px": round(2 * math.pi * 90, 1)},
            {"name": "rectangle", "centroid": [460, 190], "area_px2": 201 * 141,
             "perimeter_px": 2 * (200 + 140)},
            {"name": "triangle", "centroid": [213.3, 420.0], "area_px2": round(tri_area, 1),
             "perimeter_px": round(tri_perim, 1)},
            {"name": "star", "centroid": [560, 480], "area_px2": round(star_area, 1),
             "perimeter_px": round(star_perim, 1)},
            {"name": "ring(outer)", "centroid": [700, 330], "area_px2": round(math.pi * 79 ** 2, 1),
             "perimeter_px": round(2 * math.pi * 79, 1),
             "note": "cv2.circle 的 thickness 沿半径线向内外各扩 9px：外径 79、内径 61"},
            {"name": "ellipse", "centroid": [400, 690], "area_px2": round(math.pi * a_ * b_, 1),
             "perimeter_px": round(ell_perim, 1)},
        ],
        "ring_net_area_px2": round(math.pi * (79 ** 2 - 61 ** 2), 1),
    }
    meta["total_area_px2"] = round(sum(s["area_px2"] for s in meta["shapes"]), 1)
    meta["total_perimeter_px"] = round(sum(s["perimeter_px"] for s in meta["shapes"]), 1)
    with open(path.parent / "example1_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[meta   ] {path.parent / 'example1_meta.json'}")
    return meta


# ------------------------------------------------------------------ example2.mp4
def make_example2(path: Path, gt_path: Path) -> None:
    """运动目标检测素材：运动 → 静止 → 光照突变 → 快速运动。"""
    n_frames = 180
    bg = make_background(W, H, seed=11)
    rng = np.random.default_rng(3)
    rows = []

    with VideoWriter(path, FPS, (W, H)) as vw:
        for f in range(n_frames):
            frame = bg.copy()

            if f < 60:            # 阶段 A：匀速水平运动
                x = 70 + (f / 59.0) * 480
                y = 190 + 55 * math.sin(f / 12.0)
                phase = "A_moving"
            elif f < 90:          # 阶段 B：完全静止（帧差法会"丢"目标）
                x, y = 550.0, 190.0
                phase = "B_static"
            elif f < 92:          # 阶段 C：全局光照突变（帧差法全屏误检）
                x, y = 550.0, 190.0
                phase = "C_light_flash"
            else:                 # 阶段 D：快速往返运动
                t = f - 92
                x = 340 + 300 * math.sin(2 * math.pi * t / 88.0)
                y = 200 + 90 * math.sin(4 * math.pi * t / 88.0)
                phase = "D_fast"

            draw_ball(frame, (x, y), BALL_R)
            frame = add_noise(frame, rng)

            if phase == "C_light_flash":  # 两帧整体提亮，模拟自动曝光/灯光突变
                frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=45)

            vw.write(frame)
            rows.append([f, round(x, 2), round(y, 2), BALL_R, 1, phase])

    write_gt(gt_path, rows, header=("frame", "x", "y", "radius", "visible", "phase"))
    print(f"[video  ] {path}  ({n_frames} frames @ {FPS}fps)")


# ------------------------------------------------------------------ example3.mp4
def make_example3(path: Path, gt_path: Path) -> None:
    """常规追踪素材：小球平滑运动 + 静态干扰色块 + 自身阴影。"""
    n_frames = 300
    bg = make_background(W, H, seed=23)
    rng = np.random.default_rng(5)
    rows = []

    with VideoWriter(path, FPS, (W, H)) as vw:
        for f in range(n_frames):
            t = 2 * math.pi * f / 100.0
            x = 320 + 210 * math.sin(t)
            y = 175 + 95 * math.sin(1.6 * t + 0.4)
            frame = bg.copy()
            draw_ball(frame, (x, y), BALL_R + int(2 * math.sin(t)))
            frame = add_noise(frame, rng)
            vw.write(frame)
            rows.append([f, round(x, 2), round(y, 2), BALL_R, 1])

    write_gt(gt_path, rows)
    print(f"[video  ] {path}  ({n_frames} frames @ {FPS}fps)")


# ------------------------------------------------------------------ example4.mp4
def make_example4(path: Path, gt_path: Path) -> None:
    """困难追踪素材：高速运动 + 立柱遮挡 + 同色系干扰目标。"""
    n_frames = 240
    bg = make_background(W, H, seed=41)
    rng = np.random.default_rng(9)
    rows = []
    bar_x0, bar_x1 = 400, 460       # 遮挡立柱

    with VideoWriter(path, FPS, (W, H)) as vw:
        for f in range(n_frames):
            t = 2 * math.pi * f / 80.0
            x = 320 + 230 * math.sin(t)
            y = 175 + 105 * math.sin(1.35 * t + 1.0)
            vis = not (bar_x0 - BALL_R < x < bar_x1 + BALL_R)

            frame = bg.copy()
            # 同色系干扰目标（浅橙），与小球相向运动
            dx = 620 - (f / n_frames) * 560
            dy = 90 + 25 * math.sin(f / 9.0)
            draw_ball(frame, (dx, dy), 13, color=(60, 130, 235), shadow=False)

            draw_ball(frame, (x, y), BALL_R)
            if vis is False:  # 被立柱挡住：先画立柱再覆盖球
                cv2.rectangle(frame, (bar_x0, 0), (bar_x1, H), (105, 105, 108), -1)
                cv2.line(frame, (bar_x0, 0), (bar_x0, H), (70, 70, 72), 2)
                cv2.line(frame, (bar_x1, 0), (bar_x1, H), (70, 70, 72), 2)

            frame = add_noise(frame, rng, sigma=5.0)
            vw.write(frame)
            rows.append([f, round(x, 2), round(y, 2), BALL_R, int(vis)])

    write_gt(gt_path, rows)
    print(f"[video  ] {path}  ({n_frames} frames @ {FPS}fps)")


# ------------------------------------------------------------------ main
def main() -> None:
    p = argparse.ArgumentParser(description="生成 task-2 模拟素材（官方素材缺失时使用）")
    p.add_argument("--force", action="store_true", help="已存在也重新生成")
    args = p.parse_args()

    # 官方素材齐全时不需要生成模拟素材
    official = ["example1.jpg", "example2.mp4", "example3.mp4", "example4.mp4"]
    if not args.force and all((TASK2_DIR / f).exists() for f in official):
        print("[跳过  ] 检测到官方素材（example1.jpg / example2~4.mp4），无需生成模拟素材")
        return

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        (ASSETS_DIR / "example1.jpg", lambda f: make_example1(f)),
        (ASSETS_DIR / "example2.mp4", lambda f: make_example2(f, ASSETS_DIR / "example2_gt.csv")),
        (ASSETS_DIR / "example3.mp4", lambda f: make_example3(f, ASSETS_DIR / "example3_gt.csv")),
        (ASSETS_DIR / "example4.mp4", lambda f: make_example4(f, ASSETS_DIR / "example4_gt.csv")),
    ]
    for target, fn in jobs:
        if target.exists() and not args.force:
            print(f"[skip   ] {target} 已存在（--force 可覆盖）")
            continue
        fn(target)

    print("\n素材目录:", ASSETS_DIR)
    for f in sorted(ASSETS_DIR.iterdir()):
        print(f"  {f.name:24s} {f.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
