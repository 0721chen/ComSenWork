# -*- coding: utf-8 -*-
"""交付自检：检查 task2 / task3 的全部产出文件是否齐全、图片是否"有内容"、视频能否解码。

用法：python tools/verify_outputs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
T2 = ROOT / "task2" / "outputs"
T3 = ROOT / "task3" / "outputs"
T4 = ROOT / "task4_yolo" / "outputs"

EXPECTED_IMAGES = [
    T2 / "figures" / "fig_task2_1_open.png",
    T2 / "figures" / "fig_task2_1_channels.png",
    T2 / "figures" / "fig_task2_2_geometric.png",
    T2 / "figures" / "fig_task2_3_contours.png",
    T2 / "figures" / "fig_task2_4_motion_stats.png",
    T2 / "figures" / "fig_task2_4_frames.png",
    T2 / "figures" / "fig_task2_5_tracking_example3.png",
    T2 / "figures" / "fig_task2_5_tracking_example4.png",
    T2 / "images" / "task2_3_contours_annotated.png",
    T2 / "images" / "task2_3_mask.png",
    T2 / "images" / "task2_3_hierarchy.png",
    T2 / "images" / "channel_B.png",
    T2 / "images" / "channel_G.png",
    T2 / "images" / "channel_R.png",
    T3 / "figures" / "fig_task3_2_surface.png",
    T3 / "figures" / "fig_task3_2_slice.png",
    T3 / "figures" / "fig_task3_2_loss.png",
    T3 / "figures" / "fig_task3_2_scatter.png",
    T3 / "figures" / "fig_task3_3_curves.png",
    T3 / "figures" / "fig_task3_3_predictions.png",
    T3 / "figures" / "fig_task3_3_confusion.png",
    T3 / "figures" / "fig_task3_3_misclassified.png",
    T3 / "figures" / "fig_task3_3_dataset.png",
    T3 / "figures" / "fig_task3_3_class_dist.png",
]

EXPECTED_VIDEOS = [
    T2 / "videos" / "example2_motion_compare.mp4",
    T2 / "videos" / "example3_tracking.mp4",
    T2 / "videos" / "example4_tracking.mp4",
]

EXPECTED_JSON = [
    T2 / "data" / "task2_1_info.json",
    T2 / "data" / "task2_3_contours.json",
    T2 / "data" / "task2_4_motion_stats.json",
    T2 / "data" / "task2_5_tracking_metrics.json",
    T3 / "logs" / "task3_2_result.json",
    T3 / "logs" / "task3_3_metrics.json",
    T4 / "metrics" / "task4_1_model_info.json",
    T4 / "metrics" / "task4_2_detections.json",
    T4 / "metrics" / "task4_3_metrics.json",
]

# task-4（YOLO 方向）题1~题3 的产出
EXPECTED_T4_IMAGES = [
    T4 / "figures" / "task4_1_api_cheatsheet.png",
    T4 / "figures" / "task4_2_predictions.png",
    T4 / "figures" / "task4_3_results.png",
    T4 / "figures" / "task4_3_confusion_matrix.png",
    T4 / "figures" / "task4_3_PR_curve.png",
    T4 / "figures" / "task4_3_summary.png",
    T4 / "images" / "pred_coco-bear.jpg",
    T4 / "images" / "pred_coco-bus.jpg",
    T4 / "images" / "pred_coco-person.jpg",
]
EXPECTED_T4_WEIGHTS = [
    T4 / "runs" / "drone_yolov8n" / "weights" / "best.pt",
    T4 / "runs" / "drone_yolov8n" / "weights" / "best.onnx",
]


def check_images() -> list[str]:
    problems = []
    print(f"{'图片':58s} {'尺寸':>12s} {'灰度std':>9s} {'结论':>6s}")
    for p in EXPECTED_IMAGES:
        if not p.exists():
            problems.append(f"缺失: {p.relative_to(ROOT)}")
            print(f"{str(p.relative_to(ROOT)):58s} {'-':>12s} {'-':>9s} {'缺失':>6s}")
            continue
        img = Image.open(p).convert("L")
        arr = np.asarray(img, dtype=np.float32)
        std = float(arr.std())
        ok = std > 2.0
        if not ok:
            problems.append(f"疑似空白图: {p.relative_to(ROOT)} (std={std:.3f})")
        print(f"{str(p.relative_to(ROOT)):58s} {f'{img.width}x{img.height}':>12s} "
              f"{std:>9.2f} {'OK' if ok else '空白':>6s}")
    return problems


def check_videos() -> list[str]:
    problems = []
    print(f"\n{'视频':40s} {'分辨率':>10s} {'帧数':>6s} {'fps':>6s} {'时长':>7s}")
    for p in EXPECTED_VIDEOS:
        if not p.exists():
            problems.append(f"缺失: {p.relative_to(ROOT)}")
            continue
        cap = cv2.VideoCapture(str(p))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = cap.read()
        cap.release()
        if not ok or n == 0:
            problems.append(f"无法解码: {p.relative_to(ROOT)}")
        print(f"{str(p.relative_to(ROOT)):40s} {f'{w}x{h}':>10s} {n:>6d} {fps:>6.1f} "
              f"{n / max(fps, 1):>6.1f}s")
    return problems


def check_json() -> list[str]:
    problems = []
    print()
    for p in EXPECTED_JSON:
        if not p.exists():
            problems.append(f"缺失: {p.relative_to(ROOT)}")
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        keys = list(data.keys())[:6]
        print(f"{str(p.relative_to(ROOT)):52s} keys={keys}")
    return problems


def check_task4() -> list[str]:
    """task-4：图片非空白 + 权重/ONNX 文件存在且体积合理。"""
    problems = []
    print("\n[task-4 产出]")
    for p in EXPECTED_T4_IMAGES:
        if not p.exists():
            problems.append(f"缺失: {p.relative_to(ROOT)}")
            print(f"{str(p.relative_to(ROOT)):56s} 缺失")
            continue
        img = Image.open(p).convert("L")
        arr = np.asarray(img, dtype=np.float32)
        std = float(arr.std())
        ok = std > 2.0
        if not ok:
            problems.append(f"疑似空白图: {p.relative_to(ROOT)}")
        print(f"{str(p.relative_to(ROOT)):56s} {f'{img.width}x{img.height}':>12s} "
              f"std={std:6.2f} {'OK' if ok else '空白'}")
    for p in EXPECTED_T4_WEIGHTS:
        if not p.exists():
            problems.append(f"缺失权重: {p.relative_to(ROOT)}")
        else:
            mb = p.stat().st_size / 1024 / 1024
            print(f"{str(p.relative_to(ROOT)):56s} {mb:8.2f} MB")
            if mb < 1:
                problems.append(f"权重大小异常: {p.relative_to(ROOT)} ({mb:.2f} MB)")
    return problems


def main() -> None:
    problems = []
    problems += check_images()
    problems += check_videos()
    problems += check_json()
    problems += check_task4()

    print("\n" + "=" * 60)
    if problems:
        print(f"发现 {len(problems)} 个问题：")
        for x in problems:
            print("  -", x)
        sys.exit(1)
    print("全部产出检查通过 ✅")


if __name__ == "__main__":
    main()
