# -*- coding: utf-8 -*-
"""把图片/视频帧"翻译"成 ASCII 灰阶图，方便在没有图形界面的环境下判断画面内容；
同时探测偏红目标（小球）在各帧中的位置与大小。

用法：
    python tools/ascii_preview.py img  task2/example1.jpg
    python tools/ascii_preview.py vid  task2/example3.mp4 --frames 0 60 120 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

CHARS = "@%#*+=-:. "   # 越暗字符越"实"


def ascii_art(gray: np.ndarray, cols: int = 70) -> str:
    h, w = gray.shape
    rows = max(1, int(cols * h / w / 2.1))
    small = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
    lines = []
    for r in small:
        lines.append("".join(CHARS[min(len(CHARS) - 1, int(v / 256 * len(CHARS)))] for v in r))
    return "\n".join(lines)


def red_blobs(frame: np.ndarray, min_area=40):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 110, 50), (15, 255, 255)) | cv2.inRange(hsv, (165, 110, 50), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        fill = a / (np.pi * r * r + 1e-6)
        out.append((a, x, y, r, fill))
    out.sort(key=lambda t: -t[0])
    return out


def do_image(path: Path) -> None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print(f"\n=== {path.name}  {img.shape[1]}x{img.shape[0]} ===")
    print(ascii_art(gray, 78))
    print(f"灰度: mean={gray.mean():.1f} std={gray.std():.1f} "
          f"min={gray.min()} max={gray.max()}")
    print(f"HSV: H mean={hsv[..., 0].mean():.1f} std={hsv[..., 0].std():.1f}  "
          f"S mean={hsv[..., 1].mean():.1f}  V mean={hsv[..., 2].mean():.1f}")
    edges = cv2.Canny(gray, 50, 150)
    print(f"Canny 边缘像素占比 = {(edges > 0).mean() * 100:.2f}%")
    rb = red_blobs(img, min_area=200)
    print(f"偏红连通域(area>=200): {len(rb)} 个")
    for a, x, y, r, f in rb[:6]:
        print(f"   area={a:8.0f} center=({x:.0f},{y:.0f}) r={r:.1f} fill={f:.2f}")
    # 灰度直方图分段
    hist = np.histogram(gray, bins=8, range=(0, 256))[0]
    print("灰度分布(0-32,32-64,...):", (hist / gray.size * 100).round(1).tolist())


def do_video(path: Path, frame_idxs) -> None:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n=== {path.name}  {n} 帧 @ {fps:.2f}fps ===")
    if not frame_idxs:
        frame_idxs = [int(n * f) for f in (0.02, 0.2, 0.4, 0.6, 0.8, 0.98)]
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"  帧 {idx} 读取失败")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(f"\n--- frame {idx} ---")
        print(ascii_art(gray, 78))
        rb = red_blobs(frame, min_area=60)
        print(f"frame {idx}: 偏红连通域 {len(rb)} 个")
        for a, x, y, r, f in rb[:5]:
            print(f"   area={a:8.0f} center=({x:.0f},{y:.0f}) r={r:.1f} fill={f:.2f}")
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["img", "vid"])
    ap.add_argument("path")
    ap.add_argument("--frames", nargs="*", type=int, default=None)
    args = ap.parse_args()
    p = Path(args.path)
    if args.kind == "img":
        do_image(p)
    else:
        do_video(p, args.frames)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
