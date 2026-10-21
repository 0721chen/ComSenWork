# -*- coding: utf-8 -*-
"""探查官方素材：图片尺寸、视频帧率/帧数/分辨率、画面里的运动目标颜色与外观、
以及 task3 的 MNIST parquet 文件结构。用于给后续脚本选参数（不做任何修改）。

用法：python tools/probe_assets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T2 = ROOT / "task2"
T3 = ROOT / "task3"


def probe_image(path: Path) -> None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    print(f"\n[图片] {path.name}: {img.shape[1]}x{img.shape[0]}x{img.shape[2]}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(f"  Otsu 阈值 = {t:.0f}，灰度均值 {gray.mean():.1f}，std {gray.std():.1f}")
    for thr in (60, 90, 120, t):
        _, b = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = sorted([cv2.contourArea(c) for c in cnts], reverse=True)[:8]
        print(f"  阈值 {thr:6.0f} → 轮廓 {len(cnts):3d} 个，最大面积 {areas[:8]}")


def probe_video(path: Path, samples=6) -> None:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n[视频] {path.name}: {w}x{h}  {fps:.2f} fps  {n} 帧  时长 {n/max(fps,1):.1f}s  "
          f"{path.stat().st_size/1024/1024:.1f}MB")

    mog2 = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=True)
    idxs = np.linspace(0, max(n - 1, 1), samples).astype(int)
    frame_list = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frame_list.append(f)
    cap.release()

    for i, f in enumerate(frame_list):
        fg = mog2.apply(f, learningRate=0.02)
        m = np.where(fg == 255, 255, 0).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if i not in idxs:
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > 60]
        info = []
        for c in sorted(cnts, key=lambda c: -cv2.contourArea(c))[:3]:
            a = cv2.contourArea(c)
            x, y, bw, bh = cv2.boundingRect(c)
            mask = np.zeros(m.shape, np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            mean_bgr = cv2.mean(f, mask=mask)[:3]
            patch = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[mask > 0]
            hsv_med = np.median(patch, axis=0) if len(patch) else [0, 0, 0]
            info.append(f"area={a:7.0f} bbox=({x},{y},{bw},{bh}) BGR={tuple(round(v) for v in mean_bgr)} "
                        f"HSV中位=({hsv_med[0]:.0f},{hsv_med[1]:.0f},{hsv_med[2]:.0f})")
        print(f"  frame {i:4d}: 前景连通域 {len(cnts)} 个")
        for s in info:
            print(f"      {s}")
        if not cnts:
            print(f"      画面均值 BGR={tuple(round(v) for v in f.reshape(-1,3).mean(axis=0))}")


def probe_parquet(path: Path) -> None:
    import pandas as pd

    print(f"\n[parquet] {path.name}: {path.stat().st_size/1024/1024:.1f}MB")
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print("  pandas 读取失败:", e)
        try:
            import pyarrow.parquet as pq
            print("  schema:", pq.read_schema(path))
        except Exception as e2:
            print("  pyarrow 也失败:", e2)
        return
    print(f"  shape={df.shape}  columns={list(df.columns)[:12]}")
    print("  dtypes:", {c: str(t) for c, t in df.dtypes.items()})
    print(df.head(3).to_string(max_colwidth=60))
    for c in df.columns:
        v = df[c].iloc[0]
        print(f"   列 {c!r}: type={type(v).__name__}", end="")
        if isinstance(v, dict):
            print(f" keys={list(v.keys())}", end="")
            for k, vv in v.items():
                if isinstance(vv, (bytes, bytearray)):
                    print(f" {k}=bytes(len={len(vv)}, head={bytes(vv[:8])!r})", end="")
                else:
                    print(f" {k}={vv!r}", end="")
        elif isinstance(v, (bytes, bytearray)):
            print(f" len={len(v)} head={bytes(v[:8])!r}", end="")
        print()
    for c in df.columns:
        s = df[c]
        if s.dtype.kind in "iu" or s.nunique() < 20:
            vc = s.value_counts().head(12)
            print(f"   {c} 取值分布: {dict(vc)}")


def main() -> None:
    if (T2 / "example1.jpg").exists():
        probe_image(T2 / "example1.jpg")
    for name in ("example2.mp4", "example3.mp4", "example4.mp4"):
        p = T2 / name
        if p.exists():
            probe_video(p)
    pq = T3 / "train-00000-of-00001.parquet"
    if pq.exists():
        probe_parquet(pq)


if __name__ == "__main__":
    main()
