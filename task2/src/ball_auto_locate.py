# -*- coding: utf-8 -*-
"""自动定位"来回滚动的小球"（没有人工标注时使用）。

判别依据（四条同时成立才像"球"）：

1. **颜色均匀 + 圆形**：按 HSV 色相扇区分别做颜色分割，取圆度 >= 0.55 的连通块作为候选
   （球是"单一颜色 + 圆形"的物体，这一点能过滤掉绝大多数场景纹理）；
2. **尺寸稳定**：球在整段视频里半径几乎不变（半径变异系数 CV 小）；
3. **连续可见**：球一直在画面里滚动 → 轨迹链的帧覆盖率接近 100%；
4. **往复滚动签名**：轨迹是一条线段（垂直方向漂移 << 平行方向跨度），
   且沿主运动方向的符号反复翻转（>= 2 次）。

对每条候选链加权打分，取最高者作为初始框与参考轨迹。

用法（独立调试，会打印候选链明细）：
    python task2/src/ball_auto_locate.py --video example3.mp4
    python task2/src/ball_auto_locate.py --video example4.mp4 --top 5
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from common import TASK2_DIR, DATA_DIR, dump_json, ensure_dirs, resolve_asset

CACHE_DIR = TASK2_DIR / "outputs" / "data" / "cache"

SCALE = 0.5               # 检测在小图上做（速度快、抗噪）
AREA_RANGE = (60, 5000)   # 小图上的连通域面积范围
FILL_MIN = 0.55           # 圆度下限
MIN_R_VAL = 90            # 至少要有一定亮度
GAP_MAX = 6               # 允许漏检帧数
HUE_STEP = 30


@dataclass
class Candidate:
    frame: int
    x: float
    y: float
    r: float
    area: float
    fill: float
    bgr: tuple
    hue_bin: int


@dataclass
class Chain:
    pts: list = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.pts)


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    k = int(max(3, min(k, max(3, len(x) // 2 * 2 + 1))))
    return np.convolve(x, np.ones(k) / k, mode="valid")


# ---------------------------------------------------------------- 候选提取
def build_candidates(video_path: Path, use_cache: bool = True):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{video_path.stem}_colorcands.pkl"
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            raw, meta = pickle.load(f)
        per_frame = {k: [Candidate(*t) for t in v] for k, v in raw.items()}
        print(f"[auto] 载入候选缓存 {cache.name}（{meta['frames']} 帧, {meta['W']}x{meta['H']}）")
        return per_frame, meta["fps"], (meta["W"], meta["H"])

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bins = [(h0, h0 + HUE_STEP) for h0 in range(0, 180, HUE_STEP)]

    per_frame, idx = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        cands = []
        for (h0, h1) in bins:
            m = cv2.inRange(hsv, (h0, 60, MIN_R_VAL), (h1, 255, 255))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                a = cv2.contourArea(c)
                if not (AREA_RANGE[0] <= a <= AREA_RANGE[1]):
                    continue
                (cx, cy), r = cv2.minEnclosingCircle(c)
                if r < 4:
                    continue
                fill = a / (np.pi * r * r + 1e-6)
                if fill < FILL_MIN:
                    continue
                mask_c = np.zeros(m.shape, np.uint8)
                cv2.drawContours(mask_c, [c], -1, 255, -1)
                bgr = tuple(round(float(v), 1) for v in cv2.mean(small, mask=mask_c)[:3])
                cands.append(Candidate(idx, cx / SCALE, cy / SCALE, r / SCALE, a / SCALE ** 2,
                                       float(fill), bgr, h0))
        per_frame[idx] = cands
        idx += 1
    cap.release()

    meta = {"fps": fps, "W": W, "H": H, "frames": idx}
    with open(cache, "wb") as f:
        pickle.dump(({k: [(c.frame, c.x, c.y, c.r, c.area, c.fill, c.bgr, c.hue_bin) for c in v]
                      for k, v in per_frame.items()}, meta), f)
    print(f"[auto] 候选提取完成 {idx} 帧（平均 {sum(len(v) for v in per_frame.values())/max(idx,1):.2f} 个/帧），"
          f"已缓存 {cache.name}")
    return per_frame, fps, (W, H)


# ---------------------------------------------------------------- 串链
def link_candidates(per_frame: dict, W: int, H: int) -> list:
    max_step = 0.05 * max(W, H)          # 单帧最大位移（1080p≈96px）
    chains: list = []
    for idx in sorted(per_frame):
        for c in per_frame[idx]:
            best, best_d = None, None
            for ch in chains:
                last = ch.pts[-1]
                gap = idx - last.frame
                if not (0 < gap <= GAP_MAX):
                    continue
                d = float(np.hypot(c.x - last.x, c.y - last.y))
                if d > max_step * gap:
                    continue
                if abs(c.r - last.r) > 0.5 * last.r + 3:
                    continue
                if np.linalg.norm(np.array(c.bgr) - np.array(last.bgr)) > 40:
                    continue
                if best_d is None or d < best_d:
                    best, best_d = ch, d
            if best is not None:
                best.pts.append(c)
            else:
                chains.append(Chain([c]))
    return chains


def score_chain(ch: Chain) -> dict:
    pts = ch.pts
    fr = np.array([p.frame for p in pts])
    xs = np.array([p.x for p in pts], np.float32)
    ys = np.array([p.y for p in pts], np.float32)
    rs = np.array([p.r for p in pts], np.float32)

    along, perp, axis = (xs, ys, "水平") if np.ptp(xs) >= np.ptp(ys) else (ys, xs, "垂直")
    perp_ext = float(np.percentile(perp, 95) - np.percentile(perp, 5))
    along_ext = float(np.ptp(along))
    straightness = max(0.0, 1.0 - perp_ext / max(along_ext, 1e-6))

    flips = 0
    if len(along) >= 12:
        sm = _smooth(along, 7)
        v = np.diff(sm)
        s = np.sign(v[np.abs(v) > max(np.abs(v).mean() * 0.3, 1e-6)])
        s = s[s != 0]
        flips = int((np.diff(s) != 0).sum()) if len(s) > 1 else 0

    coverage = float(len(pts) / max(1, fr[-1] - fr[0] + 1))
    size_cv = float(np.std(rs) / max(np.mean(rs), 1e-6))
    colors = np.array([p.bgr for p in pts], np.float32)
    color_std = float(np.mean(np.std(colors, axis=0))) if len(colors) > 2 else 999.0
    speed = float(np.mean(np.abs(np.diff(along)))) if len(along) > 2 else 0.0

    score = (3.0 * straightness + 1.5 * min(flips, 6) + 3.0 * coverage
             + 2.0 * max(0.0, 1 - size_cv * 3) + 1.5 * max(0.0, 1 - color_std / 40))
    return {"flips": flips, "axis": axis, "straightness": round(straightness, 3),
            "coverage": round(coverage, 3), "size_cv": round(size_cv, 4),
            "color_std": round(color_std, 1), "speed_px_per_frame": round(speed, 1),
            "score": round(float(score), 3), "span": [int(fr[0]), int(fr[-1])]}


def locate(video_path: Path, top: int = 5, use_cache: bool = True,
           min_length: int = 25, min_coverage: float = 0.7) -> dict:
    per_frame, fps, (W, H) = build_candidates(video_path, use_cache=use_cache)
    chains = link_candidates(per_frame, W, H)
    scored = [(c, score_chain(c)) for c in chains if c.length >= min_length]
    scored = [(c, s) for c, s in scored if s["coverage"] >= min_coverage]
    scored.sort(key=lambda t: -t[1]["score"])

    print(f"[auto] {video_path.name}: {W}x{H}@{fps:.1f}fps，满足条件的候选链 {len(scored)} 条")
    for c, s in scored[:top]:
        xs = [p.x for p in c.pts]
        ys = [p.y for p in c.pts]
        print(f"   帧 {s['span'][0]:3d}..{s['span'][1]:3d} 长度 {c.length:3d} 翻转 {s['flips']:2d}"
              f"({s['axis']}) 直线 {s['straightness']:.2f} 覆盖 {s['coverage']:.2f} "
              f"尺寸CV {s['size_cv']:.3f} 颜色std {s['color_std']:.1f} → 分数 {s['score']:.2f} | "
              f"x[{min(xs):.0f}..{max(xs):.0f}] y[{min(ys):.0f}..{max(ys):.0f}] "
              f"r中位 {np.median([p.r for p in c.pts]):.0f}px "
              f"BGR={tuple(int(v) for v in np.median(np.array([p.bgr for p in c.pts]), axis=0))}")

    if not scored:
        return {"ok": False, "reason": "没有满足条件的候选链"}
    best, best_score = scored[0]
    return {
        "ok": True, "fps": fps, "size": [W, H], "score": best_score,
        "start": {"frame": best.pts[0].frame, "x": round(best.pts[0].x, 1),
                  "y": round(best.pts[0].y, 1), "r": round(best.pts[0].r, 1)},
        "track": [{"frame": p.frame, "x": round(p.x, 1), "y": round(p.y, 1), "r": round(p.r, 1)}
                  for p in best.pts],
        "candidates": [{"length": c.length, "score": s, "start": [c.pts[0].frame,
                                                                 round(c.pts[0].x),
                                                                 round(c.pts[0].y),
                                                                 round(c.pts[0].r)]}
                       for c, s in scored[:top]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="自动定位来回滚动的小球")
    ap.add_argument("--video", default="example4.mp4")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    path = resolve_asset(args.video)
    res = locate(path, top=args.top, use_cache=not args.no_cache)
    if res.get("ok"):
        dump_json(res, DATA_DIR / f"{path.stem}_auto_locate.json")


if __name__ == "__main__":
    main()
