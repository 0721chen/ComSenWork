# -*- coding: utf-8 -*-
"""task2-4：example2.mp4 运动目标检测 —— 对比"不使用背景减除(帧差法)"与"使用 MOG2 背景减除"。

三种处理方法：
    A. 不使用背景减除：|I_t − I_{t−1}| > T                    （两帧差分，无记忆）
    B. MOG2 背景减除：逐像素多高斯背景模型，前景=偏离模型；detectShadows 剔除阴影
    C. MOG2 + 亮度补偿：先把每帧全局平均亮度对齐到参考帧，再送进 MOG2（抗光照突变）

example2.mp4 是 1280×720@29.97fps、约 33 秒的真实监控/街景片段，**没有人工真值**，
所以用下面这些可量化的代理指标做对比：

    前景面积均值     —— 检出目标"胖不胖"（帧差法只得到运动边缘，面积偏小）
    连通域个数均值   —— 目标是否被碎成多块
    单目标一致性     —— "恰好 1 个显著连通域"的帧占比
    轮廓填充度       —— 最大连通域面积 / 其外接框面积
    轮廓实心度       —— 轮廓面积 / 凸包面积（帧差的"月牙"低，MOG2 的完整剪影高）
    质心抖动         —— 相邻帧最大目标质心位移均值（越小越稳）
    全屏误检帧数     —— 前景面积 > 15% 画面 的帧数（曝光/光照突变症状）
    单帧耗时         —— 三种方法的计算成本对比

输出：
    outputs/videos/example2_motion_compare.mp4      四分屏标注视频（原图/A/B/C）
    outputs/figures/fig_task2_4_motion_stats.png    逐帧曲线 + 指标对比
    outputs/figures/fig_task2_4_frames.png          关键帧掩码对比
    outputs/data/task2_4_motion_stats.json/.csv     数值结果
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import (
    DATA_DIR, FIG_DIR, VID_DIR, VideoWriter, dump_json, ensure_dirs, find_gt,
    open_video, put_text, resolve_asset, save_fig, setup_matplotlib, show_panel,
)

MIN_AREA = 120           # 连通域面积阈值（720p 下的噪点过滤）
DIFF_THRESHOLD = 25      # 帧差法二值化阈值
BIG_AREA_RATIO = 0.15    # 前景面积占比超过该值视为"全屏误检"

PANEL_W, PANEL_H = 480, 270
CANVAS_W, CANVAS_H, HEADER = PANEL_W * 2, PANEL_H * 2 + 40, 40
K_OPEN = np.ones((3, 3), np.uint8)
K_CLOSE = np.ones((5, 5), np.uint8)

METHODS = {
    "naive": dict(label="A 帧差法(无背景减除)", color="tab:orange", panel_color=(0, 0, 255)),
    "mog2": dict(label="B MOG2 背景减除", color="tab:blue", panel_color=(255, 0, 200)),
    "comp": dict(label="C MOG2+亮度补偿", color="tab:green", panel_color=(0, 200, 0)),
}


# ---------------------------------------------------------------- 三种方法
def postprocess(m: np.ndarray) -> np.ndarray:
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K_OPEN, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K_CLOSE, iterations=2)
    return m


def mask_naive(prev_gray: np.ndarray, gray: np.ndarray, thr: int = DIFF_THRESHOLD) -> np.ndarray:
    m = cv2.absdiff(prev_gray, gray)
    _, m = cv2.threshold(m, thr, 255, cv2.THRESH_BINARY)
    return postprocess(m)


def mog2_mask(mog2, frame: np.ndarray) -> np.ndarray:
    fg = mog2.apply(frame, learningRate=-1)
    m = np.where(fg == 255, 255, 0).astype(np.uint8)      # 127 = 阴影，直接丢弃
    return postprocess(m)


def illum_compensate(frame: np.ndarray, ref_mean: float) -> np.ndarray:
    """减去全局亮度偏移：等价于"先对亮度做归一化，再做背景减除"。"""
    delta = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()) - ref_mean
    return np.clip(frame.astype(np.float32) - delta, 0, 255).astype(np.uint8)


def blob_stats(mask: np.ndarray, min_area: int = MIN_AREA):
    """返回 (前景面积, 连通域个数, 最大连通域(面积, 外接框, 质心, 实心度))。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total, kept = 0.0, []
    for c in contours:
        a = cv2.contourArea(c)
        total += a
        if a >= min_area:
            x, y, bw, bh = cv2.boundingRect(c)
            M = cv2.moments(c)
            cx = M["m10"] / M["m00"] if M["m00"] else x + bw / 2
            cy = M["m01"] / M["m00"] if M["m00"] else y + bh / 2
            hull = cv2.convexHull(c)
            solidity = float(a / max(cv2.contourArea(hull), 1e-6))
            kept.append((a, (x, y, bw, bh), (cx, cy), solidity))
    if not kept:
        return float(total), 0, None
    kept.sort(key=lambda t: -t[0])
    return float(total), len(kept), kept[0]


def render_panel(frame: np.ndarray, mask: np.ndarray, blob, color) -> np.ndarray:
    vis = frame.copy()
    vis[mask == 255] = (0.35 * vis[mask == 255] + 0.65 * np.array(color)).astype(np.uint8)
    if blob is not None:
        area, (x, y, bw, bh), (cx, cy), _sol = blob
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2, cv2.LINE_AA)
        put_text(vis, f"area={area:.0f}", (x, max(14, y - 6)), 0.45, (0, 255, 0))
    return vis


def panelize(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="运动目标检测：帧差法 vs MOG2 背景减除")
    ap.add_argument("--video", default="example2.mp4")
    ap.add_argument("--max-frames", type=int, default=0, help="只处理前 N 帧（0=全部）")
    args = ap.parse_args()

    ensure_dirs(FIG_DIR, VID_DIR, DATA_DIR)
    setup_matplotlib()

    video_path = resolve_asset(args.video)
    cap, fps, w, h, n_frames = open_video(video_path)
    n_use = min(n_frames, args.max_frames) if args.max_frames else n_frames
    print(f"输入视频: {video_path.name}  {w}×{h}  {fps:.2f} fps  {n_frames} 帧，处理 {n_use} 帧")

    gt = {}
    gt_path = find_gt(video_path)
    if gt_path.exists():
        with open(gt_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gt[int(row["frame"])] = (float(row["x"]), float(row["y"]))
        print(f"发现真值 {gt_path.name}：将额外统计命中率")

    mog2_b = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)
    mog2_c = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)

    rows, key_frames = [], {}
    prev_gray, ref_mean, idx = None, None, 0
    cost = {k: 0.0 for k in METHODS}

    out_video = VID_DIR / f"{video_path.stem}_motion_compare.mp4"
    with VideoWriter(out_video, fps, (CANVAS_W, CANVAS_H)) as vw:
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and idx >= args.max_frames):
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if ref_mean is None:
                ref_mean = float(gray.mean())

            t0 = time.perf_counter()
            m_a = np.zeros_like(gray) if prev_gray is None else mask_naive(prev_gray, gray)
            t1 = time.perf_counter()
            m_b = mog2_mask(mog2_b, frame)
            t2 = time.perf_counter()
            m_c = mog2_mask(mog2_c, illum_compensate(frame, ref_mean))
            t3 = time.perf_counter()
            cost["naive"] += t1 - t0
            cost["mog2"] += t2 - t1
            cost["comp"] += t3 - t2

            prev_gray = gray
            stat = {"naive": blob_stats(m_a), "mog2": blob_stats(m_b), "comp": blob_stats(m_c)}

            row = {"frame": idx}
            for k, (area, nblob, best) in stat.items():
                row[f"area_{k}"] = round(area, 1)
                row[f"blobs_{k}"] = nblob
                row[f"fill_{k}"] = round(best[0] / max(1, best[1][2] * best[1][3]), 3) if best else 0.0
                row[f"solidity_{k}"] = round(best[3], 3) if best else 0.0
                row[f"cx_{k}"] = round(best[2][0], 1) if best else ""
                row[f"cy_{k}"] = round(best[2][1], 1) if best else ""
                if idx in gt:
                    gx, gy = gt[idx]
                    row[f"hit_{k}"] = bool(best is not None and
                                           np.hypot(best[2][0] - gx, best[2][1] - gy) < 40)
            rows.append(row)

            # ---- 四分屏标注视频
            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
            canvas[:HEADER] = (30, 30, 30)
            put_text(canvas,
                     f"frame {idx:03d}  A(frame diff)={row['area_naive']:.0f}px2  "
                     f"B(MOG2)={row['area_mog2']:.0f}px2  C(MOG2+illum)={row['area_comp']:.0f}px2",
                     (12, 27), 0.55, (0, 255, 255))

            panels = [
                (0, 0, frame, "original"),
                (0, 1, render_panel(frame, m_a, stat["naive"][2], METHODS["naive"]["panel_color"]),
                 "A. frame diff (NO background subtraction)"),
                (1, 0, render_panel(frame, m_b, stat["mog2"][2], METHODS["mog2"]["panel_color"]),
                 "B. MOG2 background subtraction"),
                (1, 1, render_panel(frame, m_c, stat["comp"][2], METHODS["comp"]["panel_color"]),
                 "C. MOG2 + illumination compensation"),
            ]
            for r_, c_, img_p, tag in panels:
                y0, x0 = HEADER + r_ * PANEL_H, c_ * PANEL_W
                canvas[y0:y0 + PANEL_H, x0:x0 + PANEL_W] = panelize(img_p)
                put_text(canvas, tag, (x0 + 8, y0 + PANEL_H - 8), 0.42, (230, 230, 230))
            cv2.line(canvas, (PANEL_W, HEADER), (PANEL_W, CANVAS_H), (90, 90, 90), 1)
            cv2.line(canvas, (0, HEADER + PANEL_H), (CANVAS_W, HEADER + PANEL_H), (90, 90, 90), 1)
            vw.write(canvas)

            for tag, at in (("early", 30), ("mid", n_use // 2), ("late", n_use - 30)):
                if idx == at:
                    key_frames[tag] = dict(frame=frame.copy(), a=m_a.copy(),
                                           b=m_b.copy(), c=m_c.copy(), idx=idx)
            idx += 1
    cap.release()

    # ---------------- 指标统计
    def agg(k: str) -> dict:
        areas = np.array([r[f"area_{k}"] for r in rows], float)
        blobs = np.array([r[f"blobs_{k}"] for r in rows], float)
        fill = np.array([r[f"fill_{k}"] for r in rows], float)
        sol = np.array([r[f"solidity_{k}"] for r in rows], float)
        cx = np.array([r[f"cx_{k}"] if r[f"cx_{k}"] != "" else np.nan for r in rows], float)
        cy = np.array([r[f"cy_{k}"] if r[f"cy_{k}"] != "" else np.nan for r in rows], float)
        jump = float(np.nanmean(np.hypot(np.diff(cx), np.diff(cy)))) if len(cx) > 2 else float("nan")
        res = {
            "mean_area_px2": round(float(areas.mean()), 1),
            "median_area_px2": round(float(np.median(areas)), 1),
            "mean_blobs": round(float(blobs.mean()), 2),
            "median_blobs": float(np.median(blobs)),
            "single_target_rate": round(float((blobs == 1).mean()), 3),
            "empty_rate": round(float((blobs == 0).mean()), 3),
            "mean_fill_ratio": round(float(fill[fill > 0].mean()) if (fill > 0).any() else 0.0, 3),
            "mean_solidity": round(float(sol[sol > 0].mean()) if (sol > 0).any() else 0.0, 3),
            "centroid_jitter_px": round(jump, 2) if jump == jump else None,
            "fullscreen_false_frames": int((areas > BIG_AREA_RATIO * w * h).sum()),
            "ms_per_frame": round(cost[k] / max(1, len(rows)) * 1000, 2),
        }
        if gt:
            res["hit_rate"] = round(float(np.mean([r.get(f"hit_{k}", False) for r in rows])), 3)
        return res

    summary = {k: agg(k) for k in METHODS}
    result = {
        "video": video_path.name, "size": [w, h], "fps": round(float(fps), 2),
        "processed_frames": len(rows),
        "params": {"diff_threshold": DIFF_THRESHOLD, "min_area_px2": MIN_AREA,
                   "big_area_ratio": BIG_AREA_RATIO,
                   "mog2": {"history": 300, "varThreshold": 16, "detectShadows": True}},
        "metrics": {METHODS[k]["label"]: v for k, v in summary.items()},
        "has_ground_truth": bool(gt),
    }
    dump_json(result, DATA_DIR / "task2_4_motion_stats.json")
    with open(DATA_DIR / "task2_4_motion_stats.csv", "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))

    # ---------------- 曲线图
    fr = [r["frame"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(13, 11.5), gridspec_kw={"height_ratios": [1.1, 1, 1]})

    for k in METHODS:
        axes[0].plot(fr, [r[f"area_{k}"] for r in rows], color=METHODS[k]["color"], lw=1.0,
                     label=METHODS[k]["label"])
    axes[0].set_ylabel("前景面积 / px²")
    axes[0].set_yscale("symlog", linthresh=2e3)
    axes[0].axhline(BIG_AREA_RATIO * w * h, color="red", ls="--", lw=1, label="全屏误检阈值(15% 画面)")
    axes[0].set_title("逐帧前景面积（同一段视频，三种方法）", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(alpha=0.3)

    for k in METHODS:
        axes[1].plot(fr, [r[f"blobs_{k}"] for r in rows], color=METHODS[k]["color"], lw=1.0,
                     label=METHODS[k]["label"])
    axes[1].set_ylabel("连通域个数")
    axes[1].set_xlabel("帧序号")
    axes[1].set_title("逐帧显著连通域个数（目标越集中越接近 1）", fontsize=11)
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(alpha=0.3)

    ax = axes[2]
    metric_keys = [("mean_blobs", "连通域均值"), ("single_target_rate", "单目标一致性"),
                   ("mean_fill_ratio", "轮廓填充度"), ("mean_solidity", "轮廓实心度"),
                   ("fullscreen_false_frames", "全屏误检帧数"), ("ms_per_frame", "单帧耗时")]
    xs = np.arange(len(metric_keys))
    width = 0.26
    maxima = [max(float(summary[kk][mk] or 0) for kk in METHODS) or 1.0 for mk, _ in metric_keys]
    for i, k in enumerate(METHODS):
        vals = [float(summary[k][mk] or 0) for mk, _ in metric_keys]
        norm = [v / m for v, m in zip(vals, maxima)]
        ax.bar(xs + (i - 1) * width, norm, width, label=METHODS[k]["label"],
               color=METHODS[k]["color"])
        for x_, v, nv in zip(xs, vals, norm):
            ax.text(x_ + (i - 1) * width, nv + 0.015, f"{v:.2f}" if v < 10 else f"{v:.0f}",
                    ha="center", fontsize=7, rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels([m[1] for m in metric_keys], fontsize=9)
    ax.set_ylabel("归一化到三种方法的最大值")
    ax.set_title("代理指标对比（柱上为原始数值）", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"task2-4 帧差法 vs MOG2 vs MOG2+亮度补偿 —— {video_path.name}", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig_task2_4_motion_stats.png")

    # ---------------- 关键帧对比图
    if key_frames:
        tags = list(key_frames)
        fig, axes = plt.subplots(4, len(tags), figsize=(4.2 * len(tags), 11.5))
        if len(tags) == 1:
            axes = axes.reshape(-1, 1)
        for j, tag in enumerate(tags):
            d = key_frames[tag]
            show_panel(axes[0, j], d["frame"], f"frame {d['idx']} 原始帧")
            for i, (key, name) in enumerate((("a", "A 帧差"), ("b", "B MOG2"), ("c", "C MOG2+亮度补偿")), start=1):
                area = blob_stats(d[key])[0]
                show_panel(axes[i, j], d[key], f"{name} 面积={area:.0f}px²")
        fig.suptitle("task2-4 关键帧掩码对比", fontsize=13)
        fig.tight_layout()
        save_fig(fig, "fig_task2_4_frames.png")


if __name__ == "__main__":
    main()
