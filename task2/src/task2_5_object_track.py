# -*- coding: utf-8 -*-
"""task2-5：用 OpenCV 的 tracker 追踪 example3.mp4 / example4.mp4 里的小球，
保存追踪结果视频，并用实验说明"如何提高追踪准确度"。

**球是怎么被找到的（不需要人工画框）**：
    用 COCO 预训练检测器（torchvision `fasterrcnn_mobilenet_v3_large_fpn`）里的
    `sports ball` 类直接检出球 → 取第 0 帧（或前若干帧）里分数最高的框作为初始框。
    这一步既解决了"自动初始化"，又顺带提供了逐帧的"参考轨迹"（见下）。

三条流水线（同一段视频、同一个初始框，一次遍历同时跑完）：

    1. CSRT 原始用法：init 后逐帧 tracker.update(frame)
    2. KCF  原始用法：同上（更快，但对快速运动/遮挡/相机运动脆弱）
    3. 改进流水线 = CSRT + 检测器重检测 + 外观模型 + 匀速预测
         · 每 `--detect-step` 帧用 **COCO 检测器**给一次"绝对位置"（语义级、可靠）；
         · 中间帧用**从初始框现场学习的 HSV 外观模型**做轻量检测（便宜）；
         · 与追踪框偏差 > 25px 时用检测结果重新 init tracker（纠漂移）；
         · 都检测不到时用匀速模型外推（抗遮挡/漏检）；
         · 检测与追踪同时可用时 0.5/0.5 融合，抑制框抖动。

评价（相对**逐帧 COCO 参考轨迹**，参考覆盖不到 100% 会注明）：
    平均/最大/95 分位中心误差、精度(<20px)、成功率(IoU>0.5)、原始 tracker 的丢失率。

输出：
    outputs/videos/{example3,example4}_tracking.mp4
    outputs/figures/fig_task2_5_tracking_{example3,example4}.png
    outputs/data/task2_5_tracking_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import pickle
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import (
    DATA_DIR, FIG_DIR, VID_DIR, VideoWriter, bgr2rgb, dump_json, ensure_dirs, find_gt,
    open_video, put_text, resolve_asset, save_fig, setup_matplotlib,
)
from ball_detector import CocoBallDetector, available as coco_available

REDETECT_DIST = 25          # 追踪与检测偏差超过该值 → 重检测纠正
TRAIL = 60                  # 轨迹拖尾长度
OUT_MAX_W = 1280            # 输出视频最大宽度
CACHE_DIR = DATA_DIR / "cache"
COLOR_REF = (80, 220, 60)
COLOR_RAW = (255, 160, 40)
COLOR_HYB = (0, 235, 255)
COLOR_DET = (255, 0, 255)


# ---------------------------------------------------------------- 基础工具
def make_tracker(kind: str = "CSRT"):
    for module in (cv2, getattr(cv2, "legacy", None)):
        if module is None:
            continue
        fn = getattr(module, f"Tracker{kind}_create", None)
        if fn is not None:
            return fn()
    raise RuntimeError(f"当前 OpenCV 不支持 {kind} tracker")


def box_from_center(c, r):
    return (int(c[0] - r), int(c[1] - r), int(2 * r), int(2 * r))


def center_of_box(box):
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------- 外观模型（轻量检测，中间帧用）
class AppearanceModel:
    """从初始框内像素学 HSV 模型，用于两次 COCO 检测之间的快速复核。"""

    def __init__(self, frame: np.ndarray, center, radius: float):
        x0 = int(max(0, center[0] - radius * 0.6))
        x1 = int(min(frame.shape[1], center[0] + radius * 0.6))
        y0 = int(max(0, center[1] - radius * 0.6))
        y1 = int(min(frame.shape[0], center[1] + radius * 0.6))
        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).reshape(-1, 3)
        h, s, v = hsv[:, 0].astype(np.float32), hsv[:, 1].astype(np.float32), hsv[:, 2].astype(np.float32)
        rad = h / 180.0 * 2 * np.pi
        ang = np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())
        hue_c = float((ang / (2 * np.pi) * 180) % 180)
        d = np.abs((h - hue_c + 90) % 180 - 90)
        hue_half = float(np.percentile(d, 90)) + 10
        self.hue_lo, self.hue_hi = float((hue_c - hue_half) % 180), float((hue_c + hue_half) % 180)
        self.wraps = self.hue_lo > self.hue_hi
        self.s_lo = max(30, int(np.percentile(s, 5)) - 30)
        self.v_lo = max(30, int(np.percentile(v, 5)) - 35)
        self.area_ref = float(np.pi * radius * radius)
        self.r_ref = float(radius)
        # 判别力自检：颜色范围太宽（色相跨 > 50° 或饱和度下限很低）说明这个模型的
        # 颜色假设不可分，直接禁用，避免在复杂场景里把背景当球（example4 就是这个情况）。
        hue_width = hue_half * 2
        self.weak = (hue_width > 50) or (self.s_lo < 60) or (self.v_lo < 70)

    def detect(self, frame: np.ndarray, hint=None):
        if self.weak:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if not self.wraps:
            m = cv2.inRange(hsv, (int(self.hue_lo), self.s_lo, self.v_lo), (int(self.hue_hi), 255, 255))
        else:
            m = cv2.inRange(hsv, (0, self.s_lo, self.v_lo), (int(self.hue_hi), 255, 255)) | \
                cv2.inRange(hsv, (int(self.hue_lo), self.s_lo, self.v_lo), (180, 255, 255))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        a_lo, a_hi = 0.3 * self.area_ref, 2.5 * self.area_ref
        cands = []
        for c in cnts:
            a = cv2.contourArea(c)
            if not (a_lo <= a <= a_hi):
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            fill = a / (np.pi * r * r + 1e-6)
            if fill < 0.4 or r < 4:
                continue
            cands.append((float(cx), float(cy), float(r), fill))
        if not cands:
            return None
        if hint is None:
            return max(cands, key=lambda t: t[3])[:3]
        return min(cands, key=lambda t: (t[0] - hint[0]) ** 2 + (t[1] - hint[1]) ** 2)[:3]


# ---------------------------------------------------------------- COCO 参考轨迹（逐帧缓存）
def build_reference(video: Path, detector: CocoBallDetector, step: int = 1,
                    use_cache: bool = True) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{video.stem}_coco_ref_step{step}.pkl"
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            ref = pickle.load(f)
        print(f"[coco] 载入参考轨迹缓存 {cache.name}（{len(ref)} 帧有检测）")
        return ref

    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref, idx = {}, 0
    t0 = time.time()

    # 单目标"跟踪式"检测：尺寸一致性 + 匀速预测 + 距离闸门 + 连续帧确认，
    # 这样绿圈只会出现在"那一刻确实就是同一个球"的位置上，背景误检直接丢弃。
    pos, vel, r_ref = None, np.zeros(2), None
    miss = 0                    # 距离上次接受检测的帧数
    pending = None              # 丢失后等待确认的候选（连续 2 帧一致才接受）
    n_reject = 0                # 被丢弃的候选次数（诊断用）

    def _cost(c, pred, r):
        return float(np.hypot(c[0] - pred[0], c[1] - pred[1])) / (2.5 * r) + (1.0 - c[3])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            cands = detector._infer(frame)
            accepted = None
            n_plausible = 0
            if cands:
                if r_ref is None:                       # 用第一个最可信的球确定尺度
                    best = max(cands, key=lambda c: c[3])
                    r_ref = best[2]
                    accepted = best
                else:
                    # ① 尺寸一致性（球在整段视频里半径基本不变）
                    kept = [c for c in cands if abs(c[2] - r_ref) <= 0.45 * r_ref + 8]
                    n_plausible = len(kept)
                    if kept:
                        pred = (pos[0] + vel[0], pos[1] + vel[1]) if pos is not None else (kept[0][0], kept[0][1])
                        # ② 运动闸门：离预测位置太远的不信；跟丢太久（miss 大）则进入
                        #    "全局重捕获"模式：不再限制距离，但仍要求尺寸一致 + 连续两帧确认
                        if miss <= 12:
                            gate = r_ref * min(2.0 + 0.4 * miss, 6.0)
                            near = [c for c in kept
                                    if float(np.hypot(c[0] - pred[0], c[1] - pred[1])) <= gate]
                        else:
                            near = kept
                        if near:
                            best = min(near, key=lambda c: _cost(c, pred, r_ref))
                            if miss == 0:
                                accepted = best
                            else:
                                # ③ 连续确认：跟丢后要两帧都指向同一处才认（挡掉单帧误检）
                                if pending is not None and float(np.hypot(best[0] - pending[0],
                                                                          best[1] - pending[1])) <= r_ref:
                                    accepted = best
                                else:
                                    pending = best
                        else:
                            n_reject += 1
                    else:
                        n_reject += 1
            if accepted is not None:
                prev_pos = pos
                pos = (accepted[0], accepted[1])
                if prev_pos is not None:
                    d = np.array([pos[0] - prev_pos[0], pos[1] - prev_pos[1]])
                    vmax = 1.5 * r_ref
                    vel = 0.6 * vel + 0.4 * np.clip(d, -vmax, vmax)
                    vel = np.clip(vel, -vmax, vmax)
                miss, pending = 0, None
                ref[idx] = (round(accepted[0], 1), round(accepted[1], 1),
                            round(accepted[2], 1), max(1, n_plausible))
            else:
                miss += 1
                if pos is not None:
                    # 这一帧没有可信检测：速度衰减，避免"预测越飘越远"把后面的正确检测挡在闸门外
                    vel = np.clip(vel * 0.85, -1.5 * r_ref, 1.5 * r_ref)
                    pos = (pos[0] + vel[0], pos[1] + vel[1])
        idx += 1
    cap.release()
    with open(cache, "wb") as f:
        pickle.dump(ref, f)
    print(f"[coco] 参考轨迹：{len(ref)}/{idx} 帧（丢弃候选 {n_reject} 次），"
          f"用时 {time.time() - t0:.1f}s，已缓存 {cache.name}")
    return ref


# ---------------------------------------------------------------- 三条流水线（一次遍历）
def run_tracking(frame_iter, init_box, model: AppearanceModel, start_idx: int, out_video: Path,
                 fps: float, size, title: str, ref: dict, scale: float, detector,
                 detect_step: int):
    csrt = make_tracker("CSRT")
    kcf = make_tracker("KCF")
    hyb_trk = make_tracker("CSRT")

    first = next(frame_iter)
    csrt.init(first, init_box)
    kcf.init(first, init_box)
    hyb_trk.init(first, init_box)

    hyb_center = center_of_box(init_box)
    hyb_box = init_box
    vel = np.zeros(2)
    anchor_center = hyb_center          # 最近一次检测得到的绝对位置
    anchor_tracker = hyb_center         # 该检测时刻追踪器自己的位置（锚点法用）
    since_det = 0                       # 距离上次成功检测的帧数
    trail_raw, trail_hyb, trail_ref = deque(maxlen=TRAIL), deque(maxlen=TRAIL), deque(maxlen=TRAIL)
    records, det_records = {}, {}
    stats = {"reinit": 0, "predict": 0, "coco": 0, "appearance": 0, "csrt_fail": 0, "kcf_fail": 0}
    out_w, out_h = int(size[0] * scale), int(size[1] * scale)

    def handle(frame, idx, is_first=False):
        nonlocal csrt, kcf, hyb_trk, hyb_center, hyb_box, vel, anchor_center, anchor_tracker, since_det

        if is_first:
            ok_c, box_c, ok_k, box_k = True, init_box, True, init_box
        else:
            ok_c, box_c = csrt.update(frame)
            box_c = tuple(int(v) for v in box_c)
            ok_k, box_k = kcf.update(frame)
            box_k = tuple(int(v) for v in box_k)
            if not ok_c:
                stats["csrt_fail"] += 1
            if not ok_k:
                stats["kcf_fail"] += 1
        c_csrt = center_of_box(box_c) if ok_c else None
        c_kcf = center_of_box(box_k) if ok_k else None

        # ---- 改进流水线（检测为主、追踪器为辅、匀速预测兜底，全部带硬闸门防跑飞）
        prev = hyb_center
        r = model.r_ref
        pred = (hyb_center[0] + vel[0], hyb_center[1] + vel[1])
        # 检测距离闸门：正常情况容许 2 个半径的偏差；长时间没检测到（被遮挡/跟丢）逐步放宽，
        # 上限 4 个半径，保证还能重新捕获目标
        gate = r * min(2.0 + 0.5 * since_det, 4.0)

        det, det_src = None, ""
        if not is_first and detector is not None and (idx - start_idx) % detect_step == 0:
            det = detector.detect_filtered(frame, hint=pred, r_ref=r, size_tol=0.45, max_dist=gate)
            det_src = "coco"
        if det is None and model.detect is not None and not model.weak:
            det = model.detect(frame, hint=pred)
            det_src = "appearance" if det is not None else ""
            if det is not None and float(np.hypot(det[0] - pred[0], det[1] - pred[1])) > gate:
                det, det_src = None, ""
        if det is not None:
            stats[det_src] = stats.get(det_src, 0) + 1
            since_det = 0
        else:
            since_det += 1
        det_records[idx] = det

        state = "init" if is_first else "track"
        if not is_first:
            ok_h, tbox = hyb_trk.update(frame)
            tbox = tuple(int(v) for v in tbox)
            if det is not None:
                # 1) 有检测：位置直接换成检测结果（检测器没有累积漂移）
                dcx, dcy, dr = det
                hyb_center = (dcx, dcy)
                hyb_box = box_from_center(hyb_center, dr)
                need = True
                if ok_h:
                    tc = center_of_box(tbox)
                    need = float(np.hypot(tc[0] - dcx, tc[1] - dcy)) > 0.5 * r
                if need:
                    hyb_trk = make_tracker("CSRT")
                    hyb_trk.init(frame, hyb_box)
                    stats["reinit"] += 1
                    state = f"reinit({det_src})"
                else:
                    state = f"det({det_src})"
                anchor_center = (dcx, dcy)
                anchor_tracker = center_of_box(tbox) if ok_h else (dcx, dcy)
            else:
                # 2) 没检测：刚落检不久 → 短时匀速预测最准；否则退回追踪器 / 锚点
                tc = center_of_box(tbox) if ok_h else None
                if since_det <= 2:
                    hyb_center = pred
                    hyb_box = box_from_center(hyb_center, r)
                    state = "predict(short)"
                elif tc is not None and float(np.hypot(tc[0] - pred[0], tc[1] - pred[1])) <= gate:
                    hyb_center = tc
                    hyb_box = tbox
                    state = "track(no det)"
                else:
                    # 3) 追踪器也不可用：锚点 + 追踪器位移（如果有），否则纯匀速预测
                    cand = None
                    if tc is not None:
                        cand = (anchor_center[0] + (tc[0] - anchor_tracker[0]),
                                anchor_center[1] + (tc[1] - anchor_tracker[1]))
                        if float(np.hypot(cand[0] - pred[0], cand[1] - pred[1])) > gate:
                            cand = None
                    hyb_center = cand if cand is not None else pred
                    hyb_box = box_from_center(hyb_center, r)
                    state = "anchor" if cand is not None else "predict"
                    stats["predict"] += 1

        # 速度估计 + 硬钳制（单帧位移不超过 1.5 个半径，彻底避免"预测飞走"的正反馈）
        delta = np.array([hyb_center[0] - prev[0], hyb_center[1] - prev[1]])
        vmax = 1.5 * r
        delta = np.clip(delta, -vmax, vmax)
        vel = 0.7 * vel + 0.3 * delta
        vel = np.clip(vel, -vmax, vmax)
        records[idx] = dict(csrt=(ok_c, c_csrt, box_c), kcf=(ok_k, c_kcf, box_k),
                            hyb=(hyb_center, hyb_box, state))

        # ---- 绘制
        small = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((out_h + 46, out_w, 3), np.uint8)
        canvas[46:] = small
        canvas[:46] = (28, 28, 28)
        k = scale

        if idx in ref:
            gx, gy, gr = ref[idx][0] * k, ref[idx][1] * k + 46, ref[idx][2] * k
            gr_i = max(6, int(gr))
            # 绿色参考画得更醒目：半透明填充 + 粗描边 + 圆心十字 + 标注
            overlay = canvas.copy()
            cv2.circle(overlay, (int(gx), int(gy)), gr_i, COLOR_REF, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)
            ring = max(3, int(round(gr_i * 0.09)))
            cv2.circle(canvas, (int(gx), int(gy)), gr_i, COLOR_REF, ring, cv2.LINE_AA)
            cv2.drawMarker(canvas, (int(gx), int(gy)), COLOR_REF, cv2.MARKER_TILTED_CROSS,
                           max(12, gr_i), max(2, ring), cv2.LINE_AA)
            put_text(canvas, f"REF r={ref[idx][2]:.0f}", (int(gx - gr_i), int(gy - gr_i - 8)),
                     0.48, COLOR_REF)
            trail_ref.append((int(gx), int(gy)))
        if det is not None:
            dx, dy, _ = det
            cv2.drawMarker(canvas, (int(dx * k), int(dy * k + 46)), COLOR_DET,
                           cv2.MARKER_CROSS, 18, 3, cv2.LINE_AA)

        if ok_c:
            x, y, bw, bh = [int(v * k) for v in box_c]
            if bw > 2 and bh > 2:
                cv2.rectangle(canvas, (x, y + 46), (x + bw, y + bh + 46), COLOR_RAW,
                              max(2, int(round(bh * 0.05))), cv2.LINE_AA)
            trail_raw.append((int(c_csrt[0] * k), int(c_csrt[1] * k + 46)))

        x, y, bw, bh = [int(v * k) for v in hyb_box]
        cv2.rectangle(canvas, (x, y + 46), (x + bw, y + bh + 46), COLOR_HYB,
                      max(2, int(round(bh * 0.05))), cv2.LINE_AA)
        trail_hyb.append((int(hyb_center[0] * k), int(hyb_center[1] * k + 46)))

        for tr, col, th in ((trail_ref, COLOR_REF, max(3, int(round(3 * max(k, 0.6))))),
                            (trail_raw, COLOR_RAW, 2), (trail_hyb, COLOR_HYB, 3)):
            pts = list(tr)
            for a_, b_ in zip(pts, pts[1:]):
                cv2.line(canvas, a_, b_, col, th, cv2.LINE_AA)

        err_c = err_h = None
        if idx in ref:
            gx, gy = ref[idx][0], ref[idx][1]
            if c_csrt:
                err_c = float(np.hypot(c_csrt[0] - gx, c_csrt[1] - gy))
            err_h = float(np.hypot(hyb_center[0] - gx, hyb_center[1] - gy))
        txt = f"{title}  frame {idx:03d}  state={state}"
        if idx in ref:
            txt += f"  err: CSRT={'-' if err_c is None else f'{err_c:.0f}'}px  " \
                   f"hybrid={'-' if err_h is None else f'{err_h:.0f}'}px"
        put_text(canvas, txt, (10, 30), 0.5, (0, 255, 200))
        put_text(canvas, "GREEN=COCO reference  blue=CSRT  yellow=hybrid(CSRT+detector+predict)  magenta=current detection",
                 (10, out_h + 40), 0.42, (205, 205, 205))
        vw.write(canvas)

    with VideoWriter(out_video, fps, (out_w, out_h + 46)) as vw:
        handle(first, start_idx, is_first=True)
        idx = start_idx + 1
        for frame in frame_iter:
            handle(frame, idx)
            idx += 1
    return records, det_records, stats


# ---------------------------------------------------------------- 指标与图
def compute_metrics(tracks: dict, ref: dict, allow=None) -> dict:
    errs, ious = [], []
    for i, (c, box) in tracks.items():
        g = ref.get(i)
        if g is None or c is None:
            continue
        if allow is not None and i not in allow:
            continue
        gt_box = box_from_center((g[0], g[1]), g[2])
        errs.append(float(np.hypot(c[0] - g[0], c[1] - g[1])))
        valid = box is not None and box[2] > 0 and box[3] > 0
        ious.append(iou(box, gt_box) if valid else 0.0)
    errs, ious = np.array(errs), np.array(ious)
    if errs.size == 0:
        return {"n_eval_frames": 0}
    return {
        "mean_err_px": round(float(errs.mean()), 2),
        "median_err_px": round(float(np.median(errs)), 2),
        "max_err_px": round(float(errs.max()), 2),
        "p95_err_px": round(float(np.percentile(errs, 95)), 2),
        "precision_20px": round(float((errs < 20).mean()), 3),
        "success_iou50": round(float((ious > 0.5).mean()), 3),
        "n_eval_frames": int(errs.size),
    }


def make_figure(video: Path, ref: dict, records: dict, metrics: dict, first_frame,
                start_idx: int, out_name: str) -> None:
    idxs = sorted(records)
    err_raw, err_hyb = [], []
    for i in idxs:
        g = ref.get(i)
        c_raw = records[i]["csrt"][1]
        c_hyb = records[i]["hyb"][0]
        err_raw.append(np.hypot(c_raw[0] - g[0], c_raw[1] - g[1]) if (g and c_raw) else np.nan)
        err_hyb.append(np.hypot(c_hyb[0] - g[0], c_hyb[1] - g[1]) if g else np.nan)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.3, wspace=0.2)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(idxs, err_raw, color="tab:blue", lw=1.1, label="CSRT 原始")
    ax.plot(idxs, err_hyb, color="tab:orange", lw=1.5, label="CSRT+检测纠正+预测")
    ax.axhline(20, color="gray", ls="--", lw=1)
    ax.set_xlabel("帧序号")
    ax.set_ylabel("中心误差 / px")
    ax.set_title("逐帧中心误差（相对 COCO 检测器参考轨迹）", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(bgr2rgb(first_frame))
    if ref:
        ax2.plot([ref[i][0] for i in idxs if i in ref], [ref[i][1] for i in idxs if i in ref],
                 color="lime", lw=2.0, label="参考轨迹")
    xs_raw = [(i, records[i]["csrt"][1]) for i in idxs if records[i]["csrt"][1]]
    ax2.plot([p[1][0] for p in xs_raw], [p[1][1] for p in xs_raw], color="tab:blue", lw=1.0, label="CSRT")
    ax2.plot([records[i]["hyb"][0][0] for i in idxs], [records[i]["hyb"][0][1] for i in idxs],
             color="gold", lw=1.3, label="改进")
    ax2.set_title("轨迹对比（叠加在起始帧上）", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 1])
    names = ["CSRT", "KCF", "CSRT+检测纠正+预测"]
    keys = ["csrt", "kcf", "hybrid"]
    mean_err = [metrics[k].get("mean_err_px", 0) for k in keys]
    prec = [metrics[k].get("precision_20px", 0) * 100 for k in keys]
    xs = np.arange(3)
    b1 = ax3.bar(xs - 0.19, mean_err, 0.36, color="tab:blue", label="平均中心误差/px")
    ax3b = ax3.twinx()
    b2 = ax3b.bar(xs + 0.19, prec, 0.36, color="tab:green", alpha=0.75, label="精度(<20px)/%")
    ax3.set_xticks(xs)
    ax3.set_xticklabels(names, fontsize=9)
    ax3.set_ylabel("平均中心误差 / px")
    ax3b.set_ylabel("精度 / %")
    ax3b.set_ylim(0, 112)
    ax3.set_title("三种追踪方式指标", fontsize=11)
    for bars, axis in ((b1, ax3), (b2, ax3b)):
        for b in bars:
            axis.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.1f}",
                      ha="center", va="bottom", fontsize=8)
    ax3.grid(axis="y", alpha=0.3)
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    fig.suptitle(f"task2-5 小球追踪结果：{video.name}", fontsize=13)
    save_fig(fig, out_name)


# ---------------------------------------------------------------- 主流程
def load_gt(video: Path) -> dict:
    gt_path = find_gt(video)
    if not gt_path.exists():
        return {}
    out = {}
    with open(gt_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["frame"])] = {"x": float(row["x"]), "y": float(row["y"]),
                                      "r": float(row["radius"])}
    return out


def auto_init_from_coco(video: Path, detector: CocoBallDetector, scan: int = 60):
    """在第 0 帧找球；找不到就往后扫若干帧，返回 (frame_idx, center, r, score)。"""
    cap = cv2.VideoCapture(str(video))
    for idx in range(scan):
        ok, frame = cap.read()
        if not ok:
            break
        if idx % 3 == 0 or idx < 6:
            box = detector.detect_box(frame)
            if box is not None:
                x, y, w, h, sc = box
                cap.release()
                return idx, (x + w / 2.0, y + h / 2.0), max(4.0, (w + h) / 4.0), sc
    cap.release()
    return None


def process(video_name: str, cli_init, detect_step: int, ref_step: int, res: dict) -> None:
    video = resolve_asset(video_name)
    cap, fps, W, H, n = open_video(video)
    detector = CocoBallDetector(score_thr=0.05, max_side=1280) if coco_available() else None
    if detector is None:
        raise SystemExit("需要 torch + torchvision 才能用 COCO 检测器；也可以 --init 手动指定")

    gt = load_gt(video)
    start_idx = 0
    if cli_init is not None:
        cx, cy, r = cli_init
        source = "命令行 --init"
    elif gt:
        cx, cy, r = gt[0]["x"], gt[0]["y"], gt[0]["r"]
        source = "真值 csv"
    else:
        got = auto_init_from_coco(video, detector)
        if got is None:
            raise SystemExit(f"{video.name}: COCO 检测器前 60 帧没找到 sports ball，请用 --init 指定")
        start_idx, (cx, cy), r, sc = got
        source = f"COCO 检测器（sports ball, score={sc:.2f}, frame {start_idx}）"

    print(f"\n=== {video.name}: {W}×{H} {n} 帧 @{fps:.2f}fps")
    print(f"初始框 center=({cx:.1f},{cy:.1f}) r={r:.1f}  来源: {source}")

    # 参考轨迹（逐帧 COCO，缓存）
    ref = build_reference(video, detector, step=ref_step)

    if start_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    ok, first = cap.read()
    if not ok:
        raise SystemExit("起始帧读取失败")
    model = AppearanceModel(first, (cx, cy), r)
    print(f"外观模型: H∈[{model.hue_lo:.0f},{model.hue_hi:.0f}]{'(跨0/180)' if model.wraps else ''} "
          f"S≥{model.s_lo} V≥{model.v_lo}")

    def frame_iter():
        while True:
            ok_, f_ = cap.read()
            if not ok_:
                break
            yield f_

    scale = min(1.0, OUT_MAX_W / W)
    out_video = VID_DIR / f"{video.stem}_tracking.mp4"
    records, det_records, stats = run_tracking(
        frame_iter(), box_from_center((cx, cy), r), model, start_idx, out_video,
        fps, (W, H), video.stem, ref, scale, detector, detect_step)
    cap.release()

    tracks = {i: (rec["csrt"][1], rec["csrt"][2]) for i, rec in records.items()}
    tracks_kcf = {i: (rec["kcf"][1], rec["kcf"][2]) for i, rec in records.items()}
    tracks_hyb = {i: (rec["hyb"][0], rec["hyb"][1]) for i, rec in records.items()}

    # 歧义帧：画面里同时存在多个 "sports ball" 候选时，谁是"那个球"本身无法判定，
    # 评测时单独给一份"无歧义帧"的指标（example4 就是这种多球场景）。
    unamb = {i for i, v in ref.items() if len(v) < 4 or v[3] <= 1}
    metrics = {
        "csrt": compute_metrics(tracks, ref),
        "kcf": compute_metrics(tracks_kcf, ref),
        "hybrid": compute_metrics(tracks_hyb, ref),
        "csrt_unamb": compute_metrics(tracks, ref, allow=unamb),
        "kcf_unamb": compute_metrics(tracks_kcf, ref, allow=unamb),
        "hybrid_unamb": compute_metrics(tracks_hyb, ref, allow=unamb),
        "ambiguous_frames": len(ref) - len(unamb),
        "csrt_lost_rate": round(stats["csrt_fail"] / max(1, len(records)), 3),
        "kcf_lost_rate": round(stats["kcf_fail"] / max(1, len(records)), 3),
        "hybrid_events": {"reinit": stats["reinit"], "predict": stats["predict"],
                          "coco_detect": stats.get("coco", 0),
                          "appearance_detect": stats.get("appearance", 0)},
        "reference_coverage": round(len(ref) / max(1, len(records)), 3),
        "reference": "COCO sports-ball detector",
    }
    print(f"参考轨迹覆盖 {metrics['reference_coverage']:.1%}"
          f"（歧义帧 {metrics['ambiguous_frames']} 个）")
    print(f"CSRT  : {metrics['csrt']}")
    print(f"KCF   : {metrics['kcf']}")
    print(f"HYBRID: {metrics['hybrid']}  {metrics['hybrid_events']}")
    print(f"无歧义帧上 HYBRID: {metrics['hybrid_unamb']}")

    make_figure(video, ref, records, metrics, first, start_idx,
                f"fig_task2_5_tracking_{video.stem}.png")

    # 逐帧误差（只保留有参考的帧），方便定位问题帧
    per_frame = []
    for i in sorted(ref):
        c_csrt = records.get(i, {}).get("csrt", (False, None, None))[1]
        c_hyb = records.get(i, {}).get("hyb", ((np.nan, np.nan), None, ""))[0]
        g = ref[i]
        per_frame.append({
            "frame": i,
            "err_csrt": round(float(np.hypot(c_csrt[0] - g[0], c_csrt[1] - g[1])), 1) if c_csrt else None,
            "err_hybrid": round(float(np.hypot(c_hyb[0] - g[0], c_hyb[1] - g[1])), 1),
            "state": records.get(i, {}).get("hyb", (None, None, ""))[2],
        })

    res[video.stem] = {
        "video": video.name, "size": [W, H], "fps": round(float(fps), 2),
        "tracked_frames": len(records), "init_source": source,
        "init": {"frame": start_idx, "x": round(cx, 1), "y": round(cy, 1), "r": round(r, 1)},
        "detect_step": detect_step, "ref_step": ref_step,
        "metrics": metrics,
        "per_frame_error": per_frame,
        "output_video": str(out_video),
    }


def parse_init(text: str):
    if not text:
        return None
    parts = [float(v) for v in text.replace("，", ",").split(",")]
    if len(parts) != 3:
        raise SystemExit("--init 需要形如 900,700,45 的三元组 (x,y,r)")
    return tuple(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenCV tracker 小球追踪（COCO 检测器辅助）")
    ap.add_argument("--videos", nargs="*", default=["example3.mp4", "example4.mp4"])
    ap.add_argument("--init", default="", help="手动指定初始框 x,y,r")
    ap.add_argument("--detect-step", type=int, default=2, help="改进流水线每隔多少帧做一次 COCO 检测")
    ap.add_argument("--ref-step", type=int, default=1,
                    help="参考轨迹每隔多少帧做一次 COCO 检测（1=逐帧，绿色参考最密也最慢）")
    args = ap.parse_args()

    ensure_dirs(FIG_DIR, VID_DIR, DATA_DIR)
    setup_matplotlib()
    cli_init = parse_init(args.init)
    res = {}
    for name in args.videos:
        process(name, cli_init, args.detect_step, args.ref_step, res)
    dump_json(res, DATA_DIR / "task2_5_tracking_metrics.json")


if __name__ == "__main__":
    main()
