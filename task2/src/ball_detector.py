# -*- coding: utf-8 -*-
"""基于 COCO 预训练检测器的"小球检测器"（task-2 题五用）。

为什么用它：example3 / example4 里的"球"是**真正的球类物体**，
而 COCO 数据集里正好有 `sports ball` 这一类，用 torchvision 的现成预训练权重就能：
    · 自动找到球（不需要人工画框，也不需要靠颜色/形状瞎猜）；
    · 在追踪过程中提供"绝对位置"的重新检测（纠漂移、抗遮挡）。

实现要点：
    * 模型：`fasterrcnn_mobilenet_v3_large_fpn`（比 resnet50 版快 3~5 倍，够用）；
    * 推理前把长边缩到 `max_side`（默认 960），CPU 上单帧约 0.2~0.4 s；
    * 只保留 `sports ball` 类、且分数 ≥ `score_thr` 的框；
    * 多个候选时按"离预测位置最近"选择（避免误检到别的球）。

`available()` 用于在 torch/torchvision 缺失时优雅降级为颜色模型。
"""

from __future__ import annotations

import cv2
import numpy as np


class CocoBallDetector:
    def __init__(self, score_thr: float = 0.35, max_side: int = 960, model_name: str = "mobilenet"):
        import torch
        import torchvision
        from torchvision.models.detection import (
            fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
        )

        self.torch = torch
        torch.set_num_threads(max(1, torch.get_num_threads()))
        if model_name == "resnet50":
            self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights).eval()
        else:
            self.weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=self.weights).eval()
        self.cats = self.weights.meta["categories"]
        self.ball_ids = {i for i, c in enumerate(self.cats) if c == "sports ball"}
        self.score_thr = score_thr
        self.max_side = max_side
        self.calls = 0

    # ---------------------------------------------------------------- 内部
    def _infer(self, frame: np.ndarray):
        torch = self.torch
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            out = self.model([t])[0]
        self.calls += 1
        dets = []
        for box, score, lab in zip(out["boxes"], out["scores"], out["labels"]):
            if int(lab) not in self.ball_ids:
                continue
            if float(score) < self.score_thr:
                continue
            x1, y1, x2, y2 = [float(v) / scale for v in box.tolist()]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            r = (max(1.0, x2 - x1) + max(1.0, y2 - y1)) / 4.0
            dets.append((cx, cy, r, float(score), (x1, y1, x2 - x1, y2 - y1)))
        dets.sort(key=lambda t: -t[3])
        return dets

    # ---------------------------------------------------------------- 对外
    def detect(self, frame: np.ndarray, hint=None):
        """返回 (cx, cy, r) 或 None；hint=(x,y) 时在多个候选里选最近的。"""
        dets = self._infer(frame)
        if not dets:
            return None
        if hint is None:
            best = dets[0]
            return best[0], best[1], best[2]
        cx, cy = hint
        best = min(dets, key=lambda d: (d[0] - cx) ** 2 + (d[1] - cy) ** 2)
        return best[0], best[1], best[2]

    def detect_filtered(self, frame: np.ndarray, hint, r_ref: float,
                        size_tol: float = 0.45, max_dist: float | None = None):
        """带"尺寸一致性 + 距离闸门 + 分数"综合代价的检测（追踪时用）。

        * 尺寸：球的半径在整段视频里基本不变，偏离 r_ref 超过 size_tol 的候选直接丢弃
          （低阈值召回出的"别的圆东西"大多尺寸不对）；
        * 距离：离预测位置超过 max_dist 的丢弃；
        * 代价：`距离/(2.5·r) + (1 - score)`，越小越好 —— 兼顾"离预测近"和"检测置信度高"。
        """
        dets = self._infer(frame)
        if not dets:
            return None
        cands = [d for d in dets if abs(d[2] - r_ref) <= size_tol * r_ref + 6]
        if max_dist is not None:
            cands = [d for d in cands if float(np.hypot(d[0] - hint[0], d[1] - hint[1])) <= max_dist]
        if not cands:
            return None
        best = min(cands, key=lambda d: float(np.hypot(d[0] - hint[0], d[1] - hint[1])) / (2.5 * r_ref)
                   + (1.0 - d[3]))
        return best[0], best[1], best[2]

    def detect_box(self, frame: np.ndarray, hint=None):
        """返回 (x, y, w, h, score) 形式的框（用于初始化 tracker）。"""
        dets = self._infer(frame)
        if not dets:
            return None
        if hint is None:
            best = dets[0]
        else:
            cx, cy = hint
            best = min(dets, key=lambda d: (d[0] - cx) ** 2 + (d[1] - cy) ** 2)
        return (*best[4], best[3])


def available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except Exception:      # pragma: no cover
        return False


if __name__ == "__main__":      # 手动检查用
    import sys
    from pathlib import Path
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "example4.mp4")
    det = CocoBallDetector()
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{path.name}: {n} 帧")
    hits = 0
    for idx in range(0, n, max(1, n // 12)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            break
        d = det._infer(f)
        if d:
            hits += 1
        print(f"  frame {idx:4d}: " + (f"({d[0][0]:.0f},{d[0][1]:.0f}) r={d[0][2]:.0f} score={d[0][3]:.2f}" if d else "无"))
    cap.release()
    print(f"命中 {hits} 个采样帧")
