# -*- coding: utf-8 -*-
"""task4-题2：用 YOLO 官方预训练权重对 example 目录下的三张图片推理并保存。

做法：
    · 权重：`yolov8n.pt`（COCO 80 类，官方预训练）；
    · 推理：`model.predict(imgsz=640, conf=0.25, iou=0.45)`；
    · 保存：每张图的**标注结果图**（`results[0].plot()`）到 `outputs/images/pred_*.jpg`，
      以及检测结果明细 json（类别 / 置信度 / 左上右下坐标）；
    · 另做一张"原图 vs 推理结果"的对比图，方便写报告。

输出：
    outputs/images/pred_coco-bear.jpg、pred_coco-bus.jpg、pred_coco-person.jpg
    outputs/figures/task4_2_predictions.png
    outputs/metrics/task4_2_detections.json
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt

from common4 import (
    EXAMPLE_DIR, FIG_DIR, IMG_DIR, METRIC_DIR, dump_json, ensure_dirs,
    load_yolo, save_fig, setup_matplotlib,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ensure_dirs()
    setup_matplotlib()

    model = load_yolo("yolov8n.pt")
    images = sorted([p for p in EXAMPLE_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not images:
        raise SystemExit(f"{EXAMPLE_DIR} 里没有图片")
    print(f"待推理图片 {len(images)} 张：{[p.name for p in images]}")

    all_det = {}
    panels = []
    for p in images:
        results = model.predict(source=str(p), imgsz=640, conf=0.25, iou=0.45, verbose=False)
        r = results[0]
        annotated = r.plot()                     # BGR ndarray，已画好框+标签
        out_img = IMG_DIR / f"pred_{p.stem}{p.suffix}"
        cv2.imwrite(str(out_img), annotated)
        print(f"[image ] {out_img}")

        dets = []
        for xyxy, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            dets.append({"class": r.names[int(cls)], "conf": round(float(conf), 4),
                         "xyxy": [round(float(v), 1) for v in xyxy]})
        all_det[p.name] = {
            "n_objects": len(dets),
            "detections": dets,
            "speed_ms": {k: round(float(v), 1) for k, v in getattr(r, "speed", {}).items()},
        }
        print(f"  {p.name}: 检出 {len(dets)} 个目标 → " +
              ", ".join(f"{d['class']}({d['conf']:.2f})" for d in dets[:6]))
        panels.append((p, annotated, dets))

    # ---- 对比图：上排原图，下排推理结果
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 8.4))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for i, (p, annotated, dets) in enumerate(panels):
        orig = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(orig)
        axes[0, i].set_title(f"原图 {p.name}\n{orig.shape[1]}×{orig.shape[0]}", fontsize=10)
        axes[0, i].axis("off")
        axes[1, i].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        names = ", ".join(f"{d['class']} {d['conf']:.2f}" for d in dets[:4]) or "（无检测）"
        axes[1, i].set_title(f"yolov8n 推理结果：{len(dets)} 个目标\n{names}", fontsize=10)
        axes[1, i].axis("off")
    fig.suptitle("task4-题2  官方预训练权重对 example 三张图片的推理结果", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "task4_2_predictions.png")

    dump_json({"weights": "yolov8n.pt", "imgsz": 640, "conf": 0.25, "iou": 0.45,
               "images": all_det},
              METRIC_DIR / "task4_2_detections.json")


if __name__ == "__main__":
    main()
