# -*- coding: utf-8 -*-
"""task4-题1：学习 Ultralytics YOLO 的 API 接口（并留下可运行的示例）。

本脚本把"用 YOLO 做检测"需要的接口都跑一遍并打印出来，作为学习记录：
    1. 加载模型            YOLO("yolov8n.pt")            —— 权重、任务类型、类别名、参数量/GFLOPs
    2. 推理（预测）         model.predict(...) / model(source) —— conf / iou / imgsz / classes / device / max_det
    3. 结果结构            results[0].boxes.xyxy / .conf / .cls、results[0].plot()、results[0].names
    4. 训练                model.train(data=..., epochs=..., imgsz=..., batch=..., device=..., project=...)
    5. 验证                model.val(data=..., split="val") → mAP50、mAP50-95、precision、recall
    6. 导出                model.export(format="onnx" | "torchscript" | "engine", half=True, int8=True)
    7. 其它                model.names、model.task、model.model.info()

输出：
    outputs/figures/task4_1_api_cheatsheet.png   API 速查表
    outputs/metrics/task4_1_model_info.json      模型信息
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common4 import EXAMPLE_DIR, METRIC_DIR, dump_json, ensure_dirs, load_yolo, save_fig, setup_matplotlib

API_TABLE = [
    ("加载模型", 'YOLO("yolov8n.pt")', "按权重文件名推断任务（detect/segment/classify/pose）"),
    ("推理", 'model.predict(source, conf=0.25, iou=0.45, imgsz=640, classes=[...], device="cpu")',
     "source 可以是路径/目录/numpy/PIL/URL；返回 list[Results]"),
    ("快捷推理", 'model(source, conf=0.25, save=True, project=..., name=...)',
     "等价于 predict，save=True 直接把标注图写到 project/name"),
    ("结果读取", "r = results[0]; r.boxes.xyxy / r.boxes.conf / r.boxes.cls",
     "xyxy 是左上右下坐标，conf 是置信度，cls 是类别索引"),
    ("可视化", "r.plot()", "返回画好框的 BGR 图像（可直接 cv2.imwrite）"),
    ("类别名", "r.names", "dict{index: name}，COCO 为 80 类"),
    ("训练", 'model.train(data="data.yaml", epochs=50, imgsz=640, batch=16, device="0")',
     "project/name 决定输出目录；会生成 results.png、PR 曲线、混淆矩阵"),
    ("验证", 'model.val(data="data.yaml", split="val")',
     "返回 mAP50 / mAP50-95 / precision / recall / per-class 指标"),
    ("导出", 'model.export(format="onnx", half=True, int8=False, imgsz=640)',
     "支持 onnx / torchscript / openvino / engine(TensorRT) / tflite 等"),
    ("断点续训", 'YOLO("runs/detect/train/weights/last.pt").train(resume=True)', "从上次的 last.pt 继续训练"),
    ("模型信息", "model.model.info()", "打印各层参数量与 GFLOPs"),
]


def main() -> None:
    ensure_dirs()
    setup_matplotlib()

    print("=" * 78)
    print("题1：Ultralytics YOLO API 学习与演示")
    print("=" * 78)

    # ---- 1. 加载模型
    model = load_yolo("yolov8n.pt")
    print(f"\n[1] 加载模型: yolov8n.pt   task={getattr(model, 'task', '?')}   类别数={len(model.names)}")
    print("    前 10 个类别:", {i: model.names[i] for i in list(model.names)[:10]})
    try:
        from ultralytics.utils.torch_utils import model_info
        info = model_info(model.model, verbose=False)
        n_param = int(info[0])
        gflops = float(info[3]) if len(info) > 3 else None
    except Exception:
        n_param, gflops = sum(p.numel() for p in model.model.parameters()), None
    print(f"    参数量: {n_param/1e6:.2f} M   GFLOPs: {gflops}")

    # ---- 2/3. 推理 + 结果结构（用一张官方图片演示，不落盘）
    img_path = next(iter(sorted(EXAMPLE_DIR.glob("*.jpg"))), None)
    demo = {}
    if img_path is not None:
        results = model.predict(source=str(img_path), conf=0.25, iou=0.45, imgsz=640, verbose=False)
        r = results[0]
        boxes = r.boxes
        print(f"\n[2] 推理 {img_path.name}: 检出 {len(boxes)} 个目标")
        rows = []
        for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
            rows.append({"class": r.names[int(cls)], "conf": round(conf, 3),
                         "xyxy": [round(v, 1) for v in xyxy]})
            print(f"    {r.names[int(cls)]:<12s} conf={conf:.3f}  xyxy={[round(v,1) for v in xyxy]}")
        demo = {"image": img_path.name, "n_objects": int(len(boxes)), "detections": rows,
                "boxes_attrs": ["xyxy", "xywh", "conf", "cls"],
                "speed_ms": {k: round(float(v), 1) for k, v in getattr(r, "speed", {}).items()}}

    # ---- 4/5/6 只打印调用示例（真正执行分别在各题脚本里）
    print("\n[4] 训练接口示例:")
    print('    model.train(data="dataset/data.yaml", epochs=50, imgsz=640, batch=16, '
          'device="0", project="outputs/runs", name="drone")')
    print("[5] 验证接口示例: model.val(data=\"dataset/data.yaml\", split=\"val\")")
    print('    → 返回 mAP50 / mAP50-95 / precision / recall')
    print("[6] 导出接口示例: model.export(format=\"onnx\", imgsz=640, half=False, int8=False)")

    # ---- 7. 速查表图片
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.axis("off")
    cell = [[a, b, c] for a, b, c in API_TABLE]
    tbl = ax.table(cellText=cell, colLabels=["能力", "调用方式", "说明"], loc="center",
                   cellLoc="left", colWidths=[0.13, 0.5, 0.37])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.55)
    for (row, col), cell_obj in tbl.get_celld().items():
        if row == 0:
            cell_obj.set_facecolor("#1f4e79")
            cell_obj.set_text_props(color="white", weight="bold")
        elif col == 1:
            cell_obj.set_text_props(family="monospace")
    ax.set_title("task4-题1  Ultralytics YOLO API 速查表", fontsize=14, pad=14)
    save_fig(fig, "task4_1_api_cheatsheet.png")

    dump_json({"model": "yolov8n.pt", "params_M": round(n_param / 1e6, 3), "gflops": gflops,
               "n_classes": len(model.names), "task": getattr(model, "task", None),
               "demo_inference": demo,
               "api_notes": [{"capability": a, "call": b, "note": c} for a, b, c in API_TABLE]},
              METRIC_DIR / "task4_1_model_info.json")


if __name__ == "__main__":
    main()
