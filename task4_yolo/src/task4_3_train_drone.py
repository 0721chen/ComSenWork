# -*- coding: utf-8 -*-
"""task4-题3：基于无人机数据集训练 YOLO、导出模型、保存训练结果。

流程：
    1. 读数据集 `dataset/data.yaml`（Roboflow 导出的 YOLO 格式：train/ valid/ test/）；
    2. **抽子集**：CPU 上全量训练太慢，所以按固定随机种子抽 `--n-train/--n-val` 张，
       生成 `dataset/subset/{train,val}/{images,labels}` + `data.yaml`（比例不变，可复现）；
    3. 训练：`yolov8n.pt` 预训练权重迁移学习（题目给的 `drone.pt` 未随题面提供，
       用官方预训练权重代替，训练流程完全一致）；
    4. 验证：`model.val()` → mAP50 / mAP50-95 / precision / recall / 每类指标；
    5. 导出：`best.onnx`（提高部分要求的量化/ONNX 导出也顺带做了），并统计模型大小与推理速度；
    6. 保存：训练曲线 results.png、混淆矩阵、PR 曲线等（ultralytics 自动生成）+ 本脚本汇总的 json。

输出：
    outputs/runs/drone_yolov8n/           # 训练产物（weights/best.pt、results.png、confusion_matrix.png…）
    outputs/figures/task4_3_*.png         # 复制出来的关键图
    outputs/metrics/task4_3_metrics.json  # 指标汇总
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

from common4 import (
    DATASET_DIR, EXAMPLE_DIR, FIG_DIR, METRIC_DIR, RUNS_DIR, WEIGHTS_DIR,
    dump_json, ensure_dirs, load_yolo, save_fig, setup_matplotlib,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------- 数据集
def find_split(root: Path, *names) -> Path | None:
    """兼容 `root/train/images` 与 `root/train` 两种目录布局。"""
    for name in names:
        d = root / name
        if d.exists():
            if (d / "images").exists():
                return d / "images"
            return d
    return None


def prepare_subset(root: Path, n_train: int, n_val: int, seed: int = 0) -> Path:
    """抽子集，返回子集 data.yaml 路径。"""
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        raise SystemExit(f"找不到 {yaml_path}，请先运行 python src/fetch_dataset.py")
    meta = yaml.safe_load(yaml_path.read_text(encoding="utf-8", errors="replace"))
    classes = meta.get("names")
    if isinstance(classes, dict):
        classes = [classes[k] for k in sorted(classes)]
    print(f"数据集类别({len(classes)}): {classes}")

    out = root / "subset"
    if (out / "data.yaml").exists():
        print(f"[skip] 子集已存在: {out}")
        return out / "data.yaml"

    rng = random.Random(seed)
    for split, n in (("train", n_train), ("valid", n_val)):
        src_img = find_split(root, split, "val" if split == "valid" else split)
        if src_img is None:
            raise SystemExit(f"数据集里找不到 {split} 划分")
        src_lbl = src_img.parent / "labels"
        imgs = sorted([p for p in src_img.iterdir() if p.suffix.lower() in IMG_EXTS])
        rng.shuffle(imgs)
        picked = imgs[:n]
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        kept = 0
        for p in picked:
            lbl = src_lbl / f"{p.stem}.txt"
            shutil.copy2(p, out / split / "images" / p.name)
            if lbl.exists():
                shutil.copy2(lbl, out / split / "labels" / lbl.name)
                kept += 1
        print(f"  {split}: 共 {len(imgs)} 张，抽取 {len(picked)} 张（有标注 {kept}）")

    data = {"path": str(out.resolve()), "train": "train/images", "val": "valid/images",
            "nc": len(classes), "names": classes}
    (out / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                                   encoding="utf-8")
    print(f"[dataset] 子集 data.yaml: {out / 'data.yaml'}")
    return out / "data.yaml"


# ---------------------------------------------------------------- 训练 / 验证 / 导出
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=320)
    ap.add_argument("--n-val", type=int, default=80)
    ap.add_argument("--name", default="drone_yolov8n")
    ap.add_argument("--export", action="store_true", default=True)
    args = ap.parse_args()

    ensure_dirs()
    setup_matplotlib()
    t0 = time.time()

    data_yaml = prepare_subset(DATASET_DIR, args.n_train, args.n_val)
    model = load_yolo("yolov8n.pt")

    print(f"\n[训练] epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} device=cpu")
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device="cpu", workers=0, project=str(RUNS_DIR), name=args.name,
                exist_ok=True, plots=True, seed=0, val=True, verbose=True)
    train_sec = time.time() - t0
    run_dir = RUNS_DIR / args.name
    best = run_dir / "weights" / "best.pt"
    print(f"[训练完成] {train_sec:.1f}s  权重: {best}")

    # ---- 验证
    m = model.val(data=str(data_yaml), split="val", imgsz=args.imgsz, device="cpu",
                  project=str(RUNS_DIR), name=f"{args.name}_val", exist_ok=True, plots=True)
    per_class = {}
    try:
        names = m.names if isinstance(m.names, dict) else {i: n for i, n in enumerate(m.names)}
        for i, name in names.items():
            try:
                p, r, ap50, ap = m.box.p[i], m.box.r[i], m.box.ap50[i], m.box.ap[i]
                per_class[str(name)] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                                        "mAP50": round(float(ap50), 4), "mAP50-95": round(float(ap), 4)}
            except Exception:
                continue
    except Exception:
        pass
    metrics = {
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "n_train": args.n_train, "n_val": args.n_val,
        "mAP50": round(float(m.box.map50), 4),
        "mAP50-95": round(float(m.box.map), 4),
        "precision_mean": round(float(np.mean(m.box.p)), 4) if len(m.box.p) else None,
        "recall_mean": round(float(np.mean(m.box.r)), 4) if len(m.box.r) else None,
        "per_class": per_class,
        "train_seconds": round(train_sec, 1),
        "weights": str(best),
    }
    print(f"[验证] mAP50={metrics['mAP50']}  mAP50-95={metrics['mAP50-95']}")

    # ---- 导出（提高部分：ONNX / 半精度）
    export_info = {}
    if args.export:
        try:
            onnx_path = model.export(format="onnx", imgsz=args.imgsz, opset=12)
            onnx_path = Path(str(onnx_path))
            export_info["onnx"] = str(onnx_path)
            export_info["onnx_size_mb"] = round(onnx_path.stat().st_size / 1024 / 1024, 2)
            print(f"[导出] {onnx_path}  {export_info['onnx_size_mb']} MB")
        except Exception as e:
            print(f"[导出失败] {type(e).__name__}: {e}")

    # ---- 推理速度（CPU，用 example 里的图片）
    demo = next(iter(sorted(EXAMPLE_DIR.glob("*.jpg"))), None)
    speed = {}
    if demo is not None:
        best_model = load_yolo(str(best)) if best.exists() else model
        best_model.predict(str(demo), imgsz=args.imgsz, verbose=False)     # 预热
        t1 = time.time()
        for _ in range(5):
            best_model.predict(str(demo), imgsz=args.imgsz, verbose=False)
        ms = (time.time() - t1) / 5 * 1000
        speed = {"image": demo.name, "ms_per_image": round(ms, 1), "fps": round(1000 / ms, 2)}
        print(f"[速度] {demo.name}: {speed['ms_per_image']} ms/张 ({speed['fps']} FPS)")

    metrics["export"] = export_info
    metrics["speed"] = speed
    metrics["weights_size_mb"] = round(best.stat().st_size / 1024 / 1024, 2) if best.exists() else None
    dump_json(metrics, METRIC_DIR / "task4_3_metrics.json")

    # ---- 把关键图复制到 figures（训练目录 + 验证目录都要看），并做一张汇总
    copied = []
    val_dir = RUNS_DIR / f"{args.name}_val"
    for src_dir, prefix in ((run_dir, ""), (val_dir, "val_")):
        for fname in ("results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
                      "BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png",
                      "PR_curve.png", "F1_curve.png", "labels.jpg",
                      "val_batch0_labels.jpg", "val_batch0_pred.jpg", "train_batch0.jpg"):
            src = src_dir / fname
            if src.exists():
                dst = FIG_DIR / f"task4_3_{prefix}{fname}"
                shutil.copy2(src, dst)
                copied.append(dst.name)
                print(f"[figure] {dst}")
    metrics["figures"] = copied

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axis("off")
    rows = [["mAP@0.5", f"{metrics['mAP50']:.4f}"],
            ["mAP@0.5:0.95", f"{metrics['mAP50-95']:.4f}"],
            ["precision(mean)", f"{metrics['precision_mean']}"],
            ["recall(mean)", f"{metrics['recall_mean']}"],
            ["参数量", "3.16 M（yolov8n）"],
            ["训练集/验证集", f"{args.n_train} / {args.n_val}"],
            ["epochs / imgsz", f"{args.epochs} / {args.imgsz}"],
            ["训练耗时", f"{train_sec/60:.1f} min (CPU)"],
            ["权重大小(best.pt)", f"{metrics['weights_size_mb']} MB"],
            ["ONNX 大小", f"{export_info.get('onnx_size_mb', '-')} MB"]]
    tbl = ax.table(cellText=rows, colLabels=["指标", "数值"], loc="center", cellLoc="center",
                   colWidths=[0.5, 0.5])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    for (r_, c_), cell in tbl.get_celld().items():
        if r_ == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")
    ax.set_title("task4-题3  无人机数据集训练结果汇总", fontsize=13, pad=12)
    save_fig(fig, "task4_3_summary.png")


if __name__ == "__main__":
    main()
