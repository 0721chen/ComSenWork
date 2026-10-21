# -*- coding: utf-8 -*-
"""task-4（YOLO 方向）公共工具：目录约定、字体、json/图片保存。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK4_DIR = Path(__file__).resolve().parents[1]         # .../task4_yolo
EXAMPLE_DIR = TASK4_DIR / "example"                     # 官方素材（3 张图 + 视频）
DATASET_DIR = TASK4_DIR / "dataset"                     # 无人机数据集（Roboflow 导出）
WEIGHTS_DIR = TASK4_DIR / "weights"                     # 预训练/训练得到的权重
OUT_DIR = TASK4_DIR / "outputs"
IMG_DIR = OUT_DIR / "images"
FIG_DIR = OUT_DIR / "figures"
RUNS_DIR = OUT_DIR / "runs"
METRIC_DIR = OUT_DIR / "metrics"


def ensure_dirs() -> None:
    for d in (DATASET_DIR, WEIGHTS_DIR, IMG_DIR, FIG_DIR, RUNS_DIR, METRIC_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 130
    plt.rcParams["savefig.bbox"] = "tight"


def save_fig(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"[figure] {path}")
    return path


def dump_json(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    print(f"[json  ] {path}")
    return path


def load_yolo(weights: str = "yolov8n.pt"):
    """加载 YOLO 模型；权重不存在时 ultralytics 会自动下载到 weights/ 目录。"""
    from ultralytics import YOLO

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    name = Path(weights).name
    local = WEIGHTS_DIR / name
    if Path(weights).exists():
        return YOLO(str(weights))
    if local.exists():
        return YOLO(str(local))
    model = YOLO(name)                       # 触发下载
    src = Path(name)
    if src.exists():
        src.replace(local)                   # 把自动下载的权重挪进 weights/
    return YOLO(str(local)) if local.exists() else model
