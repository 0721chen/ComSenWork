# -*- coding: utf-8 -*-
"""task-3 公共工具：目录约定、随机种子、matplotlib 中文字体、保存辅助。"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK3_DIR = Path(__file__).resolve().parents[1]     # .../task3
DATA_DIR = TASK3_DIR / "data"                       # MNIST 原始数据
OUT_DIR = TASK3_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
LOG_DIR = OUT_DIR / "logs"


def ensure_dirs() -> None:
    for d in (DATA_DIR, FIG_DIR, MODEL_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 130
    plt.rcParams["savefig.bbox"] = "tight"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


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
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[json  ] {path}")
    return path
