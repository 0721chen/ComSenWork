# -*- coding: utf-8 -*-
"""task-2 公共工具模块。

包含：目录约定、matplotlib 中文字体设置、BGR/RGB 转换、带文字标签的绘图辅助、
以及统一的"保存图片/视频"接口，保证所有脚本都能从任意工作目录直接运行。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib

# Windows 控制台默认 GBK，统一改成 UTF-8，避免中文/特殊符号打印报错
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

matplotlib.use("Agg")  # 无界面环境下渲染到文件
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 目录约定
TASK2_DIR = Path(__file__).resolve().parents[1]          # .../task2
ASSETS_DIR = TASK2_DIR / "assets"                        # 输入素材
OUT_DIR = TASK2_DIR / "outputs"                          # 全部输出
FIG_DIR = OUT_DIR / "figures"                            # 结果图
VID_DIR = OUT_DIR / "videos"                             # 结果视频
IMG_DIR = OUT_DIR / "images"                             # 单张结果图
DATA_DIR = OUT_DIR / "data"                              # 数值结果 json/csv


def ensure_dirs(*dirs: Path) -> None:
    """创建输出目录（默认创建全部输出子目录）。"""
    for d in (*dirs,) or (FIG_DIR, VID_DIR, IMG_DIR, DATA_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


def setup_matplotlib() -> None:
    """配置 matplotlib：中文字体、负号显示、高 dpi。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"


# ---------------------------------------------------------------- 图像工具
def bgr2rgb(img: np.ndarray) -> np.ndarray:
    """OpenCV 默认 BGR 顺序，转成 matplotlib/PIL 使用的 RGB 顺序。"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def put_text(img: np.ndarray, text: str, org, scale=0.6, color=(0, 255, 0), thick=2) -> np.ndarray:
    """在图像上绘制带黑色描边的文字（中文会显示为问号，故一律用英文/数字）。"""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return img


def save_fig(fig, name: str, subdir: Path = FIG_DIR) -> Path:
    """保存 matplotlib 图像并关闭，避免内存堆积。"""
    subdir.mkdir(parents=True, exist_ok=True)
    path = Path(subdir) / name
    fig.savefig(path)
    plt.close(fig)
    print(f"[figure] {path}")
    return path


def show_panel(ax, img_bgr: np.ndarray, title: str, cmap=None) -> None:
    """在一个 subplot 上显示 OpenCV 图像。"""
    if img_bgr.ndim == 2:
        ax.imshow(img_bgr, cmap=cmap or "gray", vmin=0, vmax=255)
    else:
        ax.imshow(bgr2rgb(img_bgr))
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def imwrite(path: Path, img: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"保存图片失败: {path}")
    print(f"[image ] {path}")
    return path


def dump_json(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[json  ] {path}")
    return path


# ---------------------------------------------------------------- 视频工具
class VideoWriter:
    """cv2.VideoWriter 的上下文管理器封装（自动按源视频帧率/尺寸建 writer）。"""

    def __init__(self, path: Path, fps: float, size, fourcc: str = "mp4v"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc), float(fps), (int(size[0]), int(size[1]))
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"无法创建视频文件: {path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        if self.path.exists():
            print(f"[video  ] {self.path}  ({self.path.stat().st_size / 1024:.0f} KB)")
        return False


def open_video(path: Path):
    """打开视频，返回 (cap, fps, width, height, n_frames)。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, w, h, n


def read_first_frame(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"读取首帧失败: {path}")
    return frame


# ---------------------------------------------------------------- 素材定位
# 优先使用题目提供的官方素材（task2/example*.jpg|mp4），
# 找不到时再退回 make_assets.py 生成的模拟素材（task2/assets/）。
SEARCH_DIRS = [TASK2_DIR, TASK2_DIR / "assets"]


def resolve_asset(name: str) -> Path:
    """按优先级查找素材（官方素材优先），都找不到时给出清晰提示。"""
    for d in SEARCH_DIRS:
        path = d / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"缺少素材 {name}\n"
        f"  · 官方素材请放到 {TASK2_DIR}\n"
        f"  · 或运行 `python src/make_assets.py` 生成可复现的模拟素材到 {ASSETS_DIR}"
    )


def find_gt(video_path: Path) -> Path:
    """真值 csv 与视频同目录（官方素材没有真值，模拟素材有）。"""
    return video_path.with_name(video_path.stem + "_gt.csv")
