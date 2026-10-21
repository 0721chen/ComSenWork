# -*- coding: utf-8 -*-
"""一键运行 task-2 全部题目（生成素材 -> 题1~题5 -> 汇总输出清单）。

用法：
    python task2/src/run_all.py            # 全部跑一遍
    python task2/src/run_all.py --skip-assets   # 素材已存在时不重新生成
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
TASK2 = SRC.parent
OUT = TASK2 / "outputs"

STEPS = [
    ("生成/检查素材", "make_assets.py"),
    ("题1 图像读取与三通道分离", "task2_1_image_basics.py"),
    ("题2 缩放/旋转/平移/翻转", "task2_2_geometric.py"),
    ("题3 轮廓检测与面积周长", "task2_3_contours.py"),
    ("题4 运动目标检测(MOG2 对比)", "task2_4_motion_detect.py"),
    ("题5 tracker 小球追踪", "task2_5_object_track.py"),
]


def run(script: str, extra: list[str] | None = None) -> float:
    cmd = [sys.executable, str(SRC / script), *(extra or [])]
    t0 = time.time()
    print(f"\n>>> {script}")
    r = subprocess.run(cmd, cwd=str(SRC))
    if r.returncode != 0:
        raise SystemExit(f"[失败] {script} 退出码 {r.returncode}")
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-assets", action="store_true", help="跳过素材生成")
    args = ap.parse_args()

    total = 0.0
    for title, script in STEPS:
        if args.skip_assets and script == "make_assets.py":
            continue
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
        total += run(script)
    print(f"\n全部完成，用时 {total:.1f}s")

    print("\n输出清单：")
    for d in (OUT / "figures", OUT / "videos", OUT / "images", OUT / "data"):
        if not d.exists():
            continue
        files = sorted(d.glob("*"))
        print(f"\n[{d.name}] {len(files)} 个文件")
        for f in files:
            print(f"  {f.name:44s} {f.stat().st_size / 1024:9.1f} KB")


if __name__ == "__main__":
    main()
