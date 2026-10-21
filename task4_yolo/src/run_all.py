# -*- coding: utf-8 -*-
"""一键运行 task-4（YOLO 方向）的题1~题3。

用法：
    python task4_yolo/src/run_all.py                 # 题1 + 题2 + 题3（默认 20 epoch）
    python task4_yolo/src/run_all.py --epochs 40     # 训练更久
    python task4_yolo/src/run_all.py --skip-dataset  # 数据集已就绪时跳过下载
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
TASK4 = SRC.parent
OUT = TASK4 / "outputs"


def run(cmd: list[str]) -> float:
    t0 = time.time()
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(SRC))
    if r.returncode != 0:
        raise SystemExit(f"[失败] {' '.join(cmd)} (exit {r.returncode})")
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--skip-dataset", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    total = 0.0
    if not args.skip_dataset and not (TASK4 / "dataset" / "data.yaml").exists():
        total += run([py, str(SRC / "fetch_dataset.py")])
    total += run([py, str(SRC / "task4_1_api_guide.py")])
    total += run([py, str(SRC / "task4_2_predict_images.py")])
    total += run([py, str(SRC / "task4_3_train_drone.py"), "--epochs", str(args.epochs),
                  "--imgsz", str(args.imgsz), "--n-train", str(args.n_train),
                  "--n-val", str(args.n_val)])
    print(f"\n全部完成，用时 {total/60:.1f} min")

    print("\n输出清单：")
    for d in (OUT / "images", OUT / "figures", OUT / "metrics"):
        if d.exists():
            print(f"\n[{d.name}]")
            for f in sorted(d.glob("*")):
                print(f"  {f.name:44s} {f.stat().st_size / 1024:9.1f} KB")


if __name__ == "__main__":
    main()
