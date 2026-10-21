# -*- coding: utf-8 -*-
"""一键运行 task-3 全部题目（数据集下载 -> 题2 最小二乘拟合 -> 题3 MLP-MNIST）。"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
TASK3 = SRC.parent
OUT = TASK3 / "outputs"


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
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    total = run("task3_2_least_squares.py")
    total += run("task3_3_mlp_mnist.py", ["--epochs", str(args.epochs)])
    print(f"\n全部完成，用时 {total:.1f}s")

    print("\n输出清单：")
    for d in (OUT / "figures", OUT / "models", OUT / "logs"):
        if d.exists():
            print(f"\n[{d.name}]")
            for f in sorted(d.glob("*")):
                print(f"  {f.name:42s} {f.stat().st_size / 1024:9.1f} KB")


if __name__ == "__main__":
    main()
