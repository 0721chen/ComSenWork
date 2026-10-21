# -*- coding: utf-8 -*-
"""一键跑完 task-2、task-3（可选 task-4），并做交付自检。

用法：
    python run_all.py                  # task-2 + task-3 + 自检
    python run_all.py --epochs 20      # 改变 task3 MLP 训练轮数
    python run_all.py --with-task4     # 额外跑 task-4 题1~题3（CPU 上训练约 16 分钟）
    python run_all.py --skip-verify    # 跳过最后的自检
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"[失败] {' '.join(cmd)} (exit {r.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--with-task4", action="store_true", help="额外运行 task-4（YOLO 方向，题1~3）")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    t0 = time.time()
    run([py, str(ROOT / "task2" / "src" / "run_all.py")])
    run([py, str(ROOT / "task3" / "src" / "run_all.py"), "--epochs", str(args.epochs)])
    if args.with_task4:
        run([py, str(ROOT / "task4_yolo" / "src" / "run_all.py")])
    if not args.skip_verify:
        run([py, str(ROOT / "tools" / "verify_outputs.py")])
    print(f"\n全部完成，总耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
