# -*- coding: utf-8 -*-
"""下载并解压无人机数据集（Roboflow 导出，YOLO 格式）。

题目推荐的链接是 Roboflow 的直链导出包（约 236 MB）。网络不稳定时用**断点续传**：
按 Range 请求从已下载的字节处继续，最多重试 30 次，直到文件长度等于 Content-Length。

用法：
    python src/fetch_dataset.py            # 下载 + 解压（已存在则跳过）
    python src/fetch_dataset.py --force    # 重新下载
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

from common4 import DATASET_DIR, WEIGHTS_DIR, ensure_dirs

URL = "https://app.roboflow.com/ds/nSMQrzUp0X?key=TTcKC8vGXi"
ZIP_PATH = DATASET_DIR / "_drone_dataset.zip"
EXPECTED = 236_501_971          # 服务器 Content-Length（字节）


def download(target: Path = ZIP_PATH, expected: int = EXPECTED, force: bool = False) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if force and target.exists():
        target.unlink()
    for attempt in range(1, 31):
        pos = target.stat().st_size if target.exists() else 0
        if pos >= expected:
            print(f"[ok] 已下载完整: {target}  {pos/1024/1024:.1f} MB")
            return target
        headers = {"User-Agent": "Mozilla/5.0", "Range": f"bytes={pos}-"}
        try:
            req = urllib.request.Request(URL, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r, open(target, "ab") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            print(f"[try {attempt}] 到 {target.stat().st_size/1024/1024:.1f} MB")
        except Exception as e:
            print(f"[try {attempt}] 中断: {type(e).__name__}: {e} → 续传")
    if target.exists() and target.stat().st_size >= expected * 0.99:
        return target
    raise SystemExit(f"下载失败，已下载 {target.stat().st_size if target.exists() else 0} 字节")


def extract(zip_path: Path = ZIP_PATH, out: Path = DATASET_DIR, force: bool = False) -> Path:
    marker = out / "data.yaml"
    if marker.exists() and not force:
        print(f"[ok] 数据集已解压: {out}")
        return out
    out.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {zip_path.name} → {out}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    print(f"[ok] 解压完成: {out}")
    return out


def summarize(root: Path = DATASET_DIR) -> None:
    yaml_path = root / "data.yaml"
    if yaml_path.exists():
        print("\n--- data.yaml ---")
        print(yaml_path.read_text(encoding="utf-8", errors="replace"))
    for split in ("train", "valid", "test"):
        img_dir = root / split / "images"
        if img_dir.exists():
            n = len([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            print(f"{split:6s}: {n} 张图片")
        else:
            # Roboflow 有些导出把图片直接放在 split/ 下
            if (root / split).exists():
                n = len([p for p in (root / split).iterdir() if p.is_file()])
                print(f"{split:6s}: {n} 个文件（无 images/ 子目录）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    p = download(force=args.force)
    extract(p, force=args.force)
    summarize()


if __name__ == "__main__":
    main()
