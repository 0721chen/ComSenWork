# -*- coding: utf-8 -*-
"""MNIST 数据加载：**优先读题目提供的 parquet**，读不到再回退到官方 idx 文件下载。

* 训练集：`task3/train-00000-of-00001.parquet`
  —— HuggingFace MNIST 格式，2 列：
       image: struct<bytes: binary(PNG 编码的 28×28 灰度图), path: string>
       label: int64
  用 pyarrow 读、用 Pillow 解码 PNG（顺便满足"用图像库读数据"的要求），共 60000 张。
* 测试集：MNIST 官方 `t10k` 的 idx 文件（题目只给了训练集，测试集按官方协议补全，
  `download()` 会自动从两个镜像之一下载，共 10000 张）。
* 最终都组织成 **pandas DataFrame**：`label` + `p0..p783` 共 785 列。

若要完全离线，也可以只用 idx 文件：删除 parquet 即可自动回退。
"""

from __future__ import annotations

import gzip
import io
import struct
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common import DATA_DIR, TASK3_DIR

PARQUET = TASK3_DIR / "train-00000-of-00001.parquet"

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
IDX_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


# ---------------------------------------------------------------- parquet 读取
def load_parquet_images(path: Path = PARQUET) -> tuple[np.ndarray, np.ndarray]:
    """读取 HuggingFace 格式的 parquet：PNG bytes -> (N,28,28) uint8 数组。"""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["image", "label"]).to_pandas()
    labels = table["label"].to_numpy(dtype=np.int64)
    recs = table["image"].to_numpy()
    imgs = np.zeros((len(recs), 28, 28), np.uint8)
    for i, rec in enumerate(recs):
        arr = np.asarray(Image.open(io.BytesIO(rec["bytes"])).convert("L"), dtype=np.uint8)
        if arr.shape != (28, 28):
            arr = np.asarray(Image.fromarray(arr).resize((28, 28)), dtype=np.uint8)
        imgs[i] = arr[:, :, 0] if arr.ndim == 3 else arr
    print(f"[parquet] {path.name}: {imgs.shape[0]} 张 28×28 灰度图, 标签 {len(labels)} 个")
    return imgs, labels


# ---------------------------------------------------------------- idx 回退 / 测试集
def download(root: Path = DATA_DIR, names: list[str] | None = None, timeout: int = 60) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for fn in (names or list(IDX_FILES.values())):
        target = root / fn
        if target.exists() and target.stat().st_size > 1000:
            continue
        last_err = None
        for mirror in MIRRORS:
            url = mirror + fn
            try:
                print(f"[download] {url}")
                req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
                with urllib.request.urlopen(req, timeout=timeout) as r, open(target, "wb") as f:
                    f.write(r.read())
                print(f"[ok      ] {target.name}  {target.stat().st_size / 1024:.0f} KB")
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"[fail    ] {url}: {e}")
        if last_err is not None:
            raise RuntimeError(f"MNIST 下载失败，请手动把 {fn} 放入 {root}") from last_err


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"不是 idx3 图像文件: {path}"
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"不是 idx1 标签文件: {path}"
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8)


# ---------------------------------------------------------------- DataFrame
def to_dataframe(images: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """把 (N,28,28) 图像摊平成 (N,785) 的 DataFrame：label + p0..p783。"""
    flat = images.reshape(len(images), -1)
    df = pd.DataFrame(flat, columns=[f"p{i}" for i in range(flat.shape[1])])
    df.insert(0, "label", labels.astype(np.int64))
    return df


def load_dataframes(root: Path = DATA_DIR):
    """返回 (df_train, df_test, info)。训练集优先用 parquet，测试集用官方 idx。"""
    info = {"source_train": None, "source_test": None}
    if PARQUET.exists():
        x_train, y_train = load_parquet_images(PARQUET)
        info["source_train"] = f"parquet:{PARQUET.name}"
    else:
        download(root, [IDX_FILES["train_images"], IDX_FILES["train_labels"]])
        x_train = read_idx_images(root / IDX_FILES["train_images"])
        y_train = read_idx_labels(root / IDX_FILES["train_labels"])
        info["source_train"] = "idx:train-*-ubyte.gz"

    download(root, [IDX_FILES["test_images"], IDX_FILES["test_labels"]])
    x_test = read_idx_images(root / IDX_FILES["test_images"])
    y_test = read_idx_labels(root / IDX_FILES["test_labels"])
    info["source_test"] = "idx:t10k-*-ubyte.gz"
    print(f"[data] 训练 {info['source_train']}  测试 {info['source_test']}")

    df_train = to_dataframe(x_train, y_train)
    df_test = to_dataframe(x_test, y_test)
    print(f"训练集 DataFrame: {df_train.shape}, 测试集 DataFrame: {df_test.shape}")
    print("类别分布(训练集):", df_train["label"].value_counts().sort_index().to_dict())
    return df_train, df_test, info


if __name__ == "__main__":
    load_dataframes()
