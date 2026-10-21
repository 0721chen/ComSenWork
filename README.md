# ComSen2026 算法组招新题 · 计算机视觉实践

ComSen2026 算法组招新题的完成作品：**task-2（OpenCV 与 Pillow）**、
**task-3（MLP 识别 MNIST）**、**task-4 方向1 题1~3（基于 YOLO 的目标识别）**。
全部实验**只用 CPU** 即可复现。

## 做了什么

| 任务 | 内容 |
| --- | --- |
| **task-2** | ① 图像读取与 BGR/RGB 通道分离；② 缩放/旋转/平移/翻转与综合仿射；③ 真实照片的轮廓检测：从「天空+飞机+云」中**只提取飞机**并算面积/周长；④ 帧差法 vs MOG2 背景减除（另加亮度补偿分支）；⑤ 用 CSRT/KCF 与「CSRT + COCO 检测器 + 匀速预测」追踪 example3/4 的小球 |
| **task-3** | ① 深度学习核心概念梳理；② 只用 PyTorch 自动求导 + 梯度下降解最小二乘拟合；③ pandas + PIL 读数据，784-256-128-10 MLP 识别 MNIST |
| **task-4** | ① 学习 Ultralytics YOLO 接口；② 官方预训练权重推理三张图片；③ 无人机数据集（Roboflow drone-and-bird，2 类）微调训练、导出 ONNX、保存训练曲线与混淆矩阵 |

## 关键结果

| 题目 | 指标 |
| --- | --- |
| task-2 题三 | 只保留飞机轮廓：**面积 71 772.5 px²、周长 2 300.3 px**（3 个云候选被剔除） |
| task-2 题四 | 帧差法 vs MOG2：实心度 **0.725 vs 0.871**、质心抖动 **70.5 vs 54.8 px** |
| task-2 题五 | example3 **6.66 px**、example4 **13.59 px**（CSRT 分别为 15.02 / 44.47 px） |
| task-3 题二 | **R²(test) = 0.9499**（噪声理论上限 0.9501） |
| task-3 题三 | MLP **测试准确率 98.15%**（12 epoch，CPU 约 20 s） |
| task-4 题二 | bear 0.951 / bus 0.930 / person 0.896 |
| task-4 题三 | **mAP@0.5 = 0.6844**、mAP@0.5:0.95 = 0.2983（drone 类 0.763） |

## 快速开始

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

python run_all.py                 # task-2 + task-3 + 交付自检（约 5~8 分钟）
python run_all.py --with-task4    # 再加 task-4 题1~题3（CPU 上 YOLO 训练约 16 分钟）
```

无人机数据集不在仓库里（236 MB），跑 `python task4_yolo/src/fetch_dataset.py` 一键下载。

## 文档

- **详细说明**（目录结构、题目↔交付物对照、运行方式、评估方式、提交材料包、结论速览）：[`SUBMISSION.md`](SUBMISSION.md)
- **实验报告**：[`task2/REPORT.md`](task2/REPORT.md) · [`task3/REPORT.md`](task3/REPORT.md) · [`task4_yolo/REPORT.md`](task4_yolo/REPORT.md)
- **题目原文**：[`docs/题目原文.md`](docs/题目原文.md)

## 环境

Windows 11 · Python 3.13.7 · opencv-contrib-python 4.11 · PyTorch 2.14.1（CPU）· ultralytics 8.4.182 · pandas 3.0 · pyarrow。

> 已知坑：`pip install ultralytics` 会把 `opencv-python 5.x` 装进来并覆盖 contrib 版 `cv2`
> （OpenCV 5 里没有 CSRT/KCF tracker）。修复：
> `pip uninstall -y opencv-python opencv-contrib-python && pip install opencv-contrib-python==4.11.0.86`
