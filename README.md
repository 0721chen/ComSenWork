# ComSen2026 算法组招新题 · 计算机视觉实践

本仓库是 ComSen2026 算法组招新题的完成作品，包含 **task-2（OpenCV 与 Pillow）**、
**task-3（MLP 识别 MNIST）** 与 **task-4 方向1 题1~3（基于 YOLO 的目标识别）** 的全部源码、结果图、
结果视频、训练模型与实验报告。

> 题目原文：[`docs/题目原文.md`](docs/题目原文.md)　·　提交说明：[`SUBMISSION.md`](SUBMISSION.md)

---

## 目录结构

```
.
├── task2/                     # OpenCV + Pillow（题1~题5）
│   ├── example1.jpg  example2~4.mp4     # 官方素材
│   ├── REPORT.md                        # 实验报告
│   ├── src/                             # 8 个脚本（图像/几何/轮廓/运动检测/追踪 + 工具）
│   └── outputs/{figures,images,videos,data}
├── task3/                     # MLP 识别 MNIST（题1~题3）
│   ├── train-00000-of-00001.parquet      # 官方 MNIST 训练集
│   ├── REPORT.md
│   ├── src/                             # 概念/最小二乘/MLP/数据工具
│   └── outputs/{figures,models,logs}
├── task4_yolo/                # YOLO（题1~题3）
│   ├── example/                         # 官方 3 张图 + 视频
│   ├── REPORT.md
│   ├── src/                             # API/推理/训练/数据集下载
│   └── outputs/{figures,images,metrics,runs}
├── tools/                     # 交付自检、提交包生成、素材探查
├── docs/题目原文.md            # 题目原文
├── run_all.py                 # 一键运行
└── requirements.txt
```

---

## 环境与安装

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

实测环境：Windows 11 · Python 3.13.7 · opencv-contrib-python 4.11 · PyTorch 2.14.1（CPU）·
ultralytics 8.4.182 · pandas 3.0 · pyarrow。全部实验**只用 CPU** 即可复现。

> 已知坑：`pip install ultralytics` 会把 `opencv-python 5.x` 装进来并覆盖 contrib 版 `cv2`
> （OpenCV 5 里没有 CSRT/KCF tracker），导致 task-2 题五报错。修复：
> `pip uninstall -y opencv-python opencv-contrib-python && pip install opencv-contrib-python==4.11.0.86`

---

## 快速开始

```bash
python run_all.py                 # task-2 + task-3 + 交付自检（约 5~8 分钟）
python run_all.py --with-task4    # 再加 task-4 题1~题3（CPU 上 YOLO 训练约 16 分钟）

# 也可以单独跑
python task2/src/run_all.py
python task3/src/run_all.py --epochs 12
python task4_yolo/src/run_all.py
python tools/verify_outputs.py    # 只做交付自检
```

数据说明：

* **素材**用题目提供的官方文件，放在各 `taskN/` 目录下；
* **MNIST** 训练集用题目给的 `task3/train-00000-of-00001.parquet`，测试集用官方 idx（首次运行自动下载）；
* **无人机数据集**（Roboflow `drone-and-bird`，3893/649/196 张，2 类）不在仓库里（体积 236 MB），
  跑 `python task4_yolo/src/fetch_dataset.py` 一键下载解压。

---

## 题目要求 → 交付物对照

### task-2

| 题目要求 | 交付物 | 关键结果 |
| --- | --- | --- |
| 打开显示图片、分离三通道 | `task2/outputs/figures/fig_task2_1_open.png`、`fig_task2_1_channels.png`、`images/channel_{B,G,R}.png` | B/G/R 均值 198/160/112（蓝天背景） |
| 缩放/旋转/平移/翻转 | `figures/fig_task2_2_geometric.png`、`images/task2_2_*.jpg` | 9 种变换 + 综合仿射 |
| 轮廓检测 + 总面积/总周长 | `images/task2_3_contours_annotated.png`、`task2_3_mask.png`、`task2_3_rejected.png`、`figures/fig_task2_3_contours.png` | **只保留飞机：71 772.5 px² / 2 300.3 px**；3 个云候选被剔除 |
| 运动检测（MOG2 vs 不用背景减除） | **`videos/example2_motion_compare.mp4`**、`figures/fig_task2_4_*.png` | 实心度 0.725 vs 0.871、质心抖动 70.5 vs 54.8 px |
| example3/4 小球追踪 + 提高准确度 | **`videos/example3_tracking.mp4`**、**`example4_tracking.mp4`**、`figures/fig_task2_5_*.png` | example3 **6.66 px**、example4 **13.59 px**（CSRT 分别 15.02 / 44.47 px） |

报告：[`task2/REPORT.md`](task2/REPORT.md)

### task-3

| 题目要求 | 交付物 | 关键结果 |
| --- | --- | --- |
| 题1 深度学习概念理解 | [`task3/REPORT.md`](task3/REPORT.md) §题一 | 概念逐条 + 与实验数字对应 |
| 题2 最小二乘拟合（图 + 参数 + R²） | `figures/fig_task3_2_{surface,slice,loss,scatter}.png`、`logs/task3_2_result.json` | **R²(test)=0.9499**（噪声理论上限 0.9501） |
| 题3 MLP 识别 MNIST | `figures/fig_task3_3_{curves,predictions,confusion,misclassified}.png`、`models/mlp_mnist.pt` | **测试准确率 98.15%**（12 epoch / CPU 20 s） |

报告：[`task3/REPORT.md`](task3/REPORT.md)

### task-4（方向1 题1~3）

| 题目要求 | 交付物 | 关键结果 |
| --- | --- | --- |
| 题1 学习 YOLO API | `figures/task4_1_api_cheatsheet.png`、`metrics/task4_1_model_info.json` | 加载/推理/训练/验证/导出全接口速查 |
| 题2 预训练权重推理三张图片 | `images/pred_coco-{bear,bus,person}.jpg`、`figures/task4_2_predictions.png` | bear 0.951 / bus 0.930 / person 0.896 |
| 题3 无人机数据集训练 + 模型导出 | `runs/drone_yolov8n/weights/{best.pt,best.onnx}`、`figures/task4_3_*.png`、`metrics/task4_3_metrics.json` | **mAP@0.5=0.6844**、mAP@0.5:0.95=0.2983、drone 类 0.763 |

> task-4 题面提到的 `drone.pt` 未随题面提供，因此以官方 `yolov8n.pt` 预训练权重为起点做迁移学习，
> 训练 / 验证 / 导出流程与"用 drone.pt 微调"完全一致。

报告：[`task4_yolo/REPORT.md`](task4_yolo/REPORT.md)

---

## 实现要点（详见各报告）

**task-2**

* 题三：先「Otsu 暗色 ∪ 低饱和高亮」得到候选（云也会进来），再用**边缘密度（飞机 12.26% / 云 0.00%）+
  平均亮度 V（128 / 197~210）**两条可解释判据剔除云，只保留飞机轮廓；
* 题四：帧差法只给"运动边缘"（实心度低、质心抖动大），MOG2 给完整剪影；另加"亮度补偿"分支处理光照突变；
* 题五：用 COCO 预训练检测器（`sports ball`）自动定位球，并生成**跟踪式**参考轨迹（尺寸一致性 + 运动闸门 +
  连续确认）；改进流水线 = CSRT + 检测器纠正 + 匀速预测。

**task-3**

* 题二：只用 PyTorch 自动求导 + Adam 解最小二乘，R² 已贴到噪声上限，说明残差就是噪声本身；
* 题三：784-256-128-10 MLP（Dropout 0.2，235 146 参数），pandas 组织数据、PIL 解码 parquet 里的 PNG。

**task-4**

* 数据集抽 600/150 子集（CPU 现实折中），yolov8n 20 epoch @416 → mAP@0.5 0.6844；
* 顺带完成 ONNX 导出（12.09 MB）与 CPU 推理速度基准（28.2 ms/张）。

---

## 自检

```bash
python tools/verify_outputs.py
```

检查：全部结果图存在且非空白、3 个结果视频可解码、9 份指标 json 齐全、task-4 权重/ONNX 体积正常。
当前**全部通过**。
