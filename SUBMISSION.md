# ComSen2026 算法组招新题 · task-2 & task-3 & task-4（题1~3）提交说明

本仓库包含 **task-2（OpenCV 与 Pillow）**、**task-3（MLP 识别 MNIST）** 与 **task-4 方向1 的题1~题3（YOLO 目标识别）**
的全部源码、结果图、结果视频、训练权重与实验报告。

> 本次已按**题目提供的官方素材**重跑全部结果：
> `task2/example1.jpg`、`task2/example2.mp4`、`task2/example3.mp4`、`task2/example4.mp4`、
> `task3/train-00000-of-00001.parquet`（MNIST 训练集）。

## 一、目录结构

```
.
├── README.md                    # 题目原文（未改动）
├── requirements.txt             # 依赖
├── run_all.py                   # ★ 一键跑完 task-2 + task-3 + 自检
├── task2/
│   ├── example1.jpg             # ★ 官方素材
│   ├── example2.mp4             # ★ 官方素材（1280×720，984 帧）
│   ├── example3.mp4             # ★ 官方素材（700×394，229 帧，红球竖直来回）
│   ├── example4.mp4             # ★ 官方素材（1920×1080，581 帧，复杂场景）
│   ├── REPORT.md                # ★ task-2 实验报告（题1~题5）
│   ├── assets/                  # 备用：模拟素材 + 逐帧真值（官方素材缺失时才用）
│   ├── src/
│   │   ├── common.py                 # 公共工具（路径/字体/绘图/视频读写；官方素材优先）
│   │   ├── make_assets.py            # 生成模拟素材（检测到官方素材会自动跳过）
│   │   ├── task2_1_image_basics.py   # 题1 读取显示 + 三通道分离
│   │   ├── task2_2_geometric.py      # 题2 缩放/旋转/平移/翻转
│   │   ├── task2_3_contours.py       # 题3 轮廓检测 + 总面积/总周长
│   │   ├── task2_4_motion_detect.py  # 题4 帧差法 vs MOG2（+亮度补偿）
│   │   ├── task2_5_object_track.py   # 题5 CSRT/KCF/改进追踪 + 指标
│   │   ├── ball_detector.py          # 题5 辅助：COCO 预训练检测器定位 "sports ball"
│   │   ├── ball_auto_locate.py       # 题5 备用：颜色+形状+往复轨迹的启发式定位（例4 失效，见报告）
│   │   └── run_all.py
│   └── outputs/
│       ├── figures/  images/  videos/  data/     # 结果（报告引用的就是这些）
├── task3/
│   ├── train-00000-of-00001.parquet  # ★ 官方 MNIST 训练集（HuggingFace 格式）
│   ├── REPORT.md                # ★ task-3 实验报告（概念理解 + 题2 + 题3）
│   ├── data/                    # 官方 idx 测试集（首次运行自动下载）
│   ├── src/
│   │   ├── common.py、mnist_data.py（parquet→PIL 解码→pandas DataFrame）
│   │   ├── task3_2_least_squares.py  # 题2 自动求导 + 梯度下降拟合
│   │   ├── task3_3_mlp_mnist.py      # 题3 MLP 训练/评估/可视化
│   │   └── run_all.py
│   └── outputs/figures/ models/ logs/
├── task4_yolo/                  # ★ task-4 方向1（YOLO）题1~题3
│   ├── REPORT.md                # 报告（API 学习 / 三张图推理 / 无人机数据集训练）
│   ├── example/                 # 官方素材（3 张图 + 视频）
│   ├── dataset/                 # 无人机数据集（Roboflow drone-and-bird：3893/649/196）+ subset（600/150）
│   ├── weights/                 # pretrained 权重缓存（yolov8n.pt）
│   ├── src/
│   │   ├── common4.py                # 公共工具
│   │   ├── fetch_dataset.py          # 数据集下载（断点续传）+ 解压
│   │   ├── task4_1_api_guide.py      # 题1 YOLO API 学习与速查表
│   │   ├── task4_2_predict_images.py # 题2 预训练权重推理三张图
│   │   ├── task4_3_train_drone.py    # 题3 训练 + 验证 + ONNX 导出
│   │   └── run_all.py
│   └── outputs/{images,figures,metrics,runs}
└── tools/
    ├── verify_outputs.py        # 交付自检（图片非空白 / 视频可解码 / 指标齐全 / 权重存在）
    ├── probe_assets.py          # 素材探查（分辨率、帧率、前景统计、parquet 结构）
    └── ascii_preview.py         # 无图形界面时把图片/视频帧转成 ASCII 预览
```

## 二、如何运行

```bash
# 1) 安装依赖（opencv 必须用 contrib 版，才有 CSRT/KCF tracker；parquet 需要 pyarrow；
#    task-4 需要 ultralytics）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2) 一键跑完 task-2 + task-3（约 5~8 分钟，CPU 即可；1080p 追踪最慢）
python run_all.py

# 3) 加上 task-4 题1~题3（CPU 上 YOLO 训练约 16 分钟）
python run_all.py --with-task4

# 也可以单独跑
python task2/src/run_all.py             # 题1~题5
python task3/src/run_all.py --epochs 12 # 题2 + 题3
python task4_yolo/src/run_all.py        # 题1 + 题2 + 题3
python tools/verify_outputs.py          # 交付自检

# 若想手动指定 example4 的初始框（自动定位不理想时）
python task2/src/task2_5_object_track.py --videos example4.mp4 --init 1351,19,45
```

环境：Windows 11 / Python 3.13.7 / opencv-contrib-python 4.11.0.86 / PyTorch 2.14.1 (CPU) /
pandas 3.0 / numpy 2.4 / pyarrow / **ultralytics 8.4.182**。

> ⚠️ 注意：`pip install ultralytics` 会把 `opencv-python` 装进来并**覆盖** `opencv-contrib-python` 的 `cv2`
> （OpenCV 5 里没有 CSRT/KCF tracker）。装完 ultralytics 后如果 task-2 题五报 `TrackerCSRT_create` 不存在，
> 执行一次：`pip uninstall -y opencv-python opencv-contrib-python && pip install opencv-contrib-python==4.11.0.86`。

## 三、题目要求与交付物对照

### task-2

| 题目要求 | 交付物 |
| --- | --- |
| 打开并显示图片，分离三个通道 | `outputs/figures/fig_task2_1_open.png`、`fig_task2_1_channels.png`、`images/channel_{B,G,R}.png` |
| 缩放 / 旋转 / 平移 / 翻转 | `outputs/figures/fig_task2_2_geometric.png`、`images/task2_2_*.jpg`（8 张） |
| 检测 example1.jpg 轮廓并保存，计算总面积与总周长 | `images/task2_3_contours_annotated.png`、`images/task2_3_mask.png`、`images/task2_3_rejected.png`、`figures/fig_task2_3_contours.png`、`data/task2_3_contours.json`（**只保留 1 个飞机轮廓：总面积 71 772.5 px²，总周长 2 300.3 px**；3 个云候选已剔除） |
| example2.mp4 运动检测，比较 MOG2 与不使用背景减除 | **`videos/example2_motion_compare.mp4`**（984 帧四分屏）、`figures/fig_task2_4_motion_stats.png`、`figures/fig_task2_4_frames.png`、`data/task2_4_motion_stats.{json,csv}` |
| example3/4.mp4 小球追踪，保存视频，思考如何提高准确度 | **`videos/example3_tracking.mp4`**、**`videos/example4_tracking.mp4`**、`figures/fig_task2_5_tracking_example{3,4}.png`、`data/task2_5_tracking_metrics.json`、报告 §5.4 |
| 文档 | `task2/REPORT.md` |

### task-3

| 题目要求 | 交付物 |
| --- | --- |
| 题1 深度学习概念理解 | `task3/REPORT.md` §1（神经元/权重/激活/前向传播/损失/梯度下降/反向传播/lr/优化器/Epoch/过拟合/正则化/BN/数据集划分/张量/自动求导/CPU-GPU/混合精度） |
| 题2 自动求导 + 梯度下降做最小二乘 | `figures/fig_task3_2_{surface,slice,loss,scatter}.png`、`logs/task3_2_result.json`（**R²(train)=0.9511，R²(test)=0.9499**，含参数表） |
| 题3 MLP 识别 MNIST | `figures/fig_task3_3_curves.png`、`fig_task3_3_predictions.png`、`fig_task3_3_confusion.png`、`fig_task3_3_misclassified.png`、`models/mlp_mnist.pt`、`logs/task3_3_metrics.json`（**测试准确率 98.15%**） |
| 文档 | `task3/REPORT.md` |

### task-4（方向 1：目标追踪与识别 · 题 1~3）

| 题目要求 | 交付物 |
| --- | --- |
| 题1 学习 YOLO 的 API 接口 | `task4_yolo/outputs/figures/task4_1_api_cheatsheet.png`、`outputs/metrics/task4_1_model_info.json`（加载/推理/结果结构/训练/验证/导出/断点续训 全接口速查表） |
| 题2 官方预训练权重推理三张图片并保存 | `outputs/images/pred_coco-{bear,bus,person}.jpg`、`outputs/figures/task4_2_predictions.png`、`outputs/metrics/task4_2_detections.json`（bear 0.951 / bus 0.930 / person 0.896） |
| 题3 无人机数据集训练、保存模型与训练结果绘图 | `outputs/runs/drone_yolov8n/weights/best.pt`（6.16 MB）、`best.onnx`（12.09 MB）、`outputs/figures/task4_3_{results,confusion_matrix,PR_curve,F1_curve,labels}.png/jpg`、`outputs/metrics/task4_3_metrics.json`（**mAP@0.5 = 0.6844，mAP@0.5:0.95 = 0.2983**，drone mAP@0.5 = 0.763） |
| 文档 | `task4_yolo/REPORT.md` |

> 说明：题面提到的无人机预训练权重 `drone.pt` **未随题面提供**，题3 用官方 `yolov8n.pt` 迁移学习代替，
> 训练/验证/导出流程与"用 drone.pt 微调"完全一致（换权重只需改一行）。
> 题4（视频推理与 EAO/速度分析）与题5（量化）本次未做，但 **ONNX 导出**已作为题3 的一部分完成。

## 四、关于"没有标注"的评估方式（说明为什么这样写）

官方素材只有视频/图片、**没有人工真值**，所以：

1. **题三**：直接在真实照片上做双分支分割（暗色 + 低饱和高亮）+ 轮廓提取，
   给出总面积/总周长/每个轮廓的属性表，并解释"为什么只用 Otsu 会漏掉白色主体部分"。
2. **题四**：用**代理指标**做定量对比（前景面积、连通域个数、填充度、实心度、质心抖动、
   全屏误检帧数、单帧耗时），这些指标都能从掩码直接算出来，不需要真值；
   同时用模拟素材（`task2/assets/`，带真值）补充"命中率/光照突变"这类需要真值的对比。
3. **题五**：先用 **COCO 预训练检测器**（torchvision，`sports ball` 类）在第 0 帧自动定位球
   —— example3 是红球（r≈58 px，score 0.64），example4 是**大球**（r≈149 px，score 0.88）；
   再把参考做成**跟踪式单目标检测**（尺寸一致性 + 运动预测闸门 + 连续两帧确认 + 全局重捕获），
   example4 里丢弃了 4 个误检候选、绿圈不再跳到别的物体上；参考覆盖 77.1%（447/581 帧）。
   报告中明确标注 `reference = COCO sports-ball detector`，以及其中 75 个"多球歧义帧"（单独统计）。
   想要人工指定初始框时可用 `--init x,y,r` 覆盖自动结果。

## 五、提交材料包（可直接上交）

```bash
python tools/build_submission.py     # 自动生成 提交/ 目录 + 压缩包
```

生成结果（`提交/`）：

| 文件 | 说明 |
| --- | --- |
| `ComSen2026_提交报告.md` | 合并报告（task-2 + task-3 + task-4 题1~3） |
| `ComSen2026_提交报告.html` | 网页版，浏览器打开 → 打印 → **另存为 PDF** |
| `ComSen2026_提交报告.docx` | **Word 版（28 张图已内嵌，可直接交）** |
| `ComSen2026_task2-3-4_提交包.zip` | **76.9 MB**：报告 + 36 张图 + 3 个视频 + 2 个模型（不含代码） |
| `提交说明与模板.md` | 邮件正文模板、GitHub Issue 模板（`班别-姓名-学号-仓库链接`）、提交前检查清单 |
| `图片/` `视频/` `模型/` | 按题目要求命名的结果图、task-2 的三个结果视频、task-4 的 `best.pt`/`best.onnx` |

报告开头有一张 **"题目要求 → 本包文件"对照表**，老师可以逐条对号入座；
封面上留了 **班别 / 姓名 / 学号 / 仓库链接** 的填写位。

## 六、自检结果

`python tools/verify_outputs.py`：所有图片存在且非空白（灰度 std > 2）、三个结果视频可解码
（帧数/分辨率正确）、9 份指标 json 齐全、task-4 的权重/ONNX 体积正常 —— 当前**全部通过**。

## 七、主要结论速览

* **task2-3**：example1.jpg 是"天空+飞机+云"：用「Otsu 暗色 ∪ 低饱和高亮」得到候选，再用**边缘密度（飞机 12.3% vs 云 0.0%）+ 亮度 V（飞机 128 vs 云 197~210）** 剔除 3 个云候选，只留飞机 → **面积 71 772.5 px²（11.86% 画面）、周长 2 300.3 px**；
  多边形近似后周长少 7.3%；`RETR_TREE` 扣孔后净面积少 432 px²。
* **task2-4**（984 帧真实街景）：帧差法的**实心度 0.725 vs MOG2 0.871**、**填充度 0.552 vs 0.692**、
  **质心抖动 70.5 px vs 54.8 px** —— 帧差只给"运动边缘"，MOG2 给"完整剪影"；
  成本约 1.2 / 7.0 / 18 ms；亮度补偿在本片段无收益（+165% 耗时），但在光照突变场景能把误检从 22.9 万 px² 降到 0。
* **task2-5**：球用 COCO `sports ball` 检测器自动定位（example3 红球 r≈58、example4 大球 r≈149）。
  绿色参考做成**跟踪式单目标检测**（尺寸一致性 + 运动预测闸门 + 连续两帧确认 + 跟丢后全局重捕获 +
  速度衰减），example4 里**丢弃了 4 个误检候选**，绿圈不再跳到别的物体上（误差上限 1 010 → 42 px）；
  画法也加强了（半透明填充 + 粗描边 + 圆心十字 + `REF r=` 标注 + 加粗轨迹）。
  example3：CSRT 15.02 px → **改进 6.66 px（中位 4.57，精度 95.6%，成功率 100%）**，KCF 31.69 px；
  example4（1080p60、多球、相机在动）：CSRT 44.47 px（框心系统性偏差）、KCF 第 2 帧后失败，
  **改进 13.59 px（中位 12.39，最大 42 px，精度 83%，成功率 99.3%）**，参考覆盖 77.1%。
* **task3-2**：R²(test) = **0.9499**，与噪声理论上限 0.9501 只差 0.0002；残差 σ = 1.73 ≈ 理论 √3。
* **task3-3**：MLP(784-256-128-10, Dropout 0.2)，12 epoch，**测试准确率 98.15%**，CPU 训练约 20 s。
* **task4-2**：yolov8n 预训练权重对三张图的主目标全部正确（bear 0.951 / bus 0.930 / person 0.896），
  单张 CPU 推理 15~38 ms。
* **task4-3**：Roboflow `drone-and-bird` 数据集（3893/649/196，2 类）抽 600/150 子集，
  yolov8n 训练 20 epoch（imgsz 416，**CPU 954 s**）→ **mAP@0.5 0.6844、mAP@0.5:0.95 0.2983**，
  其中 drone mAP@0.5 = **0.763**、bird = 0.606；导出 `best.pt` 6.16 MB / `best.onnx` 12.09 MB，
  CPU 推理 28.2 ms/张（35.5 FPS）。
