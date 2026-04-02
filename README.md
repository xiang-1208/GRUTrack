# GRUTrack

基于神经网络的卡尔曼滤波 3D 多目标跟踪系统，依赖于 nuScenes 数据集。

## 项目简介

GRUTrack 将深度学习与传统卡尔曼滤波跟踪相结合，实现鲁棒的 3D 多目标跟踪。该系统使用神经网络（KalmanNet）自适应估计卡尔曼增益，相比固定增益的传统方法具有更好的跟踪性能。

## 项目结构

```
GRUTrack/
├── KalmanNet_nn.py           # 神经网络增强的卡尔曼滤波 (KalmanNet)
├── train.py                  # 训练脚本
├── test.py                   # 测试脚本 (跟踪 + 可选评估)
├── config/
│   └── nusc_config.yaml     # 配置文件
├── tracking/
│   ├── nusc_tracker.py       # 核心跟踪器 (数据关联、轨迹管理)
│   ├── nusc_trajectory.py    # 轨迹类
│   ├── nusc_life_manage.py  # 轨迹生命周期管理
│   └── nusc_score_manage.py # 轨迹评分管理
├── motion_module/
│   ├── kalman_filter.py      # 卡尔曼滤波实现
│   ├── motion_model.py       # 运动模型 (CTRA, CV, CA 等)
│   └── nusc_object.py        # 单帧目标表示
├── geometry/
│   ├── nusc_box.py          # NuScenes 边界框工具
│   ├── nusc_distance.py     # 距离/IoU 度量 (IoU, GIoU)
│   └── nusc_utils.py        # 几何工具
├── dataloader/
│   └── nusc_loader_self.py  # NuScenes 数据加载器
├── pre_processing/
│   ├── nusc_nms.py          # 非极大值抑制
│   └── nusc_data_conversion.py # 数据格式转换
├── utils/
│   ├── io.py                # 文件输入输出
│   ├── matching.py           # 匹配算法 (匈牙利算法、贪婪匹配)
│   ├── math.py              # 数学工具
│   └── script.py            # 脚本工具
└── data/
    └── script/              # 数据处理脚本
```

## 主要特性

- **神经网络增强的卡尔曼滤波**: 使用基于 GRU 的网络自适应学习卡尔曼增益
- **3D 多目标跟踪**: 支持 7 类目标 (汽车、卡车、公交车、拖车、行人、自行车、摩托车)
- **多种运动模型**: 支持 CTRA (恒定转角速率和加速度)、CV、CA、CTRV、自行车模型
- **先进的数据关联**: 基于类别特异度量的双阶段匈牙利匹配
- **nuScenes 数据集支持**: 内置 nuScenes 3D 跟踪数据集支持

## 安装

```bash
pip install -r requirements.txt
```

### 依赖

- PyTorch
- NumPy
- Pandas
- Numba
- Shapely
- pyquaternion
- nuscenes-devkit
- motmetrics

## 使用方法

### 数据准备

#### 1. 下载数据集

下载 [nuScenes 数据集](https://www.nuscenes.org/nuscenes.html) (v1.0-trainval 或 v1.0-test)。

#### 2. 生成检测结果

使用 3D 检测器 (如 CenterPoint, VoxelNeXt 等) 生成检测结果，输出格式需为 nuScenes 检测格式。

#### 3. 使用数据处理脚本

项目提供了 `data/script/` 目录下的脚本进行数据预处理：

| 脚本 | 功能 |
|------|------|
| `split.py` | 提取每个序列的首帧 token |
| `first_frame.py` | 按时间顺序提取每个序列的所有 token |
| `final_frame.py` | 提取每个序列的尾帧 token |
| `reorder_detection.py` | 按时间顺序重组检测结果 |
| `gt.py` | 导出 Ground Truth (用于训练) |

**处理步骤：**

```bash
cd data/script

# 步骤1: 提取每个序列的首帧 token
python split.py \
    --dataset_path /path/to/nuscenes/v1.0-trainval/ \
    --detector_path /path/to/detection.json \
    --dataset_name NuScenes \
    --dataset_version train

# 步骤2: 提取所有 token (按时间顺序)
python first_frame.py \
    --dataset_path /path/to/nuscenes/v1.0-trainval/ \
    --detector_path /path/to/detection.json \
    --dataset_name NuScenes \
    --dataset_version train

# 步骤3: 提取尾帧 token
python final_frame.py \
    --dataset_path /path/to/nuscenes/v1.0-trainval/ \
    --detector_path /path/to/detection.json \
    --dataset_name NuScenes \
    --dataset_version train

# 步骤4: 重组检测结果为时间顺序
python reorder_detection.py \
    --dataset_path /path/to/nuscenes/v1.0-trainval/ \
    --detector_path /path/to/detection.json \
    --dataset_name NuScenes \
    --dataset_version train \
    --detector_name centerpoint

# 步骤5: 导出 Ground Truth (训练时需要)
python gt.py
```

**输出文件结构：**

```
data/utils/
├── train/
│   ├── detector/
│   │   └── train_centerpoint.json    # 重组后的检测结果
│   ├── nusc_first_train_token.json  # 首帧 token 列表
│   ├── nusc_train_token.json         # 所有 token (按序列分组)
│   ├── nusc_final_train_token.json  # 尾帧 token 列表
│   └── gt_train.json                 # Ground Truth
├── val/
│   └── ...                           # 同上
└── test/
    └── ...                           # 同上
```

**注意：**
- 修改脚本中的硬编码路径为实际路径
- `dataset_version` 支持: `train`, `trainval`, `test`
- Ground Truth 仅在训练时需要

### 训练

```bash
python train.py \
    --config_path config/nusc_config.yaml \
    --train_path data/utils/train \
    --val_path data/utils/val \
    --nusc_path data/nuscenes/v1.0-trainval/ \
    --epochs 100
```

主要参数:
- `--config_path`: 配置文件路径
- `--train_path`: 训练数据目录
- `--val_path`: 验证数据目录
- `--nusc_path`: nuScenes 数据集路径
- `--model`: 预训练模型路径 (可选)
- `--epochs`: 训练轮数

### 测试

```bash
# 仅跟踪 (test 集)
python test.py \
    --model output/model/step_XXXX.pth \
    --data_split test

# 仅跟踪 (val 集)
python test.py \
    --model output/model/step_XXXX.pth \
    --data_split val

# 跟踪 + 评估
python test.py \
    --mode eval \
    --model output/model/step_XXXX.pth \
    --data_split val \
    --nusc_path data/nuscenes/v1.0-trainval/
```

主要参数:
- `--mode`: `"test"` 仅跟踪, `"eval"` 跟踪 + nuScenes 评估
- `--data_split`: `"test"`, `"val"`, `"train"` 自动配置数据路径
- `--model`: 训练好的模型权重路径

## 配置说明

`config/nusc_config.yaml` 中的主要配置项:

### 基础设置
- `LiDAR_interval`: 帧间时间间隔 (nuScenes 为 0.5s)
- `has_velo`: 检测器是否提供速度信息
- `CLASS_NUM`: 目标类别数 (nuScenes 为 7)

### 数据关联
- `two_stage`: 启用双阶段匹配
- `first_thre`: 类别特定的第一阶段阈值
- `second_thre`: 第二阶段阈值
- `algorithm`: 匹配算法 (Hungarian, Greedy, MNN)

### 运动模型
- `filter`: 每类别的卡尔曼滤波类型 (LinearKalmanFilter, ExtendKalmanFilter)
- `model`: 每类别的运动模型 (CTRA, CV, CA, CTRV, BICYCLE)

### 生命周期管理
- `max_age`: 轨迹被认为消亡前的最大帧数
- `min_hit`: 轨迹被确认前的最小检测数
- `decay_rate`: 每帧的分数衰减率

## 运动模型

系统支持多种运动模型:

| 模型 | 描述 |
|------|------|
| CTRA  | 恒定转角速率和加速度 |
| CV    | 恒定速度 |
| CA    | 恒定加速度 |
| CTRV  | 恒定转角速率和速度 |
| BICYCLE | 自行车模型 |

状态向量: `[x, y, z, w, l, h, v, a, yaw, yaw_rate]`

## 跟踪流程

1. **状态预测**: 使用运动模型预测所有活跃轨迹
2. **数据关联**: 使用双阶段匈牙利匹配将检测与轨迹关联
3. **状态更新**: 用检测更新匹配的轨迹
4. **轨迹管理**: 处理新轨迹、未匹配轨迹和轨迹生命周期
5. **评分管理**: 根据检测置信度更新轨迹评分

## 输出格式

跟踪结果以 nuScenes 跟踪格式保存:

```json
{
    "results": {
        "<sample_token>": [
            {
                "sample_token": "<token>",
                "translation": [x, y, z],
                "size": [w, l, h],
                "rotation": [qw, qx, qy, qz],
                "velocity": [vx, vy],
                "tracking_id": "<id>",
                "tracking_name": "<category>",
                "tracking_score": <score>
            }
        ]
    },
    "meta": {
        "use_camera": false,
        "use_lidar": true,
        "use_radar": false,
        "use_map": false,
        "use_external": false
    }
}
```

## 许可证

本项目遵循 LICENSE 文件中的条款。
