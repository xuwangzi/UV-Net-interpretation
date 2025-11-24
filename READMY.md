# UV-Net 项目代码结构说明

## 📁 项目目录结构

```
UV-Net/
├── uvnet/              # 核心模型代码
│   ├── encoders.py     # 编码器模块（曲线、曲面、图）
│   └── models.py       # 模型定义（分类、分割）
├── datasets/            # 数据集处理
│   ├── base.py         # 数据集基类
│   ├── solidletters.py # SolidLetters 数据集
│   ├── mfcad.py        # MFCAD 数据集
│   ├── fusiongallery.py # Fusion Gallery 数据集
│   └── util.py         # 工具函数（旋转、缩放等）
├── process/             # 数据预处理工具
│   ├── solid_to_graph.py      # STEP → DGL 图
│   ├── solid_to_pointcloud.py # STEP → 点云
│   ├── solid_to_rendermesh.py # STEP → 渲染网格
│   ├── visualize.py           # 可视化工具
│   └── visualize_uvgrid_graph.py # UV-grid 可视化
├── classification.py   # 分类任务训练/测试脚本
├── segmentation.py     # 分割任务训练/测试脚本
└── environment.yml     # Conda 环境配置
```

---

## 🛠️ 环境依赖安装说明

### 1. 先安装以下依赖

- python
- pytorch
- pytorch-lightning
- torchmetrics
- joblib
- matplotlib
- matplotlib-base
- scikit-learn
- tqdm
- trimesh

### 2. 再安装

- dgl-cuda
    - [dglteam channel 地址](https://anaconda.org/dglteam/repo)
- occwl
    - 步骤 1：github 
    ``git clone git@github.com:AutodeskAILab/occwl.git``
    - 步骤 2：安装 occwl 
    `pip install git+https://github.com/AutodeskAILab/occwl`

### 3. 安装 python-occ

- 安装 pythonOCC：https://github.com/tpaviot/pythonocc-core?tab=readme-ov-file#install-with-conda

- 可能需要指定 PYTHONOCC 和 NUMPY 的路径

```
export PYTHONOCC_INSTALL_DIRECTORY=$CONDA_PREFIX/lib/python3.9/site-packages/OCC
export _INCLUDE_DIR=$(python -c "import numpy; print(numpy.get_include())")

cmake \
    -DOCCT_INCLUDE_DIR=/opt/occt790/include/opencascade \
    -DOCCT_LIBRARY_DIR=/opt/occt790/lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHONOCC_INSTALL_DIRECTORY=$PYTHONOCC_INSTALL_DIRECTORY \
    -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python \
    -DPython3_NumPy_INCLUDE_DIRS=$NUMPY_INCLUDE_DIR \
    -DPYTHONOCC_MESHDS_NUMPY=ON \
    ..
```

## 🔧 核心模块详解

### 1. `uvnet/` - 核心模型代码

#### `uvnet/encoders.py` - 编码器模块
**功能**：实现三种编码器，将 B-rep 数据转换为特征向量

**主要类**：
- **`UVNetCurveEncoder`** (1D CNN)
  - 输入：边 UV-grid (batch × 6 × 10)
  - 输出：边嵌入向量 (batch × 64)
  - 结构：3层 1D 卷积 + 全局平均池化 + 全连接层
  
- **`UVNetSurfaceEncoder`** (2D CNN)
  - 输入：面 UV-grid (batch × 7 × 10 × 10)
  - 输出：面嵌入向量 (batch × 64)
  - 结构：3层 2D 卷积 + 全局平均池化 + 全连接层
  
- **`UVNetGraphEncoder`** (图神经网络)
  - 输入：面嵌入、边嵌入、DGL 图
  - 输出：节点嵌入、图级嵌入
  - 结构：多层消息传递（NodeConv + EdgeConv）+ 图池化

**辅助类**：
- `_NodeConv`: 节点特征更新（使用边特征）
- `_EdgeConv`: 边特征更新（使用节点特征）
- `_MLP`: 多层感知机
- `_conv1d`, `_conv2d`, `_fc`: 卷积层和全连接层构建函数

---

#### `uvnet/models.py` - 模型定义
**功能**：定义分类和分割任务的完整模型

**主要类**：
- **`UVNetClassifier`** - 分类模型
  - 组成：曲线编码器 + 曲面编码器 + 图编码器 + 分类器
  - 输入：DGL 图（包含面 UV-grid 和边 UV-grid）
  - 输出：整个模型的类别 logits
  
- **`Classification`** (PyTorch Lightning)
  - 封装 `UVNetClassifier`
  - 实现训练/验证/测试步骤
  - 使用交叉熵损失和准确率指标

- **`UVNetSegmenter`** - 分割模型
  - 组成：曲线编码器 + 曲面编码器 + 图编码器 + 分割器
  - 输入：DGL 图
  - 输出：每个面的类别 logits（节点级预测）
  
- **`Segmentation`** (PyTorch Lightning)
  - 封装 `UVNetSegmenter`
  - 实现训练/验证/测试步骤
  - 使用交叉熵损失和 IoU 指标

**辅助类**：
- `_NonLinearClassifier`: 3层 MLP 分类器

---

### 2. `datasets/` - 数据集处理

#### `datasets/base.py` - 数据集基类
**功能**：提供数据集的基础功能

**`BaseDataset` 类**：
- `load_graphs()`: 批量加载 DGL 图文件
- `load_one_graph()`: 加载单个图文件
- `center_and_scale()`: 中心化和缩放（归一化）
- `convert_to_float32()`: 数据类型转换
- `_collate()`: 批处理函数（将多个图合并为 batch）
- `get_dataloader()`: 创建 PyTorch DataLoader

**数据格式**：
- 输入：`.bin` 文件（DGL 图格式）
- 图节点数据：`ndata['x']` - 面 UV-grid (num_faces × 10 × 10 × 7)
- 图边数据：`edata['x']` - 边 UV-grid (num_edges × 10 × 6)

---

#### `datasets/solidletters.py` - SolidLetters 数据集
**功能**：字母分类数据集（26 个类别）

**特点**：
- 从文件名提取标签（首字母）
- 自动划分训练/验证集（80/20）
- 测试集使用独立文件列表

**数据格式**：
- 标签：`sample['label']` - 整数标签 (0-25)

---

#### `datasets/mfcad.py` - MFCAD 数据集
**功能**：机械特征识别分割数据集（16 个类别）

**特点**：
- 从 JSON 文件加载标签
- 标签存储在图的节点数据中：`ndata['y']`
- 使用 `split.json` 划分数据集

**数据格式**：
- 标签：`graph.ndata['y']` - 每个面的类别标签

---

#### `datasets/fusiongallery.py` - Fusion Gallery 数据集
**功能**：Fusion 360 Gallery 分割数据集（8 个类别）

**特点**：
- 从 `.seg` 文件加载标签
- 支持数据集版本 s1.0.0 和 s2.0.0
- 自动划分训练/验证集

**数据格式**：
- 标签：`graph.ndata['y']` - 每个面的类别标签

---

#### `datasets/util.py` - 工具函数
**功能**：提供数据预处理和增强工具

**主要函数**：
- `bounding_box_uvgrid()`: 计算 UV-grid 的边界框
- `center_and_scale_uvgrid()`: 中心化和缩放 UV-grid
- `get_random_rotation()`: 生成随机旋转（90度倍数）
- `rotate_uvgrid()`: 旋转 UV-grid（点和法向量）
- `valid_font()`: 验证字体名称（SolidLetters 数据集用）

---

### 3. `process/` - 数据预处理工具

#### `process/solid_to_graph.py` - STEP 转 DGL 图
**功能**：将 STEP 格式的 CAD 文件转换为 DGL 图格式

**处理流程**：
1. 加载 STEP 文件（使用 occwl）
2. 构建面邻接图
3. 为每个面生成 2D UV-grid
4. 为每条边生成 1D UV-grid
5. 保存为 DGL `.bin` 格式

**输出**：
- `.bin` 文件：包含面 UV-grid 和边 UV-grid 的 DGL 图

---

#### `process/solid_to_pointcloud.py` - STEP 转点云
**功能**：从 STEP 文件提取点云数据

**输出**：
- `.npz` 文件：包含点坐标和法向量

---

#### `process/solid_to_rendermesh.py` - STEP 转渲染网格
**功能**：将 STEP 文件转换为 STL 格式的渲染网格

**输出**：
- `.stl` 文件：非水密网格（用于渲染）

---

#### `process/visualize.py` - 可视化工具
**功能**：可视化 STEP 文件和 DGL 图

---

#### `process/visualize_uvgrid_graph.py` - UV-grid 可视化
**功能**：可视化 UV-grid 数据

---

### 4. 训练/测试脚本

#### `classification.py` - 分类任务脚本
**功能**：训练和测试分类模型

**主要功能**：
- 解析命令行参数
- 创建 PyTorch Lightning Trainer
- 加载数据集（SolidLetters）
- 训练/测试模型
- 保存检查点和日志

**使用示例**：
```bash
# 训练
python classification.py train --dataset solidletters \
    --dataset_path /path/to/solidletters \
    --max_epochs 100 --batch_size 64

python classification.py train --dataset solidletters \
    --dataset_path /home/d3010/code/CAD/datasets/SolidLetters \
    --max_epochs 100 --batch_size 64

# 测试
python classification.py test --dataset solidletters \
    --dataset_path /path/to/solidletters \
    --checkpoint ./results/classification/best.ckpt
```

---

#### `segmentation.py` - 分割任务脚本
**功能**：训练和测试分割模型

**主要功能**：
- 支持 MFCAD 和 Fusion Gallery 数据集
- 支持随机旋转数据增强
- 可配置曲线输入通道数

**使用示例**：
```bash
# 训练
python segmentation.py train --dataset mfcad \
    --dataset_path /path/to/mfcad \
    --max_epochs 100 --batch_size 64 \
    --random_rotate

# 测试
python segmentation.py test --dataset mfcad \
    --dataset_path /path/to/mfcad \
    --checkpoint ./results/segmentation/best.ckpt
```

---

## 🔗 模块依赖关系

```
训练/测试脚本
├── classification.py
│   ├── datasets.solidletters → datasets.base
│   └── uvnet.models.Classification → uvnet.models.UVNetClassifier
│       └── uvnet.encoders (CurveEncoder, SurfaceEncoder, GraphEncoder)
│
└── segmentation.py
    ├── datasets.mfcad → datasets.base
    ├── datasets.fusiongallery → datasets.base
    └── uvnet.models.Segmentation → uvnet.models.UVNetSegmenter
        └── uvnet.encoders (CurveEncoder, SurfaceEncoder, GraphEncoder)

数据集模块
├── datasets.base (基类)
│   └── datasets.util (工具函数)
│
├── datasets.solidletters → datasets.base
├── datasets.mfcad → datasets.base
└── datasets.fusiongallery → datasets.base

数据预处理
└── process/
    ├── solid_to_graph.py (STEP → DGL 图)
    ├── solid_to_pointcloud.py (STEP → 点云)
    └── solid_to_rendermesh.py (STEP → 网格)
```

---

## 📊 数据流

### 训练流程

```
STEP 文件
  ↓ (process/solid_to_graph.py)
DGL 图 (.bin)
  ↓ (datasets/*.py)
PyTorch Dataset
  ↓ (DataLoader)
Batch of Graphs
  ↓ (模型前向传播)
  ├── 边 UV-grid → UVNetCurveEncoder → 边嵌入
  ├── 面 UV-grid → UVNetSurfaceEncoder → 面嵌入
  └── 图结构 + 嵌入 → UVNetGraphEncoder → 图嵌入
  ↓
分类/分割输出
  ↓ (损失计算)
模型更新
```

### 分类任务数据流

```
输入：DGL 图
  ├── ndata['x']: 面 UV-grid (N×10×10×7)
  └── edata['x']: 边 UV-grid (E×10×6)
  ↓
编码阶段
  ├── 边嵌入 (E×64)
  ├── 面嵌入 (N×64)
  └── 图嵌入 (1×128)
  ↓
分类器
  └── 类别 logits (batch_size×num_classes)
```

### 分割任务数据流

```
输入：DGL 图
  ├── ndata['x']: 面 UV-grid (N×10×10×7)
  └── edata['x']: 边 UV-grid (E×10×6)
  ↓
编码阶段
  ├── 边嵌入 (E×64)
  ├── 面嵌入 (N×64)
  ├── 节点嵌入 (N×128)
  └── 图嵌入 (1×128)
  ↓
特征融合
  └── 节点嵌入 + 图嵌入 (N×256)
  ↓
分割器
  └── 每个面的类别 logits (N×num_classes)
```

---

## 🎯 关键设计模式

1. **继承模式**：所有数据集继承 `BaseDataset`，复用基础功能
2. **组合模式**：模型由多个编码器组合而成
3. **Lightning 模式**：使用 PyTorch Lightning 简化训练流程
4. **模块化设计**：编码器、模型、数据集分离，便于扩展

---

## 📝 扩展指南

### 添加新数据集
1. 继承 `BaseDataset`
2. 实现 `num_classes()` 静态方法
3. 重写 `load_one_graph()` 加载标签
4. 重写 `_collate()` 处理批数据（如需要）

### 添加新模型
1. 在 `uvnet/models.py` 中定义新模型类
2. 继承 `pl.LightningModule`
3. 实现 `training_step`, `validation_step`, `test_step`
4. 创建对应的训练脚本

### 处理新数据格式
1. 在 `process/` 目录添加转换脚本
2. 使用 `occwl` 库读取 CAD 文件
3. 生成 UV-grid 和面邻接图
4. 保存为 DGL `.bin` 格式

