# 安装和使用指南

## 环境配置

### 1. Python 环境

推荐使用 Python 3.8 或更高版本：

```bash
python --version  # 检查Python版本
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用 venv
python -m venv otlga_env
source otlga_env/bin/activate  # Linux/Mac
# 或
otlga_env\Scripts\activate  # Windows

# 使用 conda
conda create -n otlga python=3.8
conda activate otlga
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置路径

在使用代码前，需要修改以下文件中的路径配置：

#### train_otlga.py

```python
# 修改以下路径
data_root = "/path/to/your/data"  # 数据根目录
csv_path = os.path.join(data_root, "path/to/your/dataset.csv")  # CSV文件路径
save_dir = "/path/to/save/checkpoints"  # 模型保存目录
```

#### test_otlga.py

```python
# 修改以下路径
data_root = "/path/to/your/data"
csv_path = os.path.join(data_root, "path/to/your/dataset.csv")
checkpoint_path = "/path/to/your/checkpoint.pth"  # 模型权重路径
```

#### train_ablation.py 和 test_ablation.py

同样需要修改 `data_root` 和 `csv_path`。

## 数据准备

### 1. 数据集格式

CSV文件需要包含以下列：

- `filename`: 图像文件名（相对于data_root的路径）
- `split`: 数据集划分（'train', 'valid', 'test'）
- `label`: 标签信息（可选，用于辅助任务）
- `org_caption`: 原始报告文本

示例：

```csv
filename,split,label,org_caption
images/patient1/study1.jpg,train,"[1,0,1,0,...]","No acute cardiopulmonary process..."
images/patient1/study2.jpg,valid,"[0,1,0,1,...]","Mild cardiomegaly..."
```

### 2. 目录结构

推荐的数据目录结构：

```
data_root/
├── merged_dataset/
│   └── merged_dataset_real.csv
├── images/
│   ├── patient1/
│   │   ├── study1.jpg
│   │   └── study2.jpg
│   └── patient2/
│       └── ...
```

## 快速开始

### 1. 训练完整模型

```bash
python train_otlga.py
```

### 2. 测试模型

```bash
python test_otlga.py
```

### 3. 运行消融实验

```bash
# 训练消融实验
python train_ablation.py --config baseline

# 测试消融实验
python test_ablation.py --config baseline
```

## 常见问题

### Q: 如何修改模型参数？

A: 在 `train_otlga.py` 或 `train_ablation.py` 中修改模型初始化参数：

```python
model = OTLGAModel(
    vit_type='vit_base',  # 可改为 'vit_large'
    freeze_vit=False,
    freeze_layers=0,
    c_embed_dim=256  # 可修改嵌入维度
)
```

### Q: 如何调整训练超参数？

A: 在训练脚本中修改：

```python
batch_size = 64      # 批次大小
epochs = 30          # 训练轮数
lr = 5e-5            # 学习率
weight_decay = 0.01  # 权重衰减
```

### Q: 内存不足怎么办？

A: 可以：
1. 减小 `batch_size`
2. 减小 `img_size`（例如从224改为192）
3. 使用梯度累积

### Q: 如何使用预训练模型？

A: 在测试脚本中指定checkpoint路径：

```python
checkpoint_path = "/path/to/your/pretrained_model.pth"
```

## 技术支持

如有问题，请提交 Issue 或联系作者。

