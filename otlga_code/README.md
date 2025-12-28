# OTLGA: Optimal Transport with Local-Global Attention

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0+-orange.svg)](https://pytorch.org/)

**OTLGA** (Optimal Transport with Local-Global Attention) 是一个用于医学影像和放射学报告双向检索的深度学习模型。

## 🎯 核心创新

1. **局部-全局注意力 (LGA)**: 增强局部特征与全局上下文的交互
2. **熵正则最优传输 (OT)**: 实现细粒度的图像-文本对齐
3. **OT引导门控融合**: 动态融合对齐后的跨模态特征

## 📁 文件结构

```
otlga_github/
├── otlga_model.py              # 核心模型定义 (OTLGAModel)
├── otlga_model_ablation.py     # 消融实验模型 (OTLGAModelAblation)
├── otlga_dataset.py            # 数据集类 (OTLGADataset)
├── vit_custom.py               # Vision Transformer 实现
├── modules.py                  # 辅助损失模块 (SentenceContrastive, UncertaintyAuxiliary)
├── train_otlga.py              # 主训练脚本
├── test_otlga.py               # 测试评估脚本
├── train_ablation.py           # 消融实验训练脚本
├── test_ablation.py            # 消融实验测试脚本
├── requirements.txt            # 依赖包列表
├── README.md                   # 本文档
└── ABLATION_STUDY.md           # 消融实验说明
```

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备

使用 MIMIC-CXR 数据集，需要准备：

1. **图像数据**: 组织在数据目录下的图像文件
2. **CSV标注文件**: 包含以下字段的CSV文件
   - `filename`: 图像文件名
   - `split`: 数据集划分 (train/valid/test)
   - `label`: 标签信息
   - `org_caption`: 原始报告文本

**注意**: 需要修改代码中的路径配置：
- `train_otlga.py`: 修改 `data_root` 和 `csv_path`
- `test_otlga.py`: 修改 `data_root` 和 `csv_path`
- `train_ablation.py`: 修改 `data_root` 和 `csv_path`
- `test_ablation.py`: 修改 `data_root` 和 `csv_path`

### 训练模型

#### 训练完整 OTLGA 模型

```bash
python train_otlga.py
```

#### 训练消融实验配置

```bash
# 训练单个配置
python train_ablation.py --config baseline
python train_ablation.py --config lga
python train_ablation.py --config ot
python train_ablation.py --config full

# 训练所有配置
python train_ablation.py --config all
```

### 测试模型

#### 测试完整模型

```bash
python test_otlga.py
```

#### 测试消融实验配置

```bash
# 测试单个配置
python test_ablation.py --config baseline
python test_ablation.py --config full

# 测试所有配置
python test_ablation.py --config all
```

## 🏗️ 模型架构

- **视觉编码器**: ViT-Base (768维)
- **文本编码器**: BERT-Base (768维)
- **共同嵌入空间**: 256维
- **核心模块**: 
  - Local-Global Attention (LGA)
  - Entropic Optimal Transport (OT)
  - OT-guided Gated Fusion


## 🔬 消融实验

项目包含完整的消融实验框架，可以评估各个模块的贡献：

- **baseline**: 基线模型（投影层 + ITC损失 + 句子级对比损失）
- **lga**: 仅LGA模块
- **ot**: 仅OT模块
- **gated_fusion**: OT + Gated Fusion
- **lga_ot**: LGA + OT
- **lga_gated**: LGA + OT + Gated Fusion
- **ot_gated**: OT + Gated Fusion
- **full**: 完整OTLGA模型

详细说明请参考 `ABLATION_STUDY.md`。

## 💾 依赖

主要依赖包（详见 `requirements.txt`）：

- `torch >= 1.9.0`
- `torchvision >= 0.10.0`
- `transformers >= 4.20.0`
- `pandas >= 1.3.0`
- `numpy >= 1.21.0`
- `tqdm >= 4.62.0`
- `Pillow >= 8.3.0`

## 📝 使用示例

### 训练示例

```python
from otlga_model import OTLGAModel
from otlga_dataset import OTLGADataset
from torch.utils.data import DataLoader

# 初始化模型
model = OTLGAModel(
    vit_type='vit_base',
    freeze_vit=False,
    freeze_layers=0,
    c_embed_dim=256
)

# 加载数据
dataset = OTLGADataset(
    data_root="path/to/data",
    csv_path="path/to/data.csv",
    split='train',
    img_size=224
)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 训练循环
for batch in dataloader:
    image, text_input, label = batch
    v_final, t_final, ot_loss, T_fused = model(image, text_input)
    # ... 计算损失并反向传播
```

## 📄 引用

如果使用本模型，请引用：

```bibtex
@article{otlga2024,
  title={OTLGA: Optimal Transport with Local-Global Attention for Medical Image-Text Retrieval},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 📜 许可证

[请添加您的许可证信息]

## 👥 作者

[请添加作者信息]

## 🙏 致谢

感谢 MIMIC-CXR 数据集提供者以及相关开源项目的贡献。
