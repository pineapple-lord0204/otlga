# OTLGA: Optimal Transport with Local-Global Attention

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0+-orange.svg)](https://pytorch.org/)

**OTLGA** (Optimal Transport with Local-Global Attention) is a deep learning model for bidirectional retrieval between medical images and radiology reports.

## 🎯 Core Innovations

1. **Local-Global Attention (LGA)**: Enhances interaction between local features and global context
2. **Entropic Optimal Transport (OT)**: Achieves fine-grained image-text alignment
3. **OT-guided Gated Fusion**: Dynamically fuses aligned cross-modal features

## 📁 File Structure

```
otlga_github/
├── otlga_model.py              # Core model definition (OTLGAModel)
├── otlga_model_ablation.py     # Ablation experiment model (OTLGAModelAblation)
├── otlga_dataset.py            # Dataset class (OTLGADataset)
├── vit_custom.py               # Vision Transformer implementation
├── modules.py                  # Auxiliary loss modules (SentenceContrastive, UncertaintyAuxiliary)
├── train_otlga.py              # Main training script
├── test_otlga.py               # Testing and evaluation script
├── train_ablation.py           # Ablation experiment training script
├── test_ablation.py            # Ablation experiment testing script
├── requirements.txt            # Dependencies list
├── README.md                   # This document
└── ABLATION_STUDY.md           # Ablation experiment documentation
```

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Data Preparation

Using the MIMIC-CXR dataset, you need to prepare:

1. **Image Data**: Image files organized in the data directory
2. **CSV Annotation File**: A CSV file containing the following fields
   - `filename`: Image filename
   - `split`: Dataset split (train/valid/test)
   - `label`: Label information
   - `org_caption`: Original report text

**Note**: You need to modify path configurations in the code:
- `train_otlga.py`: Modify `data_root` and `csv_path`
- `test_otlga.py`: Modify `data_root` and `csv_path`
- `train_ablation.py`: Modify `data_root` and `csv_path`
- `test_ablation.py`: Modify `data_root` and `csv_path`

### Train Model

#### Train Full OTLGA Model

```bash
python train_otlga.py
```

#### Train Ablation Experiment Configurations

```bash
# Train single configuration
python train_ablation.py --config baseline
python train_ablation.py --config lga
python train_ablation.py --config ot
python train_ablation.py --config full

# Train all configurations
python train_ablation.py --config all
```

### Test Model

#### Test Full Model

```bash
python test_otlga.py
```

#### Test Ablation Experiment Configurations

```bash
# Test single configuration
python test_ablation.py --config baseline
python test_ablation.py --config full

# Test all configurations
python test_ablation.py --config all
```

## 🏗️ Model Architecture

- **Visual Encoder**: ViT-Base (768 dimensions)
- **Text Encoder**: BERT-Base (768 dimensions)
- **Common Embedding Space**: 256 dimensions
- **Core Modules**: 
  - Local-Global Attention (LGA)
  - Entropic Optimal Transport (OT)
  - OT-guided Gated Fusion

## 📊 Performance

Performance on MIMIC-CXR test set:

| Metric | Image-to-Text | Text-to-Image | Average |
|--------|---------------|---------------|---------|
| **R@1** | 45.81% | 81.29% | 63.55% |
| **R@5** | - | - | - |
| **R@10** | - | - | - |

For detailed ablation experiment results, please refer to `ABLATION_STUDY.md`.

## 🔬 Ablation Experiments

The project includes a complete ablation experiment framework to evaluate the contribution of each module:

- **baseline**: Baseline model (projection layers + ITC loss + sentence-level contrastive loss)
- **lga**: LGA module only
- **ot**: OT module only
- **gated_fusion**: OT + Gated Fusion
- **lga_ot**: LGA + OT
- **lga_gated**: LGA + OT + Gated Fusion
- **ot_gated**: OT + Gated Fusion
- **full**: Full OTLGA model

For detailed description, please refer to `ABLATION_STUDY.md`.

## 💾 Dependencies

Main dependencies (see `requirements.txt` for details):

- `torch >= 1.9.0`
- `torchvision >= 0.10.0`
- `transformers >= 4.20.0`
- `pandas >= 1.3.0`
- `numpy >= 1.21.0`
- `tqdm >= 4.62.0`
- `Pillow >= 8.3.0`

## 📝 Usage Example

### Training Example

```python
from otlga_model import OTLGAModel
from otlga_dataset import OTLGADataset
from torch.utils.data import DataLoader

# Initialize model
model = OTLGAModel(
    vit_type='vit_base',
    freeze_vit=False,
    freeze_layers=0,
    c_embed_dim=256
)

# Load data
dataset = OTLGADataset(
    data_root="path/to/data",
    csv_path="path/to/data.csv",
    split='train',
    img_size=224
)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training loop
for batch in dataloader:
    image, text_input, label = batch
    v_final, t_final, ot_loss, T_fused = model(image, text_input)
    # ... compute loss and backpropagate
```

## 📄 Citation

If you use this model, please cite:

```bibtex
@article{otlga2024,
  title={OTLGA: Optimal Transport with Local-Global Attention for Medical Image-Text Retrieval},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 📜 License

[Please add your license information]

## 👥 Authors

[Please add author information]

## 🙏 Acknowledgments

Thanks to the MIMIC-CXR dataset providers and contributors of related open-source projects.
