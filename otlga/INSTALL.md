# Installation and Usage Guide

## Environment Setup

### 1. Python Environment

Python 3.8 or higher is recommended:

```bash
python --version  # Check Python version
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv otlga_env
source otlga_env/bin/activate  # Linux/Mac
# or
otlga_env\Scripts\activate  # Windows

# Using conda
conda create -n otlga python=3.8
conda activate otlga
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Paths

Before using the code, you need to modify path configurations in the following files:

#### train_otlga.py

```python
# Modify the following paths
data_root = "/path/to/your/data"  # Data root directory
csv_path = os.path.join(data_root, "path/to/your/dataset.csv")  # CSV file path
save_dir = "/path/to/save/checkpoints"  # Model save directory
```

#### test_otlga.py

```python
# Modify the following paths
data_root = "/path/to/your/data"
csv_path = os.path.join(data_root, "path/to/your/dataset.csv")
checkpoint_path = "/path/to/your/checkpoint.pth"  # Model weights path
```

#### train_ablation.py and test_ablation.py

Similarly, modify `data_root` and `csv_path`.

## Data Preparation

### 1. Dataset Format

The CSV file needs to contain the following columns:

- `filename`: Image filename (path relative to data_root)
- `split`: Dataset split ('train', 'valid', 'test')
- `label`: Label information (optional, for auxiliary tasks)
- `org_caption`: Original report text

Example:

```csv
filename,split,label,org_caption
images/patient1/study1.jpg,train,"[1,0,1,0,...]","No acute cardiopulmonary process..."
images/patient1/study2.jpg,valid,"[0,1,0,1,...]","Mild cardiomegaly..."
```

### 2. Directory Structure

Recommended data directory structure:

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

## Quick Start

### 1. Train Full Model

```bash
python train_otlga.py
```

### 2. Test Model

```bash
python test_otlga.py
```

### 3. Run Ablation Experiments

```bash
# Train ablation experiment
python train_ablation.py --config baseline

# Test ablation experiment
python test_ablation.py --config baseline
```

## FAQ

### Q: How to modify model parameters?

A: Modify model initialization parameters in `train_otlga.py` or `train_ablation.py`:

```python
model = OTLGAModel(
    vit_type='vit_base',  # Can be changed to 'vit_large'
    freeze_vit=False,
    freeze_layers=0,
    c_embed_dim=256  # Can modify embedding dimension
)
```

### Q: How to adjust training hyperparameters?

A: Modify in the training script:

```python
batch_size = 64      # Batch size
epochs = 30          # Training epochs
lr = 5e-5            # Learning rate
weight_decay = 0.01  # Weight decay
```

### Q: What to do if out of memory?

A: You can:
1. Reduce `batch_size`
2. Reduce `img_size` (e.g., from 224 to 192)
3. Use gradient accumulation

### Q: How to use pre-trained models?

A: Specify checkpoint path in the test script:

```python
checkpoint_path = "/path/to/your/pretrained_model.pth"
```

## Technical Support

If you have questions, please submit an Issue or contact the authors.
