# Code Structure Documentation

## Core Modules

### 1. Model Definitions

#### `otlga_model.py`
Complete OTLGA model implementation, including:

- **LocalGlobalAttention**: Local-Global Attention module
  - Input: Local features [B, n, d] and global feature [B, d]
  - Output: Enhanced local and global features

- **EntropicOT**: Entropic Regularized Optimal Transport module
  - Implements Sinkhorn algorithm for optimal transport
  - Computes image-text alignment matrix

- **OTGatedFusion**: OT-guided Gated Fusion module
  - Dynamically fuses cross-modal features using OT alignment matrix

- **OTLGAModel**: Complete model
  - Visual encoder (ViT)
  - Text encoder (BERT)
  - Projection layers
  - LGA + OT + Gated Fusion modules

#### `otlga_model_ablation.py`
Ablation experiment model supporting selective enabling/disabling of modules:

- **OTLGAModelAblation**: Ablation experiment model class
  - Controls enabling/disabling of LGA, OT, Gated Fusion via configuration parameters
  - Used to evaluate the contribution of each module

### 2. Dataset

#### `otlga_dataset.py`
Dataset class implementation:

- **OTLGADataset**: PyTorch Dataset class
  - Image preprocessing and augmentation
  - Text cleaning and encoding
  - Supports single-view and multi-view data

- **Text Processing Functions**:
  - `clean_medical_text()`: Cleans medical report text
  - `augment_text_for_retrieval()`: Text augmentation (random sentence dropping)

- **Image Processing Functions**:
  - `get_transforms()`: Gets image transforms (training/validation)

### 3. Auxiliary Modules

#### `modules.py`
Auxiliary loss and task modules:

- **SentenceContrastive**: Sentence-level contrastive loss
  - Computes contrast between global image features and text token-level features

- **UncertaintyAuxiliary**: Uncertainty auxiliary task
  - Auxiliary task for multi-label classification

#### `vit_custom.py`
Custom Vision Transformer implementation:

- **get_ViT()**: Gets standard ViT model
- **create_eva_vit_g()**: Creates EVA ViT-G model (optional)

### 4. Training and Testing Scripts

#### `train_otlga.py`
Training script for complete model:

- Data loading and preprocessing
- Model initialization
- Training loop (including validation)
- Loss computation:
  - ITC loss (Image-Text Contrastive)
  - OT loss
  - Sentence Contrastive loss
  - Uncertainty auxiliary loss (optional)
- Model saving and logging

#### `test_otlga.py`
Testing script for complete model:

- Load trained model
- Extract image and text features
- Compute retrieval metrics (R@1, R@5, R@10, MedR, MeanR, etc.)
- Save test results

#### `train_ablation.py`
Ablation experiment training script:

- Supports training multiple configurations
- Configurations defined in `ABLATION_CONFIGS` dictionary
- Can train single configuration or all configurations

#### `test_ablation.py`
Ablation experiment testing script:

- Tests specified ablation configuration
- Generates comparison reports
- Saves test results to JSON file

## Code Flow

### Training Flow

1. **Data Loading**
   ```
   OTLGADataset → DataLoader → Batch
   ```

2. **Forward Propagation**
   ```
   Image → ViT → v_features
   Text → BERT → t_features
   v_features, t_features → Projection → v_proj, t_proj
   v_proj, t_proj → LGA → v_lga, t_lga
   v_lga, t_lga → OT → ot_loss, T_fused
   v_lga, T_fused → Gated Fusion → v_final, t_final
   ```

3. **Loss Computation**
   ```
   ITC Loss (v_final, t_final)
   OT Loss (from OT module)
   Sentence Contrastive Loss (v_final, T_fused)
   Total Loss = ITC + λ_ot * OT + λ_sent * Sent + ...
   ```

4. **Backward Propagation and Optimization**

### Testing Flow

1. **Feature Extraction**
   - Iterate through test set, extract all image and text features

2. **Similarity Computation**
   - Compute image-text similarity matrix

3. **Retrieval Evaluation**
   - For each query, compute top-K retrieval results
   - Compute various retrieval metrics

## Configuration Parameters

### Model Parameters

- `vit_type`: ViT type ('vit_base', 'vit_large')
- `freeze_vit`: Whether to freeze ViT parameters
- `freeze_layers`: Number of layers to freeze
- `c_embed_dim`: Common embedding space dimension (default 256)

### Training Parameters

- `batch_size`: Batch size (default 64)
- `epochs`: Training epochs (default 30)
- `lr`: Learning rate (default 5e-5)
- `weight_decay`: Weight decay (default 0.01)

### Loss Weights

- `lambda_ot`: OT loss weight (default 0.5)
- `lambda_sent`: Sentence-level contrastive loss weight (default 0.3)
- `lambda_uncertainty`: Uncertainty auxiliary loss weight (default 0.1)

## Data Format Requirements

### CSV File Format

| Column Name | Description | Required |
|-------------|-------------|----------|
| `filename` | Image file path (relative to data_root) | ✅ |
| `split` | Dataset split ('train'/'valid'/'test') | ✅ |
| `org_caption` | Original report text | ✅ |
| `label` | Labels (for auxiliary tasks) | ❌ |

### Image Format

- Supports common image formats (JPG, PNG, etc.)
- Default input size: 224x224
- Normalization: ImageNet standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Text Format

- Uses BERT tokenizer for encoding
- Maximum length: 256 tokens (configurable)
- Automatically handles text cleaning and preprocessing

## Extension and Modification

### Adding New Loss Functions

1. Define new loss class in `modules.py`
2. Initialize loss function in training script
3. Add loss computation in training loop

### Modifying Model Architecture

1. Modify model structure in `otlga_model.py`
2. Ensure forward propagation return format is consistent
3. Update feature extraction logic in test script (if needed)

### Adding New Datasets

1. Inherit from `OTLGADataset` class
2. Implement necessary data loading logic
3. Ensure return format is consistent with original dataset
