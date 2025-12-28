# OTLGA Ablation Study

## Experiment Design

This ablation study evaluates the contribution of three core modules:
- **LGA (Local-Global Attention)**: Local-Global Attention module
- **OT (Optimal Transport)**: Entropic Regularized Optimal Transport module
- **Gated Fusion**: OT-guided Gated Fusion module

## Ablation Configurations

| Configuration | LGA | OT | Gated Fusion | Description |
|---------------|-----|----|--------------|-------------|
| `baseline` | ❌ | ❌ | ❌ | Baseline model (projection layers + ITC loss + sentence-level contrastive loss) |
| `lga` | ✅ | ❌ | ❌ | LGA module only |
| `ot` | ❌ | ✅ | ❌ | OT module only |
| `gated_fusion` | ❌ | ✅ | ✅ | OT + Gated Fusion (Gated Fusion requires OT) |
| `lga_ot` | ✅ | ✅ | ❌ | LGA + OT |
| `lga_gated` | ✅ | ✅ | ✅ | LGA + OT + Gated Fusion |
| `ot_gated` | ❌ | ✅ | ✅ | OT + Gated Fusion |
| `full` | ✅ | ✅ | ✅ | Full OTLGA model |

## Usage

### 1. Train Single Configuration

```bash
cd /root/autodl-fs/MIMIC/otlga
python3 train_ablation.py --config baseline
python3 train_ablation.py --config lga
python3 train_ablation.py --config ot
python3 train_ablation.py --config gated_fusion
python3 train_ablation.py --config lga_ot
python3 train_ablation.py --config lga_gated
python3 train_ablation.py --config ot_gated
python3 train_ablation.py --config full
```

### 2. Train All Configurations

```bash
python3 train_ablation.py --config all
```

### 3. Test Single Configuration

```bash
python3 test_ablation.py --config baseline
python3 test_ablation.py --config full
```

### 4. Test All Configurations

```bash
python3 test_ablation.py --config all
```

## Output Files

### Training Outputs

- `checkpoints_ablation/{config_name}/best_model.pth` - Best model weights
- `checkpoints_ablation/{config_name}/training_results.json` - Training process records
- `checkpoints_ablation/summary.json` - Training summary for all configurations

### Testing Outputs

- `checkpoints_ablation/{config_name}/test_results.json` - Detailed test results
- `checkpoints_ablation/test_summary.json` - Test summary for all configurations

## Experiment Parameters

- **Training Epochs**: 30 epochs
- **Batch Size**: 64
- **Learning Rate**: 5e-5
- **Optimizer**: AdamW (weight_decay=0.01)
- **Learning Rate Scheduler**: CosineAnnealingLR

## Loss Weights

Loss weights are adjusted according to configuration:
- **Baseline**: ITC loss + Sentence Contrastive loss (lambda_sent=0.3)
- **Other configurations**: ITC loss + OT loss (lambda_ot) + Sentence Contrastive loss (lambda_sent)

## Evaluation Metrics

Each configuration computes:
- R@1, R@5, R@10 (Recall@K)
- MedR, MeanR (Median/Mean Rank)
- P@1, P@5, P@10 (Precision@K)
- Image-to-Text and Text-to-Image retrieval metrics

## Expected Results

By comparing the performance of different configurations, we can evaluate:
1. The independent contribution of each module
2. Synergistic effects between modules
3. Advantages of the full model
