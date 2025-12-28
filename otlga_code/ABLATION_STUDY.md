# OTLGA 消融实验

## 实验设计

本消融实验评估三个核心模块的贡献：
- **LGA (Local-Global Attention)**: 局部-全局注意力模块
- **OT (Optimal Transport)**: 熵正则最优传输模块
- **Gated Fusion**: OT引导门控融合模块

## 消融配置

| 配置名称 | LGA | OT | Gated Fusion | 说明 |
|---------|-----|----|--------------|------|
| `baseline` | ❌ | ❌ | ❌ | 基线模型（投影层+ITC损失+句子级对比损失） |
| `lga` | ✅ | ❌ | ❌ | 仅LGA模块 |
| `ot` | ❌ | ✅ | ❌ | 仅OT模块 |
| `gated_fusion` | ❌ | ✅ | ✅ | OT + Gated Fusion（Gated Fusion需要OT） |
| `lga_ot` | ✅ | ✅ | ❌ | LGA + OT |
| `lga_gated` | ✅ | ✅ | ✅ | LGA + OT + Gated Fusion |
| `ot_gated` | ❌ | ✅ | ✅ | OT + Gated Fusion |
| `full` | ✅ | ✅ | ✅ | 完整OTLGA模型 |

## 使用方法

### 1. 训练单个配置

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

### 2. 训练所有配置

```bash
python3 train_ablation.py --config all
```

### 3. 测试单个配置

```bash
python3 test_ablation.py --config baseline
python3 test_ablation.py --config full
```

### 4. 测试所有配置

```bash
python3 test_ablation.py --config all
```

## 输出文件

### 训练输出
- `checkpoints_ablation/{config_name}/best_model.pth` - 最佳模型权重
- `checkpoints_ablation/{config_name}/training_results.json` - 训练过程记录
- `checkpoints_ablation/summary.json` - 所有配置的训练总结

### 测试输出
- `checkpoints_ablation/{config_name}/test_results.json` - 测试结果详情
- `checkpoints_ablation/test_summary.json` - 所有配置的测试总结

## 实验参数

- **训练轮数**: 10 epochs
- **批次大小**: 64
- **学习率**: 5e-5
- **优化器**: AdamW (weight_decay=0.01)
- **学习率调度**: CosineAnnealingLR

## 损失权重

根据配置不同，损失权重会调整：
- **Baseline**: 仅 ITC 损失
- **+LGA**: ITC + Sentence Contrastive (λ=0.3)
- **+OT**: ITC + OT Loss (λ=0.1)
- **+Gated Fusion**: ITC + OT Loss (λ=0.1)
- **+LGA+OT**: ITC + OT Loss (λ=0.1) + Sentence Contrastive (λ=0.3)
- **Full**: ITC + OT Loss (λ=0.1) + Sentence Contrastive (λ=0.3)

## 评估指标

每个配置会计算：
- **Recall@K** (K=1, 5, 10)
- **Precision@K** (K=1, 5, 10)
- **MedR** (Median Rank)
- **MeanR** (Mean Rank)

## 预期结果

通过对比不同配置的性能，可以评估：
1. 每个模块的独立贡献
2. 模块之间的协同效应
3. 完整模型的优势




