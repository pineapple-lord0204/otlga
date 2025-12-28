# 代码结构说明

## 核心模块

### 1. 模型定义

#### `otlga_model.py`
完整的OTLGA模型实现，包含：

- **LocalGlobalAttention**: 局部-全局注意力模块
  - 输入: 局部特征 [B, n, d] 和全局特征 [B, d]
  - 输出: 增强后的局部和全局特征

- **EntropicOT**: 熵正则最优传输模块
  - 实现Sinkhorn算法进行最优传输
  - 计算图像-文本对齐矩阵

- **OTGatedFusion**: OT引导的门控融合模块
  - 使用OT对齐矩阵动态融合跨模态特征

- **OTLGAModel**: 完整模型
  - 视觉编码器 (ViT)
  - 文本编码器 (BERT)
  - 投影层
  - LGA + OT + Gated Fusion模块

#### `otlga_model_ablation.py`
消融实验模型，支持选择性启用/禁用各个模块：

- **OTLGAModelAblation**: 消融实验模型类
  - 通过配置参数控制LGA、OT、Gated Fusion的启用/禁用
  - 用于评估各个模块的贡献

### 2. 数据集

#### `otlga_dataset.py`
数据集类实现：

- **OTLGADataset**: PyTorch Dataset类
  - 图像预处理和增强
  - 文本清洗和编码
  - 支持单视角和多视角数据

- **文本处理函数**:
  - `clean_medical_text()`: 清洗医学报告文本
  - `augment_text_for_retrieval()`: 文本增强（随机丢弃句子）

- **图像处理函数**:
  - `get_transforms()`: 获取图像变换（训练/验证）

### 3. 辅助模块

#### `modules.py`
辅助损失和任务模块：

- **SentenceContrastive**: 句子级对比损失
  - 计算图像全局特征与文本token级别的对比

- **UncertaintyAuxiliary**: 不确定性辅助任务
  - 用于多标签分类的辅助任务

#### `vit_custom.py`
Vision Transformer自定义实现：

- **get_ViT()**: 获取标准ViT模型
- **create_eva_vit_g()**: 创建EVA ViT-G模型（可选）

### 4. 训练和测试脚本

#### `train_otlga.py`
完整模型的训练脚本：

- 数据加载和预处理
- 模型初始化
- 训练循环（包含验证）
- 损失计算：
  - ITC损失（Image-Text Contrastive）
  - OT损失
  - Sentence Contrastive损失
  - Uncertainty辅助损失（可选）
- 模型保存和日志记录

#### `test_otlga.py`
完整模型的测试脚本：

- 加载训练好的模型
- 提取图像和文本特征
- 计算检索指标（R@1, R@5, R@10, MedR, MeanR等）
- 保存测试结果

#### `train_ablation.py`
消融实验训练脚本：

- 支持多个配置的训练
- 配置定义在 `ABLATION_CONFIGS` 字典中
- 可以训练单个配置或所有配置

#### `test_ablation.py`
消融实验测试脚本：

- 测试指定的消融配置
- 生成对比报告
- 保存测试结果到JSON文件

## 代码流程

### 训练流程

1. **数据加载**
   ```
   OTLGADataset → DataLoader → Batch
   ```

2. **前向传播**
   ```
   Image → ViT → v_features
   Text → BERT → t_features
   v_features, t_features → Projection → v_proj, t_proj
   v_proj, t_proj → LGA → v_lga, t_lga
   v_lga, t_lga → OT → ot_loss, T_fused
   v_lga, T_fused → Gated Fusion → v_final, t_final
   ```

3. **损失计算**
   ```
   ITC Loss (v_final, t_final)
   OT Loss (from OT module)
   Sentence Contrastive Loss (v_final, T_fused)
   Total Loss = ITC + λ_ot * OT + λ_sent * Sent + ...
   ```

4. **反向传播和优化**

### 测试流程

1. **特征提取**
   - 遍历测试集，提取所有图像和文本特征

2. **相似度计算**
   - 计算图像-文本相似度矩阵

3. **检索评估**
   - 对每个查询，计算top-K检索结果
   - 计算各种检索指标

## 配置参数

### 模型参数

- `vit_type`: ViT类型 ('vit_base', 'vit_large')
- `freeze_vit`: 是否冻结ViT参数
- `freeze_layers`: 冻结的层数
- `c_embed_dim`: 共同嵌入空间维度（默认256）

### 训练参数

- `batch_size`: 批次大小（默认64）
- `epochs`: 训练轮数（默认30）
- `lr`: 学习率（默认5e-5）
- `weight_decay`: 权重衰减（默认0.01）

### 损失权重

- `lambda_ot`: OT损失权重（默认0.5）
- `lambda_sent`: 句子级对比损失权重（默认0.3）
- `lambda_uncertainty`: 不确定性辅助损失权重（默认0.1）

## 数据格式要求

### CSV文件格式

| 列名 | 说明 | 必需 |
|------|------|------|
| `filename` | 图像文件路径（相对于data_root） | ✅ |
| `split` | 数据集划分 ('train'/'valid'/'test') | ✅ |
| `org_caption` | 原始报告文本 | ✅ |
| `label` | 标签（用于辅助任务） | ❌ |

### 图像格式

- 支持常见图像格式（JPG, PNG等）
- 默认输入尺寸：224x224
- 归一化：ImageNet标准（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）

### 文本格式

- 使用BERT tokenizer进行编码
- 最大长度：256 tokens（可配置）
- 自动处理文本清洗和预处理

## 扩展和修改

### 添加新的损失函数

1. 在 `modules.py` 中定义新的损失类
2. 在训练脚本中初始化损失函数
3. 在训练循环中添加损失计算

### 修改模型架构

1. 在 `otlga_model.py` 中修改模型结构
2. 确保前向传播返回格式一致
3. 更新测试脚本中的特征提取逻辑（如需要）

### 添加新的数据集

1. 继承 `OTLGADataset` 类
2. 实现必要的数据加载逻辑
3. 确保返回格式与原始数据集一致

