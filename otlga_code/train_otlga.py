import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from otlga_dataset import OTLGADataset
from otlga_model import OTLGAModel
from modules import UncertaintyAuxiliary
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

class SimpleSentenceContrastive(nn.Module):
    """简化的句子级对比损失"""
    def __init__(self, temperature=0.5):  # 增大温度避免数值不稳定
        super().__init__()
        self.temperature = temperature
    
    def forward(self, v_features, t_features):
        """
        v_features: [B, d] 图像全局特征
        t_features: [B, seq_len, d] 文本序列特征
        """
        t_global = t_features[:, 0, :]  # [B, d]
        
        # 归一化
        v_features = F.normalize(v_features, p=2, dim=-1)
        t_global = F.normalize(t_global, p=2, dim=-1)
        
        # 计算相似度矩阵
        logits = torch.matmul(v_features, t_global.t()) / self.temperature
        
        # 对比学习损失
        labels = torch.arange(len(v_features)).to(logits.device)
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.t(), labels)
        
        return (loss_v2t + loss_t2v) / 2

def train():
    # --- 配置参数 ---
    data_root = "/root/autodl-fs/MIMIC"
    csv_path = os.path.join(data_root, "merged_dataset/merged_dataset_real.csv")
    save_dir = "/root/autodl-fs/MIMIC/research_implementation/checkpoints_real_fixed"
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64  # 增加负样本数量
    epochs = 30      # 增加训练轮数
    lr = 5e-5        # 降低学习率避免overshooting
    
    print("=" * 70)
    print("训练配置（修复版 - 真实MIMIC-CXR数据）")
    print("=" * 70)
    print(f"数据集: {csv_path}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"学习率: {lr}")
    print(f"设备: {device}")
    print(f"保存目录: {save_dir}")
    
    # --- 数据准备 ---
    print("\n" + "=" * 70)
    print("加载数据集")
    print("=" * 70)
    
    train_dataset = OTLGADataset(
        data_root=data_root,
        csv_path=csv_path,
        split='train',
        img_size=224,
        is_multiview=False
    )
    
    val_dataset = OTLGADataset(
        data_root=data_root,
        csv_path=csv_path,
        split='valid',
        img_size=224,
        is_multiview=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # --- 模型初始化 ---
    print("\n" + "=" * 70)
    print("初始化模型")
    print("=" * 70)
    
    model = OTLGAModel(
        vit_type='vit_base',
        freeze_vit=False,
        freeze_layers=0,  # 完全解冻ViT
        c_embed_dim=256,
        bert_model='base'
    ).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params/1e6:.2f}M")
    print(f"可训练参数: {trainable_params/1e6:.2f}M")
    
    # --- 损失函数 ---
    sent_contrast = SimpleSentenceContrastive(temperature=0.5).to(device)  # 增大温度
    # cls_loss_fn = UncertaintyAuxiliary(d=256, num_labels=11).to(device)
    
    # --- 优化器 ---
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # --- 学习率调度器 ---
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    
    # --- 训练策略（方案A：强化对比学习）---
    lambda_ot = 0.1
    lambda_sent = 0.3  # 辅助细粒度对齐
    lambda_cls = 0.0
    
    print("\n训练策略（方案A - 强化对比学习）:")
    print(f"  批次大小: {batch_size} (更多负样本)")
    print(f"  训练轮数: {epochs} (充分训练)")
    print(f"  学习率: {lr} (加速收敛)")
    print(f"\n损失权重:")
    print(f"  全局对比损失权重: 1.0 (主导)")
    print(f"  OT对齐损失权重: {lambda_ot}")
    print(f"  句子级对比损失权重: {lambda_sent}")
    print(f"  特征多样性正则化: 0.01 (防止特征崩溃)")
    print(f"  不确定性学习权重: {lambda_cls} (禁用)")
    print(f"\n预期损失分布: ITC≈70% | Sent≈25% | OT≈5% | Div≈0.5%")
    
    # --- 训练循环 ---
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        train_total_loss = 0
        train_itc_loss = 0
        train_ot_loss = 0
        train_sent_loss = 0
        train_diversity_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            images = batch['image'].to(device)
            texts = batch['text']
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            v_final, t_final, ot_loss, t_fused = model(images, texts, is_multiview=False)
            
            # 1. 全局对比损失 (ITC)
            logits = torch.matmul(v_final, t_final.t()) / model.temp
            target = torch.arange(len(v_final)).to(device)
            loss_itc = (
                F.cross_entropy(logits, target) +
                F.cross_entropy(logits.t(), target)
            ) / 2
            
            # 2. OT对齐损失
            loss_ot = ot_loss.mean()
            
            # 3. 句子级对比损失
            loss_sent = sent_contrast(v_final, t_fused)
            
            # 4. 特征多样性正则化（防止特征崩溃）
            # 鼓励不同样本的特征有足够的差异
            v_sim_matrix = torch.matmul(v_final, v_final.t())
            t_sim_matrix = torch.matmul(t_final, t_final.t())
            # 提取上三角矩阵（排除对角线）
            v_sim_upper = v_sim_matrix[torch.triu(torch.ones_like(v_sim_matrix, dtype=torch.bool), diagonal=1)]
            t_sim_upper = t_sim_matrix[torch.triu(torch.ones_like(t_sim_matrix, dtype=torch.bool), diagonal=1)]
            # 惩罚过高的相似度（鼓励特征多样性）
            lambda_diversity = 0.01
            loss_diversity = (v_sim_upper.abs().mean() + t_sim_upper.abs().mean()) * lambda_diversity
            
            # 总损失
            loss = loss_itc + lambda_ot * loss_ot + lambda_sent * loss_sent + loss_diversity
            
            loss.backward()
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_total_loss += loss.item()
            train_itc_loss += loss_itc.item()
            train_ot_loss += loss_ot.item()
            train_sent_loss += loss_sent.item()
            train_diversity_loss += loss_diversity.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'itc': f'{loss_itc.item():.3f}',
                'ot': f'{loss_ot.item():.3f}',
                'sent': f'{loss_sent.item():.3f}',
                'div': f'{loss_diversity.item():.4f}'
            })
        
        avg_train_loss = train_total_loss / len(train_loader)
        avg_train_itc = train_itc_loss / len(train_loader)
        avg_train_ot = train_ot_loss / len(train_loader)
        avg_train_sent = train_sent_loss / len(train_loader)
        avg_train_diversity = train_diversity_loss / len(train_loader)
        
        # --- 验证阶段 ---
        model.eval()
        val_total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in pbar:
                images = batch['image'].to(device)
                texts = batch['text']
                labels = batch['labels'].to(device)
                
                v_final, t_final, ot_loss, t_fused = model(images, texts, is_multiview=False)
                
                logits = torch.matmul(v_final, t_final.t()) / model.temp
                target = torch.arange(len(v_final)).to(device)
                loss_itc = (
                    F.cross_entropy(logits, target) +
                    F.cross_entropy(logits.t(), target)
                ) / 2
                
                loss_ot = ot_loss.mean()
                loss_sent = sent_contrast(v_final, t_fused)
                
                loss = loss_itc + lambda_ot * loss_ot + lambda_sent * loss_sent
                
                val_total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_total_loss / len(val_loader)
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} (ITC:{avg_train_itc:.3f} | OT:{avg_train_ot:.3f} | Sent:{avg_train_sent:.3f} | Div:{avg_train_diversity:.4f})")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  学习率: {current_lr:.2e}")
        
        # --- 保存最佳模型 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(save_dir, "best_model.pth")
            # 只保存模型权重和关键信息（不保存optimizer以节省空间和inode）
            try:
                # 如果旧文件存在，先删除以释放inode
                if os.path.exists(best_model_path):
                    try:
                        os.remove(best_model_path)
                    except:
                        pass  # 如果删除失败，继续尝试覆盖
                
                # 只保存必要的状态以节省空间
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    # 不保存optimizer_state_dict以节省空间和inode
                }, best_model_path, _use_new_zipfile_serialization=False)
                print(f"  ✓ 保存最佳模型 (Val Loss: {best_val_loss:.4f})")
            except Exception as e:
                print(f"  ⚠️  保存模型失败: {e}")
                try:
                    backup_path = f"/tmp/best_model_epoch_{epoch+1}.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'best_val_loss': best_val_loss,
                    }, backup_path, _use_new_zipfile_serialization=False)
                    print(f"  ⚠️  已保存到临时目录: {backup_path}")
                    print(f"  ⚠️  注意：临时文件在重启后会丢失，请及时复制到持久化目录")
                except Exception as e2:
                    print(f"  ❌ 所有保存尝试都失败: {e2}")
                    print(f"  ⚠️  建议：清理磁盘空间或inode后重新训练")
    
    print("\n" + "=" * 70)
    print(f"训练完成！最佳验证损失: {best_val_loss:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    train()

