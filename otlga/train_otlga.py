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
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, v_features, t_features):
        
        t_global = t_features[:, 0, :]


        v_features = F.normalize(v_features, p=2, dim=-1)
        t_global = F.normalize(t_global, p=2, dim=-1)


        logits = torch.matmul(v_features, t_global.t()) / self.temperature


        labels = torch.arange(len(v_features)).to(logits.device)
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.t(), labels)

        return (loss_v2t + loss_t2v) / 2

def train():

    data_root = "/root/autodl-fs/MIMIC"
    csv_path = os.path.join(data_root, "merged_dataset/merged_dataset_real.csv")
    save_dir = "/root/autodl-fs/MIMIC/research_implementation/checkpoints_real_fixed"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 30
    lr = 5e-5

    print("=" * 70)
    print("Training Configuration (Fixed - Real MIMIC-CXR Data)")
    print("=" * 70)
    print(f"Dataset: {csv_path}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {lr}")
    print(f"Device: {device}")
    print(f"Save Directory: {save_dir}")


    print("\n" + "=" * 70)
    print("Loading Dataset")
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

    print(f"Training Set: {len(train_dataset)} Samples")
    print(f"Validation Set: {len(val_dataset)} Samples")


    print("\n" + "=" * 70)
    print("Initializing Model")
    print("=" * 70)

    model = OTLGAModel(
        vit_type='vit_base',
        freeze_vit=False,
        freeze_layers=0,
        c_embed_dim=256,
        bert_model='base'
    ).to(device)


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params/1e6:.2f}M")
    print(f"Trainable Parameters: {trainable_params/1e6:.2f}M")


    sent_contrast = SimpleSentenceContrastive(temperature=0.5).to(device)



    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)


    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)


    lambda_ot = 0.1
    lambda_sent = 0.3
    lambda_cls = 0.0

    print("\nTraining Strategy (Plan A - Enhanced Contrastive Learning):")
    print(f"  Batch Size: {batch_size} (More Negative Samples)")
    print(f"  Epochs: {epochs} (Sufficient Training)")
    print(f"  Learning Rate: {lr} (Accelerate Convergence)")
    print(f"\nLoss Weights:")
    print(f"  Global Contrastive Loss Weight: 1.0 (Dominant)")
    print(f"  OT Alignment Loss Weight: {lambda_ot}")
    print(f"  Sentence Contrastive Loss Weight: {lambda_sent}")
    print(f"  Feature Diversity Regularization: 0.01 (Prevent Feature Collapse)")
    print(f"  Uncertainty Learning Weight: {lambda_cls} (Disabled)")
    print(f"\nExpected Loss Distribution: ITC≈70% | Sent≈25% | OT≈5% | Div≈0.5%")


    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    best_val_loss = float('inf')

    for epoch in range(epochs):

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


            v_final, t_final, ot_loss, t_fused = model(images, texts, is_multiview=False)


            logits = torch.matmul(v_final, t_final.t()) / model.temp
            target = torch.arange(len(v_final)).to(device)
            loss_itc = (
                F.cross_entropy(logits, target) +
                F.cross_entropy(logits.t(), target)
            ) / 2


            loss_ot = ot_loss.mean()


            loss_sent = sent_contrast(v_final, t_fused)



            v_sim_matrix = torch.matmul(v_final, v_final.t())
            t_sim_matrix = torch.matmul(t_final, t_final.t())

            v_sim_upper = v_sim_matrix[torch.triu(torch.ones_like(v_sim_matrix, dtype=torch.bool), diagonal=1)]
            t_sim_upper = t_sim_matrix[torch.triu(torch.ones_like(t_sim_matrix, dtype=torch.bool), diagonal=1)]

            lambda_diversity = 0.01
            loss_diversity = (v_sim_upper.abs().mean() + t_sim_upper.abs().mean()) * lambda_diversity


            loss = loss_itc + lambda_ot * loss_ot + lambda_sent * loss_sent + loss_diversity

            loss.backward()

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


        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} (ITC:{avg_train_itc:.3f} | OT:{avg_train_ot:.3f} | Sent:{avg_train_sent:.3f} | Div:{avg_train_diversity:.4f})")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(save_dir, "best_model.pth")

            try:

                if os.path.exists(best_model_path):
                    try:
                        os.remove(best_model_path)
                    except:
                        pass


                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,

                }, best_model_path, _use_new_zipfile_serialization=False)
                print(f"  ✓ Saving Best Model (Val Loss: {best_val_loss:.4f})")
            except Exception as e:
                print(f"  ⚠️  Failed to Save Model: {e}")
                try:
                    backup_path = f"/tmp/best_model_epoch_{epoch+1}.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'best_val_loss': best_val_loss,
                    }, backup_path, _use_new_zipfile_serialization=False)
                    print(f"  ⚠️  Saved to Temporary Directory: {backup_path}")
                    print(f"  ⚠️  Note: Temporary files will be lost after reboot, please copy to persistent directory in time")
                except Exception as e2:
                    print(f"  ❌ All Save Attempts Failed: {e2}")
                    print(f"  ⚠️  Suggestion: Free up disk space or inodes before retraining")

    print("\n" + "=" * 70)
    print(f"Training Complete! Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    train()

