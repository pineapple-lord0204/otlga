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
from otlga_model_ablation import OTLGAModelAblation
from modules import UncertaintyAuxiliary
import warnings
import logging
import json

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

ABLATION_CONFIGS = {
    'baseline': {
        'name': 'Baseline + Sentence Loss',
        'use_lga': False,
        'use_ot': False,
        'use_gated_fusion': False,
        'lambda_ot': 0.0,
        'lambda_sent': 0.3,
    },
    'lga': {
        'name': '+LGA',
        'use_lga': True,
        'use_ot': False,
        'use_gated_fusion': False,
        'lambda_ot': 0.0,
        'lambda_sent': 0.3,
    },
    'ot': {
        'name': '+OT',
        'use_lga': False,
        'use_ot': True,
        'use_gated_fusion': False,
        'lambda_ot': 0.1,
        'lambda_sent': 0.0,
    },
    'gated_fusion': {
        'name': '+Gated Fusion',
        'use_lga': False,
        'use_ot': True,
        'use_gated_fusion': True,
        'lambda_ot': 0.1,
        'lambda_sent': 0.0,
    },
    'lga_ot': {
        'name': '+LGA+OT',
        'use_lga': True,
        'use_ot': True,
        'use_gated_fusion': False,
        'lambda_ot': 0.1,
        'lambda_sent': 0.3,
    },
    'lga_gated': {
        'name': '+LGA+Gated Fusion',
        'use_lga': True,
        'use_ot': True,
        'use_gated_fusion': True,
        'lambda_ot': 0.1,
        'lambda_sent': 0.3,
    },
    'ot_gated': {
        'name': '+OT+Gated Fusion',
        'use_lga': False,
        'use_ot': True,
        'use_gated_fusion': True,
        'lambda_ot': 0.1,
        'lambda_sent': 0.0,
    },
    'full': {
        'name': 'Full (OTLGA)',
        'use_lga': True,
        'use_ot': True,
        'use_gated_fusion': True,
        'lambda_ot': 0.1,
        'lambda_sent': 0.3,
    },
}

def train_ablation(config_name, config, resume_from=None, epochs=10):
    data_root = "/root/autodl-fs/MIMIC"
    csv_path = os.path.join(data_root, "merged_dataset/merged_dataset_real.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    lr = 5e-5

    print("\n" + "=" * 70)
    print(f"Ablation Experiment: {config['name']} ({config_name})")
    print("=" * 70)
    print(f"Configuration: LGA={config['use_lga']}, OT={config['use_ot']}, Gated Fusion={config['use_gated_fusion']}")

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

    model = OTLGAModelAblation(
        vit_type='vit_base',
        freeze_vit=False,
        freeze_layers=0,
        c_embed_dim=256,
        bert_model='base',
        use_lga=config['use_lga'],
        use_ot=config['use_ot'],
        use_gated_fusion=config['use_gated_fusion']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params/1e6:.2f}M")
    print(f"Trainable Parameters: {trainable_params/1e6:.2f}M")

    sent_contrast = SimpleSentenceContrastive(temperature=0.5).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    lambda_ot = config['lambda_ot']
    lambda_sent = config['lambda_sent']
    lambda_cls = 0.0

    save_dir = f"/root/autodl-fs/MIMIC/otlga/checkpoints_ablation/{config_name}"
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming Training from Checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            print(f"  Loaded Model State (Epoch: {start_epoch}, Best Val Loss: {best_val_loss:.4f})")
        else:
            model.load_state_dict(checkpoint)
            print(f"  Loaded Model State")

    total_epochs = start_epoch + epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=lr*0.01)

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        train_total_loss = 0.0
        train_itc_loss = 0.0
        train_ot_loss = 0.0
        train_sent_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            images = batch['image'].to(device)
            texts = batch['text']

            optimizer.zero_grad()

            v_final, t_final, ot_loss, t_fused, attention_mask = model(images, texts, is_multiview=False)

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

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'itc': f'{loss_itc.item():.3f}',
                'ot': f'{loss_ot.item():.3f}',
                'sent': f'{loss_sent.item():.3f}',
            })

        scheduler.step()

        avg_train_loss = train_total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = batch['image'].to(device)
                texts = batch['text']

                v_final, t_final, ot_loss, t_fused, attention_mask = model(images, texts, is_multiview=False)

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

        avg_val_loss = val_total_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"\nEpoch {epoch+1}/{start_epoch + epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} (ITC: {train_itc_loss/len(train_loader):.3f} | OT: {train_ot_loss/len(train_loader):.3f} | Sent: {train_sent_loss/len(train_loader):.3f})")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(save_dir, "best_model.pth")

            if os.path.exists(best_model_path):
                try:
                    os.remove(best_model_path)
                except:
                    pass

            try:
                tmp_model_path = best_model_path + ".tmp"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'config_name': config_name
                }, tmp_model_path)
                os.replace(tmp_model_path, best_model_path)
                print(f"  ✓ Saving Best Model (Val Loss: {best_val_loss:.4f})")
            except Exception as e:
                print(f"  ⚠️  Failed to Save Model: {e}")

    results = {
        'config_name': config_name,
        'config': config,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    results_path = os.path.join(save_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining Complete! Best Validation Loss: {best_val_loss:.4f}")
    print(f"Results saved to: {results_path}")

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='all',
                       choices=['all'] + list(ABLATION_CONFIGS.keys()),
                       help='Ablation experiment configuration to run')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint (checkpoint path)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Epochs (default: 30, consistent with normal training)')
    args = parser.parse_args()

    if args.config == 'all':
        configs_to_run = ABLATION_CONFIGS
    else:
        configs_to_run = {args.config: ABLATION_CONFIGS[args.config]}

    all_results = {}
    for config_name, config in configs_to_run.items():
        try:
            results = train_ablation(config_name, config, resume_from=args.resume, epochs=args.epochs)
            all_results[config_name] = results
        except Exception as e:
            print(f"\n❌ Configuration {config_name}  training failed: {e}")
            import traceback
            traceback.print_exc()

    summary_path = "/root/autodl-fs/MIMIC/otlga/checkpoints_ablation/summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("Ablation Experiment Summary")
    print("=" * 70)
    print(f"{'Configuration':<20} {'Best Validation Loss':<15}")
    print("-" * 70)
    for config_name, results in all_results.items():
        if results:
            print(f"{config_name:<20} {results['best_val_loss']:<15.4f}")
    print(f"\nComplete Results Saved to: {summary_path}")

if __name__ == "__main__":
    main()


