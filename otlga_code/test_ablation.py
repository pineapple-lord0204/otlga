import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from otlga_dataset import OTLGADataset
from otlga_model_ablation import OTLGAModelAblation
import json

def compute_retrieval_metrics(similarity_matrix, return_precision=True):
    n_samples = similarity_matrix.shape[0]
    ranks = []
    
    for i in range(n_samples):
        scores = similarity_matrix[i]
        sorted_indices = np.argsort(scores)[::-1]
        rank = np.where(sorted_indices == i)[0][0]
        ranks.append(rank)
    
    r1 = 100.0 * len([r for r in ranks if r < 1]) / n_samples
    r5 = 100.0 * len([r for r in ranks if r < 5]) / n_samples
    r10 = 100.0 * len([r for r in ranks if r < 10]) / n_samples
    medr = np.median(ranks) + 1
    mean_rank = np.mean(ranks) + 1
    
    metrics = {
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
        "MedR": medr,
        "MeanR": mean_rank
    }
    
    if return_precision:
        p1_list = [1.0 if r < 1 else 0.0 for r in ranks]
        p5_list = [1.0/5 if r < 5 else 0.0 for r in ranks]
        p10_list = [1.0/10 if r < 10 else 0.0 for r in ranks]
        
        metrics["P@1"] = 100.0 * np.mean(p1_list)
        metrics["P@5"] = 100.0 * np.mean(p5_list)
        metrics["P@10"] = 100.0 * np.mean(p10_list)
    
    return metrics

@torch.no_grad()
def evaluate_ablation(model, dataloader, device, config_name):
    model.eval()
    all_v = []
    all_t = []
    
    print(f"\n正在提取特征 [{config_name}]...")
    for batch in tqdm(dataloader, desc="特征提取"):
        images = batch['image'].to(device)
        texts = batch['text']
        
        v_final, t_final, _, _, _ = model(images, texts, is_multiview=False)
        
        all_v.append(v_final.cpu())
        all_t.append(t_final.cpu())
    
    all_v = torch.cat(all_v, dim=0)
    all_t = torch.cat(all_t, dim=0)
    
    print(f"特征提取完成 [{config_name}]")
    print(f"  图像特征: {all_v.shape}")
    print(f"  文本特征: {all_t.shape}")
    
    sim_matrix = torch.matmul(all_v, all_t.t()).numpy()
    
    print(f"\n{'='*60}")
    print(f"[{config_name}] Image-to-Text 检索")
    print(f"{'='*60}")
    i2t_metrics = compute_retrieval_metrics(sim_matrix)
    for k, v in i2t_metrics.items():
        print(f"  {k}: {v:.2f}")
    
    print(f"\n{'='*60}")
    print(f"[{config_name}] Text-to-Image 检索")
    print(f"{'='*60}")
    t2i_metrics = compute_retrieval_metrics(sim_matrix.T)
    for k, v in t2i_metrics.items():
        print(f"  {k}: {v:.2f}")
    
    avg_metrics = {
        "Avg R@1": (i2t_metrics["R@1"] + t2i_metrics["R@1"]) / 2,
        "Avg R@5": (i2t_metrics["R@5"] + t2i_metrics["R@5"]) / 2,
        "Avg R@10": (i2t_metrics["R@10"] + t2i_metrics["R@10"]) / 2,
        "Avg P@1": (i2t_metrics["P@1"] + t2i_metrics["P@1"]) / 2,
        "Avg P@5": (i2t_metrics["P@5"] + t2i_metrics["P@5"]) / 2,
        "Avg P@10": (i2t_metrics["P@10"] + t2i_metrics["P@10"]) / 2,
    }
    
    print(f"\n{'='*60}")
    print(f"[{config_name}] 平均检索性能")
    print(f"{'='*60}")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.2f}")
    
    return {
        "i2t": i2t_metrics,
        "t2i": t2i_metrics,
        "average": avg_metrics
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='all',
                       choices=['all', 'baseline', 'lga', 'ot', 'gated_fusion', 
                               'lga_ot', 'lga_gated', 'ot_gated', 'full'],
                       help='要测试的消融实验配置')
    args = parser.parse_args()
    
    data_root = "/root/autodl-fs/MIMIC"
    csv_path = os.path.join(data_root, "merged_dataset/merged_dataset_real.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ABLATION_CONFIGS = {
        'baseline': {'name': 'Baseline', 'use_lga': False, 'use_ot': False, 'use_gated_fusion': False},
        'lga': {'name': '+LGA', 'use_lga': True, 'use_ot': False, 'use_gated_fusion': False},
        'ot': {'name': '+OT', 'use_lga': False, 'use_ot': True, 'use_gated_fusion': False},
        'gated_fusion': {'name': '+Gated Fusion', 'use_lga': False, 'use_ot': True, 'use_gated_fusion': True},
        'lga_ot': {'name': '+LGA+OT', 'use_lga': True, 'use_ot': True, 'use_gated_fusion': False},
        'lga_gated': {'name': '+LGA+Gated Fusion', 'use_lga': True, 'use_ot': True, 'use_gated_fusion': True},
        'ot_gated': {'name': '+OT+Gated Fusion', 'use_lga': False, 'use_ot': True, 'use_gated_fusion': True},
        'full': {'name': 'Full (OTLGA)', 'use_lga': True, 'use_ot': True, 'use_gated_fusion': True},
    }
    
    if args.config == 'all':
        configs_to_test = ABLATION_CONFIGS
    else:
        configs_to_test = {args.config: ABLATION_CONFIGS[args.config]}
    
    test_dataset = OTLGADataset(
        data_root=data_root,
        csv_path=csv_path,
        split='test',
        img_size=224,
        is_multiview=False
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print("=" * 70)
    print("消融实验测试")
    print("=" * 70)
    print(f"测试集: {len(test_dataset)} 样本")
    
    all_results = {}
    
    for config_name, config in configs_to_test.items():
        checkpoint_path = f"/root/autodl-fs/MIMIC/otlga/checkpoints_ablation/{config_name}/best_model.pth"
        
        if not os.path.exists(checkpoint_path):
            print(f"\n⚠️  配置 {config_name} 的模型不存在: {checkpoint_path}")
            print("   请先运行训练: python3 train_ablation.py --config {config_name}")
            continue
        
        print(f"\n{'='*70}")
        print(f"测试配置: {config['name']} ({config_name})")
        print(f"{'='*70}")
        
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
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ 成功加载模型: {checkpoint_path}")
            if 'epoch' in checkpoint:
                print(f"  训练轮数: Epoch {checkpoint['epoch']}")
            if 'best_val_loss' in checkpoint:
                print(f"  最佳验证损失: {checkpoint['best_val_loss']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✓ 成功加载模型: {checkpoint_path}")
        
        results = evaluate_ablation(model, test_loader, device, config['name'])
        all_results[config_name] = results
        
        results_path = f"/root/autodl-fs/MIMIC/otlga/checkpoints_ablation/{config_name}/test_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ 测试结果已保存到: {results_path}")
    
    summary_path = "/root/autodl-fs/MIMIC/otlga/checkpoints_ablation/test_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("消融实验测试总结")
    print("=" * 70)
    print(f"\n{'配置':<25} {'Avg R@1':<12} {'Avg R@5':<12} {'Avg R@10':<12}")
    print("-" * 70)
    
    for config_name, results in all_results.items():
        if results and 'average' in results:
            avg_r1 = results['average'].get('Avg R@1', 0)
            avg_r5 = results['average'].get('Avg R@5', 0)
            avg_r10 = results['average'].get('Avg R@10', 0)
            config_display = ABLATION_CONFIGS[config_name]['name']
            print(f"{config_display:<25} {avg_r1:>10.2f}%  {avg_r5:>10.2f}%  {avg_r10:>10.2f}%")
    
    print(f"\n完整结果已保存到: {summary_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()



