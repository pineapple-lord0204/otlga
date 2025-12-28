import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from otlga_dataset import OTLGADataset
from otlga_model import OTLGAModel
import json

def compute_retrieval_metrics(similarity_matrix, return_precision=True):
    """
    计算检索指标，包括 Recall 和 Precision
    similarity_matrix: [N_img, N_txt] 相似度矩阵
    """
    n_samples = similarity_matrix.shape[0]
    ranks = []
    
    # Image to Text (I2T) - 找到正确匹配的排名
    for i in range(n_samples):
        scores = similarity_matrix[i]
        sorted_indices = np.argsort(scores)[::-1]  # 降序
        rank = np.where(sorted_indices == i)[0][0]
        ranks.append(rank)
    
    # Recall@K: 正确答案在前 K 个结果中的比例
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
    
    # 计算 Precision@K
    if return_precision:
        # Precision@K: 前 K 个结果中正确答案的比例
        # 所以 P@K = 1/K if 正确答案在前K个，否则 0
        p1_list = [1.0 if r < 1 else 0.0 for r in ranks]
        p5_list = [1.0/5 if r < 5 else 0.0 for r in ranks]
        p10_list = [1.0/10 if r < 10 else 0.0 for r in ranks]
        
        metrics["P@1"] = 100.0 * np.mean(p1_list)
        metrics["P@5"] = 100.0 * np.mean(p5_list)
        metrics["P@10"] = 100.0 * np.mean(p10_list)
    
    return metrics

@torch.no_grad()
def evaluate(model, dataloader, device, dataset_name="All"):
    """
    在测试集上进行双向检索评估
    """
    model.eval()
    all_v = []
    all_t = []
    all_study_ids = []
    
    print(f"\n正在提取特征 [{dataset_name}]...")
    for batch in tqdm(dataloader, desc="特征提取"):
        images = batch['image'].to(device)
        texts = batch['text']
        study_ids = batch['study_id']
        
        v_final, t_final, _, _ = model(images, texts, is_multiview=False)
        
        all_v.append(v_final.cpu())
        all_t.append(t_final.cpu())
        all_study_ids.extend(study_ids)
    
    all_v = torch.cat(all_v, dim=0)
    all_t = torch.cat(all_t, dim=0)
    
    print(f"特征提取完成 [{dataset_name}]")
    print(f"  图像特征: {all_v.shape}")
    print(f"  文本特征: {all_t.shape}")
    
    # 计算相似度矩阵
    print(f"计算相似度矩阵 [{dataset_name}]...")
    sim_matrix = torch.matmul(all_v, all_t.t()).numpy()
    
    # 计算双向检索指标
    print(f"\n{'='*60}")
    print(f"[{dataset_name}] Image-to-Text 检索")
    print(f"{'='*60}")
    i2t_metrics = compute_retrieval_metrics(sim_matrix)
    for k, v in i2t_metrics.items():
        print(f"  {k}: {v:.2f}")
    
    print(f"\n{'='*60}")
    print(f"[{dataset_name}] Text-to-Image 检索")
    print(f"{'='*60}")
    t2i_metrics = compute_retrieval_metrics(sim_matrix.T)
    for k, v in t2i_metrics.items():
        print(f"  {k}: {v:.2f}")
    
    # 计算平均指标
    print(f"\n{'='*60}")
    print(f"[{dataset_name}] 平均检索性能")
    print(f"{'='*60}")
    avg_metrics = {
        "Avg R@1": (i2t_metrics["R@1"] + t2i_metrics["R@1"]) / 2,
        "Avg R@5": (i2t_metrics["R@5"] + t2i_metrics["R@5"]) / 2,
        "Avg R@10": (i2t_metrics["R@10"] + t2i_metrics["R@10"]) / 2,
        "Avg P@1": (i2t_metrics["P@1"] + t2i_metrics["P@1"]) / 2,
        "Avg P@5": (i2t_metrics["P@5"] + t2i_metrics["P@5"]) / 2,
        "Avg P@10": (i2t_metrics["P@10"] + t2i_metrics["P@10"]) / 2,
    }
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.2f}")
    
    return {
        "i2t": i2t_metrics,
        "t2i": t2i_metrics,
        "average": avg_metrics
    }

def main():
    # 配置
    data_root = "/root/autodl-fs/MIMIC"
    csv_path = os.path.join(data_root, "merged_dataset/merged_dataset_real.csv")
    checkpoint_path = "/root/autodl-fs/MIMIC/research_implementation/checkpoints_real_fixed/best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("开始测试研究模型（分数据集评估）")
    print("=" * 70)
    
    # 1. 加载模型
    print("\n[1/4] 加载训练好的模型...")
    model = OTLGAModel(
        vit_type='vit_base',
        freeze_vit=False,
        freeze_layers=0,  # 匹配训练配置
        c_embed_dim=256
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        # 现在只保存了state_dict，不是完整的checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ 成功加载模型: {checkpoint_path}")
            if 'epoch' in checkpoint:
                print(f"  训练轮数: Epoch {checkpoint['epoch']}")
            if 'best_val_loss' in checkpoint:
                print(f"  最佳验证损失: {checkpoint['best_val_loss']:.4f}")
        else:
            # 直接是state_dict
            model.load_state_dict(checkpoint)
            print(f"✓ 成功加载模型: {checkpoint_path}")
            print(f"  (仅包含模型权重)")
    else:
        print(f"✗ 未找到模型文件: {checkpoint_path}")
        return
    
    # 2. 加载完整测试集
    print("\n[2/4] 加载测试数据集...")
    test_dataset_full = OTLGADataset(
        data_root=data_root,
        csv_path=csv_path,
        split='test',  # 现在有独立的 test 了
        img_size=224,
        is_multiview=False
    )
    
    # 3. 按 source 分割数据集
    print("\n[3/4] 按数据源划分测试集...")
    import pandas as pd
    df = pd.read_csv(csv_path)
    test_df = df[df['split'] == 'test']
    
    # 根据merged_dataset_real.csv的实际source列名
    dataset1_df = test_df[test_df['source'] == 'mimic-cxr']
    dataset2_df = test_df[test_df['source'] == 'dataset2']
    
    print(f"  Dataset1 (MIMIC-CXR): {len(dataset1_df)} 样本")
    print(f"  Dataset2 (Cleaned):   {len(dataset2_df)} 样本")
    
    # 4. 在各个数据集上评估
    print("\n[4/4] 在各数据集上进行评估...")
    
    all_results = {}
    
    # 4.1 在整个测试集上评估
    print("\n" + "="*70)
    print("评估: 整个测试集")
    print("="*70)
    test_loader_full = DataLoader(test_dataset_full, batch_size=32, shuffle=False, num_workers=4)
    results_all = evaluate(model, test_loader_full, device, "All Test Set")
    all_results['all'] = results_all
    
    # 4.2 在 Dataset1 上评估
    if len(dataset1_df) > 0:
        print("\n" + "="*70)
        print("评估: Dataset1 (MIMIC-CXR)")
        print("="*70)
        
        # 创建临时 CSV
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            dataset1_df.to_csv(f.name, index=False)
            temp_csv1 = f.name
        
        test_dataset1 = OTLGADataset(
            data_root=data_root,
            csv_path=temp_csv1,
            split='test',
            img_size=224,
            is_multiview=False
        )
        test_loader1 = DataLoader(test_dataset1, batch_size=32, shuffle=False, num_workers=4)
        results_dataset1 = evaluate(model, test_loader1, device, "Dataset1 (MIMIC-CXR)")
        all_results['dataset1'] = results_dataset1
        
        os.remove(temp_csv1)
    
    # 4.3 在 Dataset2 上评估
    if len(dataset2_df) > 0:
        print("\n" + "="*70)
        print("评估: Dataset2 (Cleaned)")
        print("="*70)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            dataset2_df.to_csv(f.name, index=False)
            temp_csv2 = f.name
        
        test_dataset2 = OTLGADataset(
            data_root=data_root,
            csv_path=temp_csv2,
            split='test',
            img_size=224,
            is_multiview=False
        )
        test_loader2 = DataLoader(test_dataset2, batch_size=32, shuffle=False, num_workers=4)
        results_dataset2 = evaluate(model, test_loader2, device, "Dataset2 (Cleaned)")
        all_results['dataset2'] = results_dataset2
        
        os.remove(temp_csv2)
    
    # 5. 保存结果
    output_path = os.path.join(os.path.dirname(checkpoint_path), "test_results_detailed.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ 测试结果已保存到: {output_path}")
    
    # 6. 打印汇总对比
    print("\n" + "="*70)
    print("测试结果汇总对比")
    print("="*70)
    print(f"\n{'指标':<15} {'整体':<12} {'Dataset1':<12} {'Dataset2':<12}")
    print("-" * 70)
    
    metrics_to_show = ["Avg R@1", "Avg R@5", "Avg R@10", "Avg P@1", "Avg P@5", "Avg P@10"]
    for metric in metrics_to_show:
        all_val = all_results['all']['average'].get(metric, 0)
        d1_val = all_results.get('dataset1', {}).get('average', {}).get(metric, 0)
        d2_val = all_results.get('dataset2', {}).get('average', {}).get(metric, 0)
        print(f"{metric:<15} {all_val:>10.2f}%  {d1_val:>10.2f}%  {d2_val:>10.2f}%")
    
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)

if __name__ == "__main__":
    main()
