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
def evaluate(model, dataloader, device, dataset_name="All"):
    
    model.eval()
    all_v = []
    all_t = []
    all_study_ids = []

    print(f"\nExtracting Features [{dataset_name}]...")
    for batch in tqdm(dataloader, desc="Feature Extraction"):
        images = batch['image'].to(device)
        texts = batch['text']
        study_ids = batch['study_id']

        v_final, t_final, _, _ = model(images, texts, is_multiview=False)

        all_v.append(v_final.cpu())
        all_t.append(t_final.cpu())
        all_study_ids.extend(study_ids)

    all_v = torch.cat(all_v, dim=0)
    all_t = torch.cat(all_t, dim=0)

    print(f"Feature Extraction Complete [{dataset_name}]")
    print(f"  Image Features: {all_v.shape}")
    print(f"  Text Features: {all_t.shape}")


    print(f"Computing Similarity Matrix [{dataset_name}]...")
    sim_matrix = torch.matmul(all_v, all_t.t()).numpy()


    print(f"\n{'='*60}")
    print(f"[{dataset_name}] Image-to-Text Retrieval")
    print(f"{'='*60}")
    i2t_metrics = compute_retrieval_metrics(sim_matrix)
    for k, v in i2t_metrics.items():
        print(f"  {k}: {v:.2f}")

    print(f"\n{'='*60}")
    print(f"[{dataset_name}] Text-to-Image Retrieval")
    print(f"{'='*60}")
    t2i_metrics = compute_retrieval_metrics(sim_matrix.T)
    for k, v in t2i_metrics.items():
        print(f"  {k}: {v:.2f}")


    print(f"\n{'='*60}")
    print(f"[{dataset_name}] Average Retrieval Performance")
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

    data_root = "/root/autodl-fs/MIMIC"
    csv_path = os.path.join(data_root, "merged_dataset/merged_dataset_real.csv")
    checkpoint_path = "/root/autodl-fs/MIMIC/research_implementation/checkpoints_real_fixed/best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Starting Model Testing (Dataset-wise Evaluation)")
    print("=" * 70)


    print("\n[1/4] Loading Trained Model...")
    model = OTLGAModel(
        vit_type='vit_base',
        freeze_vit=False,
        freeze_layers=0,
        c_embed_dim=256
    ).to(device)

    if os.path.exists(checkpoint_path):

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Successfully Loaded Model: {checkpoint_path}")
            if 'epoch' in checkpoint:
                print(f"  Epochs: Epoch {checkpoint['epoch']}")
            if 'best_val_loss' in checkpoint:
                print(f"  Best Validation Loss: {checkpoint['best_val_loss']:.4f}")
        else:

            model.load_state_dict(checkpoint)
            print(f"✓ Successfully Loaded Model: {checkpoint_path}")
            print(f"  ((Model weights only))")
    else:
        print(f"✗ Model file not found: {checkpoint_path}")
        return


    print("\n[2/4] Loading Test Dataset...")
    test_dataset_full = OTLGADataset(
        data_root=data_root,
        csv_path=csv_path,
        split='test',
        img_size=224,
        is_multiview=False
    )


    print("\n[3/4] Splitting Test Set by Data Source...")
    import pandas as pd
    df = pd.read_csv(csv_path)
    test_df = df[df['split'] == 'test']


    dataset1_df = test_df[test_df['source'] == 'mimic-cxr']
    dataset2_df = test_df[test_df['source'] == 'dataset2']

    print(f"  Dataset1 (MIMIC-CXR): {len(dataset1_df)} Samples")
    print(f"  Dataset2 (Cleaned):   {len(dataset2_df)} Samples")


    print("\n[4/4] Evaluating on Each Dataset...")

    all_results = {}


    print("\n" + "="*70)
    print("Evaluation: Full Test Set")
    print("="*70)
    test_loader_full = DataLoader(test_dataset_full, batch_size=32, shuffle=False, num_workers=4)
    results_all = evaluate(model, test_loader_full, device, "All Test Set")
    all_results['all'] = results_all


    if len(dataset1_df) > 0:
        print("\n" + "="*70)
        print("Evaluation: Dataset1 (MIMIC-CXR)")
        print("="*70)


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


    if len(dataset2_df) > 0:
        print("\n" + "="*70)
        print("Evaluation: Dataset2 (Cleaned)")
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


    output_path = os.path.join(os.path.dirname(checkpoint_path), "test_results_detailed.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Test Results Saved to: {output_path}")


    print("\n" + "="*70)
    print("Test Results Summary Comparison")
    print("="*70)
    print(f"\n{'Metric':<15} {'Overall':<12} {'Dataset1':<12} {'Dataset2':<12}")
    print("-" * 70)

    metrics_to_show = ["Avg R@1", "Avg R@5", "Avg R@10", "Avg P@1", "Avg P@5", "Avg P@10"]
    for metric in metrics_to_show:
        all_val = all_results['all']['average'].get(metric, 0)
        d1_val = all_results.get('dataset1', {}).get('average', {}).get(metric, 0)
        d2_val = all_results.get('dataset2', {}).get('average', {}).get(metric, 0)
        print(f"{metric:<15} {all_val:>10.2f}%  {d1_val:>10.2f}%  {d2_val:>10.2f}%")

    print("\n" + "="*70)
    print("Testing Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
