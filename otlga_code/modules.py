import torch
import torch.nn as nn
import torch.nn.functional as F

class SentenceContrastive(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, v_final, T_fused, text_list, tokenizer):
        """
        v_final: [B, d] 全局图像特征
        T_fused: [B, m, d] 文本 token 特征
        """
        # 简化版：计算图像全局特征与文本每个 token (句子成分) 的对比
        # 在医学报告中，T_fused 的每个 token 可以看作一个局部语义
        B, m, d = T_fused.shape
        
        # 图像与所有文本 token 的相似度 [B, B, m]
        # v_final: [B, 1, d], T_fused: [B, m, d]
        sim = torch.einsum('id,jmd->ijm', v_final, T_fused) / self.temp
        
        # 寻找每个图像最匹配的文本 token (Max-pooling 策略)
        sim_max, _ = sim.max(dim=-1) # [B, B]
        
        targets = torch.arange(B).to(v_final.device)
        loss_i2t = F.cross_entropy(sim_max, targets)
        loss_t2i = F.cross_entropy(sim_max.t(), targets)
        
        return (loss_i2t + loss_t2i) / 2

class UncertaintyAuxiliary(nn.Module):
    def __init__(self, d, num_labels=14):
        super().__init__()
        self.classifier = nn.Linear(d, num_labels)
        
    def forward(self, v_final, labels=None):
        """
        labels: [B, num_labels] 软标签 (0, 1, 0.5)
        """
        logits = self.classifier(v_final)
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            return loss, torch.sigmoid(logits)
        return torch.sigmoid(logits)


