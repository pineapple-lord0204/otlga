import torch
import torch.nn as nn
import torch.nn.functional as F

class SentenceContrastive(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, v_final, T_fused, text_list, tokenizer):
        B, m, d = T_fused.shape
        
        sim = torch.einsum('id,jmd->ijm', v_final, T_fused) / self.temp
        
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
        logits = self.classifier(v_final)
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            return loss, torch.sigmoid(logits)
        return torch.sigmoid(logits)


