import os

# 必须在导入 transformers 之前设置镜像环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import logging

# 导入本地定义的 ViT
from vit_custom import get_ViT, create_eva_vit_g

# ==========================================
# 1. 局部-全局注意力模块 (LGA)
# ==========================================
class LocalGlobalAttention(nn.Module):
    def __init__(self, d, num_heads=8, dropout=0.2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, local_feats, global_feat):
        """
        local_feats: [B, n, d]
        global_feat: [B, d]
        """
        # 将全局特征拼接到局部序列中进行交互
        query = torch.cat([global_feat.unsqueeze(1), local_feats], dim=1)
        attn_out, _ = self.self_attn(query, query, query)
        attn_out = self.dropout(attn_out)
        out = self.norm(query + attn_out)
        
        # 分离增强后的特征
        v_enhanced = out[:, 0, :]
        V_enhanced = out[:, 1:, :]
        return V_enhanced, v_enhanced

# ==========================================
# 2. 熵正则最优传输 (OT)
# ==========================================
class EntropicOT(nn.Module):
    def __init__(self, epsilon=0.05, max_iter=50):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter

    def forward(self, C):
        """
        C: 代价矩阵 [B, n, m], 比如 1 - cosine_similarity
        """
        B, n, m = C.shape
        device = C.device
        
        # 均匀边际分布
        a = torch.ones((B, n), device=device) / n
        b = torch.ones((B, m), device=device) / m
        
        K = torch.exp(-C / self.epsilon)
        u = torch.ones((B, n), device=device) / n
        
        for _ in range(self.max_iter):
            v = b / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
            u = a / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
            
        P = u.unsqueeze(-1) * K * v.unsqueeze(1) # [B, n, m]
        ot_loss = torch.sum(P * C, dim=(1, 2))
        return P, ot_loss

# ==========================================
# 3. OT 引导门控融合 (Gated Fusion)
# ==========================================
class OTGatedFusion(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate_v = nn.Sequential(nn.Linear(2*d, d), nn.Sigmoid())
        self.gate_t = nn.Sequential(nn.Linear(2*d, d), nn.Sigmoid())
        
    def forward(self, V, T, P):
        """
        V: [B, n, d], T: [B, m, d], P: [B, n, m]
        """
        # 视觉侧：通过 P 聚合文本信息
        # P: [B, n, m], T: [B, m, d] -> [B, n, d]
        T_aligned = torch.bmm(P, T) 
        g_v = self.gate_v(torch.cat([V, T_aligned], dim=-1))
        V_fused = V + g_v * T_aligned
        
        # 文本侧：通过 P^T 聚合视觉信息
        # P^T: [B, m, n], V: [B, n, d] -> [B, m, d]
        V_aligned = torch.bmm(P.transpose(1, 2), V)
        g_t = self.gate_t(torch.cat([T, V_aligned], dim=-1))
        T_fused = T + g_t * V_aligned
        
        return V_fused, T_fused

# ==========================================
# 核心模型：OTLGAModel (Optimal Transport with Local-Global Attention)
# ==========================================
class OTLGAModel(nn.Module):
    def __init__(self, 
                 vit_type='vit_base', 
                 vit_path='', 
                 freeze_vit=True, 
                 freeze_layers=8, # 若部分冻结，指定冻结前几层
                 c_embed_dim=256,
                 max_txt_len=128,
                 bert_model='base'):  # 新增：支持医学领域BERT
        super().__init__()
        
        # 1. 视觉编码器 (ViT)
        if vit_type == 'eva_vit':
            self.visual_encoder = create_eva_vit_g(vit_path, 224, precision="fp16")
            vision_dim = 1408
        else:
            self.visual_encoder = get_ViT(vit_path, 224)
            vision_dim = 768
            
        # --- 策略 1: ViT 冻结策略 ---
        if freeze_vit:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            logging.info("Visual Encoder 全冻结")
        elif freeze_layers > 0:
            # 部分冻结示例 (针对 ViT 结构)
            if hasattr(self.visual_encoder, 'blocks'):
                for i, block in enumerate(self.visual_encoder.blocks):
                    if i < freeze_layers:
                        for param in block.parameters():
                            param.requires_grad = False
                logging.info(f"Visual Encoder 冻结前 {freeze_layers} 层")

        # 2. 文本编码器 - 支持医学领域BERT
        from transformers import AutoTokenizer, BertModel
        
        if bert_model == 'clinical':
            model_name = 'emilyalsentzer/Bio_ClinicalBERT'
            print(f"  📚 使用ClinicalBERT (在MIMIC-III上预训练)")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_encoder = BertModel.from_pretrained(model_name)
        elif bert_model == 'bio':
            model_name = 'dmis-lab/biobert-v1.1'
            print(f"  📚 使用BioBERT (在生物医学文献上预训练)")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_encoder = BertModel.from_pretrained(model_name)
        else:
            self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        text_dim = self.text_encoder.config.hidden_size
        
        # 3. 投影层
        self.vision_proj = nn.Linear(vision_dim, c_embed_dim)
        self.text_proj = nn.Linear(text_dim, c_embed_dim)
        
        # 4. 创新模块
        self.lga = LocalGlobalAttention(c_embed_dim)
        self.ot = EntropicOT()
        self.gated_fusion = OTGatedFusion(c_embed_dim)
        
        self.temp = nn.Parameter(torch.ones([]) * 0.5)  # 增大初始温度避免特征崩溃
        self.max_txt_len = max_txt_len

    def forward(self, image, text_input, is_multiview=False):
        """
        image: [B, 3, H, W] 或 [B, V, 3, H, W] (V为视角数)
        text_input: List[str]
        """
        device = image.device
        B = image.size(0)
        
        # --- 策略 4: 多视角支持 ---
        if is_multiview and image.dim() == 5:
            # 简单策略：分别通过编码器后取平均或拼接
            B, V, C, H, W = image.shape
            image = image.view(-1, C, H, W)
            
        if hasattr(self.visual_encoder, 'image_size'):  # 判断是否为 EVA-ViT
            image = image.to(torch.float16)
            v_embeds = self.visual_encoder(image) # [B*V, n, dim]
        else:
            v_embeds = self.visual_encoder(image) # [B*V, n, dim]
        
        if v_embeds.dtype == torch.float16:
            v_embeds = v_embeds.to(torch.float32)
        
        if is_multiview:
            v_embeds = v_embeds.view(B, -1, v_embeds.size(-2), v_embeds.size(-1))
            V_local = v_embeds.mean(dim=1) # [B, n, dim] 跨视角平均
        else:
            V_local = v_embeds


        # 映射与归一化
        V_local = self.vision_proj(V_local)
        v_global = V_local.mean(dim=1)
        
        # 文本处理
        tokens = self.tokenizer(text_input, padding=True, truncation=True, 
                               max_length=self.max_txt_len, return_tensors="pt").to(device)
        T_full = self.text_encoder(**tokens).last_hidden_state
        T_local = self.text_proj(T_full)
        
        # attention_mask: 1 for real tokens, 0 for padding
        attention_mask = tokens['attention_mask'].unsqueeze(-1).float()  # [B, seq_len, 1]
        t_global = (T_local * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)  # [B, d]
        
        # LGA 增强
        V_local, v_global = self.lga(V_local, v_global)
        T_local, t_global = self.lga(T_local, t_global)
        
        # OT 对齐
        # 计算代价矩阵 (1 - cosine similarity)
        V_norm = F.normalize(V_local, p=2, dim=-1)
        T_norm = F.normalize(T_local, p=2, dim=-1)
        C = 1.0 - torch.bmm(V_norm, T_norm.transpose(1, 2))
        P, ot_loss = self.ot(C)
        
        # 门控融合
        V_fused, T_fused = self.gated_fusion(V_local, T_local, P)
        
        v_final = F.normalize(V_fused.mean(dim=1), p=2, dim=-1)
        attention_mask = tokens['attention_mask'].unsqueeze(-1).float()  # [B, seq_len, 1]
        t_final = (T_fused * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)  # [B, d]
        t_final = F.normalize(t_final, p=2, dim=-1)
        
        return v_final, t_final, ot_loss, T_fused

