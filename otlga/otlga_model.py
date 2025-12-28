import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import logging

from vit_custom import get_ViT, create_eva_vit_g

class LocalGlobalAttention(nn.Module):
    def __init__(self, d, num_heads=8, dropout=0.2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, local_feats, global_feat):
        query = torch.cat([global_feat.unsqueeze(1), local_feats], dim=1)
        attn_out, _ = self.self_attn(query, query, query)
        attn_out = self.dropout(attn_out)
        out = self.norm(query + attn_out)
        
        v_enhanced = out[:, 0, :]
        V_enhanced = out[:, 1:, :]
        return V_enhanced, v_enhanced

class EntropicOT(nn.Module):
    def __init__(self, epsilon=0.05, max_iter=50):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter

    def forward(self, C):
        B, n, m = C.shape
        device = C.device
        
        a = torch.ones((B, n), device=device) / n
        b = torch.ones((B, m), device=device) / m
        
        K = torch.exp(-C / self.epsilon)
        u = torch.ones((B, n), device=device) / n
        
        for _ in range(self.max_iter):
            v = b / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
            u = a / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
            
        P = u.unsqueeze(-1) * K * v.unsqueeze(1)
        ot_loss = torch.sum(P * C, dim=(1, 2))
        return P, ot_loss

class OTGatedFusion(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate_v = nn.Sequential(nn.Linear(2*d, d), nn.Sigmoid())
        self.gate_t = nn.Sequential(nn.Linear(2*d, d), nn.Sigmoid())
        
    def forward(self, V, T, P):
        T_aligned = torch.bmm(P, T) 
        g_v = self.gate_v(torch.cat([V, T_aligned], dim=-1))
        V_fused = V + g_v * T_aligned
        
        V_aligned = torch.bmm(P.transpose(1, 2), V)
        g_t = self.gate_t(torch.cat([T, V_aligned], dim=-1))
        T_fused = T + g_t * V_aligned
        
        return V_fused, T_fused

class OTLGAModel(nn.Module):
    def __init__(self, 
                 vit_type='vit_base', 
                 vit_path='', 
                 freeze_vit=True, 
                 freeze_layers=8,
                 c_embed_dim=256,
                 max_txt_len=128,
                 bert_model='base'):
        super().__init__()
        
        if vit_type == 'eva_vit':
            self.visual_encoder = create_eva_vit_g(vit_path, 224, precision="fp16")
            vision_dim = 1408
        else:
            self.visual_encoder = get_ViT(vit_path, 224)
            vision_dim = 768
            
        if freeze_vit:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            logging.info("Visual Encoder fully frozen")
        elif freeze_layers > 0:
            if hasattr(self.visual_encoder, 'blocks'):
                for i, block in enumerate(self.visual_encoder.blocks):
                    if i < freeze_layers:
                        for param in block.parameters():
                            param.requires_grad = False
                logging.info(f"Visual Encoder frozen first {freeze_layers} layers")

        from transformers import AutoTokenizer, BertModel
        
        if bert_model == 'clinical':
            model_name = 'emilyalsentzer/Bio_ClinicalBERT'
            print(f"Using ClinicalBERT (pre-trained on MIMIC-III)")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_encoder = BertModel.from_pretrained(model_name)
        elif bert_model == 'bio':
            model_name = 'dmis-lab/biobert-v1.1'
            print(f"Using BioBERT (pre-trained on biomedical literature)")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_encoder = BertModel.from_pretrained(model_name)
        else:
            self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        text_dim = self.text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(vision_dim, c_embed_dim)
        self.text_proj = nn.Linear(text_dim, c_embed_dim)
        
        self.lga = LocalGlobalAttention(c_embed_dim)
        self.ot = EntropicOT()
        self.gated_fusion = OTGatedFusion(c_embed_dim)
        
        self.temp = nn.Parameter(torch.ones([]) * 0.5)
        self.max_txt_len = max_txt_len

    def forward(self, image, text_input, is_multiview=False):
        device = image.device
        B = image.size(0)
        
        if is_multiview and image.dim() == 5:
            B, V, C, H, W = image.shape
            image = image.view(-1, C, H, W)
            
        if hasattr(self.visual_encoder, 'image_size'):
            image = image.to(torch.float16)
            v_embeds = self.visual_encoder(image)
        else:
            v_embeds = self.visual_encoder(image)
        
        if v_embeds.dtype == torch.float16:
            v_embeds = v_embeds.to(torch.float32)
        
        if is_multiview:
            v_embeds = v_embeds.view(B, -1, v_embeds.size(-2), v_embeds.size(-1))
            V_local = v_embeds.mean(dim=1)
        else:
            V_local = v_embeds

        V_local = self.vision_proj(V_local)
        v_global = V_local.mean(dim=1)
        
        tokens = self.tokenizer(text_input, padding=True, truncation=True, 
                               max_length=self.max_txt_len, return_tensors="pt").to(device)
        T_full = self.text_encoder(**tokens).last_hidden_state
        T_local = self.text_proj(T_full)
        
        attention_mask = tokens['attention_mask'].unsqueeze(-1).float()
        t_global = (T_local * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
        
        V_local, v_global = self.lga(V_local, v_global)
        T_local, t_global = self.lga(T_local, t_global)
        
        V_norm = F.normalize(V_local, p=2, dim=-1)
        T_norm = F.normalize(T_local, p=2, dim=-1)
        C = 1.0 - torch.bmm(V_norm, T_norm.transpose(1, 2))
        P, ot_loss = self.ot(C)
        
        V_fused, T_fused = self.gated_fusion(V_local, T_local, P)
        
        v_final = F.normalize(V_fused.mean(dim=1), p=2, dim=-1)
        attention_mask = tokens['attention_mask'].unsqueeze(-1).float()
        t_final = (T_fused * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
        t_final = F.normalize(t_final, p=2, dim=-1)
        
        return v_final, t_final, ot_loss, T_fused
