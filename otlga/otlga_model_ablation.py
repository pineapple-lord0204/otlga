import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import logging

from vit_custom import get_ViT, create_eva_vit_g
from otlga_model import LocalGlobalAttention, EntropicOT, OTGatedFusion

class OTLGAModelAblation(nn.Module):
    def __init__(self,
                 vit_type='vit_base',
                 vit_path='',
                 freeze_vit=False,
                 freeze_layers=0,
                 c_embed_dim=256,
                 max_txt_len=128,
                 bert_model='base',
                 use_lga=True,
                 use_ot=True,
                 use_gated_fusion=True):
        super().__init__()

        self.use_lga = use_lga
        self.use_ot = use_ot
        self.use_gated_fusion = use_gated_fusion

        if vit_type == 'eva_vit':
            self.visual_encoder = create_eva_vit_g(vit_path, 224, precision="fp16")
            vision_dim = 1408
        else:
            self.visual_encoder = get_ViT(vit_path, 224)
            vision_dim = 768

        if freeze_vit:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            logging.info("Visual Encoder Fully Frozen")
        elif freeze_layers > 0:
            if hasattr(self.visual_encoder, 'blocks'):
                for i, block in enumerate(self.visual_encoder.blocks):
                    if i < freeze_layers:
                        for param in block.parameters():
                            param.requires_grad = False
                logging.info(f"Visual Encoder Frozen First {freeze_layers} layers")

        from transformers import AutoTokenizer, BertModel

        if bert_model == 'clinical':
            model_name = 'emilyalsentzer/Bio_ClinicalBERT'
            print(f"  📚 Using ClinicalBERT")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_encoder = BertModel.from_pretrained(model_name)
        elif bert_model == 'bio':
            model_name = 'dmis-lab/biobert-v1.1'
            print(f"  📚 Using BioBERT")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_encoder = BertModel.from_pretrained(model_name)
        else:
            self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        text_dim = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_dim, c_embed_dim)
        self.text_proj = nn.Linear(text_dim, c_embed_dim)

        if self.use_lga:
            self.lga = LocalGlobalAttention(c_embed_dim)
        if self.use_ot:
            self.ot = EntropicOT()
        if self.use_gated_fusion:
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

        if self.use_lga:
            V_local, v_global = self.lga(V_local, v_global)
            T_local, t_global = self.lga(T_local, t_global)

        if self.use_ot:
            V_norm = F.normalize(V_local, p=2, dim=-1)
            T_norm = F.normalize(T_local, p=2, dim=-1)
            C = 1.0 - torch.bmm(V_norm, T_norm.transpose(1, 2))
            P, ot_loss = self.ot(C)
        else:
            P = None
            ot_loss = torch.zeros(B, device=device)

        if self.use_gated_fusion and P is not None:
            V_fused, T_fused = self.gated_fusion(V_local, T_local, P)
        else:
            V_fused = V_local
            T_fused = T_local

        v_final = F.normalize(V_fused.mean(dim=1), p=2, dim=-1)
        attention_mask = tokens['attention_mask'].unsqueeze(-1).float()
        t_final = (T_fused * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
        t_final = F.normalize(t_final, p=2, dim=-1)

        return v_final, t_final, ot_loss, T_fused, tokens['attention_mask']



