from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModel

@dataclass
class ClipBatchOutputs:
    image_emb: torch.Tensor   # (B, D)
    text_emb: torch.Tensor    # (B, D)
    logit_scale: torch.Tensor # scalar (already exp)

class ResNet50Backbone(nn.Module):
    def __init__(self, train_backbone: bool):
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # (B, 2048, 1, 1)
        self.out_dim = 2048
        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)  # (B, 2048)
        return feat

class BERTBackbone(nn.Module):
    def __init__(self, model_name: str, train_backbone: bool):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.out_dim = self.bert.config.hidden_size
        if not train_backbone:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS]
        return cls

class ClipResNetBert(nn.Module):
    def __init__(
        self,
        text_model_name: str,
        embed_dim: int = 256,
        train_image_backbone: bool = False,
        train_text_backbone: bool = False,
    ):
        super().__init__()
        self.image_encoder = ResNet50Backbone(train_backbone=train_image_backbone)
        self.text_encoder = BERTBackbone(model_name=text_model_name, train_backbone=train_text_backbone)

        self.image_proj = nn.Linear(self.image_encoder.out_dim, embed_dim)
        self.text_proj = nn.Linear(self.text_encoder.out_dim, embed_dim)

        # CLIP temperature parameter (learned)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))  # ~log(1/0.07)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> ClipBatchOutputs:
        img_feat = self.image_encoder(images)
        txt_feat = self.text_encoder(input_ids, attention_mask)

        img_emb = F.normalize(self.image_proj(img_feat), dim=-1)
        txt_emb = F.normalize(self.text_proj(txt_feat), dim=-1)

        return ClipBatchOutputs(
            image_emb=img_emb,
            text_emb=txt_emb,
            logit_scale=self.logit_scale.exp(),
        )

def clip_contrastive_loss(image_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    logits = logit_scale * (image_emb @ text_emb.T)  # (B, B)
    targets = torch.arange(image_emb.size(0), device=image_emb.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_i + loss_t)
