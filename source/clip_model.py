from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    vit_b_16,
    vit_b_32,
    ViT_B_16_Weights,
    ViT_B_32_Weights,
)
from transformers import AutoModel

@dataclass
class ClipBatchOutputs:
    image_emb: torch.Tensor   
    text_emb: torch.Tensor    
    logit_scale: torch.Tensor



# ----------------- Image Backbones --------------------

class ResNet50Backbone(nn.Module):
    def __init__(self, train_backbone: bool):
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # (B, 2048, 1, 1)
        self.out_dim = 2048
        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        feat = self.backbone(x).flatten(1)  # (B, 2048)
        return feat

class ViTBackbone(nn.Module):
    def __init__(self, model_name: str, train_backbone: bool):
        super().__init__()

        if model_name == "vit_b_16":
            m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        elif model_name == "vit_b_32":
            m = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported ViT model: {model_name}")

        self.out_dim = m.heads.head.in_features
        m.heads.head = nn.Identity()
        self.backbone = m

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


def build_image_backbone(name: str, train_backbone: bool):
    name = name.lower()

    if name == "resnet50":
        return ResNet50Backbone(train_backbone)
    elif name in ["vit_b_16", "vit_b_32"]:
        return ViTBackbone(name, train_backbone)
    else:
        raise ValueError(f"Unsupported image encoder: {name}")
    

    

# ------------------ Text Backbones --------------------
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

# ------------------ Contrastive Loss ----------------------
def clip_contrastive_loss(image_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    logits = logit_scale * (image_emb @ text_emb.T)  # (B, B)
    targets = torch.arange(image_emb.size(0), device=image_emb.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_i + loss_t)



# ----------------- Multimodal MES Classifier -----------------
class MultimodalMESHead(nn.Module):
    # fully multimodal MES classifier that uses image and text embeddings and predicts MES

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 256, num_classes: int = 4):
        super().__init__()

        # concatenation of image + text embeddings into 2 * embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        image_emb: (B, D)
        text_emb:  (B, D)
        returns:   (B, 4) logits
        """
        fused = torch.cat([image_emb, text_emb], dim=-1)
        logits = self.classifier(fused)
        return logits