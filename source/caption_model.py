import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class ImagePrefixGPT2(nn.Module):
    def __init__(self, image_encoder: nn.Module, image_dim: int, gpt_name: str = "gpt2", prefix_len: int = 5):
        super().__init__()
        self.image_encoder = image_encoder
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_name)
        self.prefix_len = prefix_len
        gpt_dim = self.gpt.config.n_embd

        # map image feature -> (prefix_len * gpt_dim)
        self.img_to_prefix = nn.Linear(image_dim, prefix_len * gpt_dim)

    def forward(self, images, input_ids, attention_mask, labels=None):
        # image features
        img_feat = self.image_encoder(images)                 # (B, image_dim)
        prefix = self.img_to_prefix(img_feat)                 # (B, prefix_len*gpt_dim)
        prefix = prefix.view(images.size(0), self.prefix_len, -1)  # (B, prefix_len, gpt_dim)

        # token embeddings
        tok_emb = self.gpt.transformer.wte(input_ids)         # (B, T, gpt_dim)
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)   # (B, prefix_len+T, gpt_dim)

        # attention mask
        prefix_mask = torch.ones((attention_mask.size(0), self.prefix_len), device=attention_mask.device)
        attn = torch.cat([prefix_mask, attention_mask], dim=1)

        # labels need to ignore prefix positions
        if labels is not None:
            ignore = torch.full((labels.size(0), self.prefix_len), -100, device=labels.device)
            labels = torch.cat([ignore, labels], dim=1)

        return self.gpt(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)