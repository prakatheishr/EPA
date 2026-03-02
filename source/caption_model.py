import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class ImageConditionedGPT2(nn.Module):
    def __init__(self, image_encoder, gpt_model_name="gpt2"):
        super().__init__()

        self.image_encoder = image_encoder  # frozen encoder
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_model_name)

        # map image embedding into GPT hidden size
        self.image_to_gpt = nn.Linear(
            self.image_encoder.out_dim,
            self.gpt.config.n_embd
        )

    def forward(self, images, input_ids, attention_mask):
        # extract image features
        img_feat = self.image_encoder(images)  # (B, D)

        # project to GPT embedding space
        img_emb = self.image_to_gpt(img_feat)  # (B, n_embd)

        # expand to prefix token
        prefix = img_emb.unsqueeze(1)  # (B, 1, n_embd)

        # get GPT token embeddings
        inputs_embeds = self.gpt.transformer.wte(input_ids)

        # concatenate prefix + text embeddings
        inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1)

        # adjust attention mask
        prefix_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        outputs = self.gpt(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=input_ids
        )

        return outputs