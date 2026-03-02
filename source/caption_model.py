from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


@dataclass
class CaptionBatchOutputs:
    loss: torch.Tensor
    logits: torch.Tensor  # (B, T, vocab)


class PrefixCaptioner(nn.Module):

    # take a fused embedding (image_emb + text_emb) -> project it into a sequence of prefix embeddings (prefix_len, gpt_hidden) 
    # that are prepended to GPT-2's input embeddings, GPT-2 then generates the caption tokens

    def __init__(
        self,
        gpt_name: str = "gpt2",
        fused_dim: int = 512,     # image_emb(256) + text_emb(256) by default
        prefix_len: int = 10,
        dropout: float = 0.1,
        freeze_gpt: bool = False, # optional
    ):
        super().__init__()

        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_name)
        self.gpt_hidden = self.gpt.config.n_embd
        self.prefix_len = prefix_len

        self.prefix_mlp = nn.Sequential(
            nn.Linear(fused_dim, self.gpt_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.gpt_hidden, prefix_len * self.gpt_hidden),
        )

        if freeze_gpt:
            for p in self.gpt.parameters():
                p.requires_grad = False

    def build_prefix(self, fused: torch.Tensor) -> torch.Tensor:
        x = self.prefix_mlp(fused)  # (B, prefix_len * gpt_hidden)
        x = x.view(fused.size(0), self.prefix_len, self.gpt_hidden)
        return x

    def forward(
        self,
        fused: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> CaptionBatchOutputs:

        device = fused.device
        B, T = input_ids.shape

        # Build prefix embeddings
        prefix_embeds = self.build_prefix(fused)  # (B, P, H)

        # Token embeddings for caption text
        token_embeds = self.gpt.transformer.wte(input_ids)  # (B, T, H)

        # Concatenate prefix + caption embeds
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)  # (B, P+T, H)

        # Build attention mask for prefix (all ones)
        prefix_mask = torch.ones((B, self.prefix_len), dtype=attention_mask.dtype, device=device)
        attn = torch.cat([prefix_mask, attention_mask], dim=1)  # (B, P+T)

        # If labels provided, we must pad with -100 for prefix positions so loss ignores prefix
        if labels is not None:
            ignore = torch.full((B, self.prefix_len), -100, dtype=labels.dtype, device=device)
            full_labels = torch.cat([ignore, labels], dim=1)  # (B, P+T)
        else:
            full_labels = None

        out = self.gpt(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=full_labels,
        )

        return CaptionBatchOutputs(loss=out.loss, logits=out.logits)

    @torch.no_grad()
    def generate(
        self,
        fused: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 60,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> str:
        # generate a caption from fused embedding using prefix conditioning.
        device = fused.device
        prefix_embeds = self.build_prefix(fused)  # (1, P, H)

        # start with empty prompt (or BOS). GPT-2 doesn't have BOS by default, so we can start with eos_token.
        start_id = tokenizer.eos_token_id
        input_ids = torch.tensor([[start_id]], device=device)
        attn_mask = torch.ones_like(input_ids)

        for _ in range(max_new_tokens):
            token_embeds = self.gpt.transformer.wte(input_ids)  # (1, t, H)
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

            prefix_mask = torch.ones((1, self.prefix_len), dtype=attn_mask.dtype, device=device)
            attn = torch.cat([prefix_mask, attn_mask], dim=1)

            logits = self.gpt(inputs_embeds=inputs_embeds, attention_mask=attn).logits  # (1, P+t, vocab)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)

            if do_sample:
                probs = torch.softmax(next_logits, dim=-1)
                # nucleus sampling
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumsum > top_p
                cutoff[..., 0] = False
                sorted_probs[cutoff] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_id = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            else:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_id], dim=1)
            attn_mask = torch.cat([attn_mask, torch.ones_like(next_id)], dim=1)

            if next_id.item() == tokenizer.eos_token_id:
                break

        text = tokenizer.decode(input_ids[0], skip_special_tokens=True).strip()
        return text