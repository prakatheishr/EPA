from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


@dataclass
class CaptionOutputs:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class PrefixCaptioner(nn.Module):

    def __init__(
        self,
        gpt_name: str = "gpt2",
        fused_dim: int = 512,
        prefix_len: int = 10,
        dropout: float = 0.1,
        freeze_gpt: bool = False,
    ):
        super().__init__()

        self.gpt_name = gpt_name
        self.prefix_len = int(prefix_len)

        # GPT backbone
        self.gpt = AutoModelForCausalLM.from_pretrained(gpt_name)
        gpt_dim = int(self.gpt.config.n_embd)  # GPT-2 base = 768

        # Map fused multimodal features into a sequence of GPT prefix tokens
        self.prefix_mlp = nn.Sequential(
            nn.Linear(fused_dim, gpt_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(gpt_dim, self.prefix_len * gpt_dim),
        )

        if freeze_gpt:
            for p in self.gpt.parameters():
                p.requires_grad = False

    @property
    def prefix_proj(self):
        return self.prefix_mlp

    def build_prefix(self, fused: torch.Tensor) -> torch.Tensor:
        B = fused.size(0)
        gpt_dim = int(self.gpt.config.n_embd)

        prefix = self.prefix_mlp(fused)                 
        prefix = prefix.view(B, self.prefix_len, gpt_dim)
        return prefix

    def forward(
        self,
        fused: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> CaptionOutputs:

        prefix = self.build_prefix(fused)  

        # GPT token embeddings for caption tokens
        tok_emb = self.gpt.transformer.wte(input_ids)  # (B, L, D)

        # concat prefix + tokens
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)  # (B, P+L, D)

        # attention mask for prefix tokens is all ones
        prefix_mask = torch.ones(
            (input_ids.size(0), self.prefix_len),
            dtype=torch.long,
            device=input_ids.device,
        )
        attn = torch.cat([prefix_mask, attention_mask], dim=1)  # (B, P+L)

        if labels is not None:
            # Ignore prefix positions when computing language modelling loss
            ignore = torch.full(
                (labels.size(0), self.prefix_len),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            full_labels = torch.cat([ignore, labels], dim=1)  # (B, P+L)
        else:
            full_labels = None

        out = self.gpt(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=full_labels,
        )

        return CaptionOutputs(loss=out.loss, logits=out.logits)

    @torch.no_grad()
    def generate(
        self,
        fused: torch.Tensor,
        tokenizer,
        prompt: str = "Findings: ",
        max_new_tokens: int = 80,
        min_new_tokens: int = 20,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 0.9,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
    ) -> str:

        self.eval()
        device = fused.device

        prefix = self.build_prefix(fused)  # (B, P, D) typically B=1

        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)  # (B, Lp)
        prompt_emb = self.gpt.transformer.wte(prompt_ids)                         # (B, Lp, D)

        inputs_embeds = torch.cat([prefix, prompt_emb], dim=1)  # (B, P+Lp, D)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

        gen = self.gpt.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        text = tokenizer.decode(gen[0], skip_special_tokens=True).strip()

        # Remove the fixed prompt prefix from the decoded output if present
        if text.lower().startswith(prompt.strip().lower()):
            text = text[len(prompt):].strip()

        return text