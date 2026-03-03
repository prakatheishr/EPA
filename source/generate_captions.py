import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from clip_model import ClipDualEncoder
from caption_model import PrefixCaptioner

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_image():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_path", required=True)

    p.add_argument("--image_encoder", default="resnet50")
    p.add_argument("--text_encoder", default="emilyalsentzer/Bio_ClinicalBERT")
    p.add_argument("--foundation_ckpt", required=True)

    p.add_argument("--gpt_name", default="gpt2")
    p.add_argument("--caption_ckpt", required=True)
    p.add_argument("--prefix_len", type=int, default=10)

    # generation
    p.add_argument("--prompt", default="Findings: ")
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--min_new_tokens", type=int, default=20)

    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=50)

    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--no_repeat_ngram_size", type=int, default=3)

    p.add_argument("--force_cpu", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device(force_cpu=args.force_cpu)
    print("Device:", device)

    # --- Load foundation (frozen) ---
    foundation = ClipDualEncoder(
        image_encoder_name=args.image_encoder,
        text_encoder_name=args.text_encoder,
        embed_dim=256,
        train_image_backbone=False,
        train_text_backbone=False,
    ).to(device)

    foundation_ckpt = Path(args.foundation_ckpt)
    if not foundation_ckpt.exists():
        raise SystemExit(f"Foundation ckpt not found: {foundation_ckpt}")

    foundation.load_state_dict(torch.load(foundation_ckpt, map_location=device))
    foundation.eval()
    for p in foundation.parameters():
        p.requires_grad = False

    if not hasattr(foundation, "encode_image"):
        raise RuntimeError("ClipDualEncoder must expose encode_image(images).")

    # --- Load captioner ---
    captioner = PrefixCaptioner(
        gpt_name=args.gpt_name,
        fused_dim=512,
        prefix_len=args.prefix_len,
        dropout=0.0,
        freeze_gpt=True,
    ).to(device)

    caption_ckpt = Path(args.caption_ckpt)
    if not caption_ckpt.exists():
        raise SystemExit(f"Caption ckpt not found: {caption_ckpt}")

    captioner.load_state_dict(torch.load(caption_ckpt, map_location=device))
    captioner.eval()

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.gpt_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load + preprocess image ---
    img = Image.open(args.image_path).convert("RGB")
    x = preprocess_image()(img).unsqueeze(0).to(device)  # (1,3,224,224)

    # --- Encode image -> fused vector (match training: [img_emb ; zeros]) ---
    img_emb = foundation.encode_image(x)  # (1,256)
    fused = torch.cat([img_emb, torch.zeros_like(img_emb)], dim=-1)  # (1,512)

    # --- Build prefix embeddings for GPT + run generation (INLINE, no captioner.generate required) ---
    if not (hasattr(captioner, "gpt") and hasattr(captioner, "prefix_proj") and hasattr(captioner, "prefix_len")):
        raise RuntimeError("PrefixCaptioner must expose gpt, prefix_proj, prefix_len for generation.")

    gpt = captioner.gpt
    gpt.eval()

    # GPT-2 embedding dim
    gpt_dim = gpt.config.n_embd

    # Project fused -> prefix token embeddings
    prefix = captioner.prefix_proj(fused)  # (1, prefix_len * gpt_dim)
    prefix = prefix.view(1, captioner.prefix_len, gpt_dim)  # (1, prefix_len, gpt_dim)

    # prompt so generation doesn't instantly emit EOS
    prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)  # (1, L)
    prompt_emb = gpt.transformer.wte(prompt_ids)  # (1, L, gpt_dim)

    # Concatenate prefix + prompt embeddings
    inputs_embeds = torch.cat([prefix, prompt_emb], dim=1)  # (1, prefix_len+L, gpt_dim)

    # Attention mask for the embeddings
    attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    gen = gpt.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        temperature=args.temperature if args.do_sample else None,
        top_p=args.top_p if args.do_sample else None,
        top_k=args.top_k if args.do_sample else None,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(gen[0], skip_special_tokens=True).strip()

    # If the decoded string includes the prompt, strip it off
    if args.prompt and text.lower().startswith(args.prompt.strip().lower()):
        text = text[len(args.prompt) :].strip()

    print("\nGenerated caption:")
    print(text)


if __name__ == "__main__":
    main()