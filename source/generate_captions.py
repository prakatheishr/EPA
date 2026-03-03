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
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

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

    p.add_argument("--max_new_tokens", type=int, default=60)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)

    p.add_argument("--force_cpu", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    device = get_device(force_cpu=args.force_cpu)
    print("Device:", device)

    # Load foundation
    foundation = ClipDualEncoder(
        image_encoder_name=args.image_encoder,
        text_encoder_name=args.text_encoder,
        embed_dim=256,
        train_image_backbone=False,
        train_text_backbone=False,
    ).to(device)
    foundation.load_state_dict(torch.load(Path(args.foundation_ckpt), map_location=device))
    foundation.eval()
    for p in foundation.parameters():
        p.requires_grad = False

    # Load captioner
    captioner = PrefixCaptioner(
        gpt_name=args.gpt_name,
        fused_dim=512,
        prefix_len=args.prefix_len,
        dropout=0.0,
        freeze_gpt=False,
    ).to(device)
    captioner.load_state_dict(torch.load(Path(args.caption_ckpt), map_location=device))
    captioner.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.gpt_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load + preprocess image
    img = Image.open(args.image_path).convert("RGB")
    x = preprocess_image()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        if not hasattr(foundation, "encode_image"):
            raise RuntimeError("ClipDualEncoder must expose encode_image(images).")
        img_emb = foundation.encode_image(x)  # (1, 256)

        fused = torch.cat([img_emb, torch.zeros_like(img_emb)], dim=-1)  # (1, 512)

        caption = captioner.generate(
            fused=fused,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    print("\nGenerated caption:")
    print(caption)

if __name__ == "__main__":
    main()