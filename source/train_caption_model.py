import os
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import UCMultimodalDataset
from clip_model import ClipDualEncoder
from caption_model import PrefixCaptioner
from text_cleaning import clean_caption_body, format_caption_with_mes


def get_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def collate_caption(batch, tokenizer, max_len: int = 96, prompt: str = "Findings: "):
    images = torch.stack([b["image"] for b in batch], dim=0)

    targets = []
    for b in batch:
        body = clean_caption_body(b["text"])
        mes_num = int(b["label"].item()) if hasattr(b["label"], "item") else int(b["label"])
        targets.append(format_caption_with_mes(body, mes_num))

    # Full sequence fed to GPT = prompt + target
    full_texts = [prompt + t for t in targets]

    # Append EOS so model learns to stop
    full_texts = [t + tokenizer.eos_token for t in full_texts]

    tok = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
    prompt_len = int(prompt_ids.numel())
    labels[:, :prompt_len] = -100

    return {
        "images": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@torch.no_grad()
def eval_loss(foundation, captioner, loader, device):
    foundation.eval()
    captioner.eval()

    total = 0.0
    n = 0
    for batch in loader:
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        img_emb = foundation.encode_image(images)  # (B, 256)
        fused = torch.cat([img_emb, torch.zeros_like(img_emb)], dim=-1)  # (B, 512)

        out_cap = captioner(fused=fused, input_ids=input_ids, attention_mask=attn, labels=labels)
        total += float(out_cap.loss.item())
        n += 1

    return total / max(n, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_encoder", default="resnet50")
    p.add_argument("--text_encoder", default="emilyalsentzer/Bio_ClinicalBERT")
    p.add_argument("--foundation_ckpt", required=True)

    p.add_argument("--gpt_name", default="gpt2")
    p.add_argument("--out_dir", default="runs/captioner")
    p.add_argument("--prefix_len", type=int, default=10)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)

    # prefix LR
    p.add_argument("--lr", type=float, default=2e-4)
    # GPT LR (used only after unfreeze)
    p.add_argument("--gpt_lr", type=float, default=2e-5)

    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_len", type=int, default=96)
    p.add_argument("--force_cpu", action="store_true")

    # schedule
    p.add_argument("--freeze_gpt_epochs", type=int, default=5)

    # stability
    p.add_argument("--grad_clip", type=float, default=1.0)

    return p.parse_args()


def main():
    args = parse_args()

    csv_path = os.environ.get("UC_DATA_CSV")
    image_root = os.environ.get("UC_IMAGE_ROOT")
    if not csv_path or not image_root:
        raise SystemExit("Set UC_DATA_CSV and UC_IMAGE_ROOT env vars first.")

    device = get_device(force_cpu=args.force_cpu)
    print("Device:", device)
    print("Foundation encoders:", args.image_encoder, "|", args.text_encoder)
    print("GPT:", args.gpt_name)

    # tokenizer for GPT-2 captions
    tokenizer = AutoTokenizer.from_pretrained(args.gpt_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = UCMultimodalDataset("splits/train.csv", image_root=image_root, train=True)
    val_ds = UCMultimodalDataset("splits/val.csv", image_root=image_root, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        collate_fn=lambda b: collate_caption(b, tokenizer, max_len=args.max_len, prompt="Findings: "),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=lambda b: collate_caption(b, tokenizer, max_len=args.max_len, prompt="Findings: "),
    )

    # Load foundation (frozen)
    ckpt = Path(args.foundation_ckpt)
    if not ckpt.exists():
        raise SystemExit(f"Foundation checkpoint not found: {ckpt}")

    foundation = ClipDualEncoder(
        image_encoder_name=args.image_encoder,
        text_encoder_name=args.text_encoder,
        embed_dim=256,
        train_image_backbone=False,
        train_text_backbone=False,
    ).to(device)
    foundation.load_state_dict(torch.load(ckpt, map_location=device))
    foundation.eval()
    for p in foundation.parameters():
        p.requires_grad = False

    # Captioner
    captioner = PrefixCaptioner(
        gpt_name=args.gpt_name,
        fused_dim=512,
        prefix_len=args.prefix_len,
        dropout=0.1,
        freeze_gpt=False,  # we will control requires_grad manually
    ).to(device)

    # Make sure GPT knows pad token
    captioner.gpt.config.pad_token_id = tokenizer.eos_token_id

    # Split params into prefix vs GPT for different LRs
    prefix_params, gpt_params = [], []
    for name, p in captioner.named_parameters():
        if name.startswith("gpt."):
            gpt_params.append(p)
        else:
            prefix_params.append(p)

    opt = torch.optim.AdamW(
        [
            {"params": prefix_params, "lr": args.lr},
            {"params": gpt_params, "lr": args.gpt_lr},
        ],
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- freeze/unfreeze schedule ---
        freeze_now = epoch <= args.freeze_gpt_epochs
        for p in captioner.gpt.parameters():
            p.requires_grad = not freeze_now

        captioner.train()
        total = 0.0
        n = 0

        for batch in tqdm(train_loader, desc=f"Train caption epoch {epoch} (freeze_gpt={freeze_now})"):
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            img_emb = foundation.encode_image(images)  # (B, 256)
            fused = torch.cat([img_emb, torch.zeros_like(img_emb)], dim=-1)  # (B, 512)

            out_cap = captioner(fused=fused, input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out_cap.loss

            opt.zero_grad(set_to_none=True)
            loss.backward()

            # stability
            torch.nn.utils.clip_grad_norm_(captioner.parameters(), args.grad_clip)

            opt.step()

            total += float(loss.item())
            n += 1

        train_loss = total / max(n, 1)
        val_loss = eval_loss(foundation, captioner, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(captioner.state_dict(), out_dir / "best_captioner.pt")
            (out_dir / "metrics.json").write_text(json.dumps({"best_val_loss": best_val}, indent=2))
            print("Saved new best captioner:", out_dir / "best_captioner.pt")

    print("Done. Best val loss:", best_val)


if __name__ == "__main__":
    main()