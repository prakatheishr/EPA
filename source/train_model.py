import os
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from dataset import UCMultimodalDataset
from clip_model import ClipResNetBert, clip_contrastive_loss

# trains the CLIP-style model on train pairs, validates on val, saves best checkpoint.

def get_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def collate_fn(batch, tokenizer, max_len: int = 128):
    images = torch.stack([b["image"] for b in batch], dim=0)
    texts = [b["text"] for b in batch]

    tok = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return {
        "images": images,
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
    }

def main():
    # Paths from env vars
    csv_path = os.environ.get("UC_DATA_CSV")
    image_root = os.environ.get("UC_IMAGE_ROOT")
    if not csv_path or not image_root:
        raise SystemExit("Set UC_DATA_CSV and UC_IMAGE_ROOT env vars first.")

    train_csv = "splits/train.csv"
    val_csv = "splits/val.csv"

    # Medical BERT
    text_model = "emilyalsentzer/Bio_ClinicalBERT"

    # Training config (safe defaults for 981 samples)
    batch_size = 16
    epochs = 10
    lr = 1e-4
    weight_decay = 1e-4
    embed_dim = 256
    force_cpu = False

    device = get_device(force_cpu=force_cpu)
    print("Device:", device)
    print("Text model:", text_model)

    tokenizer = AutoTokenizer.from_pretrained(text_model)

    train_ds = UCMultimodalDataset(train_csv, image_root=image_root, train=True)
    val_ds = UCMultimodalDataset(val_csv, image_root=image_root, train=False)

    # drop_last=True helps keep (B,B) logits stable
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    model = ClipResNetBert(
        text_model_name=text_model,
        embed_dim=embed_dim,
        train_image_backbone=False,  # start frozen
        train_text_backbone=False,   # start frozen
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    out_dir = Path("runs/clip_rn50_clinicalbert")
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}"):
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            out = model(images, input_ids, attn)
            loss = clip_contrastive_loss(out.image_emb, out.text_emb, out.logit_scale)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item())
            n += 1

        train_loss = total / max(n, 1)

        # Val
        model.eval()
        vtotal = 0.0
        vn = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val epoch {epoch}"):
                images = batch["images"].to(device)
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)

                out = model(images, input_ids, attn)
                loss = clip_contrastive_loss(out.image_emb, out.text_emb, out.logit_scale)

                vtotal += float(loss.item())
                vn += 1

        val_loss = vtotal / max(vn, 1)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "best.pt")
            (out_dir / "metrics.json").write_text(json.dumps({"best_val_loss": best_val}, indent=2))
            print("Saved new best:", out_dir / "best.pt")

    print("Done. Best val loss:", best_val)

if __name__ == "__main__":
    main()
