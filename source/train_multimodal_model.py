import os
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from dataset import UCMultimodalDataset
from clip_model import ClipResNetBert, MultimodalMESHead


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
    labels = torch.stack([b["label"] for b in batch], dim=0)
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
        "labels": labels,
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
    }

def main():
    # Env vars (same pattern as train_model.py)
    csv_path = os.environ.get("UC_DATA_CSV")
    image_root = os.environ.get("UC_IMAGE_ROOT")
    if not csv_path or not image_root:
        raise SystemExit("Set UC_DATA_CSV and UC_IMAGE_ROOT env vars first.")

    train_csv = "splits/train.csv"
    val_csv = "splits/val.csv"
    test_csv = "splits/test.csv"

    # Medical BERT
    text_model = "emilyalsentzer/Bio_ClinicalBERT"

    # Config
    batch_size = 16
    epochs = 30
    lr = 1e-3
    weight_decay = 1e-4
    embed_dim = 256
    hidden_dim = 256
    force_cpu = False

    device = get_device(force_cpu=force_cpu)
    print("Device:", device)
    print("Text model:", text_model)

    tokenizer = AutoTokenizer.from_pretrained(text_model)

    train_ds = UCMultimodalDataset(train_csv, image_root=image_root, train=True)
    val_ds = UCMultimodalDataset(val_csv, image_root=image_root, train=False)
    test_ds = UCMultimodalDataset(test_csv, image_root=image_root, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
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
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # Load trained CLIP foundation
    ckpt = Path("runs/clip_rn50_clinicalbert/best.pt")
    if not ckpt.exists():
        raise SystemExit(f"Foundation checkpoint not found: {ckpt}. Run train_model.py first.")

    foundation = ClipResNetBert(
        text_model_name=text_model,
        embed_dim=embed_dim,
        train_image_backbone=False,
        train_text_backbone=False,
    ).to(device)
    foundation.load_state_dict(torch.load(ckpt, map_location=device))
    foundation.eval()

    for p in foundation.parameters():
        p.requires_grad = False

    # multimodal classifier head (image_emb + text_emb into MES logits)
    head = MultimodalMESHead(embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=4).to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    out_dir = Path("runs/mes_multimodal")
    out_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()