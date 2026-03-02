import os
from pathlib import Path
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from dataset import UCMultimodalDataset
from clip_model import ClipDualEncoder, MultimodalMESHead

import re

LABEL_RE = re.compile(
    r"""
    # Explicit MES / Mayo patterns (all 0–3)
    (\bmes\s*[-]?\s*[0-3]\b) |
    (\bmayo\s*endoscopic\s*score\s*[0-3]\b) |
    (\bmayo\s*score\s*[0-3]\b) |

    # "support(s) a 0/1/2/3"
    (\bsupports?\s+(a\s+)?[0-3]\b) |

    # "consistent with 0/1/2/3"
    (\bconsistent\s+with\s+[0-3]\b) |

    # "grade 0/1/2/3" or "grade of 0/1/2/3"
    (\bgrade(\s+of)?\s+[0-3]\b)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

def clean_description(text: str) -> str:
    text = "" if text is None else str(text)
    text = LABEL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    texts = [clean_description(b["text"]) for b in batch]

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


@torch.no_grad()
def evaluate(foundation, head, loader, device):
    foundation.eval()
    head.eval()

    y_true, y_pred = [], []
    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        out = foundation(images, input_ids, attn)  # ClipBatchOutputs
        logits = head(out.image_emb, out.text_emb)
        preds = logits.argmax(dim=-1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    report = classification_report(y_true, y_pred, digits=4)
    return acc, f1, cm, report




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_encoder", default="resnet50")
    parser.add_argument("--text_encoder", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--foundation_ckpt", required=True)

    args = parser.parse_args()

    csv_path = os.environ.get("UC_DATA_CSV")
    image_root = os.environ.get("UC_IMAGE_ROOT")
    if not csv_path or not image_root:
        raise SystemExit("Set UC_DATA_CSV and UC_IMAGE_ROOT env vars first.")

    train_csv = "splits/train.csv"
    val_csv = "splits/val.csv"
    test_csv = "splits/test.csv"

    image_model = args.image_encoder
    text_model = args.text_encoder
    ckpt = Path(args.foundation_ckpt)

    # config
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    if not ckpt.exists():
        raise SystemExit(f"Foundation checkpoint not found: {ckpt}")

    
    foundation = ClipDualEncoder(
        image_encoder_name=image_model,
        text_encoder_name=text_model,
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

    best_val_f1 = -1.0
    for epoch in range(1, epochs + 1):
        head.train()
        total = 0.0
        n = 0

        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}"):
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            with torch.no_grad():
                out = foundation(images, input_ids, attn)

            logits = head(out.image_emb, out.text_emb)
            loss = loss_fn(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item())
            n += 1

        train_loss = total / max(n, 1)

        val_acc, val_f1, val_cm, _ = evaluate(foundation, head, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_acc={val_acc:.4f}  val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(head.state_dict(), out_dir / "best_head.pt")
            (out_dir / "metrics.json").write_text(
                json.dumps({"best_val_macro_f1": best_val_f1}, indent=2)
            )
            print("Saved new best head:", out_dir / "best_head.pt")

    head.load_state_dict(torch.load(out_dir / "best_head.pt", map_location=device))
    test_acc, test_f1, test_cm, test_report = evaluate(foundation, head, test_loader, device)

    print("\nMultimodal MES classifier (image + paired text)")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1:  {test_f1:.4f}")
    print("Confusion:\n", test_cm)
    print(test_report)

    (out_dir / "test_results.json").write_text(
        json.dumps({"test_accuracy": test_acc, "test_macro_f1": test_f1, "confusion_matrix": test_cm.tolist()}, indent=2)
    )




if __name__ == "__main__":
    main()