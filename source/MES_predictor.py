import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer

from clip_model import ClipResNetBert

# uses trained image encoder and prompted text encoder to do MES prediction (no classifier training)
MES_MAP = {"MES-0": 0, "MES-1": 1, "MES-2": 2, "MES-3": 3}
INV_MES = {v: k for k, v in MES_MAP.items()}

PROMPTS: Dict[int, List[str]] = {
    0: ["Mayo endoscopic score 0", "normal mucosa", "no visible inflammation"],
    1: ["Mayo endoscopic score 1", "mild inflammation", "mild erythema"],
    2: ["Mayo endoscopic score 2", "moderate inflammation", "marked erythema and friability"],
    3: ["Mayo endoscopic score 3", "severe inflammation", "ulceration and bleeding"],
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def image_preprocess():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

@torch.no_grad()
def encode_prompt_prototypes(model, tokenizer, device):
    class_embs = []
    for c in range(4):
        texts = PROMPTS[c]
        tok = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids = tok["input_ids"].to(device)
        attn = tok["attention_mask"].to(device)

        # text backbone + projection
        txt_feat = model.text_encoder(input_ids, attn)
        txt_emb = F.normalize(model.text_proj(txt_feat), dim=-1)  # (k, D)
        proto = F.normalize(txt_emb.mean(dim=0, keepdim=True), dim=-1)  # (1, D)
        class_embs.append(proto)

    class_embs = torch.cat(class_embs, dim=0)  # (4, D)
    class_embs = F.normalize(class_embs, dim=-1)
    return class_embs

def main():
    image_root = os.environ["UC_IMAGE_ROOT"]
    device = get_device()

    test_csv = "splits/test.csv"
    ckpt = Path("runs/clip_rn50_clinicalbert/best.pt")
    text_model = "emilyalsentzer/Bio_ClinicalBERT"

    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}. Train first.")

    model = ClipResNetBert(text_model_name=text_model, embed_dim=256).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(text_model)
    preprocess = image_preprocess()

    class_embs = encode_prompt_prototypes(model, tokenizer, device)  # (4, D)

    df = pd.read_csv(test_csv)

    y_true = []
    y_pred = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Zero-shot MES"):
        img_path = Path(image_root) / str(row["img_url"])
        img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            img_feat = model.image_encoder(img)
            img_emb = F.normalize(model.image_proj(img_feat), dim=-1)  # (1, D)
            sims = (img_emb @ class_embs.T).squeeze(0)                # (4,)
            pred = int(torch.argmax(sims).item())

        y_true.append(MES_MAP[str(row["mes_scoring_0_3"])])
        y_pred.append(pred)

    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])

    print("Zero-shot MES using prompts (trained dual-encoder)")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1:  {f1:.4f}")
    print("Confusion:\n", cm)
    print(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    main()
