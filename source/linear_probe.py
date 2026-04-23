import os
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer

from clip_model import ClipResNetBert

MES_MAP = {"MES-0": 0, "MES-1": 1, "MES-2": 2, "MES-3": 3}

# pyTorch deafult
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def preprocess():
    # Match the image preprocessing expected by the pretrained ResNet encoder
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

@torch.no_grad()
def extract_image_embeddings(model, device, image_root: Path, df: pd.DataFrame):
    # Extract frozen image embeddings from the trained dual encoder.
    pre = preprocess()
    X = []
    y = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extract image emb"):
        img_path = image_root / str(row["img_url"])
        img = pre(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

        # Use only the image branch of the trained multimodal model
        feat = model.image_encoder(img)

        # Project into the shared embedding space learned during contrastive training
        emb = F.normalize(model.image_proj(feat), dim=-1).squeeze(0).cpu()  # (D,)
        X.append(emb)
        y.append(MES_MAP[str(row["mes_scoring_0_3"])])
    X = torch.stack(X, dim=0)
    y = torch.tensor(y, dtype=torch.long)
    return X, y

def main():
    image_root = Path(os.environ["UC_IMAGE_ROOT"])
    device = get_device()

    text_model = "emilyalsentzer/Bio_ClinicalBERT"
    ckpt = Path("runs/clip_rn50_clinicalbert/best.pt")
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}. Train first.")

    # Load dual encoder
    model = ClipResNetBert(text_model_name=text_model, embed_dim=256).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Tokenizer is loaded to keep the text side consistent with the saved model setup
    _ = AutoTokenizer.from_pretrained(text_model)

    train_df = pd.read_csv("splits/train.csv")
    test_df = pd.read_csv("splits/test.csv")

    X_train, y_train = extract_image_embeddings(model, device, image_root, train_df)
    X_test, y_test = extract_image_embeddings(model, device, image_root, test_df)

    # Linear probe
    clf = nn.Linear(X_train.size(1), 4)
    clf = clf.to(device)

    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Train probe
    epochs = 50
    batch_size = 64

    for epoch in range(1, epochs + 1):
        clf.train()
        perm = torch.randperm(X_train.size(0))
        total = 0.0
        n = 0
        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)

            logits = clf(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item())
            n += 1

        # Report a few checkpoints to track how well the frozen embeddings support classification
        if epoch in {1, 5, 10, 25, 50}:
            clf.eval()
            with torch.no_grad():
                preds = clf(X_test.to(device)).argmax(dim=-1).cpu().numpy()
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(y_test.numpy(), preds)
            f1 = f1_score(y_test.numpy(), preds, average="macro")
            print(f"Epoch {epoch:02d}: probe_train_loss={total/max(n,1):.4f}  test_acc={acc:.4f}  test_f1={f1:.4f}")

    # Final report
    clf.eval()
    with torch.no_grad():
        preds = clf(X_test.to(device)).argmax(dim=-1).cpu().numpy()

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    acc = accuracy_score(y_test.numpy(), preds)
    f1 = f1_score(y_test.numpy(), preds, average="macro")
    cm = confusion_matrix(y_test.numpy(), preds, labels=[0,1,2,3])

    print("\nLinear probe on image embeddings (trained CLIP-style model)")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1:  {f1:.4f}")
    print("Confusion:\n", cm)
    print(classification_report(y_test.numpy(), preds, digits=4))

if __name__ == "__main__":
    main()
