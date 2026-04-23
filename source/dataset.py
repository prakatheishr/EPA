from pathlib import Path
from typing import Dict, Any

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

MES_MAP = {"MES-0": 0, "MES-1": 1, "MES-2": 2, "MES-3": 3}

# pyTorch default
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

class UCMultimodalDataset(Dataset):
    def __init__(self, csv_path: str, image_root: str, train: bool):
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.transform = build_transform(train=train)

        for col in ["img_url", "description", "mes_scoring_0_3"]:
            if col not in self.df.columns:
                raise ValueError(f"Missing column in CSV: {col}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_rel = str(row["img_url"])
        img_path = self.image_root / img_rel

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Keep the clinical description as raw text for tokenisation later in the pipeline
        text = "" if pd.isna(row["description"]) else str(row["description"])
        label = torch.tensor(MES_MAP[str(row["mes_scoring_0_3"])], dtype=torch.long)

        return {"image": image, "text": text, "label": label}
