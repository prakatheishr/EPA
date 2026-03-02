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


if __name__ == "__main__":
    main()