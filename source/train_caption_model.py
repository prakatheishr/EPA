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


def collate_caption(batch, tokenizer, max_len: int = 96):
    # Builds caption targets from dataset description (cleaned) + forced MES suffix - Returns: images, input_ids, attention_mask, labels, mes_nums

    images = torch.stack([b["image"] for b in batch], dim=0)

    captions = []
    for b in batch:
        body = clean_caption_body(b["text"])
        mes_num = int(b["label"].item()) if hasattr(b["label"], "item") else int(b["label"])
        captions.append(format_caption_with_mes(body, mes_num))

    tok = tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    # in GPT-2 language modelling: labels are input_ids
    labels = tok["input_ids"].clone()

    return {
        "images": images,
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "labels": labels,
    }




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

    # datasets - use description as target caption text
    train_ds = UCMultimodalDataset("splits/train.csv", image_root=image_root, train=True)
    val_ds = UCMultimodalDataset("splits/val.csv", image_root=image_root, train=False)




if __name__ == "__main__":
    main()