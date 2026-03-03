import os
import json
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

import evaluate  # BLEU + ROUGE

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


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--split_csv", default="splits/test.csv")
    p.add_argument("--image_encoder", default="vit_b_16")
    p.add_argument("--text_encoder", default="emilyalsentzer/Bio_ClinicalBERT")
    p.add_argument("--foundation_ckpt", required=True)

    p.add_argument("--gpt_name", default="gpt2")
    p.add_argument("--caption_ckpt", required=True)
    p.add_argument("--prefix_len", type=int, default=5)

    # generation params (match what worked for you)
    p.add_argument("--prompt", default="Findings: ")
    p.add_argument("--max_new_tokens", type=int, default=60)
    p.add_argument("--min_new_tokens", type=int, default=10)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.65)
    p.add_argument("--top_p", type=float, default=0.85)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--no_repeat_ngram_size", type=int, default=4)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="0 = evaluate all rows")

    p.add_argument("--out_csv", default="runs/caption_eval/test_all_captions.csv")
    p.add_argument("--out_json", default="runs/caption_eval/test_metrics.json")
    p.add_argument("--force_cpu", action="store_true")
    return p.parse_args()


def collate_eval(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    texts = [b["text"] for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)
    img_paths = [b.get("img_path", "") for b in batch]  # may or may not exist
    return {"images": images, "texts": texts, "labels": labels, "img_paths": img_paths}



if __name__ == "__main__":
    main()