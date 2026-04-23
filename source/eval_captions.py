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


@torch.no_grad()
def main():
    args = parse_args()

    image_root = os.environ.get("UC_IMAGE_ROOT")
    if not image_root:
        raise SystemExit("Set UC_IMAGE_ROOT env var first.")

    device = get_device(force_cpu=args.force_cpu)
    print("Device:", device)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.gpt_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Foundation
    foundation = ClipDualEncoder(
        image_encoder_name=args.image_encoder,
        text_encoder_name=args.text_encoder,
        embed_dim=256,
        train_image_backbone=False,
        train_text_backbone=False,
    ).to(device)

    foundation_ckpt = Path(args.foundation_ckpt)
    if not foundation_ckpt.exists():
        raise SystemExit(f"Foundation ckpt not found: {foundation_ckpt}")
    foundation.load_state_dict(torch.load(foundation_ckpt, map_location=device))
    foundation.eval()
    for p in foundation.parameters():
        p.requires_grad = False

    if not hasattr(foundation, "encode_image"):
        raise RuntimeError("ClipDualEncoder must expose encode_image(images).")

    # Captioner
    captioner = PrefixCaptioner(
        gpt_name=args.gpt_name,
        fused_dim=512,
        prefix_len=args.prefix_len,
        dropout=0.0,
        freeze_gpt=True,
    ).to(device)

    caption_ckpt = Path(args.caption_ckpt)
    if not caption_ckpt.exists():
        raise SystemExit(f"Caption ckpt not found: {caption_ckpt}")
    captioner.load_state_dict(torch.load(caption_ckpt, map_location=device))
    captioner.eval()

    # ensure pad token is consistent
    captioner.gpt.config.pad_token_id = tokenizer.eos_token_id

    # Dataset
    ds = UCMultimodalDataset(args.split_csv, image_root=image_root, train=False)
    if args.limit and args.limit > 0:
        ds.df = ds.df.iloc[: args.limit].reset_index(drop=True)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_eval,
    )

    rows = []
    for batch in tqdm(loader, desc="Generating captions"):
        images = batch["images"].to(device)
        texts = batch["texts"]
        labels = batch["labels"].tolist()
        img_paths = batch["img_paths"]

        # conditioning vector: [img_emb ; zeros] (must match training)
        img_emb = foundation.encode_image(images)  # (B,256)
        fused = torch.cat([img_emb, torch.zeros_like(img_emb)], dim=-1)  # (B,512)

        for i in range(images.size(0)):
            pred = captioner.generate(
                fused=fused[i : i + 1],
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                num_beams=1,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )

            # Build reference in same format you trained on
            body = clean_caption_body(texts[i])
            mes_num = int(labels[i]) if not hasattr(labels[i], "item") else int(labels[i].item())
            ref = format_caption_with_mes(body, mes_num)

            rows.append(
                {
                    "img_path": img_paths[i],
                    "mes_gt": mes_num,
                    "ref_text": ref,
                    "pred_caption": pred,
                }
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("Saved captions CSV:", out_csv)

    # --- Metrics: BLEU + ROUGE ---
    preds = df["pred_caption"].astype(str).tolist()
    refs = df["ref_text"].astype(str).tolist()

    # sacreBLEU expects list-of-list references
    bleu = evaluate.load("sacrebleu")
    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
    bleu_score = float(bleu_res["score"])

    rouge = evaluate.load("rouge")
    rouge_res = rouge.compute(predictions=preds, references=refs)
    rouge1 = float(rouge_res["rouge1"])
    rouge2 = float(rouge_res["rouge2"])
    rougeL = float(rouge_res["rougeL"])

    metrics = {
        "split_csv": args.split_csv,
        "n": int(len(df)),
        "bleu": bleu_score,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "gen_params": {
            "prompt": args.prompt,
            "prefix_len": args.prefix_len,
            "do_sample": bool(args.do_sample),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))
    print("\nBLEU:", round(bleu_score, 4))
    print("ROUGE-1:", round(rouge1, 4))
    print("ROUGE-2:", round(rouge2, 4))
    print("ROUGE-L:", round(rougeL, 4))
    print("Saved metrics JSON:", out_json)


if __name__ == "__main__":
    main()