# Multimodal Learning for Ulcerative Colitis Severity Assessment and Clinical Caption Generation

## Overview

This project develops a multimodal vision–language system for analysing colonoscopy images in ulcerative colitis (UC). The system integrates image and text data to:

- Predict Mayo Endoscopic Score (MES) (0–3)
- Learn joint multimodal representations using a CLIP-style dual encoder
- Generate clinically structured captions explaining findings and severity

The pipeline progresses through three stages:

1. **Representation Learning (CLIP-style)**
2. **Multimodal Classification**
3. **Image-conditioned Caption Generation**

---

## Motivation

Traditional medical image classification models provide predictions but lack interpretability. This project aims to:

- Evaluate whether vision–language models can capture clinically meaningful features
- Compare different encoder architectures
- Extend classification into explainable caption generation

---

## Features

- CLIP-style dual encoder architecture
- Comparison of multiple encoders:
  - Image: ResNet50, ViT-B/16
  - Text: Bio_ClinicalBERT, BERT-large, GPT-2
- Zero-shot classification
- Linear probe evaluation
- Multimodal classifier (fusion-based)
- Prefix-based caption generation (GPT-2)
- Full evaluation pipeline:
  - Classification: Accuracy, Macro F1
  - Captioning: BLEU, ROUGE

---

## Methodology

### 1. Representation Learning (CLIP-style)

A dual encoder model learns aligned embeddings:

- Image encoder → image embedding
- Text encoder → text embedding
- Contrastive loss (InfoNCE) maximises similarity between matching pairs

### 2. Classification

A multimodal classifier is trained on frozen embeddings:

```
[image_emb ; text_emb] → MLP → MES class
```

### 3. Caption Generation

A GPT-2 model is conditioned on image embeddings using prefix tuning:

```
Image → Encoder → Embedding (256D)
↓
Concatenate zeros → 512D
↓
Prefix projection (MLP)
↓
GPT-2 (autoregressive decoding)
↓
Clinical caption with MES
```

> **Key design decision:** Image-only conditioning is used to prevent text leakage.

---

## Dataset

The dataset consists of:

- Colonoscopy images
- Expert-written clinical descriptions
- MES labels (0–3)

### Preprocessing

To prevent label leakage, text was cleaned to remove explicit severity indicators such as `"MES-2"`, `"supports a 3"`, and `"grade 1"`. This ensures the model learns true visual-text relationships.

---

## Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

Set dataset paths before running any scripts:

```bash
export UC_DATA_CSV=path/to/data.csv
export UC_IMAGE_ROOT=path/to/image_folder
```

---

## Training Pipeline

### Step 1: Train CLIP Model

```bash
python source/train_model.py \
  --image_encoder vit_b_16 \
  --text_encoder emilyalsentzer/Bio_ClinicalBERT \
  --output_dir runs/clip_vit16_clinicalbert
```

### Step 2: Train Multimodal Classifier

```bash
python source/train_multimodal_model.py \
  --image_encoder vit_b_16 \
  --text_encoder emilyalsentzer/Bio_ClinicalBERT \
  --foundation_ckpt runs/clip_vit16_clinicalbert/best.pt
```

### Step 3: Train Caption Model

```bash
python source/train_caption_model.py \
  --image_encoder vit_b_16 \
  --text_encoder emilyalsentzer/Bio_ClinicalBERT \
  --foundation_ckpt runs/clip_vit16_clinicalbert/best.pt \
  --gpt_name gpt2 \
  --out_dir runs/caption_vit16_gpt2 \
  --epochs 10 \
  --batch_size 8 \
  --prefix_len 5
```

---

## Inference

Generate captions for a single image:

```bash
python source/generate_captions.py \
  --image_path "path/to/image.jpg" \
  --image_encoder vit_b_16 \
  --text_encoder emilyalsentzer/Bio_ClinicalBERT \
  --foundation_ckpt runs/clip_vit16_clinicalbert/best.pt \
  --gpt_name gpt2 \
  --caption_ckpt runs/caption_vit16_gpt2/best_captioner.pt
```

---

## Evaluation

### Classification Metrics

- Accuracy
- Macro F1 Score
- Confusion Matrix

### Captioning Metrics

```bash
python source/eval_captions.py \
  --split_csv splits/test.csv \
  --caption_ckpt runs/caption_vit16_gpt2/best_captioner.pt
```

Metrics reported: BLEU, ROUGE-1, ROUGE-2, ROUGE-L

---

## Results

### Classification Performance

| Encoder Combination         | Accuracy | Macro Avg F1 | Macro Avg Recall | Macro Avg Precision |
|-----------------------------|----------|--------------|------------------|---------------------|
| ResNet50 + BERT-Large        | 0.8716   | 0.8795       | 0.8682           | 0.8972              |
| ResNet50 + GPT-2             | 0.8986   | 0.8934       | 0.8713           | 0.9260              |
| ResNet50 + ClinicalBERT      | 0.8919   | 0.8995       | 0.8937           | 0.9059              |
| ViT-B/16 + BERT-Large        | 0.8446   | 0.8615       | 0.8487           | 0.8776              |
| ViT-B/16 + GPT-2             | 0.9257   | **0.9279**   | **0.9157**       | 0.9469              |
| **ViT-B/16 + ClinicalBERT** | **0.9324** | 0.9275     | 0.9016           | **0.9655**          |


**Key observations:**
- Strong performance on clear classes (MES-0, MES-2)
- Confusion between adjacent severity levels

### Captioning Performance

| Metric  | Score  |
|---------|--------|
| BLEU    | 29.64  |
| ROUGE-1 | 0.5945 |
| ROUGE-2 | 0.4629 |
| ROUGE-L | 0.5713 |

**Example output:**
> *adequate vascular pattern normal mucosa no visible erosions and superficial ulcers mild bleeding is present MES-2.*

---

## Key Insights

- Multimodal models outperform unimodal approaches
- Vision Transformers provide stronger image representations
- Domain-specific text encoders improve performance
- Caption generation enables interpretability alongside prediction

---

## Limitations

- Small dataset (981 samples)
- Difficulty distinguishing adjacent MES grades
- GPT-2 limits fluency and diversity
- MES grading subjectivity

---

## Future Work

- Larger and more diverse datasets
- Advanced vision-language models (e.g. BLIP, Flamingo)
- Improved caption evaluation using clinical correctness metrics
- Fine-tuned domain-specific language models