This repository contains the implementation and datasets for the paper:
**"Lightweight Domain Adaptation by Knowledge Injection for Cross-modal Cultural Heritage Retrieval" (ECIR 2026 Short Paper)**.

## Overview
We propose a lightweight text-side knowledge injection framework for multimodal retrieval models.  
The method adapts only the text encoder of LLM2CLIP using LoRA, achieving efficient domain adaptation with minimal cost.

## Code Structure
- `train_injection.py` — fine-tuning script supporting three configurations: Projection-only, Model-only, Full.
- `evaluate_retrieval.py` — compute recall@K on COCO, Flickr30k, and WikiArt.
- `data/` — contains evaluation and injection datasets (see below).
- `requirements.txt` — dependencies.

## 📁 data
This repository provides the evaluation and injection datasets used in our experiments.
### WikiArt-1K
The evaluation set contains 1,000 artwork images paired with natural-language queries for image–text retrieval.

 **json/**  
- `test_1k_raw.json`: Raw dataset used in the thesis
- `test_1k_new.json`: Dataset with injected style names (With-Style condition)
- `test_1k_few.json`: Subset of samples containing only the newly added styles

 **Images**  
*(Due to anonymization, dataset links are withheld and will be shared upon acceptance.)*




