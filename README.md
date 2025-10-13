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


## 🧪 Example Usage

### 🔧 Full Finetuning (text encoder + projection)

```bash
python src/train_injection.py \
  --json finetune_800.jsonl \
  --lora_rank 4 \
  --mode "all" \
  --epochs 1
```
### 🔧 Lora-only Finetuning

```bash
python src/train_injection.py \
  --json finetune_800.jsonl \
  --lora_rank 4 \
  --mode "lora_only" \
  --epochs 1
```

### 🔧 Projection-Only Finetuning

```bash
python src/train_injection.py \
  --json finetune_800.jsonl \
  --lora_rank 4 \
  --mode "proj_only" \
  --epochs 1
```

## 🧪 Evaluation


#### 🔧 Extract embeddings

```bash
python src/extract_eval_finetune_grid.py \
  --base_model_path microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned \
  --ann_path yourpath/test.jsonl \
  --save_path yourpath/embedding.dpt \
  --mode all \
  --lora_path yourpath/final_model \
  --proj_path yourpath/final_proj.pt
```

#### 2. Run evaluation

```bash
python src/eval_llm2clip.py \
    --yaml yourpath/eval_datasets.yaml \
    --batch-size 512 \
    --save-dir output \
    --prefix llm2clip
```

### 🔧 `eval_datasets.yaml` Format

```yaml
text_feature_path: /scratch-shared/ywang4/wikiart/wikiart_new_clip_features.dpt
json_file: /scratch-shared/ywang4/wikiart/test_1k_new.json
img_root: /scratch-shared/ywang4/wikiart/images
```


