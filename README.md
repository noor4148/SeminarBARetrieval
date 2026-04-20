# Retrieval GTM for Zero-Shot Sales Forecasting

This project contains a zero-shot sales forecasting pipeline with two variants:

1. Baseline forecasting with `GTM`
2. Retrieval-augmented forecasting with `HybridRetrievalGTM`, which adds an extra analog sales curve retrieved from a retrieval memory

The code combines multiple modalities:
- product metadata (category, color, fabric)
- product images
- Google Trends time series
- temporal features
- optionally, a retrieved analog sales curve for the extension

---

## 1. Workflow Overview

### Baseline pipeline
1. Train a baseline model with `train.py`
2. Evaluate / forecast with `forecast.py`

### Retrieval-augmented pipeline
1. First train a baseline `GTM` model
2. Build a retrieval memory with `build_hybrid_retrieval_memory.py`
3. Train the hybrid model with `train_hybrid_retrieval.py`
4. Evaluate / forecast with `forecast_hybrid_retrieval.py`

---

## 2. Project Structure

The imports in the code assume the repository is organized like this:

```text
project/
├─ train.py
├─ forecast.py
├─ build_hybrid_retrieval_memory.py
├─ train_hybrid_retrieval.py
├─ forecast_hybrid_retrieval.py
├─ models/
│  ├─ GTM.py
│  ├─ GTM_hybrid_retrieval.py
├─ utils/
│  └─ data_multitrends.py
├─ dataset/
│  ├─ train.csv
│  ├─ test.csv
│  ├─ gtrends.csv
│  ├─ normalization_scale.npy
│  ├─ category_labels.pt
│  ├─ color_labels.pt
│  ├─ fabric_labels.pt
│  └─ images/
└─ log/
```
---

## 3. Required Dependencies

At minimum, you will need these Python packages:

```bash
pip install torch torchvision pytorch-lightning pandas numpy scikit-learn tqdm pillow transformers
```

Notes:
- The text encoder uses Hugging Face `transformers` with `bert-base-uncased`.
- The image encoder uses a pretrained `ResNet50` from `torchvision`.

---

## 4. Expected Input Data

### Files inside `dataset/`
- `train.csv`
- `test.csv`
- `gtrends.csv`
- `normalization_scale.npy`
- `category_labels.pt`
- `color_labels.pt`
- `fabric_labels.pt`
- `images/` directory containing product images

### CSV requirements
The preprocessing expects at least the following columns:
- `external_code`
- `season`
- `release_date`
- `image_path`
- `category`
- `color`
- `fabric`
- `extra`
- forecast target columns for the prediction horizon (by default 12 weeks, usually `0` through `11`)
- temporal feature columns

### Important: column order matters
The dataset preprocessing is written in a positional way. After dropping
`external_code`, `season`, `release_date`, and `image_path`:
- the first 12 columns are interpreted as sales targets
- columns 13 through 16 are interpreted as temporal features

Make sure your CSV layout matches this expectation.

### Google Trends
`gtrends.csv` is read with a datetime index. For each product, the code retrieves the last 52 weeks of trend data for:
- category
- color
- fabric

These three trends are then normalized independently per product and stacked together.

---

## 5. Models

### GTM
`GTM` is a multimodal forecasting model that uses:
- an image encoder (`ResNet50`)
- a text encoder based on `bert-base-uncased`
- a dummy / temporal encoder for day/week/month/year features
- a Google Trends encoder based on a Transformer encoder
- a Transformer decoder for the final forecast

### HybridRetrievalGTM
`HybridRetrievalGTM` inherits from `GTM` and adds an extra encoder for a retrieved analog sales curve. This retrieved curve is appended as extra decoder memory alongside the Google Trends encoding.

---

## 6. Training the Baseline Model

### Train GTM
```bash
python train.py \
  --data_folder dataset/ \
  --log_dir log \
  --model_type GTM \
  --batch_size 128 \
  --embedding_dim 32 \
  --hidden_dim 64 \
  --output_dim 12 \
  --wandb_run baseline_gtm
```

### What happens during training?
- `train.csv` is loaded
- the data is sorted by `release_date`
- a time-based split is created with 85% subtrain / 15% validation
- the best checkpoint is saved based on `val_wape`
- TensorBoard logging is used

Output:
- best model checkpoint in `log/<model_type>/...`

---

## 7. Forecasting with the Baseline Model

```bash
python forecast.py \
  --data_folder dataset/ \
  --ckpt_path log/GTM/<best_model>.ckpt \
  --model_type GTM \
  --model_output_dim 12 \
  --eval_horizon 6 \
  --wandb_run baseline_gtm
```

### Output
The script:
- loads `test.csv`
- generates forecasts
- rescales the predictions using `normalization_scale.npy`
- prints metrics in both normalized and original scale
- saves results to:

```text
results/<run_name>_model<model_output_dim>_eval<eval_horizon>.pth
```

The output `.pth` file contains:
- `results`
- `gts`
- `codes`

---

## 8. Building the Retrieval Memory

The retrieval memory is required for the hybrid variant.

```bash
python build_hybrid_retrieval_memory.py \
  --data_folder dataset/ \
  --train_csv dataset/train.csv \
  --test_csv dataset/test.csv \
  --checkpoint_path log/GTM/<best_baseline_model>.ckpt \
  --output_path artifacts/retrieval_memory.pth \
  --neighbors_csv artifacts/retrieval_neighbors.csv \
  --horizon_weeks 12 \
  --top_k 15 \
  --min_similarity 0.95
```

### What does this script do?
1. Loads the trained baseline `GTM` model
2. Extracts a multimodal embedding for each product based on image + text
3. Computes cosine similarity between products
4. Applies an admissibility mask so that only valid analogs remain
5. Computes a weighted retrieved sales curve

### Valid retrieval neighbors must satisfy:
- the neighbor must come from subtrain
- the neighbor must not have the same `external_code`
- the neighbor must be far enough in the past so that:

  `neighbor_release_date + horizon <= query_release_date`

### Output
- `retrieval_memory.pth` containing, among other things:
  - `metadata`
  - `embeddings`
  - `topk_scores`
  - `topk_indices`
  - `similarity_weights`
  - `retrieval_curve`
  - `retrieval_available`
- optionally: `retrieval_neighbors.csv` for easier inspection

---

## 9. Training the Hybrid Model

```bash
python train_hybrid_retrieval.py \
  --data_folder dataset/ \
  --retrieval_memory_path artifacts/retrieval_memory.pth \
  --log_dir log \
  --run_name hybrid_gtm \
  --output_dim 12
```

### What happens here?
- `train.csv` is split again using a time-based split
- for each product, the corresponding `retrieval_curve` is matched through `external_code`
- that curve is added as an extra model input
- the best checkpoint is saved based on `val_wape`

Output:
- best checkpoint in `log/GTM_hybrid/...`

---

## 10. Forecasting with the Hybrid Model

```bash
python forecast_hybrid_retrieval.py \
  --data_folder dataset/ \
  --retrieval_memory_path artifacts/retrieval_memory.pth \
  --ckpt_path log/GTM_hybrid/<best_model>.ckpt \
  --run_name hybrid_gtm \
  --model_output_dim 12 \
  --eval_horizon 6
```

### Output
The output is saved as:

```text
results/<run_name>_model<model_output_dim>_eval<eval_horizon>.pth
```

This `.pth` file contains:
- `results`
- `gts`
- `codes`
- `attns`

---

## 11. Main CLI Arguments

### Baseline training (`train.py`)
- `--data_folder`: path to the dataset directory
- `--log_dir`: directory for checkpoints and logs
- `--model_type`: `GTM` or `FCN`
- `--epochs`
- `--batch_size`
- `--embedding_dim`
- `--hidden_dim`
- `--output_dim`
- `--trend_len`
- `--num_trends`
- `--use_img`
- `--use_text`
- `--use_trends`
- `--use_encoder_mask`
- `--autoregressive`
- `--num_attn_heads`
- `--num_hidden_layers`
- `--wandb_run`

### Baseline forecasting (`forecast.py`)
- `--ckpt_path`
- `--model_output_dim`
- `--eval_horizon`

> `eval_horizon` must not be larger than `model_output_dim`.

### Retrieval memory (`build_hybrid_retrieval_memory.py`)
- `--train_csv`
- `--test_csv`
- `--checkpoint_path`
- `--output_path`
- `--neighbors_csv`
- `--val_frac`
- `--horizon_weeks`
- `--top_k`
- `--min_similarity`

### Hybrid training / forecasting
- `--retrieval_memory_path`
- `--run_name`
- plus mostly the same model arguments as the baseline scripts

---

## 12. Metrics

The evaluation scripts report both a normalized and a rescaled version of:
- WAPE
- MAE
- TS
- ERP

Rescaling uses `normalization_scale.npy`.


---

## 13. Summary

This repository implements a multimodal zero-shot forecasting pipeline for products. The baseline `GTM` combines text, images, temporal features, and Google Trends. The retrieval-augmented variant extends this with an additional analog sales memory derived from embeddings of historically similar products.
