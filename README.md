# Hybrid retrieval extension for GTM

This package implements the hybrid extension you specified:

1. Learned **image + text** embeddings from `encode_multimodal_embedding(...)` are used for similarity.
2. Retrieval is **causal**: a query product can only retrieve **subtrain** products whose full 12-week horizon is already observable before the query launch.
3. A configurable **minimum similarity threshold** is applied.
4. The remaining candidates are filtered to **top-k**, then weighted with **softmax**.
5. Their real 12-week sales curves are combined into one **weighted retrieved curve**.
6. If no valid neighbor remains, the model falls back to the baseline via `analog_available=False`.
7. The retrieved curve is fed into GTM as **extra decoder memory**.

## Files

- `models/GTM_hybrid_retrieval.py`
  - Hybrid GTM model with retrieval memory encoder and `analog_available` masking.
- `build_hybrid_retrieval_memory.py`
  - Offline builder that creates embeddings, causal filtering, thresholding, top-k neighbors, weights, retrieved curves, and an inspection CSV.
- `train_hybrid_retrieval.py`
  - Training script for the hybrid model.
- `forecast_hybrid_retrieval.py`
  - Forecast script for the hybrid model.

## Assumptions

These files are designed to live in the **same project structure as your baseline**:

- baseline `models/GTM.py` remains available
- baseline `utils/data_multitrends.py` remains available
- dataset layout matches the baseline (`train.csv`, `test.csv`, `gtrends.csv`, `images/`, label `.pt` files)

## Recommended pipeline

### 1) Train the baseline GTM first
Use your existing baseline `train.py` to get a checkpoint for the base GTM.

### 2) Build retrieval memory offline
Example:

```bash
python build_hybrid_retrieval_memory.py \
  --data_folder dataset/ \
  --train_csv dataset/train.csv \
  --test_csv dataset/test.csv \
  --checkpoint_path log/GTM/your_baseline.ckpt \
  --output_path artifacts/hybrid_retrieval_memory.pt \
  --neighbors_csv artifacts/hybrid_neighbors.csv \
  --top_k 5 \
  --min_similarity 0.2
```

### 3) Train the hybrid model

```bash
python train_hybrid_retrieval.py \
  --data_folder dataset/ \
  --retrieval_memory_path artifacts/hybrid_retrieval_memory.pt \
  --run_name HybridRun1
```

### 4) Forecast with the hybrid model

```bash
python forecast_hybrid_retrieval.py \
  --data_folder dataset/ \
  --retrieval_memory_path artifacts/hybrid_retrieval_memory.pt \
  --ckpt_path log/GTM_hybrid/your_hybrid.ckpt \
  --run_name HybridRun1 \
  --model_output_dim 12 \
  --eval_horizon 12
```

## Notes

- The offline retrieval builder reproduces the **same chronological split logic** as training: the last `val_frac` part of `train.csv` is labeled as validation, and only **subtrain** is allowed as a retrieval bank.
- Self-retrieval is explicitly blocked by `external_code`.
- The saved `.pt` contains the metadata, embeddings, top-k neighbors, weights, retrieval curves, and availability flags.
- The optional neighbors CSV makes it easy to inspect which analog products were retrieved for each query.
