# Python Research & PyTorch Optimizer Suite (`EWMABiasCorrectionResearch/python`)

This directory contains the symbolic proof verification, reference higher-dimensional operator library, reproducible Monte Carlo signal benchmarks, and a drop-in PyTorch `AdaTCAdamW` optimizer.

## Quick Start

```powershell
# 1. Verify all 5 higher-dimensional theorems symbolically (SymPy) & numerically
python verify_proofs.py

# 2. Run automated unit & edge-case tests (extreme betas, non-symmetric matrices, sparse masks)
python test_all.py

# 3. Reproduce Figures 1-4 & R^d trajectory benchmarks (d = 1 .. 1024)
python benchmark_signals_hd.py

# 4. Reproduce Figure 5 (Zipfian SFT d=64k, Non-Stationary MLP d=17.7k, Batch-Size Asymmetry)
python benchmark_optimizers_nn.py

# 5. Run CUDA GPU Mixture-of-Experts Transformer benchmark
python benchmark_gpu_transformer.py
```

## Using `AdaTCOperatorEWMA` in Python / NumPy

```python
import numpy as np
from ewma_hd import AdaTCOperatorEWMA

# Track a 128-dimensional anisotropic non-stationary stream with zero oracle hyperparameters
betas = np.linspace(0.80, 0.99, 128)
tracker = AdaTCOperatorEWMA(d=128, beta=betas)

for t in range(1, 200):
    x_t = np.sin(0.05 * t) * np.ones(128) + np.random.randn(128) * 0.2
    out = tracker.step(x_t)
    estimate = out["adatc"]       # Optimal James-Stein / Wiener trend- & init-corrected vector
    gate = out["gamma_hat"]       # Online shrinkage gate in [0, 1]
```

## Using `AdaTCAdamW` in PyTorch (Drop-In `AdamW` Replacement)

```python
import torch
from torch_ada_tc_adamw import AdaTCAdamW

# Separate dense layers (Theorems 1-3 trend + shift-aware moments)
# from sparse rows like token embeddings / MoE expert tensors (Theorem 4 exact row exposure w1, w2)
optimizer = AdaTCAdamW([
    {"params": dense_params, "row_wise_exposure": False},
    {"params": [model.tok_emb.weight, model.moe_experts], "row_wise_exposure": True},
], lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01)
```
