"""
benchmark_gpu_transformer.py
============================
End-to-End PyTorch CUDA GPU Validation on NVIDIA GeForce RTX 5060:
Trains a Multi-Head Self-Attention + Gated Mixture-of-Experts (MoE) Sequence Model
(d ~ 280,000 parameters) on Zipfian token sequences with mid-training curriculum shift,
comparing `torch.optim.AdamW` vs. `AdaTCAdamW` (drop-in PyTorch optimizer).
"""

import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_ada_tc_adamw import AdaTCAdamW

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


class TinyMoETransformer(nn.Module):
    def __init__(self, vocab_size: int = 1024, d_model: int = 64, n_heads: int = 4, n_experts: int = 8):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 32, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.router = nn.Linear(d_model, n_experts)
        # Stack expert weights as (n_experts, d_model, 2*d_model) and (n_experts, 2*d_model, d_model)
        # so row_wise_exposure=True tracks exact per-expert activation mass w1, w2!
        self.expert_w1 = nn.Parameter(torch.randn(n_experts, d_model, 2 * d_model) / (d_model ** 0.5))
        self.expert_w2 = nn.Parameter(torch.randn(n_experts, 2 * d_model, d_model) / ((2 * d_model) ** 0.5))
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, L = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :L, :]
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)

        # Top-1 Gated MoE (sparse expert activation — rare experts stay dormant until needed!)
        logits_r = self.router(x)  # (B, L, E)
        probs_r = F.softmax(logits_r, dim=-1)
        top_val, top_idx = probs_r.max(dim=-1)  # (B, L)

        moe_out = torch.zeros_like(x)
        for e in range(self.expert_w1.shape[0]):
            mask = (top_idx == e)
            if mask.any():
                xe = x[mask]  # (N_e, D)
                he = F.gelu(xe @ self.expert_w1[e])
                ye = he @ self.expert_w2[e]
                moe_out[mask] = ye * top_val[mask].unsqueeze(-1)

        x = self.ln2(x + moe_out)
        return self.head(x)


def run_gpu_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"=== PyTorch GPU Benchmark on: {gpu_name} ({device}) ===")

    vocab_size = 1024
    seq_len = 24
    batch_size = 48
    steps = 300
    n_seeds = 3

    # Zipfian vocabulary frequencies
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    probs = (ranks + 2.0) ** (-1.15)
    probs /= probs.sum()

    results = {"device": gpu_name, "std_adamw": [], "adatc_adamw": []}
    t0 = time.perf_counter()

    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        # Pre-generate synthetic Markov/bigram + non-stationary rule shift at step 150
        # Target next token depends on (tok_t * mult_a + tok_{t-1} * mult_b) mod vocab_size
        batches = []
        for s in range(steps):
            seq = rng.choice(vocab_size, size=(batch_size, seq_len), p=probs)
            if s < 150:
                y = (seq * 3 + np.roll(seq, 1, axis=1) * 5 + 7) % vocab_size
            else:
                # Mid-training task shift at step 150!
                y = (seq * 7 + np.roll(seq, 1, axis=1) * 11 + 19) % vocab_size
            batches.append((torch.tensor(seq, dtype=torch.long, device=device),
                            torch.tensor(y, dtype=torch.long, device=device)))

        torch.manual_seed(42 + seed)
        base_model = TinyMoETransformer(vocab_size=vocab_size).to(device)
        n_params = sum(p.numel() for p in base_model.parameters())
        if seed == 0:
            print(f"  Model Parameters: d = {n_params:,}")

        for opt_key in ["std_adamw", "adatc_adamw"]:
            model = TinyMoETransformer(vocab_size=vocab_size).to(device)
            model.load_state_dict(base_model.state_dict())

            if opt_key == "std_adamw":
                opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.999), weight_decay=1e-2)
            else:
                sparse_params = [model.tok_emb.weight, model.expert_w1, model.expert_w2]
                sparse_ids = {id(p) for p in sparse_params}
                dense_params = [p for p in model.parameters() if id(p) not in sparse_ids]
                opt = AdaTCAdamW([
                    {"params": dense_params, "row_wise_exposure": False},
                    {"params": sparse_params, "row_wise_exposure": True},
                ], lr=3e-3, betas=(0.9, 0.999), weight_decay=1e-2, shift_threshold=4.5)

            losses = []
            for s in range(steps):
                xb, yb = batches[s]
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

            results[opt_key].append(losses)

    elapsed = time.perf_counter() - t0
    std_arr = np.mean(results["std_adamw"], axis=0)
    ada_arr = np.mean(results["adatc_adamw"], axis=0)

    summary = {
        "gpu": gpu_name,
        "num_parameters": int(n_params),
        "elapsed_sec": float(elapsed),
        "std_adamw_overall_ce": float(np.mean(std_arr)),
        "adatc_adamw_overall_ce": float(np.mean(ada_arr)),
        "std_adamw_post_shift_150_220": float(np.mean(std_arr[150:220])),
        "adatc_adamw_post_shift_150_220": float(np.mean(ada_arr[150:220])),
        "std_adamw_final_30": float(np.mean(std_arr[-30:])),
        "adatc_adamw_final_30": float(np.mean(ada_arr[-30:])),
    }
    print(f"  Completed 3 seeds x 2 optimizers on {gpu_name} in {elapsed:.2f}s!")
    print(f"  Overall Cross-Entropy:    torch.optim.AdamW = {summary['std_adamw_overall_ce']:.4f} | AdaTCAdamW = {summary['adatc_adamw_overall_ce']:.4f}")
    print(f"  Post-Shift [150..220] CE: torch.optim.AdamW = {summary['std_adamw_post_shift_150_220']:.4f} | AdaTCAdamW = {summary['adatc_adamw_post_shift_150_220']:.4f}")
    print(f"  Final [270..300] CE:      torch.optim.AdamW = {summary['std_adamw_final_30']:.4f} | AdaTCAdamW = {summary['adatc_adamw_final_30']:.4f}")

    with open(RESULTS_DIR / "gpu_moe_transformer.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    run_gpu_benchmark()
