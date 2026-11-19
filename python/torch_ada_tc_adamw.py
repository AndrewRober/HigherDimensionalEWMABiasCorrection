"""
torch_ada_tc_adamw.py
=====================
Production-Ready PyTorch `torch.optim.Optimizer` Implementation of:
  1. AdaTCAdamW: Generalized Higher-Dimensional Adaptive Trend- & Exposure-Corrected AdamW.
     - Incorporates Theorems 1, 2, and 3 (High-Dimensional James-Stein / Wiener Shrinkage
       Trend Correction on the 1st moment m_t, pooling ||delta_t||^2 across each parameter
       tensor in R^d to eliminate non-stationary lag without noise amplification).
     - Incorporates Theorem 4 (Per-Row / Sparse Exposure Mass w1, w2 + Innovation Regime-Shift
       Damping, preventing the (1-beta1)/sqrt(1-beta2) = 3.16x - 10x wakeup shock on rare
       tokens, gated MoE experts, and abrupt distribution shifts).
"""

from __future__ import annotations
import math
import torch
from torch.optim import Optimizer


def _closed_form_moments_scalar(beta: float, t: int) -> tuple[float, float, float]:
    """Returns (W_t_0, K_t, q_delta + q_v_delta) in O(1) time for step t >= 2."""
    bt = beta ** t
    one_m_b = 1.0 - beta
    one_m_bt = max(1.0 - bt, 1e-15)
    W_t_0 = max(1.0 - bt - t * bt * one_m_b, 1e-15)

    L_t = (beta / one_m_b) - (t * bt / one_m_bt)
    M_t_num = (
        2.0 * (beta / one_m_b) * one_m_bt
        - 2.0 * t * bt
        - t * (t - 1.0) * bt * one_m_b
    )
    M_t = M_t_num / W_t_0
    K_t = L_t / max(M_t - L_t, 1e-15)

    r = beta * beta
    rt = bt * bt
    one_m_r = max(1.0 - r, 1e-15)
    S0 = (1.0 - rt) / one_m_r
    S1 = (1.0 - (t + 1.0) * rt + t * rt * r) / (one_m_r ** 2)
    S2 = (
        1.0
        + r
        - ((t + 1.0) ** 2) * rt
        + (2.0 * t * t + 2.0 * t - 1.0) * rt * r
        - (t * t) * rt * r * r
    ) / (one_m_r ** 3)

    sum_a2 = ((one_m_b / one_m_bt) ** 2) * S0
    sum_ac = ((one_m_b ** 3) / (one_m_bt * W_t_0)) * S1
    sum_c2 = ((one_m_b ** 4) / (W_t_0 ** 2)) * S2

    q_delta = (K_t ** 2) * max(sum_a2 - 2.0 * sum_ac + sum_c2, 0.0)
    q_v_delta = K_t * (sum_a2 - sum_ac)
    return W_t_0, K_t, (q_delta + q_v_delta)


class AdaTCAdamW(Optimizer):
    """
    Drop-in PyTorch replacement for `torch.optim.AdamW` implementing Higher-Dimensional
    Adaptive Trend-Corrected & Exposure-Mass-Aware Bias Correction (Theorems 1-4).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        row_wise_exposure: bool = False,
        shift_threshold: float = 5.0,
        shift_damping: float = 0.2,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            row_wise_exposure=row_wise_exposure,
            shift_threshold=shift_threshold,
            shift_damping=shift_damping,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            row_wise = group["row_wise_exposure"]
            shift_thresh = group["shift_threshold"]
            shift_damp = group["shift_damping"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["u"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["g_prev1"] = None
                    state["g_prev2"] = None
                    state["noise_tr_raw"] = 0.0
                    state["noise_steps"] = 0
                    if row_wise and p.ndim >= 2:
                        shape_w = (p.shape[0],) + (1,) * (p.ndim - 1)
                        state["row_step"] = torch.zeros(shape_w, device=p.device, dtype=torch.float32)
                    else:
                        state["w2_scalar"] = 0.0

                state["step"] += 1
                t = state["step"]
                m, u, v = state["m"], state["u"], state["v"]

                if row_wise and p.ndim >= 2:
                    # Per-Row Lazy-Clock Initialization + AdaTC/AdamW Coasting:
                    # Dormant rows (never activated yet) keep row_step == 0 so they NEVER suffer
                    # the (1-beta1)/sqrt(1-beta2) = 3.16x - 10x wakeup shock when first activated,
                    # while preserving full momentum coasting once woken!
                    reduce_dims = tuple(range(1, p.ndim))
                    active_now = (g.abs().amax(dim=reduce_dims, keepdim=True) > 0)
                    row_step = state["row_step"]
                    awake = (row_step > 0) | active_now
                    row_step.add_(awake.to(torch.float32))

                    if wd != 0.0:
                        p.sub_(p * (lr * wd * awake.to(p.dtype)))

                    m.mul_(b1).add_(g, alpha=1.0 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1.0 - b2)

                    bc1 = (1.0 - torch.pow(b1, row_step.clamp_min(1.0))).to(p.dtype)
                    bc2 = (1.0 - torch.pow(b2, row_step.clamp_min(1.0))).to(p.dtype)
                    m_hat = torch.where(awake, m / bc1, torch.zeros_like(m))
                    v_hat = torch.where(awake, v / bc2, torch.ones_like(v))

                    p.add_(awake.to(p.dtype) * (m_hat / (v_hat.sqrt().add_(eps))), alpha=-lr)
                else:
                    # Theorems 1-3 High-Dimensional AdaTC First Moment + Shift-Aware Second Moment
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)

                    # Blind 2nd-difference noise trace estimation in O(d) ops, O(1) extra scalar state
                    gp1, gp2 = state["g_prev1"], state["g_prev2"]
                    if gp1 is not None and gp2 is not None:
                        state["noise_steps"] += 1
                        d2 = g - 2.0 * gp1 + gp2
                        inst_tr = float(torch.sum(d2 * d2).item()) / 6.0
                        state["noise_tr_raw"] = 0.95 * state["noise_tr_raw"] + 0.05 * inst_tr
                        noise_tr_hat = state["noise_tr_raw"] / (1.0 - (0.95 ** state["noise_steps"]))
                    else:
                        noise_tr_hat = float(torch.sum(g * g).item())

                    state["g_prev2"] = gp1
                    state["g_prev1"] = g.clone()

                    m.mul_(b1).add_(g, alpha=1.0 - b1)
                    u.mul_(b1).add_(m, alpha=1.0 - b1)
                    m_hat_0 = m / (1.0 - (b1 ** t))

                    if t >= 3 and p.numel() >= 4:
                        W_t_0, K_t, q_sum = _closed_form_moments_scalar(b1, t)
                        u_hat_0 = u / W_t_0
                        delta_m = (m_hat_0 - u_hat_0).mul_(K_t)
                        denom_m = float(torch.sum(delta_m * delta_m).item())
                        gamma_m = max(0.0, min(1.0, 1.0 - (noise_tr_hat * q_sum) / max(denom_m, 1e-18)))
                        m_hat = m_hat_0.add(delta_m, alpha=gamma_m)
                    else:
                        m_hat = m_hat_0

                    # Shift-aware second moment exposure mass
                    w2_s = state["w2_scalar"]
                    if w2_s > 0.5 and shift_thresh > 0:
                        g_norm2 = float(torch.sum(g * g).item())
                        v_norm1 = float(torch.sum(v).item()) / w2_s
                        if g_norm2 / max(v_norm1, 1e-15) > shift_thresh:
                            v.mul_(shift_damp)
                            w2_s *= shift_damp

                    v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                    w2_s = b2 * w2_s + (1.0 - b2)
                    state["w2_scalar"] = w2_s
                    v_hat = v / max(w2_s, 1e-15)

                    p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

        return loss
