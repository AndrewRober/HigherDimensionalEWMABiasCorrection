"""
ewma_hd.py: Generalized Higher-Dimensional Operator EWMA Library
================================================================
Implements the unified operator framework in R^d (diagonal anisotropic B = diag(beta)
and dense non-commutative contraction matrices B in R^{d x d} with rho(B) < 1):

  1. UncorrectedEWMA          : v_t = B v_{t-1} + (I - B) x_t
  2. AffineBC_EWMA (Thm 1)    : v_hat^(0)_t = (I - B^t)^{-1} (v_t - B^t v_0)
  3. ExactTC_EWMA  (Thm 2)    : v_hat^(TC,1)_t = v_hat^(0)_t + K_t(B) (v_hat^(0)_t - u_hat^(0)_t)
  4. AdaTC_EWMA    (Thm 3)    : Blind high-dimensional James-Stein / Wiener optimal shrinkage
                                v_hat^(TC,gamma_hat_t)_t achieving O(1/d) excess risk over Oracle!
  5. SparseMass_EWMA (Thm 4)  : Exact cumulative-exposure bias correction for time-varying B_t,
                                sparse features/tokens/MoE experts, and regime shifts.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


def closed_form_filter_moments_diag(beta: np.ndarray, t: int):
    """
    Exact O(1)-time closed-form calculation of:
      - L_t(beta), W_t^(0)(beta), M_t(beta), K_t(beta)
      - Quadratic noise variance multipliers q_v(t), q_delta(t), q_v_delta(t)
    for coordinate-wise decay vector beta in (0, 1)^d at integer step t >= 1.
    """
    beta = np.asarray(beta, dtype=np.float64)
    bt = np.power(beta, t)
    one_m_b = 1.0 - beta
    one_m_bt = np.maximum(1.0 - bt, 1e-15)

    # Zeroth-order weight of u_t
    W_t_0 = np.maximum(1.0 - bt - t * bt * one_m_b, 1e-15)

    if t == 1:
        zeros = np.zeros_like(beta)
        ones = np.ones_like(beta)
        return {
            "bt": bt,
            "one_m_bt": one_m_bt,
            "W_t_0": W_t_0,
            "L_t": zeros,
            "M_t": zeros,
            "K_t": zeros,
            "q_v": ones,
            "q_delta": zeros,
            "q_v_delta": zeros,
        }

    # Exact first-moment lag operators L_t and M_t
    L_t = (beta / one_m_b) - (t * bt / one_m_bt)
    M_t_num = (
        2.0 * (beta / one_m_b) * one_m_bt
        - 2.0 * t * bt
        - t * (t - 1.0) * bt * one_m_b
    )
    M_t = M_t_num / W_t_0
    denom_K = np.maximum(M_t - L_t, 1e-15)
    K_t = L_t / denom_K

    # Exact O(1) polylog-geometric sums in r = beta^2
    r = beta * beta
    rt = bt * bt
    one_m_r = np.maximum(1.0 - r, 1e-15)

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

    q_v = sum_a2
    q_delta = (K_t ** 2) * np.maximum(sum_a2 - 2.0 * sum_ac + sum_c2, 0.0)
    q_v_delta = K_t * (sum_a2 - sum_ac)

    return {
        "bt": bt,
        "one_m_bt": one_m_bt,
        "W_t_0": W_t_0,
        "L_t": L_t,
        "M_t": M_t,
        "K_t": K_t,
        "q_v": q_v,
        "q_delta": q_delta,
        "q_v_delta": q_v_delta,
    }


class AdaTCOperatorEWMA:
    """
    Generalized Higher-Dimensional Operator EWMA supporting simultaneously:
      - Uncorrected EWMA: v_t
      - Theorem 1 Affine-Invariant 0th-Order Bias Correction: v_hat^(0)_t
      - Theorem 2 Exact Finite-t 1st-Order Trend Correction: v_hat^(TC,1)_t
      - Theorem 3 Blind High-Dimensional Shrinkage: v_hat^(TC,gamma_hat_t)_t
    Operates in O(d) time and O(d) memory per step for diagonal/isotropic B.
    """

    def __init__(
        self,
        d: int,
        beta: float | np.ndarray = 0.9,
        v0: np.ndarray | None = None,
        u0: np.ndarray | None = None,
        noise_beta: float = 0.95,
    ):
        self.d = d
        if np.ndim(beta) == 0:
            self.beta = np.full(d, float(beta), dtype=np.float64)
        else:
            self.beta = np.asarray(beta, dtype=np.float64).copy()
            assert self.beta.shape == (d,)

        self.v0 = np.zeros(d, dtype=np.float64) if v0 is None else np.asarray(v0, dtype=np.float64).copy()
        self.u0 = self.v0.copy() if u0 is None else np.asarray(u0, dtype=np.float64).copy()
        self.v = self.v0.copy()
        self.u = self.u0.copy()
        self.t = 0

        # Online trend-invariant 2nd-difference noise variance state (O(d) memory)
        self.noise_beta = float(noise_beta)
        self.x_prev1: np.ndarray | None = None
        self.x_prev2: np.ndarray | None = None
        self.sigma2_raw = np.zeros(d, dtype=np.float64)
        self.noise_steps = 0
        self.last_gamma_hat = 0.0

    def step(self, x_t: np.ndarray) -> dict[str, np.ndarray | float]:
        x_t = np.asarray(x_t, dtype=np.float64)
        self.t += 1
        t = self.t

        # Update raw cascaded EWMA states
        one_m_b = 1.0 - self.beta
        self.v = self.beta * self.v + one_m_b * x_t
        self.u = self.beta * self.u + one_m_b * self.v

        # Update blind trend-invariant noise variance estimator via 2nd differences:
        # Delta^2 x_t = x_t - 2 x_{t-1} + x_{t-2} cancels any affine drift mu_0 + t*g identically!
        if self.x_prev1 is not None and self.x_prev2 is not None:
            d2 = x_t - 2.0 * self.x_prev1 + self.x_prev2
            inst_var = (d2 * d2) / 6.0
            self.noise_steps += 1
            self.sigma2_raw = self.noise_beta * self.sigma2_raw + (1.0 - self.noise_beta) * inst_var
            sigma2_hat = self.sigma2_raw / (1.0 - (self.noise_beta ** self.noise_steps))
        elif self.x_prev1 is not None:
            # At t=2 fallback: first difference / 2 (conservative upper bound if trend exists)
            d1 = x_t - self.x_prev1
            sigma2_hat = (d1 * d1) / 2.0
        else:
            sigma2_hat = np.zeros(self.d, dtype=np.float64)

        self.x_prev2 = self.x_prev1
        self.x_prev1 = x_t.copy()

        # Closed-form O(1) moments
        m = closed_form_filter_moments_diag(self.beta, t)
        bt = m["bt"]

        # Theorem 1: Exact affine-invariant 0th-order bias correction
        v_hat_0 = (self.v - bt * self.v0) / m["one_m_bt"]

        if t == 1:
            self.last_gamma_hat = 0.0
            return {
                "uncorrected": self.v.copy(),
                "affine_bc": v_hat_0.copy(),
                "exact_tc": v_hat_0.copy(),
                "adatc": v_hat_0.copy(),
                "gamma_hat": 0.0,
            }

        # Theorem 2: Exact finite-t 1st-order trend correction
        u_hat_0 = (
            self.u - bt * self.u0 - t * bt * one_m_b * self.v0
        ) / m["W_t_0"]
        delta_t = m["K_t"] * (v_hat_0 - u_hat_0)
        v_hat_tc = v_hat_0 + delta_t

        # Theorem 3: High-dimensional James-Stein / Wiener optimal shrinkage gate
        if t == 2:
            # At t=2 we have no 2nd-difference noise estimate yet; use conservative shrinkage
            gamma_hat = 0.5
        else:
            denom = float(np.dot(delta_t, delta_t))
            noise_trace_term = float(np.sum(sigma2_hat * (m["q_delta"] + m["q_v_delta"])))
            if denom <= 1e-18:
                gamma_hat = 0.0
            else:
                gamma_hat = float(np.clip(1.0 - noise_trace_term / denom, 0.0, 1.0))

        self.last_gamma_hat = gamma_hat
        v_hat_adatc = v_hat_0 + gamma_hat * delta_t

        return {
            "uncorrected": self.v.copy(),
            "affine_bc": v_hat_0.copy(),
            "exact_tc": v_hat_tc.copy(),
            "adatc": v_hat_adatc.copy(),
            "gamma_hat": gamma_hat,
        }


class DenseMatrixOperatorEWMA:
    """
    Full non-commutative matrix contraction operator B in R^{d x d} (rho(B) < 1),
    supporting rotational/coupled state-space decay and arbitrary (v0, u0).
    """

    def __init__(self, B: np.ndarray, v0: np.ndarray | None = None, u0: np.ndarray | None = None):
        self.B = np.asarray(B, dtype=np.float64).copy()
        d = self.B.shape[0]
        assert self.B.shape == (d, d)
        eigvals = np.linalg.eigvals(self.B)
        assert np.max(np.abs(eigvals)) < 1.0, "Spectral radius rho(B) must be < 1"
        self.d = d
        self.I = np.eye(d, dtype=np.float64)
        self.I_m_B = self.I - self.B
        self.Inv_I_m_B = np.linalg.inv(self.I_m_B)

        self.v0 = np.zeros(d) if v0 is None else np.asarray(v0, dtype=np.float64).copy()
        self.u0 = self.v0.copy() if u0 is None else np.asarray(u0, dtype=np.float64).copy()
        self.v = self.v0.copy()
        self.u = self.u0.copy()
        self.Bt = self.I.copy()
        self.t = 0

    def step(self, x_t: np.ndarray) -> dict[str, np.ndarray]:
        x_t = np.asarray(x_t, dtype=np.float64)
        self.t += 1
        t = self.t
        self.Bt = self.Bt @ self.B

        self.v = self.B @ self.v + self.I_m_B @ x_t
        self.u = self.B @ self.u + self.I_m_B @ self.v

        I_m_Bt = self.I - self.Bt
        Inv_I_m_Bt = np.linalg.inv(I_m_Bt)
        v_hat_0 = Inv_I_m_Bt @ (self.v - self.Bt @ self.v0)

        if t == 1:
            return {"uncorrected": self.v.copy(), "affine_bc": v_hat_0.copy(), "exact_tc": v_hat_0.copy()}

        W_t_0 = I_m_Bt - t * (self.Bt @ self.I_m_B)
        Inv_W_t_0 = np.linalg.inv(W_t_0)
        u_hat_0 = Inv_W_t_0 @ (self.u - self.Bt @ self.u0 - t * (self.Bt @ self.I_m_B) @ self.v0)

        L_t = self.B @ self.Inv_I_m_B - t * (Inv_I_m_Bt @ self.Bt)
        M_t = Inv_W_t_0 @ (
            2.0 * (self.B @ self.Inv_I_m_B @ I_m_Bt)
            - 2.0 * t * self.Bt
            - t * (t - 1.0) * (self.Bt @ self.I_m_B)
        )
        K_t = L_t @ np.linalg.inv(M_t - L_t)
        v_hat_tc = v_hat_0 + K_t @ (v_hat_0 - u_hat_0)

        return {"uncorrected": self.v.copy(), "affine_bc": v_hat_0.copy(), "exact_tc": v_hat_tc.copy()}


class SparseMassEWMA:
    """
    Theorem 4: Exact Cumulative-Exposure Operator Bias Correction for time-varying B_t,
    sparse coordinate updates (e.g., rare tokens / MoE experts), and non-stationary regime resets.
    """

    def __init__(self, d: int, beta: float = 0.99, v0: np.ndarray | None = None):
        self.d = d
        self.beta = float(beta)
        self.v0 = np.zeros(d, dtype=np.float64) if v0 is None else np.asarray(v0, dtype=np.float64).copy()
        self.v = self.v0.copy()
        self.w = np.zeros(d, dtype=np.float64)  # exact accumulated probability mass in [0, 1)
        self.t = 0

    def step(self, x_t: np.ndarray, active_mask: np.ndarray | None = None, damping: float = 1.0) -> np.ndarray:
        """
        active_mask: boolean/float array in {0, 1}^d indicating which coordinates observed data at step t.
        damping in (0, 1]: optional regime-shift forgetting factor (damping < 1 re-activates bias correction!).
        """
        x_t = np.asarray(x_t, dtype=np.float64)
        self.t += 1
        if active_mask is None:
            active = np.ones(self.d, dtype=np.float64)
        else:
            active = np.asarray(active_mask, dtype=np.float64)

        if damping < 1.0:
            self.v *= damping
            self.w *= damping

        # Coordinates with active == 0 have effective beta_{t,i} = 1 (frozen state & frozen mass!)
        # Coordinates with active == 1 update with self.beta
        b_eff = 1.0 - active * (1.0 - self.beta)
        self.v = b_eff * self.v + (1.0 - b_eff) * x_t
        self.w = b_eff * self.w + (1.0 - b_eff) * 1.0

        p_init = 1.0 - self.w  # exact remaining weight on v0
        denom = np.maximum(self.w, 1e-15)
        v_hat = np.where(self.w > 1e-12, (self.v - p_init * self.v0) / denom, self.v0)
        return v_hat


if __name__ == "__main__":
    # Self-test closed-form O(1) variance formulas against explicit O(t) summation!
    betas = np.array([0.5, 0.85, 0.9, 0.98, 0.999])
    for t_chk in [2, 3, 10, 50]:
        m = closed_form_filter_moments_diag(betas, t_chk)
        ks = np.arange(1, t_chk + 1)[:, None]  # (t, 1)
        b_row = betas[None, :]
        a_k = (1 - b_row) * (b_row ** (t_chk - ks)) / (1 - b_row ** t_chk)
        W_0 = 1 - b_row ** t_chk - t_chk * (b_row ** t_chk) * (1 - b_row)
        c_k = (t_chk - ks + 1) * ((1 - b_row) ** 2) * (b_row ** (t_chk - ks)) / W_0
        h_k = m["K_t"][None, :] * (a_k - c_k)

        assert np.allclose(m["q_v"], np.sum(a_k ** 2, axis=0), atol=1e-12)
        assert np.allclose(m["q_delta"], np.sum(h_k ** 2, axis=0), atol=1e-12)
        assert np.allclose(m["q_v_delta"], np.sum(a_k * h_k, axis=0), atol=1e-12)
    print("Self-test passed: O(1) closed-form variance coefficients match exact O(t) filter norms to 1e-12!")
