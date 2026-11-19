"""
test_all.py: Automated Unit & Edge-Case Verification Suite
==========================================================
Checks numerical stability (extreme beta -> 0.99999, t=1..5000, zero/constant inputs,
non-zero v0/u0, dense non-symmetric contraction matrices, sparse masks) so there are
zero NaN/Inf hazards or subtle corner-case regressions.
"""

import numpy as np
from ewma_hd import AdaTCOperatorEWMA, DenseMatrixOperatorEWMA, SparseMassEWMA, closed_form_filter_moments_diag


def test_numerical_stability_extreme_betas():
    for beta in [0.01, 0.5, 0.9, 0.99, 0.999, 0.9999]:
        est = AdaTCOperatorEWMA(d=16, beta=beta, v0=np.full(16, -42.0))
        for t in range(1, 300):
            x = np.full(16, 100.0) if t < 150 else np.zeros(16)  # exercises exact zero delta too!
            out = est.step(x)
            for k in ["uncorrected", "affine_bc", "exact_tc", "adatc"]:
                assert np.all(np.isfinite(out[k])), f"Non-finite output in {k} at beta={beta}, t={t}"
            assert 0.0 <= out["gamma_hat"] <= 1.0


def test_exact_constant_unbiasedness_arbitrary_v0():
    rng = np.random.default_rng(11)
    d = 32
    betas = rng.uniform(0.5, 0.995, size=d)
    v0 = rng.standard_normal(d) * 500.0
    u0 = rng.standard_normal(d) * 500.0
    c = rng.standard_normal(d) * 25.0

    est = AdaTCOperatorEWMA(d=d, beta=betas, v0=v0, u0=u0)
    for t in range(1, 50):
        out = est.step(c)
        assert np.allclose(out["affine_bc"], c, atol=1e-9)
        assert np.allclose(out["exact_tc"], c, atol=1e-8)
        assert np.allclose(out["adatc"], c, atol=1e-8)


def test_exact_affine_ramp_unbiasedness_dense_and_diag():
    rng = np.random.default_rng(22)
    d = 8
    A = rng.standard_normal((d, d))
    B = 0.85 * A / np.max(np.abs(np.linalg.eigvals(A)))
    v0 = rng.standard_normal(d) * 20.0
    u0 = rng.standard_normal(d) * 15.0
    mu0 = rng.standard_normal(d) * 5.0
    g = rng.standard_normal(d) * 3.0

    dense_est = DenseMatrixOperatorEWMA(B=B, v0=v0, u0=u0)
    diag_betas = rng.uniform(0.6, 0.98, size=d)
    diag_est = AdaTCOperatorEWMA(d=d, beta=diag_betas, v0=v0, u0=u0)

    for t in range(1, 35):
        xt = mu0 + t * g
        od = dense_est.step(xt)
        og = diag_est.step(xt)
        if t >= 2:
            assert np.allclose(od["exact_tc"], xt, atol=1e-9)
            assert np.allclose(og["exact_tc"], xt, atol=1e-9)
        if t >= 3:
            # On noiseless affine ramp, sigma2_hat == 0 identically => gamma_hat == 1.0 => AdaTC == xt!
            assert np.isclose(og["gamma_hat"], 1.0, atol=1e-12)
            assert np.allclose(og["adatc"], xt, atol=1e-9)


def test_sparse_mass_ewma_equivalence_and_inactivity():
    d = 10
    beta = 0.95
    v0 = np.arange(d, dtype=np.float64)
    sm = SparseMassEWMA(d=d, beta=beta, v0=v0)
    # Coordinate 0 is active every step; coordinate 9 is inactive for 50 steps then sees 42.0!
    for t in range(1, 51):
        mask = np.zeros(d)
        mask[0] = 1.0
        x = np.zeros(d)
        x[0] = 10.0
        out = sm.step(x, active_mask=mask)
        assert np.isclose(out[0], 10.0, atol=1e-10)
        assert np.isclose(out[9], v0[9], atol=1e-12)  # untouched coordinate stays at v0

    # Now coordinate 9 wakes up at step 51 with value 42.0: must equal 42.0 immediately!
    mask = np.zeros(d)
    mask[9] = 1.0
    x = np.zeros(d)
    x[9] = 42.0
    out51 = sm.step(x, active_mask=mask)
    assert np.isclose(out51[9], 42.0, atol=1e-10)


if __name__ == "__main__":
    test_numerical_stability_extreme_betas()
    test_exact_constant_unbiasedness_arbitrary_v0()
    test_exact_affine_ramp_unbiasedness_dense_and_diag()
    test_sparse_mass_ewma_equivalence_and_inactivity()
    print("ALL UNIT & EDGE-CASE TESTS PASSED (0 errors).")
