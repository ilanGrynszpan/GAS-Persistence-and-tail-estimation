"""
Parity: Numba vs pure-Python ZAGASModel filter for the 2026-10-01 additions
  * scaling="mixed_inverse_fisher" (Fisher information of the zero-adjusted density)
  * PhiLinkedJointPiDynamics (pi_t driven by the filter's own phi_{t|t-1})
  * scaling="inverse_fisher" for one time-varying parameter (must equal the
    diagonal mode; the Numba kernel used to fall through to unit scaling)

Also checks the analytical mixed Fisher information against a Monte Carlo
estimate of E[s_t^2] for the phi-linked score.

Run:  python -m pytest tests/test_mixed_fisher_parity.py -q
"""

import sys
from pathlib import Path

import numpy as np
try:
    import pytest
except ImportError:          # allow `python tests/test_mixed_fisher_parity.py`
    import contextlib

    class pytest:  # minimal stand-in
        @staticmethod
        def raises(exc):
            @contextlib.contextmanager
            def cm():
                try:
                    yield
                except exc:
                    return
                raise AssertionError(f'{exc.__name__} not raised')
            return cm()

        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda f: f

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from distributions import GB2LogLinkPhiOnly
from models.za_gas_model import ZAGASModel
from pi_dynamics.ar_logistic import ARLogisticPiDynamics
from pi_dynamics.ar_logistic_custom_lags import ARLogisticCustomLagsPiDynamics
from pi_dynamics.phi_linked import PhiLinkedJointPiDynamics


def _series(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    wet = rng.random(n) < 0.35
    return np.where(wet, rng.gamma(0.7, 9.0, n), 0.0)


def _model(pi_dyn, scaling):
    return ZAGASModel(distribution=GB2LogLinkPhiOnly(), pi_dynamics=pi_dyn, seasonal="daily",
                      gas_lags=[1], scaling=scaling, static_params=["xi", "gamma", "zeta"])


def _theta(model, y):
    th = model.initial_theta(y).astype(float)
    names = model.parameter_names()
    rng = np.random.default_rng(0)
    th += 0.02 * rng.standard_normal(len(th))
    if "lambda1" in names:
        th[names.index("lambda1")] = 0.4
    return th


PI = {
    "A":  lambda: ARLogisticPiDynamics(),
    "A7": lambda: ARLogisticCustomLagsPiDynamics(lags=[1, 2, 3, 364, 365, 366, 367], include_ar=True),
    "C7": lambda: ARLogisticCustomLagsPiDynamics(lags=[1, 2, 3, 364, 365, 366, 367], include_ar=False),
    "D":  lambda: PhiLinkedJointPiDynamics(),
}


@pytest.mark.parametrize("variant", list(PI))
@pytest.mark.parametrize("scaling", ["unit", "mixed_inverse_fisher", "inverse_fisher"])
def test_numba_python_parity(variant, scaling):
    if variant == "D" and scaling == "inverse_fisher":
        with pytest.raises(ValueError):
            _model(PI[variant](), scaling)
        return
    y = _series()
    m = _model(PI[variant](), scaling)
    th = _theta(m, y)
    nb = m._run_filter(th, y, return_paths=True)
    m._use_nb_filter = lambda: False
    py = m._run_filter(th, y, return_paths=True)
    assert np.isclose(nb["loglik"], py["loglik"], rtol=1e-9, atol=1e-6)
    assert np.allclose(nb["phi"], py["phi"], rtol=1e-8, atol=1e-8)
    assert np.allclose(nb["pi"], py["pi"][: len(nb["pi"])], rtol=1e-10, atol=1e-12)


def test_inverse_fisher_equals_diagonal_for_one_parameter():
    y = _series()
    a = _model(PI["A"](), "inverse_fisher")
    b = _model(PI["A"](), "diagonal_inverse_fisher")
    th = _theta(a, y)
    assert np.allclose(a._run_filter(th, y, return_paths=True)["phi"],
                       b._run_filter(th, y, return_paths=True)["phi"])


def test_mixed_fisher_matches_monte_carlo():
    """E[s^2] of s = 1{y>0} dlog g/dphi + lambda1 (1{y>0} - pi) equals
    pi I_g + lambda1^2 pi (1 - pi)."""
    dist = GB2LogLinkPhiOnly()
    par = {"phi": 0.4, "xi": 0.1, "gamma": -0.2, "zeta": 1.2}
    pi, lam, n = 0.37, 0.8, 400_000
    rng = np.random.default_rng(1)
    wet = rng.random(n) < pi
    y = dist.rvs(n, **par)
    p, a, b = np.exp(-par["gamma"]), np.exp(par["xi"]), np.exp(par["zeta"])
    z = (y / np.exp(par["phi"])) ** p
    s_g = p * ((a + b) * z / (1 + z) - a)
    s = np.where(wet, s_g, 0.0) + lam * (wet - pi)
    i_g = dist.fisher_info_diag(**par)["phi"]
    assert np.isclose(np.mean(s ** 2), pi * i_g + lam ** 2 * pi * (1 - pi), rtol=0.01)


if __name__ == "__main__":   # runnable without pytest
    for v in PI:
        for sc in ["unit", "mixed_inverse_fisher", "inverse_fisher"]:
            test_numba_python_parity(v, sc)
            print("parity ok:", v, sc)
    test_inverse_fisher_equals_diagonal_for_one_parameter(); print("inverse_fisher == diagonal ok")
    np.random.seed(5)
    test_mixed_fisher_matches_monte_carlo(); print("mixed Fisher vs Monte Carlo ok")
