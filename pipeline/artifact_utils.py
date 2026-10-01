"""
Reconstruct a fitted model instance from saved artifacts (metadata.json +
estimated_parameters.csv), without re-running optimization.

Used by:
  - generate_report_extended.py (to recompute diagnostics/CDFs for plots)
  - Stage 4 (models.regime_gas.RegimeXiOnlyModel) to obtain the frozen phi
    trajectory from whichever phi-only model won Stages 1-3 (objective 3,
    prompt.md 2026-07-07): the base model may be a plain ZAGASModel
    (Stage 1), CovZAGASModel (Stage 2, weather covariates), or
    HarveyZAGASModel (Stage 3, ENSO long-short) -- this module hides that
    choice behind one function so callers do not need per-architecture logic.

Reconstruction reads only saved artifacts and never calls fit(); it is a
pure "replay" of an already-completed estimation (EXECUTION.md §22:
reports/derived computations never trigger optimization).
"""

from __future__ import annotations
import re
from typing import List, Optional

import numpy as np
import pandas as pd


def infer_gas_lags(params_df: pd.DataFrame, tv: str = "phi") -> List[int]:
    """Infer GAS lag set from parameter names, e.g. A_phi_364 -> 364."""
    pat  = re.compile(rf"^[AB]_{tv}_(\d+)$")
    lags = {int(m.group(1)) for p in params_df.get("parameter", [])
            if (m := pat.match(str(p)))}
    return sorted(lags) if lags else [1, 2, 3]


def infer_tv_from_params(params_df: pd.DataFrame) -> List[str]:
    """
    Infer which parameters are time-varying from estimated_parameters.csv.

    Falls back gracefully when metadata does not store tv_param_names.
    """
    if params_df is None or params_df.empty:
        return ["phi"]
    names: set = set()
    if "parameter" in params_df.columns:
        names = set(str(p) for p in params_df["parameter"])
    tv: List[str] = []
    if any(n.startswith("omega_phi") or n.startswith("A_phi_") for n in names):
        tv.append("phi")
    if any(n.startswith("omega_xi") or n.startswith("A_xi_") for n in names):
        tv.append("xi")
    return tv if tv else ["phi"]


def build_model_from_meta(meta: dict, params_df: pd.DataFrame, model_id: str = ""):
    """
    Reconstruct a model instance (ZAGASModel / CovZAGASModel /
    HarveyZAGASModel) from metadata.json + estimated_parameters.csv.

    The returned model is unfitted (no theta) -- callers use the model's
    .filter() / .simulate_oos() with the theta from params_df.

    `model_id` (the artifact directory name, e.g. "stage2_phi_dewpoint_short")
    is used as a secondary dispatch signal alongside metadata["model_type"]
    -- every model class's save_result() writes "model_type" (not
    "model_class"; there is no "model_id" field in metadata.json at all),
    so both must be checked as they actually exist on disk.
    """
    from distributions.gb2_log_link import GB2LogLink
    from distributions.gb2_phi_only import GB2LogLinkPhiOnly
    from pi_dynamics.factory import make_pi_dynamics

    tv = meta.get("tv_param_names") or infer_tv_from_params(params_df)
    dist = GB2LogLink() if len(tv) > 1 else GB2LogLinkPhiOnly()
    pi_dyn = make_pi_dynamics("ar_logistic", seasonal="daily")

    model_type = meta.get("model_type", "ZAGASModel")
    scaling_raw = meta.get("scaling", "diagonal_inverse_fisher")
    scaling_map = {"diagfi": "diagonal_inverse_fisher", "fullfi": "inverse_fisher"}
    scaling = scaling_map.get(scaling_raw, scaling_raw)

    if "Harvey" in model_type or "harvey" in model_id:
        from models.harvey_gas import HarveyZAGASModel
        long_names  = meta.get("long_names", [])
        short_names = meta.get("short_names", [])
        # These six default to the production pipeline's behaviour when
        # absent (all existing Stage-3 metadata.json files predate them),
        # but are REQUIRED for a structurally correct reconstruction of a
        # run_longshort_diagnostics.py artifact: e.g. a sigmoid_both fit's
        # estimated_parameters.csv holds phi_L_j/phi_S_j on the pre-sigmoid
        # raw scale, and .filter()/.cdf_series() only decode that correctly
        # if phi_L_max/phi_S_max are set to match how it was actually fit.
        return HarveyZAGASModel(
            distribution=dist, pi_dynamics=pi_dyn, seasonal="daily",
            long_names=long_names, short_names=short_names, scaling=scaling,
            restrict_persistence_upper=meta.get("restrict_persistence_upper", True),
            use_auxiliary_penalties=meta.get("use_auxiliary_penalties", True),
            fixed_phi_L=meta.get("fixed_phi_L"),
            phi_L_max=meta.get("phi_L_max"),
            phi_S_max=meta.get("phi_S_max"),
            fixed_phi_S=meta.get("fixed_phi_S"),
            raw_z_ridge=meta.get("raw_z_ridge", 0.0),
            n_harmonics=meta.get("n_harmonics", 0),
            seasonal_period=meta.get("seasonal_period", 365.25),
            kappa_gamma_bound=meta.get("kappa_gamma_bound"),
            seasonal_damping_max=meta.get("seasonal_damping_max"),
        )

    if "Cov" in model_type or model_id.startswith("stage2"):
        from models.cov_gas_model import CovZAGASModel
        lags      = meta.get("lags") or infer_gas_lags(params_df, tv[0])
        cov_names = meta.get("cov_names", [])
        return CovZAGASModel(
            distribution=dist, pi_dynamics=pi_dyn,
            seasonal="daily", gas_lags=lags, scaling=scaling,
            cov_names=cov_names,
            # Experimental per-lag sigmoid-bounded A/B/rho (default off for
            # every existing artifact, which predates these fields) -- see
            # reports/seasonality GAS/gradient_and_convergence_diagnosis.md §8.
            bound_ab=meta.get("bound_ab", False),
            a_max=meta.get("a_max", 2.0), b_max=meta.get("b_max", 1.0),
            bound_rho=meta.get("bound_rho", False), rho_max=meta.get("rho_max", 0.999),
            # Link-only ("v3") extensions -- see ZAGASModel's constructor
            # docstring; default off for every artifact predating them.
            ab_signed=meta.get("ab_signed", False), rho_signed=meta.get("rho_signed", False),
            free_zeta_gamma=meta.get("free_zeta_gamma", False),
            kurtosis_margin=meta.get("kurtosis_margin", 4.0),
            unconstrained_penalties=meta.get("unconstrained_penalties", False),
            unconstrained_state=meta.get("unconstrained_state", False),
            bound_b_joint=meta.get("bound_b_joint", False),
            b_total_max=meta.get("b_total_max", 0.98),
            bound_b_tanh_sum=meta.get("bound_b_tanh_sum", False),
            b_sum_max=meta.get("b_sum_max", 1.0),
            bound_static_box=meta.get("bound_static_box", False),
            static_box=tuple(meta.get("static_box", (-2.0, 15.0))),
        )

    if model_type in ("ZAGASModel", "RegimeZAGASModel") or model_id.startswith("stage1"):
        from models.za_gas_model import ZAGASModel
        lags = meta.get("lags") or infer_gas_lags(params_df, tv[0])
        return ZAGASModel(
            distribution=dist, pi_dynamics=pi_dyn,
            seasonal="daily", gas_lags=lags, scaling=scaling,
            bound_ab=meta.get("bound_ab", False),
            a_max=meta.get("a_max", 2.0), b_max=meta.get("b_max", 1.0),
            bound_rho=meta.get("bound_rho", False), rho_max=meta.get("rho_max", 0.999),
            ab_signed=meta.get("ab_signed", False), rho_signed=meta.get("rho_signed", False),
            free_zeta_gamma=meta.get("free_zeta_gamma", False),
            kurtosis_margin=meta.get("kurtosis_margin", 4.0),
            unconstrained_penalties=meta.get("unconstrained_penalties", False),
            unconstrained_state=meta.get("unconstrained_state", False),
            bound_b_joint=meta.get("bound_b_joint", False),
            b_total_max=meta.get("b_total_max", 0.98),
            bound_b_tanh_sum=meta.get("bound_b_tanh_sum", False),
            b_sum_max=meta.get("b_sum_max", 1.0),
            bound_static_box=meta.get("bound_static_box", False),
            static_box=tuple(meta.get("static_box", (-2.0, 15.0))),
        )

    # Fallback: plain GAS(1,2,3), phi-only or phi+xi per tv. Safe even for
    # artifact types this function does not explicitly model (e.g. Stage 4's
    # RegimeXiOnlyModel) when the caller only needs `.dist` (see callers'
    # docstrings) rather than a structurally faithful reconstruction.
    from models.za_gas_model import ZAGASModel
    return ZAGASModel(
        distribution=dist, pi_dynamics=pi_dyn,
        seasonal="daily", gas_lags=[1, 2, 3], scaling=scaling,
    )


def get_theta(params_df: pd.DataFrame) -> Optional[np.ndarray]:
    if params_df is None or params_df.empty:
        return None
    if "value" in params_df.columns:
        return params_df["value"].to_numpy(dtype=float)
    return None


def load_model_and_theta(model_dir: "Path"):
    """
    Load metadata.json + estimated_parameters.csv from `model_dir` and
    return (model, theta, meta, params_df). Any of the first two may be
    None if the artifact is incomplete.
    """
    import json
    from pathlib import Path
    model_dir = Path(model_dir)

    meta_path = model_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    params_path = model_dir / "estimated_parameters.csv"
    params_df = (
        pd.read_csv(params_path) if params_path.exists() else pd.DataFrame()
    )

    theta = get_theta(params_df)
    model = build_model_from_meta(meta, params_df, model_id=model_dir.name) if meta else None
    return model, theta, meta, params_df


def build_full_path(model, theta: np.ndarray, y_train: np.ndarray,
                     y_test: np.ndarray, extra_fit_kwargs: Optional[dict] = None,
                     extra_oos_kwargs: Optional[dict] = None) -> dict:
    """
    Run a reconstructed model's filter over train (in-sample) and
    simulate_oos over train+test, returning the frozen exogenous trajectory
    Stage 4 needs for every TV parameter, plus pi (occurrence probability).

    The in-sample filter output starts only at the base model's own warm-up
    boundary (max_lag -- up to 367 for a seasonal-lag winner), so the
    returned arrays are NOT padded back to t=0. Instead `offset` gives the
    index (into the conceptual y_train+y_test series) that array position 0
    corresponds to; callers must not query indices below `offset` -- there
    is no meaningful frozen phi/pi value before the base model's own
    warm-up completes, exactly as Stage 1-3 already exclude that window
    from every likelihood/metric computation.

    extra_fit_kwargs / extra_oos_kwargs supply covariate matrices etc. for
    CovZAGASModel / HarveyZAGASModel (must match train/test shapes).
    """
    extra_fit_kwargs = extra_fit_kwargs or {}
    extra_oos_kwargs = extra_oos_kwargs or {}

    is_paths  = model.filter(theta, y_train, **extra_fit_kwargs)
    oos_paths = model.simulate_oos(theta, y_train, y_test, **extra_oos_kwargs)

    tv_names = is_paths["tv_names"]
    offset   = len(y_train) - len(is_paths["pi"])  # base model's own max_lag

    out = {
        "tv_names": tv_names,
        "static":   is_paths["static"],
        "offset":   int(offset),
        "n_total":  len(y_train) + len(y_test),
    }
    for j, name in enumerate(tv_names):
        out[name] = np.concatenate([is_paths["f_arr"][:, j], oos_paths["f_arr_oos"][:, j]])
    out["pi"] = np.concatenate([is_paths["pi"], oos_paths["pi_oos"]])
    return out
