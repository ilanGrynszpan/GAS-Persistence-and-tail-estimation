"""
Delta-method standard errors for link-transformed parameters, shared by
generate_original_vs_v2_report.py and generate_link_only_report.py
(2026-09-02).

=============================================================================
OVERVIEW
=============================================================================

Every "bound_ab"/"bound_rho"/"free_zeta_gamma" parameter's raw theta value
IS already estimated with a perfectly ordinary BFGS approximate SE (saved
in standard_errors.csv, on the RAW/pre-link scale -- confirmed present for
100% of the v2 and link-only artifacts, so that is not the gap). The gap
was in the report generators themselves: they decoded the NATURAL-scale
value but printed "--" for its SE instead of propagating the raw SE
through the link function -- discarding real information for the (usually
large) majority of parameters where the link's derivative is well-behaved,
not just for the few that are genuinely saturated.

This module computes that propagation properly via the delta method
(Var(g(z)) ~= g'(z)^2 * Var(z) for a single-input link; a full 2-vector
Jacobian against the 2x2 covariance submatrix for zeta = gamma +
log(margin) + exp(delta), which depends on TWO raw parameters, not one),
reading the FULL saved hess_inv.npy (not just its diagonal, which is all
standard_errors.csv itself stores) so the gamma/delta covariance term is
not silently dropped.

Where the link's own derivative is (numerically) zero -- a coefficient
saturated at its sigmoid's floor or ceiling -- the delta method's own
math says Var(g(z)) ~= 0, which would print a falsely tiny, falsely
precise SE for exactly the parameter that most needs a warning, not a
number. Those cases are reported as unavailable, with the reason stated,
never silently as a small number.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Below this, the link's local derivative is treated as "saturated" --
# the coefficient is pinned close enough to its sigmoid's floor/ceiling
# that a delta-method SE for it would be a numerical artifact (near zero
# by construction), not a real precision statement. See docstring above.
DERIVATIVE_FLOOR = 1e-3

# The exact thresholds every penalty term in models/za_gas_model.py and
# models/cov_gas_model.py's _run_filter is gated on -- duplicated here
# (not imported) because this check works purely from already-saved
# metadata.json/bound_diagnostics, without reconstructing the model or
# reloading data, so it applies retroactively to every already-fit model
# at zero extra compute cost.
B_LIMIT, A_LIMIT, RHO_LIMIT = 0.98, 2.0, 0.98
STATIC_LIMIT, STATE_LIMIT, ETA_LIMIT = 20.0, 15.0, 30.0
ZETA_GAMMA_SOFT_MARGIN = 0.5  # the ORIGINAL model's soft push-away threshold on (zeta-gamma)


def check_penalty_contamination(meta: dict) -> dict:
    """
    Determines, PER PARAMETER BLOCK, whether the objective BFGS actually
    optimized (and whose curvature hess_inv approximates) had a non-zero
    penalty contribution at the reported theta -- i.e. whether the
    "approximate SE" for that block is the curvature of the LOG-LIKELIHOOD
    (valid, subject to the usual quasi-Newton caveats) or of the
    LOG-LIKELIHOOD PLUS AN ACTIVE PENALTY WALL (invalid: OPTIMIZATION.md's
    own penalty weights, e.g. LAM_PERSIST=1e5, are numerical-safety
    magnitudes, not statistically meaningful curvature -- a hess_inv built
    from that sum reflects how steeply the PENALTY curves, not how much the
    DATA informs the parameter).

    This does NOT rely on the "near_limit" bound_diagnostics flags (a 90%-
    of-threshold heuristic, meant for early warning) -- it checks EXACT
    exceedance of the same thresholds the actual penalty code in
    models/za_gas_model.py / models/cov_gas_model.py gates on, since a
    penalty of the form max(0, x-limit)**2 is EXACTLY zero for x<=limit
    and only becomes non-zero once x truly exceeds it.

    Returns {block_name: bool} for whichever blocks exist in this model's
    bound_diagnostics, where True means "this block's SE is contaminated
    by an active penalty -- do not report it as an ordinary approximate SE".
    A block whose corresponding reparameterisation flag is set (e.g.
    bound_ab=True disables the joint A/B sum penalty entirely -- see
    models/za_gas_model.py's _run_filter) is never contaminated by that
    SPECIFIC penalty, regardless of what its diagnostic sum happens to be,
    because that penalty term was never added to the objective at all.
    """
    bd = meta.get("bound_diagnostics", {}) or {}
    bound_ab = meta.get("bound_ab", False)
    bound_rho = meta.get("bound_rho", False)
    free_zeta_gamma = meta.get("free_zeta_gamma", False)
    unconstrained_penalties = meta.get("unconstrained_penalties", False)

    out = {}
    # Persistence-sum penalties (per tv-parameter): gated off entirely
    # under bound_ab.
    for key in list(bd.keys()):
        m = re.match(r"^(\w+)_B_sum$", key)
        if m:
            tv = m.group(1)
            out[f"{tv}_B"] = (not bound_ab) and (bd[key] > B_LIMIT)
        m = re.match(r"^(\w+)_A_sum$", key)
        if m:
            tv = m.group(1)
            out[f"{tv}_A"] = (not bound_ab) and (bd[key] > A_LIMIT)

    if "pi_rho" in bd:
        out["rho"] = (not bound_rho) and (abs(bd["pi_rho"]) > RHO_LIMIT)

    # Static gamma/zeta moment-existence SOFT PUSH penalty (not the bare
    # gamma>=zeta hard-validity reject, which is never a source of
    # contamination since it makes the objective 1e12, not something
    # hess_inv could ever be computed at): gated off under free_zeta_gamma
    # (the link function itself guarantees the margin there, so this
    # penalty is never even evaluated) AND, as of the 2026-09-02
    # models/za_gas_model.py / models/cov_gas_model.py fix, also under
    # unconstrained_penalties (that soft push -- unlike the hard reject --
    # is a genuine statistical restriction, not bare numerical validity,
    # and is now dropped from the objective under unconstrained_penalties
    # exactly like the |static|<20 guard just below).
    if "static_gamma" in bd and "static_zeta" in bd:
        bp_log = bd["static_zeta"] - bd["static_gamma"]
        out["gamma_zeta"] = (
            (not free_zeta_gamma) and (not unconstrained_penalties)
            and (bp_log < ZETA_GAMMA_SOFT_MARGIN)
        )

    # Static |param|<20 auxiliary guard: gated off entirely under
    # unconstrained_penalties.
    for key in list(bd.keys()):
        m = re.match(r"^static_(\w+)$", key)
        if m and not m.group(1) in ("gamma", "zeta"):
            name = m.group(1)
            out[f"static_{name}"] = (not unconstrained_penalties) and (abs(bd[key]) > STATIC_LIMIT)

    # Numerical safety guards (state/eta explosion). The state guard is now
    # gated off entirely under unconstrained_state (2026-09-02 experiment:
    # empirically the SOLE source of "(boundary optimum)" under V2 was this
    # penalty, never the A/B/rho boxes -- 0/144 contamination from those,
    # 67/144 from this one alone). The eta guard has never been observed to
    # bind in practice and is left ungated. When active, the state depends
    # on the WHOLE filtered trajectory, i.e. on every parameter, so this
    # still contaminates the ENTIRE model's Hessian, not one block, exactly
    # as before -- only the ABILITY to switch it off is new.
    unconstrained_state = meta.get("unconstrained_state", False)
    state_active = (not unconstrained_state) and (bd.get("max_state_magnitude", 0.0) > STATE_LIMIT)
    eta_active = bd.get("max_eta_magnitude", 0.0) > ETA_LIMIT
    out["__whole_model__"] = bool(state_active or eta_active)

    return out


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def _sigmoid_prime(z: float) -> float:
    s = _sigmoid(z)
    return s * (1.0 - s)


def _tanh_over_x(S: float) -> float:
    """tanh(S)/S with the S=0 removable singularity handled via its Taylor
    series -- duplicated from models/gas_filter.py's identical helper
    (this module works purely from saved metadata/hess_inv, not a live
    model instance, so its small link-function helpers are kept local
    rather than imported, matching this file's existing _sigmoid/
    _sigmoid_prime convention)."""
    if abs(S) < 1e-4:
        S2 = S * S
        return 1.0 - S2 / 3.0 + 2.0 * S2 * S2 / 15.0
    return float(np.tanh(S) / S)


def _tanh_over_x_prime(S: float) -> float:
    """d/dS [tanh(S)/S] -- see _tanh_over_x."""
    if abs(S) < 1e-4:
        return -2.0 * S / 3.0 + 8.0 * S ** 3 / 15.0
    t = np.tanh(S)
    sech2 = 1.0 - t * t
    return float((S * sech2 - t) / (S * S))


def compute_natural_and_se(
    param_names: list,
    raw_values: np.ndarray,
    hess_inv: Optional[np.ndarray],
    meta: dict,
    grad: Optional[np.ndarray] = None,
    grad_threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by parameter with columns:
        estimate  -- natural-scale value
        se        -- delta-method SE on the natural scale, or NaN
        grad      -- the RAW per-parameter gradient component (only when
                     `grad` is passed), for display alongside the SE so a
                     reader can see exactly what triggered its note.
        note      -- "" (untransformed, ordinary SE), "delta method"
                     (transformed, valid delta-method SE), "unavailable
                     (saturated)" (link derivative ~0 -- reporting a
                     number here would be misleading), or, when `grad` is
                     supplied (2026-09-02 per-parameter gradient gate):
                     "boundary optimized" / "unavailable (high gradient,
                     not at boundary)" -- see below.

    raw_values must be in the SAME order as param_names (i.e. theta
    order, matching estimated_parameters.csv's row order and hess_inv's
    row/column order).

    PER-PARAMETER GRADIENT GATE (optional, `grad` + `grad_threshold`):
    the classical requirement for a Hessian-based SE to mean anything at
    all is that theta_hat is (at least approximately) a STATIONARY POINT
    in that parameter's own direction -- not merely that the model's
    OVERALL gradient norm cleared some threshold (this project's standing
    RELIABLE_GRAD_THRESHOLD=1.0 is 1000x looser than the optimizer's own
    gtol=1e-3, and is a whole-model classification, not a per-parameter
    one). When `grad` (gradients.csv's raw per-parameter values, same
    order as param_names) is supplied, every parameter is re-classified
    by ITS OWN |grad_i| against `grad_threshold` (default 0.1, "close
    enough to zero" per direct instruction), overriding whatever the
    saturation/link-derivative logic above already decided (unless it
    ALSO cleared 0.1, this is the primary gate, not a secondary one):
        |grad_i| < grad_threshold : keep the estimate/SE computed above
            (still "unavailable (saturated)"/"...(raw SE invalid)" if the
            link itself was the problem -- small gradient does not fix a
            genuinely flat or invalid link).
        |grad_i| >= grad_threshold : SE withheld. If the corresponding
            penalty block was active (check_penalty_contamination) --
            "boundary optimized". Otherwise -- "unavailable (high
            gradient, not at boundary)", flagging that this parameter's
            large residual gradient needs a DIFFERENT explanation (e.g.
            genuine non-convergence, a flat/ill-conditioned direction, or
            collinear covariates) which this function does not attempt to
            diagnose automatically.
    """
    name_to_idx = {n: i for i, n in enumerate(param_names)}
    a_max = meta.get("a_max", 2.0)
    b_max = meta.get("b_max", 1.0)
    rho_max = meta.get("rho_max", 0.999)
    ab_signed = meta.get("ab_signed", False)
    rho_signed = meta.get("rho_signed", False)
    free_zeta_gamma = meta.get("free_zeta_gamma", False)
    kurtosis_margin = meta.get("kurtosis_margin", 4.0)
    bound_ab = meta.get("bound_ab", False)
    bound_rho = meta.get("bound_rho", False)
    bound_b_joint = meta.get("bound_b_joint", False)
    b_total_max = meta.get("b_total_max", 0.98)
    bound_b_tanh_sum = meta.get("bound_b_tanh_sum", False)
    b_sum_max = meta.get("b_sum_max", 1.0)
    bound_static_box = meta.get("bound_static_box", False)
    static_box = tuple(meta.get("static_box", (-2.0, 15.0)))

    def raw_var(i: int) -> float:
        if hess_inv is None or i >= hess_inv.shape[0]:
            return np.nan
        v = hess_inv[i, i]
        return float(v) if (np.isfinite(v) and v > 0) else np.nan

    # ── Joint-B (bound_b_joint) precompute, once per tv-name ────────────────
    # B_{name,l} = b_total_max * e_l/denom is a function of EVERY raw
    # B_{name,k} of the same tv-name jointly (softmax-with-phantom-zero,
    # see models/gas_filter.py's _ParamCodec) -- so its delta-method SE
    # needs the FULL Jacobian against the LxL raw covariance submatrix, not
    # a single-input g'(z)^2*Var(z) as for the independent-per-lag case.
    # Derivation: with w_l = e_l/denom (so B_l = b_total_max*w_l),
    #   dB_l/dz_l = b_total_max * w_l*(1-w_l)
    #   dB_l/dz_k = -b_total_max * w_l*w_k        (k != l)
    joint_b_cache: dict = {}
    if bound_ab and bound_b_joint:
        b_groups: dict = {}
        for pname in param_names:
            m = re.match(r"^B_(\w+)_(\d+)$", pname)
            if m:
                b_groups.setdefault(m.group(1), []).append((int(m.group(2)), pname))
        for tv_name, entries in b_groups.items():
            entries.sort(key=lambda t: t[0])
            lags = [l for l, _ in entries]
            pnames_l = [p for _, p in entries]
            idxs = [name_to_idx[p] for p in pnames_l]
            raw_b = np.clip([float(raw_values[i]) for i in idxs], -30.0, 30.0)
            e = np.exp(raw_b)
            denom = 1.0 + np.sum(e)
            w = e / denom
            L = len(lags)
            if hess_inv is None or max(idxs) >= hess_inv.shape[0]:
                cov_sub = None
            else:
                cov_sub = hess_inv[np.ix_(idxs, idxs)]
            for li, (lag, pname, wl) in enumerate(zip(lags, pnames_l, w)):
                natural = b_total_max * float(wl)
                jac = np.array([
                    b_total_max * wl * (1.0 - wl) if k == li else -b_total_max * wl * w[k]
                    for k in range(L)
                ])
                if cov_sub is None:
                    joint_b_cache[pname] = (natural, np.nan, "unavailable (raw SE invalid)")
                    continue
                var_b = float(jac @ cov_sub @ jac)
                if not np.isfinite(var_b) or var_b < 0:
                    joint_b_cache[pname] = (natural, np.nan, "unavailable (raw covariance invalid)")
                elif np.max(np.abs(jac)) < DERIVATIVE_FLOOR:
                    joint_b_cache[pname] = (natural, np.nan, "unavailable (saturated)")
                else:
                    joint_b_cache[pname] = (natural, np.sqrt(var_b), "delta method")

    # ── Tanh-rescale B (bound_b_tanh_sum) precompute, once per tv-name ──────
    # B_l = raw_l * scale, scale = b_sum_max*tanh(S)/S, S = sum_l raw_l --
    # see models/gas_filter.py's _ParamCodec docstring.
    #   dB_l/draw_m = (scale if m==l else 0) + raw_l * b_sum_max * f'(S)
    # where f'(S) = d/dS[tanh(S)/S], since dS/draw_m = 1 for every m.
    tanh_b_cache: dict = {}
    if bound_ab and bound_b_tanh_sum:
        b_groups_t: dict = {}
        for pname in param_names:
            m = re.match(r"^B_(\w+)_(\d+)$", pname)
            if m:
                b_groups_t.setdefault(m.group(1), []).append((int(m.group(2)), pname))
        for tv_name, entries in b_groups_t.items():
            entries.sort(key=lambda t: t[0])
            pnames_l = [p for _, p in entries]
            idxs = [name_to_idx[p] for p in pnames_l]
            raw_b = np.array([float(raw_values[i]) for i in idxs])
            S = float(np.sum(raw_b))
            scale = b_sum_max * _tanh_over_x(S)
            fprime = _tanh_over_x_prime(S)
            L = len(pnames_l)
            if hess_inv is None or max(idxs) >= hess_inv.shape[0]:
                cov_sub = None
            else:
                cov_sub = hess_inv[np.ix_(idxs, idxs)]
            for li, (r_l, pname) in enumerate(zip(raw_b, pnames_l)):
                natural = float(r_l) * scale
                jac = np.array([
                    (scale if k == li else 0.0) + r_l * b_sum_max * fprime
                    for k in range(L)
                ])
                if cov_sub is None:
                    tanh_b_cache[pname] = (natural, np.nan, "unavailable (raw SE invalid)")
                    continue
                var_b = float(jac @ cov_sub @ jac)
                if not np.isfinite(var_b) or var_b < 0:
                    tanh_b_cache[pname] = (natural, np.nan, "unavailable (raw covariance invalid)")
                elif np.max(np.abs(jac)) < DERIVATIVE_FLOOR:
                    tanh_b_cache[pname] = (natural, np.nan, "unavailable (saturated)")
                else:
                    tanh_b_cache[pname] = (natural, np.sqrt(var_b), "delta method")

    rows = []
    for pname in param_names:
        i = name_to_idx[pname]
        raw = float(raw_values[i])

        m_a = re.match(r"^A_(\w+)_(\d+)$", pname)
        m_b = re.match(r"^B_(\w+)_(\d+)$", pname)
        is_a = bool(m_a) and bound_ab
        is_b_joint = bool(m_b) and bound_ab and bound_b_joint
        is_b_tanh = bool(m_b) and bound_ab and bound_b_tanh_sum
        is_b = bool(m_b) and bound_ab and not bound_b_joint and not bound_b_tanh_sum
        is_rho = (pname == "rho") and bound_rho
        is_zeta = (pname == "zeta") and free_zeta_gamma
        # bound_static_box applies uniformly to every GB2 static parameter
        # (gamma, zeta, xi -- the only static names this codebase's GB2
        # models ever use) AND every f0_{tv_name} (initial GAS state).
        # Mutually exclusive with free_zeta_gamma at the codec level (see
        # models/gas_filter.py's _ParamCodec.__init__), so is_zeta above
        # never overlaps with this.
        is_static_box = bound_static_box and (pname in ("gamma", "zeta", "xi") or pname.startswith("f0_"))

        if is_static_box:
            lo, hi = static_box
            s = _sigmoid(raw)
            sp = _sigmoid_prime(raw)
            natural = lo + (hi - lo) * s
            deriv = (hi - lo) * sp
            v_raw = raw_var(i)
            if np.isnan(v_raw):
                rows.append({"parameter": pname, "estimate": natural, "se": np.nan,
                             "note": "unavailable (raw SE invalid)"})
            elif abs(deriv) < DERIVATIVE_FLOOR:
                rows.append({"parameter": pname, "estimate": natural, "se": np.nan,
                             "note": "unavailable (saturated)"})
            else:
                rows.append({"parameter": pname, "estimate": natural,
                             "se": abs(deriv) * np.sqrt(v_raw), "note": "delta method"})
        elif is_b_joint:
            natural, se, note = joint_b_cache[pname]
            rows.append({"parameter": pname, "estimate": natural, "se": se, "note": note})
        elif is_b_tanh:
            natural, se, note = tanh_b_cache[pname]
            rows.append({"parameter": pname, "estimate": natural, "se": se, "note": note})
        elif is_a or is_b or is_rho:
            mx = a_max if is_a else (b_max if is_b else rho_max)
            signed = ab_signed if (is_a or is_b) else rho_signed
            s = _sigmoid(raw)
            sp = _sigmoid_prime(raw)
            if signed:
                natural = mx * (2.0 * s - 1.0)
                deriv = mx * 2.0 * sp
            else:
                natural = mx * s
                deriv = mx * sp
            v_raw = raw_var(i)
            if np.isnan(v_raw):
                rows.append({"parameter": pname, "estimate": natural, "se": np.nan,
                             "note": "unavailable (raw SE invalid)"})
            elif abs(deriv) < DERIVATIVE_FLOOR:
                rows.append({"parameter": pname, "estimate": natural, "se": np.nan,
                             "note": "unavailable (saturated)"})
            else:
                rows.append({"parameter": pname, "estimate": natural,
                             "se": abs(deriv) * np.sqrt(v_raw), "note": "delta method"})

        elif is_zeta:
            gamma_val = float(raw_values[name_to_idx["gamma"]])
            delta_val = raw
            natural = gamma_val + np.log(kurtosis_margin) + np.exp(delta_val)
            gi = name_to_idx["gamma"]
            zi = i
            if hess_inv is None or gi >= hess_inv.shape[0] or zi >= hess_inv.shape[0]:
                rows.append({"parameter": pname, "estimate": natural, "se": np.nan,
                             "note": "unavailable (raw SE invalid)"})
            else:
                var_g, var_d, cov_gd = hess_inv[gi, gi], hess_inv[zi, zi], hess_inv[gi, zi]
                jac = np.array([1.0, np.exp(delta_val)])
                cov_sub = np.array([[var_g, cov_gd], [cov_gd, var_d]])
                var_zeta = float(jac @ cov_sub @ jac)
                if not np.isfinite(var_zeta) or var_zeta <= 0:
                    rows.append({"parameter": pname, "estimate": natural, "se": np.nan,
                                 "note": "unavailable (raw covariance invalid)"})
                else:
                    rows.append({"parameter": pname, "estimate": natural,
                                 "se": np.sqrt(var_zeta), "note": "delta method"})
        else:
            v_raw = raw_var(i)
            rows.append({"parameter": pname, "estimate": raw,
                         "se": np.sqrt(v_raw) if not np.isnan(v_raw) else np.nan,
                         "note": "" if not np.isnan(v_raw) else "unavailable (raw SE invalid)"})

    # ── Penalty-contamination override ──────────────────────────────────
    # Whatever the delta-method computed above assumed the Hessian it came
    # from was the LOG-LIKELIHOOD's curvature. Where a penalty was actually
    # active at this theta (check_penalty_contamination, computed from
    # EXACT threshold exceedance in bound_diagnostics, not a heuristic),
    # that assumption is false -- override to "unavailable (boundary
    # optimum)" regardless of what the raw diagonal/delta-method said,
    # since the reported "SE" would reflect the penalty wall's curvature
    # (LAM_PERSIST=1e5 etc. -- a numerical-safety magnitude), not the data.
    contamination = check_penalty_contamination(meta)
    whole_model_contaminated = contamination.get("__whole_model__", False)
    for row in rows:
        pname = row["parameter"]
        m_a = re.match(r"^A_(\w+)_(\d+)$", pname)
        m_b = re.match(r"^B_(\w+)_(\d+)$", pname)
        if m_a:
            block = f"{m_a.group(1)}_A"
        elif m_b:
            block = f"{m_b.group(1)}_B"
        elif pname == "rho":
            block = "rho"
        elif pname in ("gamma", "zeta"):
            block = "gamma_zeta"
        elif f"static_{pname}" in contamination:
            block = f"static_{pname}"  # any other static param, e.g. "xi" (phi-only branch)
        else:
            block = None
        row["_contaminated"] = whole_model_contaminated or (block is not None and contamination.get(block, False))
        if grad is None and row["_contaminated"]:
            # Old behaviour (no per-parameter gradient gate supplied):
            # whole-model / block contamination alone overrides to
            # "unavailable (boundary optimum)", exactly as before.
            row["se"] = np.nan
            row["note"] = "unavailable (boundary optimum)"

    # ── Per-parameter gradient gate (optional, see docstring) ───────────────
    if grad is not None:
        grad = np.asarray(grad, dtype=float)
        for i, row in enumerate(rows):
            g = float(grad[i]) if i < len(grad) else np.nan
            row["grad"] = g
            if not np.isfinite(g) or abs(g) >= grad_threshold:
                row["se"] = np.nan
                row["note"] = "boundary optimized" if row["_contaminated"] else \
                    "unavailable (high gradient, not at boundary)"
            # else: |g| < grad_threshold -- keep whatever estimate/SE/note
            # the link-derivative logic above already produced (including
            # a legitimate "unavailable (saturated)"/"...(raw SE invalid)"
            # if that was the reason, which a small gradient does not fix).

    for row in rows:
        row.pop("_contaminated", None)

    return pd.DataFrame(rows).set_index("parameter")


def load_param_table_with_delta_se(
    model_dir: Path,
    use_grad_gate: bool = False,
    grad_threshold: float = 0.1,
) -> Tuple[pd.DataFrame, dict]:
    """
    Reads estimated_parameters.csv, hess_inv.npy, and metadata.json from
    model_dir and returns (DataFrame[estimate, se, note (, grad)], meta).

    use_grad_gate=True additionally reads gradients.csv and applies the
    per-parameter gradient gate (see compute_natural_and_se's docstring) --
    opt-in so every report generator already built against the OLD
    (whole-model-contamination-only) behaviour is completely unaffected.
    """
    est = pd.read_csv(model_dir / "estimated_parameters.csv")
    param_names = est["parameter"].tolist()
    raw_values = est["value"].to_numpy(dtype=float)
    meta = __import__("json").loads((model_dir / "metadata.json").read_text())
    hess_path = model_dir / "hess_inv.npy"
    hess_inv = np.load(hess_path) if hess_path.exists() else None
    grad = None
    if use_grad_gate:
        grad_path = model_dir / "gradients.csv"
        if grad_path.exists():
            grad_df = pd.read_csv(grad_path).set_index("parameter")["gradient"]
            grad = grad_df.reindex(param_names).to_numpy(dtype=float)
    df = compute_natural_and_se(param_names, raw_values, hess_inv, meta,
                                 grad=grad, grad_threshold=grad_threshold)
    return df, meta


# ─────────────────────────────────────────────────────────────────────────────
# Constraint metadata (for the full Stage 1/2 parameter reference document)
# ─────────────────────────────────────────────────────────────────────────────
#
# Every constraint enforced anywhere in models/za_gas_model.py and
# models/cov_gas_model.py's _run_filter, classified as either an
# INDIVIDUAL box on one parameter (shown directly as (low, high)) or a
# JOINT constraint tying several parameters together (shown as a marker
# referencing a footnote, since no single (low, high) pair describes it).
# STAR_LEGEND is the fixed, document-wide key for the markers.

STAR_LEGEND = {
    "*":    r"Joint score-response constraint: $\sum_\ell |A_{j,\ell}| \le 2.0$, "
            r"summed across every lag $\ell$ of this dynamic parameter $j$ -- no individual box on $A_{j,\ell}$ itself.",
    "**":   r"Joint persistence constraint: $\sum_\ell |B_{j,\ell}| \le 0.98$, "
            r"summed across every lag $\ell$ of this dynamic parameter $j$ -- no individual box on $B_{j,\ell}$ itself.",
    "***":  r"GB2 moment-existence relationship (not a box on either parameter alone): "
            r"$\zeta > \gamma$ required for $E[Y]<\infty$; softly enforced below $\zeta-\gamma=0.5$.",
    "****": r"Joint covariate-magnitude constraint: $\|\Gamma_j\|_2 \le 20$, "
            r"summed (in quadrature) across every covariate coefficient of this dynamic parameter $j$.",
}


def get_constraint_info(pname: str) -> Tuple[Optional[Tuple[float, float]], Optional[str]]:
    """
    Returns (individual_box, joint_marker) for a RAW (production, non-
    bound_ab/bound_rho) parameter name:
      individual_box : (low, high) if this parameter alone has a box
                        constraint, else None.
      joint_marker   : one of STAR_LEGEND's keys if this parameter is part
                        of a JOINT (multi-parameter) constraint instead,
                        else None. A parameter never has both.
    Parameters with neither (omega_*, f0_*, omega0, omega_y_*) are fully
    free reals.
    """
    if re.match(r"^A_(\w+)_(\d+)$", pname):
        return None, "*"
    if re.match(r"^B_(\w+)_(\d+)$", pname):
        return None, "**"
    if pname == "rho":
        return (-RHO_LIMIT, RHO_LIMIT), None
    if pname in ("gamma", "zeta"):
        return None, "***"
    if re.match(r"^Gamma_[A-Za-z0-9]+_", pname) or re.match(r"^gamma_\w+_\w", pname):
        # Stage-2 covariate coefficients: e.g. "gamma_phi_dewpoint_lag1".
        return None, "****"
    if pname in ("xi",) or re.match(r"^static_", pname):
        return (-STATIC_LIMIT, STATIC_LIMIT), None
    return None, None
