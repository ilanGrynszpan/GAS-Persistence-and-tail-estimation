"""
Model selection for the marginal-effects deliverable.

=============================================================================
CRITERION  (docs/claude_code_marginal_effects_instructions.md, "TAKE CARE")
=============================================================================

"Use the model that has gradient norm of less than one and that minimized
RMSE and CRPS, prefer a model with 3 external variables. If it is not
possible, try to get at least one model that contains each variable."

Applied independently for each of the 24 (location x branch x family) cells:

    location in {bh, cruzeiro, darwin, garanhuns, manaus, salvador}
    branch   in {"phi_only" (1 TVP), "phi_xi" (2 TVP)}
    family   in {"standard" (stage1 U stage2, plain/Cov GAS),
                 "harvey"   (stage3 U stage4 U stage4S, long-short)}

Selection rule, matching this repo's own established tie-break convention
(generate_ml_comparison_report.py's select(); see also pipeline/selection.py):

    1. Hard filter: grad_norm_inf < 1 (this repo's RELIABLE_GRAD_THRESHOLD,
       inf-norm -- see generate_ml_comparison_report.py, run_polish_reoptimization.py,
       etc.) and finite OOS CRPS.
    2. Prefer the 3-variable config (dew_temp_enso / DPT+AT+ENSO) if ANY
       such candidate clears step 1.
    3. Within the resulting pool, minimize OOS CRPS (main criterion; 1%
       relative tie-tolerance, matching generate_ml_comparison_report.select()),
       then OOS RMSE as the tie-break.

=============================================================================
TWO DOCUMENTED DEVIATIONS FROM THE PURE RULE (user-approved 2026-09-16)
=============================================================================

For 2 of the 24 cells, the literal winner under steps 1-3 has ZERO external
covariates (a pure baseline model), which cannot produce a marginal-effects
row at all (every c_v is structurally zero, not just numerically small).
Since the entire point of this deliverable is covariate marginal effects,
the user chose (when asked) to use the best covariate-BEARING grad<1
alternative for these two cells instead, accepting a small RMSE/CRPS cost:

    Darwin Airport,  1 TVP, harvey  : stage4S/none (grad .149, rmse 15.00,
        crps 4.49) -> OVERRIDDEN to stage3/dew_enso (grad .601, rmse 15.03,
        crps 4.47; DPT+ENSO, no AT).
    Cruzeiro do Sul, 2 TVP, standard: stage1/L1 (grad .70, rmse 13.78,
        crps 5.62, no covariates at all) -> OVERRIDDEN to
        stage2/L1_dew_enso (grad .95, rmse 14.01, crps 5.74; DPT+ENSO, no AT).

Both overrides are covariate-bearing but still miss AT (no stage3/4/4S or
stage2 config at these two cells has DPT+AT+ENSO clearing grad<1 -- see
NO_THREE_VAR_REASON below).

=============================================================================
TWO CONFIRMED-MISSING CELLS
=============================================================================

    Cruzeiro do Sul, 2 TVP, harvey : no stage3/4/4S phi_xi config anywhere
        clears grad_norm_inf < 1.
    Manaus,          2 TVP, harvey : same.

These are left absent from the selection (None), per the user's own
instruction: "If there are no suitable models ... let me know, dont use
badly optimized models if you cant find one that suits." The report must
show these cells as not-available with a footnote, not silently drop them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "artifacts" / "stage_comparison_20260911" / "metrics.json"

SHORTS = ["bh", "cruzeiro", "darwin", "garanhuns", "manaus", "salvador"]
LOCATION_FULL = {
    "bh": "BELO HORIZONTE", "cruzeiro": "CRUZEIRO DO SUL (ACRE)",
    "darwin": "DARWIN AIRPORT", "garanhuns": "GARANHUNS (PERNAMBUCO)",
    "manaus": "MANAUS", "salvador": "SALVADOR",
}
LOCATION_DISPLAY = {
    "bh": "Belo Horizonte", "cruzeiro": "Cruzeiro do Sul",
    "darwin": "Darwin Airport", "garanhuns": "Garanhuns",
    "manaus": "Manaus", "salvador": "Salvador",
}
BRANCHES = ["phi_only", "phi_xi"]
FAMILIES = ["standard", "harvey"]

GAS_LAG_SETS = ["L1", "L123", "L1_365_366", "L123_365_366"]
STAGE2_COV = ["dew", "dew_temp", "dew_enso", "dew_temp_enso"]
HARVEY_COV5 = ["none", "dew", "dew_temp", "dew_enso", "dew_temp_enso"]
HARVEY_STAGES = ["stage3", "stage4", "stage4S"]

GRAD_THRESHOLD = 1.0
CRPS_REL_TOL = 0.01

# User-approved overrides (see module docstring). Keys: (short, branch, family).
OVERRIDES: Dict[Tuple[str, str, str], Tuple[str, str, str]] = {
    ("darwin", "phi_only", "harvey"): ("stage3", "dew_enso", "phi_only"),
    ("cruzeiro", "phi_xi", "standard"): ("stage2", "L1_dew_enso", "phi_xi"),
}


class ModelSelection:
    """One resolved slot: which (stage, cfg, branch) won, its cached metrics,
    and how many/which named external variables it uses."""

    def __init__(self, short: str, branch: str, family: str,
                 stage: Optional[str], cfg: Optional[str],
                 metrics: Optional[dict], variables: Tuple[str, ...],
                 note: str, overridden: bool = False, primary: bool = True):
        self.short = short
        self.branch = branch
        self.family = family
        self.stage = stage
        self.cfg = cfg
        self.metrics = metrics
        self.variables = variables  # subset of ("DPT", "AT", "ENSO"), present in this model
        self.note = note
        self.overridden = overridden
        # primary=True: this is the official pick (select_one's return value)
        # for its (short,branch,family) cell. primary=False: an alternative
        # grad<1, variable-bearing candidate from the same cell, computed
        # only for the reported min/max robustness range across models that
        # share a given variable -- see all_valid_models().
        self.primary = primary

    @property
    def available(self) -> bool:
        return self.stage is not None

    @property
    def n_tvp(self) -> int:
        return 1 if self.branch == "phi_only" else 2

    def __repr__(self):
        if not self.available:
            return f"<ModelSelection {self.short}/{self.branch}/{self.family}: UNAVAILABLE ({self.note})>"
        return (f"<ModelSelection {self.short}/{self.branch}/{self.family}: "
                f"{self.stage}/{self.cfg} vars={self.variables} {self.note}>")


def _load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


def _m(met: dict, short: str, stage: str, cfg: str, branch: str) -> Optional[dict]:
    return met.get(short, {}).get(stage, {}).get(cfg, {}).get(branch)


def _variables_in_cfg(cfg: str, stage: str = "") -> Tuple[str, ...]:
    """cfg like 'dew', 'dew_temp_enso', 'L1_365_366_dew_enso', 'none'.

    stage4S is HarveyZAGASModel(include_long=False) -- run_rerun_10092026.py's
    run_harvey_stage sets `incl_long = (stage != "stage4S")` unconditionally,
    but _harvey_cov() still builds and names the ENSO ("long") covariate
    block for a stage4S cfg exactly as it would for stage3/stage4 (the same
    cfg vocabulary is reused across all three stages). HarveyZAGASModel's
    own _decode_harvey then hard-sets Gamma_L (and kappa_L, phi_L, L0) to
    the CONSTANT 0 whenever include_long=False -- not a fitted value that
    happens to be small, a parameter the optimizer never had a theta slot
    for. So a stage4S "dew_temp_enso" config has exactly the same true
    covariate capacity as "dew_temp": ENSO is structurally inert. Found
    2026-09-17 (report showed several stage4S ENSO effects as EXACTLY
    0.000, not just small) -- confirmed against 3 fitted stage4S models via
    marginal_effects.coefficients.covariate_coefficients (c_phi=c_xi=0.0
    exactly for the el_nino_t/la_nina_t columns in every one of them).
    Without this stage check, `_select`'s "prefer a 3-variable config"
    step was fooled by the cfg name into crediting stage4S/dew_temp_enso
    as satisfying the 3-variable preference, sometimes ahead of a
    genuinely 3-variable stage3/stage4 candidate that also cleared
    grad_norm_inf < 1."""
    out = []
    if "dew" in cfg:
        out.append("DPT")
    if "temp" in cfg:
        out.append("AT")
    if "enso" in cfg and stage != "stage4S":
        out.append("ENSO")
    return tuple(out)


def _candidates_standard(met: dict, short: str, branch: str) -> List[Tuple[str, str, dict, Tuple[str, ...]]]:
    out = []
    for lk in GAS_LAG_SETS:
        mm = _m(met, short, "stage1", lk, branch)
        if mm is not None:
            out.append(("stage1", lk, mm, ()))
    for lk in GAS_LAG_SETS:
        for cov in STAGE2_COV:
            cfg = f"{lk}_{cov}"
            mm = _m(met, short, "stage2", cfg, branch)
            if mm is not None:
                out.append(("stage2", cfg, mm, _variables_in_cfg(cov)))
    return out


def _candidates_harvey(met: dict, short: str, branch: str) -> List[Tuple[str, str, dict, Tuple[str, ...]]]:
    out = []
    for stage in HARVEY_STAGES:
        for cov in HARVEY_COV5:
            mm = _m(met, short, stage, cov, branch)
            if mm is not None:
                out.append((stage, cov, mm, _variables_in_cfg(cov, stage=stage)))
    return out


def _select(cands: List[Tuple[str, str, dict, Tuple[str, ...]]]) -> Tuple[Optional[Tuple[str, str, dict, Tuple[str, ...]]], str]:
    pool = [c for c in cands
            if np.isfinite(c[2].get("grad_inf", np.nan)) and c[2]["grad_inf"] < GRAD_THRESHOLD
            and np.isfinite(c[2].get("crps_oos", np.nan))]
    if not pool:
        return None, "no candidate with grad_norm_inf < 1 and finite OOS CRPS"
    has3 = [c for c in pool if len(c[3]) == 3]
    used = has3 if has3 else pool
    note = ("3-variable (DPT+AT+ENSO) config available"
            if has3 else "no 3-variable config cleared grad<1; best available variable set used")
    min_crps = min(c[2]["crps_oos"] for c in used)
    close = [c for c in used if c[2]["crps_oos"] <= min_crps * (1 + CRPS_REL_TOL)]
    winner = close[0] if len(close) == 1 else min(close, key=lambda c: c[2]["rmse_oos"])
    return winner, note


def select_one(met: dict, short: str, branch: str, family: str) -> ModelSelection:
    cands = _candidates_standard(met, short, branch) if family == "standard" else _candidates_harvey(met, short, branch)
    winner, note = _select(cands)
    override_key = (short, branch, family)
    if override_key in OVERRIDES:
        o_stage, o_cfg, o_branch = OVERRIDES[override_key]
        mm = _m(met, short, o_stage, o_cfg, o_branch)
        variables = _variables_in_cfg(o_cfg, stage=o_stage)
        base_note = (f"user-approved override (2026-09-16): literal grad<1/CRPS/RMSE winner had "
                     f"zero covariates; using best covariate-bearing grad<1 alternative instead "
                     f"({o_stage}/{o_cfg}, grad={mm['grad_inf']:.3f}, rmse={mm['rmse_oos']:.3f}, "
                     f"crps={mm['crps_oos']:.3f})")
        return ModelSelection(short, branch, family, o_stage, o_cfg, mm, variables, base_note, overridden=True)
    if winner is None:
        return ModelSelection(short, branch, family, None, None, None, (), note)
    stage, cfg, mm, variables = winner
    return ModelSelection(short, branch, family, stage, cfg, mm, variables, note)


def select_all(metrics_path: Optional[Path] = None) -> Dict[Tuple[str, str, str], ModelSelection]:
    """Returns {(short, branch, family): ModelSelection} for all 6*2*2=24 cells."""
    met = json.loads((metrics_path or METRICS_PATH).read_text())
    out = {}
    for short in SHORTS:
        for branch in BRANCHES:
            for family in FAMILIES:
                out[(short, branch, family)] = select_one(met, short, branch, family)
    return out


def all_valid_models(met: dict, short: str, branch: str, family: str) -> List[ModelSelection]:
    """Every candidate for this (location, branch, family) cell that clears
    grad_norm_inf < 1, has finite OOS CRPS, and uses at least one true
    external variable (stage-aware -- see _variables_in_cfg). The official
    pick (select_one's return, respecting OVERRIDES) is always list[0] with
    .primary=True; every other qualifying candidate follows, sorted by OOS
    CRPS, with .primary=False.

    Used for the "min/max marginal effect across every correctly-optimised
    model that uses this variable" robustness range (user request,
    2026-09-17) -- NOT for re-deciding which model represents the location/
    branch/family cell in the main tables; that remains select_one's job."""
    cands = _candidates_standard(met, short, branch) if family == "standard" else _candidates_harvey(met, short, branch)
    pool = [c for c in cands
            if np.isfinite(c[2].get("grad_inf", np.nan)) and c[2]["grad_inf"] < GRAD_THRESHOLD
            and np.isfinite(c[2].get("crps_oos", np.nan)) and len(c[3]) > 0]

    primary = select_one(met, short, branch, family)
    out: List[ModelSelection] = []
    seen = set()
    if primary.available:
        out.append(primary)
        seen.add((primary.stage, primary.cfg))

    for stage, cfg, mm, variables in sorted(pool, key=lambda c: c[2]["crps_oos"]):
        if (stage, cfg) in seen:
            continue
        out.append(ModelSelection(short, branch, family, stage, cfg, mm, variables,
                                   "alternative grad<1 candidate (robustness range)", primary=False))
        seen.add((stage, cfg))
    return out


def all_valid_models_all_cells(metrics_path: Optional[Path] = None) -> Dict[Tuple[str, str, str], List[ModelSelection]]:
    """all_valid_models() for every (short, branch, family) cell."""
    met = json.loads((metrics_path or METRICS_PATH).read_text())
    return {(short, branch, family): all_valid_models(met, short, branch, family)
            for short in SHORTS for branch in BRANCHES for family in FAMILIES}


if __name__ == "__main__":
    sel = select_all()
    n_avail = sum(1 for v in sel.values() if v.available)
    print(f"{n_avail}/{len(sel)} slots available\n")
    for key, ms in sorted(sel.items()):
        print(key, "->", ms)
