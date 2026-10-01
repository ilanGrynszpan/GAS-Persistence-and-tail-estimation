"""
Reconstructs a SELECTED fitted model (marginal_effects.model_selection)
exactly as it was originally fitted -- distribution, dynamics, covariates,
theta -- with NO re-optimisation, per docs/EXECUTION.md ("Reports should
never call optimization... read only artifacts") and the marginal-effects
brief's "Reuse the existing fitted-model objects ... Do not create a
separate model implementation."

=============================================================================
WHY THIS ISN'T JUST run_rerun_10092026._build_for_augment
=============================================================================

Stage1/Stage2/Stage4S results all live under artifacts/rerun_10092026/ and
were built with the ORIGINAL (raw, unbounded) Harvey phi_L parameterisation
-- for these, run_rerun_10092026.py's own `_build_for_augment` reconstructs
the model correctly and is reused directly (no duplication).

Stage3/Stage4 results, however, come from a LATER, separate rerun
(run_stage34_phiL_sigmoid.py, 2026-09-11) that refit those two stages only,
into artifacts/rerun_10092026_phiL_sigmoid/, with HarveyZAGASModel's
phi_L_max=1.0 sigmoid-bounded reparameterisation (see that script's own
"WHY THIS EXISTS" docstring). Reconstructing a stage3/stage4 theta with
`_build_for_augment` (phi_L_max=None) would silently decode the WRONG
model -- so this module mirrors run_stage34_phiL_sigmoid.py's own
`_task_harvey_sig` construction instead for exactly those two stages.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_rerun_10092026 as D  # noqa: E402  (reuse, not reimplement -- see module docstring)
from models.harvey_gas import HarveyZAGASModel  # noqa: E402

from marginal_effects.model_selection import ModelSelection, LOCATION_FULL  # noqa: E402

OLD_TREE = D.OUT                                        # artifacts/rerun_10092026
SIGMOID_TREE = ROOT / "artifacts" / "rerun_10092026_phiL_sigmoid"
PHI_L_MAX = 1.0                                          # matches run_stage34_phiL_sigmoid.py


class LoadedModel:
    """Everything downstream (onestep.py, multistep.py) needs for one
    selected fitted model: the reconstructed model object, its cached
    theta, the loaded station data (y/dates/covariate blocks + scalers),
    and `ms_kwargs` -- the covariate dict in exactly the shape
    diagnostics.multistep_forecast.multistep_metrics / model.simulate_oos
    expect ({"X_train","X_test"} for Cov, {"X_long_train",...} for Harvey,
    {} for the plain no-covariate ZAGASModel)."""

    def __init__(self, sel: ModelSelection, model, theta: np.ndarray, data: dict,
                 ms_kwargs: dict, cfg_dir: Path):
        self.sel = sel
        self.model = model
        self.theta = theta
        self.data = data
        self.ms_kwargs = ms_kwargs
        self.cfg_dir = cfg_dir


def _sfx(branch: str) -> str:
    return D._sfx(branch)


def _resolve_stage1_dirname(short: str, cfg: str, branch: str) -> str:
    """Stage1 folders carry an extra "_<scaling>" suffix chosen by
    run_rerun_10092026.run_stage1 (per-location gradient-tier/AIC pick,
    cached in _scaling_choice*.json) -- read it back rather than
    re-deriving the choice."""
    p = OLD_TREE / short / "stage1" / f"_scaling_choice{_sfx(branch)}.json"
    sc = json.loads(p.read_text())["per_lag"][cfg]["scaling"]
    return f"{cfg}_{sc}"


def _cfg_dir(sel: ModelSelection) -> Path:
    sfx = _sfx(sel.branch)
    if sel.stage == "stage1":
        return OLD_TREE / sel.short / "stage1" / (_resolve_stage1_dirname(sel.short, sel.cfg, sel.branch) + sfx)
    if sel.stage in ("stage2", "stage4S"):
        return OLD_TREE / sel.short / sel.stage / (sel.cfg + sfx)
    if sel.stage in ("stage3", "stage4"):
        return SIGMOID_TREE / sel.short / sel.stage / (sel.cfg + sfx)
    raise ValueError(f"unrecognised stage {sel.stage!r}")


def _build_sigmoid_harvey(data: dict, short: str, stage: str, cfg: str, branch: str):
    """Mirrors run_stage34_phiL_sigmoid._task_harvey_sig's model
    construction exactly (phi_L_max=1.0), for stage3/stage4 only."""
    scaling = D._chosen_scaling(short, branch)
    harm = 0 if stage == "stage3" else D.N_HARM
    damp = None if stage == "stage3" else D.DAMP
    Xl_tr, ln, Xs_tr, sn = D._harvey_cov(data, cfg, "train")
    Xl_te, _, Xs_te, _ = D._harvey_cov(data, cfg, "test")
    model = HarveyZAGASModel(
        distribution=D._dist(branch), pi_dynamics=D.pi_dyn(), seasonal="daily",
        long_names=ln, short_names=sn, harvey_target=D._htarget(branch),
        scaling=scaling, static_params=D._static(branch),
        n_harmonics=harm, seasonal_damping_max=damp, include_long=True,
        phi_L_max=PHI_L_MAX,
    )
    ms_kwargs = {"X_long_train": Xl_tr, "X_long_test": Xl_te,
                 "X_short_train": Xs_tr, "X_short_test": Xs_te}
    return model, ms_kwargs


def load(sel: ModelSelection) -> LoadedModel:
    """Reconstruct the model + theta + data for one resolved selection.
    Raises if `sel` has no fitted model (sel.available is False)."""
    if not sel.available:
        raise ValueError(f"selection has no fitted model to load: {sel!r}")

    location = LOCATION_FULL[sel.short]
    data = D._load_data(location)  # lru_cached in run_rerun_10092026 -- one load per location
    cfg_dir = _cfg_dir(sel)
    theta = np.load(cfg_dir / "theta.npy")

    if sel.stage in ("stage3", "stage4"):
        model, ms_kwargs = _build_sigmoid_harvey(data, sel.short, sel.stage, sel.cfg, sel.branch)
    else:
        res = json.loads((cfg_dir / "result.json").read_text())
        model, filt_kwargs, oos_kwargs, ms_kwargs = D._build_for_augment(res, data, cfg_dir)

    if len(theta) != model.n_params:
        raise ValueError(
            f"theta length {len(theta)} != reconstructed model.n_params {model.n_params} "
            f"for {sel!r} at {cfg_dir} -- model reconstruction does not match the cached fit."
        )
    return LoadedModel(sel=sel, model=model, theta=theta, data=data, ms_kwargs=ms_kwargs, cfg_dir=cfg_dir)
