"""
Multi-step-ahead marginal effects via paired common-random-numbers (CRN)
Monte Carlo simulation (docs/claude_code_marginal_effects_instructions.md
Sec.3), for horizons h in {1,2,5,10,30,60,90}.

=============================================================================
DESIGN (finalised with the user 2026-09-16, overriding the instructions
doc's literal "every rolling forecast origin" reading)
=============================================================================

One forecast origin per model: the END of the in-sample (training) period
-- t0 = T_train-1, so the first simulated day is the first real
out-of-sample day. This is the model's own genuine multi-step forecast
origin, not a rolling scan through the test period.

Five simulations per model ("5 predictions per location"), all sharing the
SAME baseline draw set via common random numbers:
  1. baseline            -- no intervention at all ("one projection is
                             without changing anything").
  2. DPT +1 raw unit      -- +1 degC dew point at the intervention date.
  3. AT  +1 raw unit      -- +1 degC air temperature at the intervention date.
  4. ENSO -> El Nino      -- el_nino_t: 0->1 at the intervention date
                             (la_nina_t left at its real value).
  5. ENSO -> La Nina      -- la_nina_t: 0->1 at the intervention date
                             (el_nino_t left at its real value).
El Nino / La Nina / Neutral are three MUTUALLY EXCLUSIVE ENSO states (the
two dummies plus the omitted Neutral baseline); (4) and (5) are the two
non-trivial state switches worth reporting on their own (not a linearised
derivative -- a genuine one-time regime switch, matching the instructions
doc's "one-time impulse" language).

Every counterfactual reuses the SAME baseline path set (drawn once, with a
freshly re-seeded np.random.Generator(seed)) as its CRN pair, rather than
redrawing a separate baseline per scenario -- exactly "use what you already
have" (diagnostics.multistep_forecast's existing simulate-forward
machinery), run once for the baseline and once per scenario.

=============================================================================
REUSE, NOT REIMPLEMENTATION
=============================================================================

The endogenous-state forward simulation itself (GAS states/scores updating
along each simulated path, draws from the model's own predictive ZA-GB2
distribution) is diagnostics.multistep_forecast's existing, already-
validated machinery (`_cov_replay`/`_cov_simulate_forward` for
CovZAGASModel, `_harvey_replay`/`_harvey_simulate_forward` for
HarveyZAGASModel -- see that module's own docstring/validation note: at
h=1 its Monte Carlo RMSE/CRPS already matches the real rolling 1-step
metrics).

=============================================================================
"ONE-TIME IMPULSE" -- WHICH ROW IS PERTURBED
=============================================================================

Every covariate array here is indexed so that X_full[t] is the row read
while advancing the state from t to t+1 (see _cov_update / _harvey_step_real
in diagnostics/multistep_forecast.py: `x_t = X_full[t]` inside the loop
that also writes f_arr[t+1]). The forward simulation's first simulated step
is t=t0+1, so the intervention row is X_full[t0+1] -- the first real
out-of-sample day. Every later row (the rest of the forecast horizon,
"perfect foresight" as in this repo's existing multi-step convention) is
left at its real observed value.

=============================================================================
TERMINAL vs. CUMULATIVE TARGET (Sec.3.4-3.5)
=============================================================================

Sec.3.4 (terminal Y_{t+h}) is always computed. Sec.3.5's cumulative-target
addition applies only "if the existing forecast target is cumulative
rainfall" -- it is not (diagnostics.multistep_forecast.multistep_metrics,
this repo's existing multi-step evaluation, scores `draws[:, h-1]`, the
terminal simulated value, never a running sum), so no cumulative-target
effect is computed here.

=============================================================================
CRN CAVEAT
=============================================================================

`diagnostics.multistep_forecast._draw_y` draws one uniform for the
occurrence Bernoulli and, only when that draw is "wet", a second uniform
for the GB2 quantile. A scenario path stays exactly paired with its
baseline twin only up to the first step (if any) at which the intervention
changes that path's occurrence outcome -- from then on the two consume the
shared uniform stream at different offsets. The paired-difference estimator
remains unbiased for E[Y_scenario]-E[Y_baseline]; only its variance-
reduction benefit (and hence MC SE precision) is reduced for paths that
desynchronise. This is a property of the existing _draw_y design, not a
bug introduced here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from diagnostics.multistep_forecast import (
    _cov_replay, _cov_simulate_forward,
    _harvey_replay, _harvey_simulate_forward,
)

from marginal_effects.model_loader import LoadedModel
from marginal_effects.coefficients import column_std, column_in_model

DEFAULT_HORIZONS = (1, 2, 5, 10, 30, 60, 90)
DEFAULT_SEED = 0  # matches diagnostics.multistep_forecast.multistep_metrics's own default.
Z_975 = 1.959963984540054  # scipy.stats.norm.ppf(0.975)

SCENARIOS = ("DPT", "AT", "ENSO_EL_NINO", "ENSO_LA_NINA")
SCENARIO_COLUMN = {
    "DPT": "dewpoint_lag1",
    "AT": "temperature_lag1",
    "ENSO_EL_NINO": "el_nino_t",
    "ENSO_LA_NINA": "la_nina_t",
}
SCENARIO_LABEL = {
    "DPT": "DPT +1°C",
    "AT": "AT +1°C",
    "ENSO_EL_NINO": "ENSO: Neutral/La Niña -> El Niño",
    "ENSO_LA_NINA": "ENSO: Neutral/El Niño -> La Niña",
}


@dataclass
class ScenarioEffect:
    location: str
    branch: str
    family: str
    scenario: str
    horizon: int
    n_paths: int
    in_model: bool
    baseline_mean: float
    baseline_var: float
    delta_mu: float
    delta_mu_se: float
    delta_mu_ci_lo: float
    delta_mu_ci_hi: float
    delta_var: float
    delta_var_se: float
    delta_var_ci_lo: float
    delta_var_ci_hi: float
    n_batches: int


def _model_kind(model) -> str:
    t = type(model).__name__
    if t not in ("CovZAGASModel", "HarveyZAGASModel"):
        raise ValueError(f"multistep marginal effects only support CovZAGASModel/HarveyZAGASModel, got {t}")
    return t


def _end_of_sample_origin(loaded: LoadedModel) -> int:
    """t0 index into y_full=concat(y_train,y_test) such that the first
    simulated day is the first real OOS day -- "the origin for the
    multistep is always the end of the in-sample period" (user, 2026-09-16)."""
    return len(loaded.data["y_train"]) - 1


def _build_replay(loaded: LoadedModel):
    """One deterministic real-data replay per loaded model, shared by the
    baseline and every scenario (the perturbation only ever touches rows
    after the forecast origin)."""
    model, theta, data = loaded.model, loaded.theta, loaded.data
    y_full = np.concatenate([data["y_train"], data["y_test"]])
    kind = _model_kind(model)
    if kind == "CovZAGASModel":
        X_full = np.vstack([loaded.ms_kwargs["X_train"], loaded.ms_kwargs["X_test"]])
        replay = _cov_replay(model, theta, y_full, X_full)
        replay["_y_real"] = y_full
        return kind, replay, {"X_full": X_full}
    Xl_full = np.vstack([loaded.ms_kwargs["X_long_train"], loaded.ms_kwargs["X_long_test"]])
    Xs_full = np.vstack([loaded.ms_kwargs["X_short_train"], loaded.ms_kwargs["X_short_test"]])
    replay = _harvey_replay(model, theta, y_full, Xl_full, Xs_full)
    return kind, replay, {"Xl_full": Xl_full, "Xs_full": Xs_full, "y_full": y_full}


def _perturb_column(loaded: LoadedModel, kind: str, arrays: dict, column: str, t0: int):
    """Returns (perturbed_arrays, in_model). `arrays` (the baseline) is
    returned unchanged, with in_model=False, if this model does not use
    `column` at all -- callers should then skip simulation (effect is
    trivially 0 at every horizon)."""
    model = loaded.model
    if not column_in_model(model, column):
        return arrays, False
    delta_std = 1.0 / column_std(loaded.data, column)

    if kind == "CovZAGASModel":
        idx = list(model.cov_names).index(column)
        X_cf = arrays["X_full"].copy()
        X_cf[t0 + 1, idx] += delta_std
        return {"X_full": X_cf}, True

    long_names, short_names = list(model.long_names), list(model.short_names)
    if column in long_names:
        idx = long_names.index(column)
        Xl_cf = arrays["Xl_full"].copy()
        Xl_cf[t0 + 1, idx] += delta_std
        return {"Xl_full": Xl_cf, "Xs_full": arrays["Xs_full"]}, True
    idx = short_names.index(column)
    Xs_cf = arrays["Xs_full"].copy()
    Xs_cf[t0 + 1, idx] += delta_std
    return {"Xl_full": arrays["Xl_full"], "Xs_full": Xs_cf}, True


def _simulate(model, kind: str, replay: dict, arrays: dict, t0: int, h_max: int,
              n_paths: int, seed: int, extra: dict) -> np.ndarray:
    """Draws (n_paths, h_max) forward paths from a freshly-seeded Generator
    (CRN: call this once per branch, always with the SAME `seed`, so every
    branch consumes the identical raw-uniform stream)."""
    rng = np.random.default_rng(seed)
    if kind == "CovZAGASModel":
        return _cov_simulate_forward(model, replay, arrays["X_full"], t0, h_max, n_paths, rng)
    y_real = extra["y_full"]
    return _harvey_simulate_forward(model, replay, y_real, arrays["Xl_full"], arrays["Xs_full"],
                                     t0, h_max, n_paths, rng)


def _batched_variance(x: np.ndarray, n_batches: int):
    n = len(x)
    bs = n // n_batches
    return np.array([np.var(x[b * bs:(b + 1) * bs], ddof=1) for b in range(n_batches)])


def scenario_effects_for_model(loaded: LoadedModel, horizons: Sequence[int] = DEFAULT_HORIZONS,
                                n_paths: int = 10_000, seed: int = DEFAULT_SEED,
                                n_batches: int = 20) -> pd.DataFrame:
    """The model's 5 predictions: 1 baseline + 4 named scenarios
    (SCENARIOS), from the single end-of-training-sample origin, common
    random numbers throughout. One row per (scenario, horizon); baseline
    itself is also emitted as scenario="BASELINE" with delta_*=0 by
    construction (its own trajectory is in baseline_mean/baseline_var)."""
    model, data = loaded.model, loaded.data
    h_max = max(horizons)
    kind, replay, base_arrays = _build_replay(loaded)
    t0 = _end_of_sample_origin(loaded)
    n_test = len(data["y_test"])
    if t0 + h_max >= len(data["y_train"]) + n_test:
        raise ValueError(f"h_max={h_max} exceeds the available OOS period (n_test={n_test})")

    draws_base = _simulate(model, kind, replay, base_arrays, t0, h_max, n_paths, seed, base_arrays)

    rows = []
    sel = loaded.sel
    for h in horizons:
        base_h = draws_base[:, h - 1]
        rows.append(dict(
            location=sel.short, branch=sel.branch, family=sel.family, scenario="BASELINE",
            horizon=h, n_paths=n_paths, in_model=True,
            baseline_mean=float(np.mean(base_h)), baseline_var=float(np.var(base_h, ddof=1)),
            delta_mu=0.0, delta_mu_se=0.0, delta_mu_ci_lo=0.0, delta_mu_ci_hi=0.0,
            delta_var=0.0, delta_var_se=0.0, delta_var_ci_lo=0.0, delta_var_ci_hi=0.0,
            n_batches=n_batches,
        ))

    for scenario in SCENARIOS:
        column = SCENARIO_COLUMN[scenario]
        cf_arrays, in_model = _perturb_column(loaded, kind, base_arrays, column, t0)
        if not in_model:
            for h in horizons:
                base_h = draws_base[:, h - 1]
                rows.append(dict(
                    location=sel.short, branch=sel.branch, family=sel.family, scenario=scenario,
                    horizon=h, n_paths=0, in_model=False,
                    baseline_mean=float(np.mean(base_h)), baseline_var=float(np.var(base_h, ddof=1)),
                    delta_mu=0.0, delta_mu_se=0.0, delta_mu_ci_lo=0.0, delta_mu_ci_hi=0.0,
                    delta_var=0.0, delta_var_se=0.0, delta_var_ci_lo=0.0, delta_var_ci_hi=0.0,
                    n_batches=0,
                ))
            continue

        draws_cf = _simulate(model, kind, replay, cf_arrays, t0, h_max, n_paths, seed, base_arrays)
        for h in horizons:
            base_h, cf_h = draws_base[:, h - 1], draws_cf[:, h - 1]
            diff = cf_h - base_h
            delta_mu = float(np.mean(diff))
            se_mu = float(np.std(diff, ddof=1) / np.sqrt(n_paths))
            var_base_b = _batched_variance(base_h, n_batches)
            var_cf_b = _batched_variance(cf_h, n_batches)
            delta_var_b = var_cf_b - var_base_b
            delta_var = float(np.mean(delta_var_b))
            se_var = float(np.std(delta_var_b, ddof=1) / np.sqrt(n_batches))
            rows.append(dict(
                location=sel.short, branch=sel.branch, family=sel.family, scenario=scenario,
                horizon=h, n_paths=n_paths, in_model=True,
                baseline_mean=float(np.mean(base_h)), baseline_var=float(np.var(base_h, ddof=1)),
                delta_mu=delta_mu, delta_mu_se=se_mu,
                delta_mu_ci_lo=delta_mu - Z_975 * se_mu, delta_mu_ci_hi=delta_mu + Z_975 * se_mu,
                delta_var=delta_var, delta_var_se=se_var,
                delta_var_ci_lo=delta_var - Z_975 * se_var, delta_var_ci_hi=delta_var + Z_975 * se_var,
                n_batches=n_batches,
            ))
    return pd.DataFrame(rows)


def convergence_check_for_model(loaded: LoadedModel, horizons: Sequence[int] = DEFAULT_HORIZONS,
                                 n_paths_base: int = 10_000, n_paths_boost: int = 30_000,
                                 seed_base: int = DEFAULT_SEED, seed_boost: int = 123,
                                 se_ratio_threshold: float = 0.5, n_batches: int = 20) -> pd.DataFrame:
    """Sec.3.7's convergence check for one model, ALL 4 scenarios at once:
    repeats the single origin with 3x the paths under an independent seed,
    flagging any (scenario, horizon) whose base-run MC SE is large relative
    to the estimated effect.

    Calls scenario_effects_for_model exactly twice (base, boost) -- each
    call already computes every scenario in one pass (see that function's
    own docstring: 1 shared baseline draw + 4 scenario draws), so this
    covers all 4 scenarios for the cost of 2 full-model simulations, not
    2-per-scenario (an earlier version of this function took `scenario` as
    an argument and was called once per scenario, silently recomputing --
    and discarding -- the other 3 scenarios' draws on every call: an 8x
    waste per model, found and fixed 2026-09-17 after it contributed to a
    multi-hour convergence-check phase overrunning available memory)."""
    base = scenario_effects_for_model(loaded, horizons, n_paths_base, seed_base, n_batches)
    boost = scenario_effects_for_model(loaded, horizons, n_paths_boost, seed_boost, n_batches)
    rows = []
    for scenario in SCENARIOS:
        b_s = base[base["scenario"] == scenario].set_index("horizon")
        k_s = boost[boost["scenario"] == scenario].set_index("horizon")
        for h in horizons:
            b, k = b_s.loc[h], k_s.loc[h]
            se_ratio_mu = abs(b["delta_mu_se"] / b["delta_mu"]) if b["delta_mu"] != 0 else np.inf
            se_ratio_var = abs(b["delta_var_se"] / b["delta_var"]) if b["delta_var"] != 0 else np.inf
            rows.append({
                "scenario": scenario, "horizon": h,
                "delta_mu_base": b["delta_mu"], "delta_mu_se_base": b["delta_mu_se"],
                "delta_mu_boost": k["delta_mu"], "delta_mu_se_boost": k["delta_mu_se"],
                "se_ratio_mu": se_ratio_mu, "warn_mu": se_ratio_mu > se_ratio_threshold,
                "boost_within_base_95ci": b["delta_mu_ci_lo"] <= k["delta_mu"] <= b["delta_mu_ci_hi"],
                "delta_var_base": b["delta_var"], "delta_var_se_base": b["delta_var_se"],
                "delta_var_boost": k["delta_var"], "delta_var_se_boost": k["delta_var_se"],
                "se_ratio_var": se_ratio_var, "warn_var": se_ratio_var > se_ratio_threshold,
            })
    return pd.DataFrame(rows)
