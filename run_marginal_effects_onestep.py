"""
Computes rolling one-step-ahead marginal effects (marginal_effects.onestep)
for every selected model (marginal_effects.model_selection) and every named
covariate (DPT, AT, ENSO), and caches the results to
artifacts/marginal_effects/onestep/.

Per docs/EXECUTION.md ("DO NOT RUN ANY CODE YOU BUILT BEFORE I AUTHORIZE"),
this script is NOT executed automatically -- see the accompanying chat
message for the code-review request. It is cheap to run once authorized:
~0.2-0.5s per model (no Monte Carlo, purely analytical).

=============================================================================
OUTPUTS
=============================================================================

For every (location, branch, family) with an available PRIMARY model:
  artifacts/marginal_effects/onestep/<short>_<branch>_<family>_rows.csv
      one row per (date, variable): dm1_dx, dsigma2_dx, dM1_dx, dVM_dx,
      plus the filtered states (phi_t, xi_t_natural, pi_t) used to compute
      them, for full auditability.
  artifacts/marginal_effects/onestep/summary.csv
      one row per (location, branch, family, variable): mean/median/std and
      the 2.5/25/75/97.5% empirical quantiles of dM1_dx and dVM_dx over the
      rolling OOS window (docs/claude_code_marginal_effects_instructions.md
      Sec.2 point 4) -- this is what generate_marginal_effects_report.py's
      Tables 1-2 read for the point estimate.

For every OTHER grad<1, variable-bearing candidate in the same cell (user
request 2026-09-17: report the min/max marginal effect across every
correctly-optimised model that uses a given variable, not only the one
selected as "best"):
  artifacts/marginal_effects/onestep/alternatives/<short>_<branch>_<family>__<stage>__<cfg>_rows.csv
  artifacts/marginal_effects/onestep/alternatives_summary.csv
      same columns as summary.csv, plus `primary` (False for these rows) --
      this is what the report's tables read for the (min, max) range.

Selections with no available primary model (Cruzeiro do Sul 2-TVP Harvey,
Manaus 2-TVP Harvey) are recorded in summary.csv with all-NaN effect
columns and a "not_available" reason string, rather than silently omitted.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marginal_effects.model_selection import select_all, all_valid_models_all_cells, LOCATION_DISPLAY
from marginal_effects.model_loader import load
from marginal_effects.onestep import all_variables_one_step, summarize
from marginal_effects.coefficients import VARIABLES

OUT = ROOT / "artifacts" / "marginal_effects" / "onestep"
ALT_OUT = OUT / "alternatives"


def _summary_keys():
    keys = []
    for col in ("dm1_dx", "dsigma2_dx", "dM1_dx", "dVM_dx"):
        keys += [f"{col}_mean", f"{col}_median", f"{col}_std",
                 f"{col}_q0.025", f"{col}_q0.25", f"{col}_q0.75", f"{col}_q0.975"]
    keys += ["n_obs", "n_finite"]
    return keys


def _compute_one(sel, tag: str, rows_path: Path):
    """Computes and saves all_variables_one_step for one ModelSelection;
    returns its per-variable summary rows."""
    lm = load(sel)
    df = all_variables_one_step(lm)
    df.to_csv(rows_path, index=False)
    out = []
    for v in VARIABLES:
        sub = df[df["variable"] == v]
        s = summarize(sub)
        out.append({
            "location": sel.short, "location_display": LOCATION_DISPLAY[sel.short],
            "branch": sel.branch, "family": sel.family, "variable": v,
            "stage": sel.stage, "cfg": sel.cfg, "in_model": bool(sub["in_model"].iloc[0]),
            "primary": sel.primary, "reason": sel.note, **s,
        })
    return out, len(df)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ALT_OUT.mkdir(parents=True, exist_ok=True)
    selection = select_all()
    summary_rows = []
    t0 = time.time()

    for (short, branch, family), sel in sorted(selection.items()):
        tag = f"{short}_{branch}_{family}"
        if not sel.available:
            for v in VARIABLES:
                summary_rows.append({
                    "location": short, "location_display": LOCATION_DISPLAY[short],
                    "branch": branch, "family": family, "variable": v,
                    "stage": None, "cfg": None, "in_model": False, "primary": True,
                    "reason": sel.note, **{k: np.nan for k in _summary_keys()},
                })
            print(f"[{tag}] UNAVAILABLE: {sel.note}")
            continue
        rows, n = _compute_one(sel, tag, OUT / f"{tag}_rows.csv")
        summary_rows.extend(rows)
        print(f"[{tag}] {sel.stage}/{sel.cfg}: {n} rows written")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(f"\nWrote {OUT / 'summary.csv'} ({len(summary)} rows) in {time.time()-t0:.1f}s")

    # ---- alternative (non-primary) candidates, for the min/max range ----
    t1 = time.time()
    alt_rows = []
    all_models = all_valid_models_all_cells()
    alternatives = [ms for models in all_models.values() for ms in models if not ms.primary]
    for sel in alternatives:
        tag = f"{sel.short}_{sel.branch}_{sel.family}__{sel.stage}__{sel.cfg}"
        rows, n = _compute_one(sel, tag, ALT_OUT / f"{tag}_rows.csv")
        alt_rows.extend(rows)
        print(f"  [alt] {tag}: {n} rows")

    alt_summary = pd.DataFrame(alt_rows)
    alt_summary.to_csv(OUT / "alternatives_summary.csv", index=False)
    print(f"\nWrote {OUT / 'alternatives_summary.csv'} ({len(alt_summary)} rows, "
          f"{len(alternatives)} models) in {time.time()-t1:.1f}s")


if __name__ == "__main__":
    main()
