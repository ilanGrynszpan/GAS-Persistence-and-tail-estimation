"""
Backward compatibility of the Numba kernels after the 2026-10-01 changes
(shared _scaled_score_1/_scaled_score_2 helpers, phi-linked pi, mixed
inverse-Fisher codes 3/4): every cached candidate of the 2026-09 staged
rerun must reproduce its stored fit log-likelihood at its cached theta.

The only expected difference: one-parameter candidates fitted with
scaling="inverse_fisher" (the kernels used to run them as unit scaling).

Run:  python tests/test_kernel_backcompat.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import collect_stage_comparison_metrics as C
import run_rerun_10092026 as D

CASES = [  # (location, stage, cfg dir name without suffix, branch)
    ("MANAUS", "stage1", "L1_unit", "phi_only"),
    ("MANAUS", "stage1", "L1_diagonal_inverse_fisher", "phi_xi"),
    ("MANAUS", "stage1", "L1_inverse_fisher", "phi_xi"),
    ("SALVADOR", "stage2", "L1_365_366_dew", "phi_only"),
    ("MANAUS", "stage2", "L1_365_366_dew_temp_enso", "phi_xi"),
    ("BELO HORIZONTE", "stage3", "dew", "phi_only"),
    ("MANAUS", "stage3", "none", "phi_xi"),
    ("SALVADOR", "stage4", "none", "phi_xi"),
    ("DARWIN AIRPORT", "stage4S", "none", "phi_only"),
    ("MANAUS", "stage4S", "dew", "phi_xi"),
]


def main(dump=None):
    out = {}
    worst = 0.0
    for loc, stage, cfg, branch in CASES:
        sfx = "__xi" if branch == "phi_xi" else ""
        cdir = C._tree_for(stage) / C._short(loc) / stage / (cfg + sfx)
        res = json.loads((cdir / "result.json").read_text())
        theta = np.load(cdir / "theta.npy")
        data = D._load_data(loc)
        mcfg = cfg.rsplit("_", 0)[0]
        if stage == "stage1":
            res = dict(res, lag_key=res.get("lag_key"))
        model, fk = C.build_model(loc, stage, cfg, branch, res, data)
        ll = model.loglik(theta, data["y_train"], **fk) if fk else model.loglik(theta, data["y_train"])
        diff = abs(ll - res["loglik"])
        worst = max(worst, diff / max(1.0, abs(res["loglik"])))
        out[f"{loc}|{stage}|{cfg}|{branch}"] = ll
        print(f"{C._short(loc):9s} {stage:7s} {cfg:28s} {branch:8s} stored {res['loglik']:.6f} now {ll:.6f} diff {diff:.2e}")
    print("max relative diff vs stored", worst)
    if dump:
        Path(dump).write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
