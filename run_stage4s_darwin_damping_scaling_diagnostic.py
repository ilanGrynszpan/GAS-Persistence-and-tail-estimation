"""
Diagnostic (2026-09-13): why does Darwin / Stage 4S ("basic structural
model") / phi+xi / cov=none diverge?

WHY THIS EXISTS
================================================================================
See the conversation's diagnosis: the production fit at
artifacts/rerun_10092026/darwin/stage4S/none__xi/ (scaling="unit",
seasonal_damping_max=0.999, per run_rerun_10092026.py) has grad_inf~5.3e4,
only 22% of parameters converged, and OOS CRPS=NaN. Decoding its theta and
re-running the filter shows xi_t spikes to 11.39 at a single day (t=742),
giving GB2 mean(phi=5.64, xi=11.39, ...) ~ 1.6e13 -- one observation alone
explains the ~8e10 in-sample RMSE.

This script re-fits ONLY that one (location, stage, branch, cov) cell --
Darwin / stage4S / phi_xi / cov=none -- under every combination of:

  scaling in {unit, diagonal_inverse_fisher, inverse_fisher}
  seasonal_damping_max in {0.999 (production value), 1.0 (no artificial
      ceiling below the sigmoid's own asymptote -- "let psi be free")}

i.e. 6 fits, to see whether curvature-based score scaling and/or removing
the damping cap prevents the single-day xi spike. Everything else
(optimizer options: maxiter=300, polish=False; n_harmonics=5;
include_long=False; occurrence link; static gamma/zeta) is held IDENTICAL
to the production run_harvey_stage() call, so this isolates exactly the
two knobs asked about -- reuses run_rerun_10092026.py's own _evaluate()
for metric computation, so results are directly comparable to the cached
production result.json.

Writes to a SEPARATE artifact tree (does not touch or overwrite
artifacts/rerun_10092026/) :
    artifacts/stage4s_darwin_damping_scaling_diagnostic/<scaling>__damp<damp>/
        result.json, theta.npy, decoded_psi.json

Usage:
    python run_stage4s_darwin_damping_scaling_diagnostic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_rerun_10092026 as RR
from models.harvey_gas import HarveyZAGASModel
from distributions.gb2_log_link import GB2LogLink
from pi_dynamics.ar_logistic_custom_lags import ARLogisticCustomLagsPiDynamics
from data.station_loader import StationDataLoader

OUT = ROOT / "artifacts" / "stage4s_darwin_damping_scaling_diagnostic"

SCALINGS = ["unit", "diagonal_inverse_fisher", "inverse_fisher"]
DAMPING_OPTIONS = [("cap0999", 0.999), ("nocap1000", 1.0)]


def _pi_dyn():
    return ARLogisticCustomLagsPiDynamics(lags=[1, 2, 3], include_ar=True, link="logistic")


def _decode_psi(model: HarveyZAGASModel, theta: np.ndarray, damp: float) -> dict:
    names = model.parameter_names()
    raw = dict(zip(names, theta))
    out = {}
    for j in ("phi", "xi"):
        key = f"psi_gamma_{j}"
        if key in raw:
            out[key] = float(damp * expit(raw[key]))
            out[f"raw_{key}"] = float(raw[key])
    return out


def _extreme_state(model: HarveyZAGASModel, theta: np.ndarray, y: np.ndarray, empty: np.ndarray) -> dict:
    paths = model.filter(theta, y, X_long=empty, X_short=empty)
    phi, xi = paths["f_arr"][:, 0], paths["f_arr"][:, 1]
    i_phi, i_xi = int(np.argmax(np.abs(phi))), int(np.argmax(np.abs(xi)))
    dist = GB2LogLink()
    static = paths["static"]
    mean_at_xi_max = None
    try:
        mean_at_xi_max = float(dist.mean(phi=float(phi[i_xi]), xi=float(xi[i_xi]), **static))
    except Exception as exc:
        mean_at_xi_max = f"ERROR: {exc}"
    return {
        "phi_min": float(np.min(phi)), "phi_max": float(np.max(phi)),
        "xi_min": float(np.min(xi)), "xi_max": float(np.max(xi)),
        "argmax_abs_xi_t": i_xi, "phi_at_xi_max": float(phi[i_xi]),
        "gb2_mean_at_xi_max": mean_at_xi_max,
        "all_finite": bool(np.all(np.isfinite(phi)) and np.all(np.isfinite(xi))),
    }


def run_one(data, scaling: str, damp_label: str, damp: float) -> dict:
    y_tr, y_te = data["y_train"], data["y_test"]
    empty_tr = np.zeros((len(y_tr), 0))
    empty_te = np.zeros((len(y_te), 0))

    model = HarveyZAGASModel(
        distribution=GB2LogLink(), pi_dynamics=_pi_dyn(), seasonal="daily",
        long_names=[], short_names=[], harvey_target=["phi", "xi"],
        scaling=scaling, static_params=["gamma", "zeta"],
        n_harmonics=5, seasonal_damping_max=damp, include_long=False,
    )
    fr = model.fit(y=y_tr, X_long=empty_tr, X_short=empty_tr, verbose=False,
                   options={"maxiter": 300}, polish=False)

    res, ms, qr, theta = RR._evaluate(
        model, fr["theta"], data,
        {"X_long": empty_tr, "X_short": empty_tr},
        {"X_long_train": empty_tr, "X_long_test": empty_te,
         "X_short_train": empty_tr, "X_short_test": empty_te},
        {"X_long_train": empty_tr, "X_long_test": empty_te,
         "X_short_train": empty_tr, "X_short_test": empty_te}, fr,
    )
    res["scaling"] = scaling
    res["seasonal_damping_max"] = damp
    res.update(_decode_psi(model, fr["theta"], damp))
    res.update(_extreme_state(model, fr["theta"], y_tr, empty_tr))

    cdir = OUT / f"{scaling}__damp_{damp_label}"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "result.json").write_text(json.dumps(res, indent=2, default=float))
    np.save(cdir / "theta.npy", np.asarray(fr["theta"], dtype=float))
    return res


def main() -> None:
    print("Loading Darwin data...")
    data = StationDataLoader.from_registry(
        "DARWIN AIRPORT", precip_dir=RR.R.PRECIP_DIR, era5_dir=RR.R.ERA5_DIR,
        enso_path=RR.R.ENSO_PATH,
    ).load_all()

    rows = []
    for scaling in SCALINGS:
        for damp_label, damp in DAMPING_OPTIONS:
            print(f"\n=== scaling={scaling}  seasonal_damping_max={damp} ({damp_label}) ===")
            res = run_one(data, scaling, damp_label, damp)
            rows.append((scaling, damp_label, res))
            print(f"  grad_inf={res['grad_norm_inf_numeric']:.4g}  "
                  f"rmse_is={res['rmse_insample']:.4g}  crps_is={res['crps_insample']:.4g}  "
                  f"rmse_oos={res.get('rmse')}  crps_oos={res.get('crps_mean')}  "
                  f"psi_phi={res.get('psi_gamma_phi'):.4f}  psi_xi={res.get('psi_gamma_xi'):.4f}  "
                  f"xi_range=[{res['xi_min']:.2f}, {res['xi_max']:.2f}]  "
                  f"all_finite={res['all_finite']}")

    print("\n\n==================== SUMMARY ====================")
    hdr = f"{'scaling':24s} {'damp':10s} {'grad_inf':>12s} {'rmse_is':>14s} {'crps_is':>14s} {'rmse_oos':>10s} {'crps_oos':>10s} {'psi_phi':>8s} {'psi_xi':>8s} {'xi_max':>8s}"
    print(hdr)
    for scaling, damp_label, res in rows:
        def f(v, w):
            try:
                return f"{float(v):{w}.4g}"
            except (TypeError, ValueError):
                return f"{'nan':>{w}}"
        print(f"{scaling:24s} {damp_label:10s} {f(res['grad_norm_inf_numeric'],12)} "
              f"{f(res['rmse_insample'],14)} {f(res['crps_insample'],14)} "
              f"{f(res.get('rmse'),10)} {f(res.get('crps_mean'),10)} "
              f"{f(res.get('psi_gamma_phi'),8)} {f(res.get('psi_gamma_xi'),8)} "
              f"{f(res.get('xi_max'),8)}")

    print(f"\nWrote per-config result.json/theta.npy under {OUT}")


if __name__ == "__main__":
    main()
