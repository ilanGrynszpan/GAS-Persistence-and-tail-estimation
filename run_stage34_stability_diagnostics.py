"""
Stage 3/4 (Harvey long-short) stability diagnostics (2026-09-11).

WHY THIS EXISTS
================================================================================
Stage 3 (long+short, no seasonal) and Stage 4 (long+short + damped
trigonometric seasonal) came out of the 2026-09-10 rerun with checked
gradient norms of order 1e4-1e6 at the reported optimum, versus ~1-100 for
Stage 1/2 and ~5 for Stage 4S (the structural model with NO long component:
HarveyZAGASModel(include_long=False)). This script tests four candidate
explanations, all evaluated from the ALREADY-FITTED theta.npy cached under
artifacts/rerun_10092026/ -- no refitting, per docs/REPORTING.md.

  H1  Boundary: one or more of {phi_L, phi_S, kappa_L, kappa_S, |Gamma_L|,
      |Gamma_S|} sits at/past its soft-penalty threshold (MODELS.md sec 20 /
      harvey_gas.py's LAM_LONG/LAM_SHORT/LAM_SCORE/LAM_COV terms:
      phi_L<0.995, phi_S<phi_L, |kappa|<2, |Gamma|<20) at the optimum, so the
      large reported gradient is dominated by the penalty's own (very
      steep, LAM=1e5/1e3/1e2) slope rather than a genuine likelihood
      optimization failure.
  H2  Long/short non-identification: the PURE log-likelihood surface (no
      penalty) is nearly flat along a (phi_L, phi_S) trade-off direction --
      tested via the local 2x2 finite-difference Hessian of loglik in
      (phi_L, phi_S); a large eigenvalue-ratio (condition number) with the
      near-zero eigenvector close to (1,1)/sqrt(2) means the optimizer can
      raise phi_L and phi_S together almost for free but cannot tell how
      much belongs to which component.
  H3  Covariate/kappa collinearity: the covariate series correlates with
      the realised GAS score, so kappa (score loading) and Gamma (covariate
      loading) can trade off against each other. Tested both by raw
      correlation(covariate, score) and by the same curvature-ridge test as
      H2 applied to (kappa_S, Gamma_S[largest]).
  H4  Seasonality/long confound (Stage 4 only, since Stage 3 has
      n_harmonics=0): correlation between the damped seasonal contribution
      Gamma_{j,t} = sum_h seas_h and the long-run level L_{j,t}.

Natural-experiment control: Stage 4S removes the long component entirely
(no phi_L/kappa_L/Gamma_L in theta at all), so H1/H2 involving the long
block cannot apply there by construction, while H3 (covariates) and H4-like
seasonal collinearity with the remaining single component still could. Its
much smaller gradient norms are the single strongest piece of evidence in
this analysis; see generate_stage34_stability_report.py.

Output: artifacts/stage34_stability_diagnostics/<loc>/<stage>/<cfg>[__xi].json
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_rerun_10092026 as D  # reuse data loading + cached-model reconstruction
from data.station_loader import STATION_REGISTRY
from diagnostics.numerical_diff import numerical_gradient_checked

OUT = ROOT / "artifacts" / "stage34_stability_diagnostics"
# stage4S (include_long=False) is diagnosed too, purely as the natural-
# experiment CONTROL: it has no phi_L/kappa_L/Gamma_L at all, so H1/H2
# involving the long block cannot apply there by construction, while H3
# (covariate/score correlation on the surviving short-only block) still
# can -- comparing stage3/4 against it isolates what the long component
# specifically contributes to the instability.
STAGES = ("stage3", "stage4", "stage4S")
TOL_PHI = 0.02      # "at/near boundary" tolerance for phi order/upper-bound (natural scale)
TOL_KAPPA = 0.05    # for |kappa| - 2.0
TOL_GAMMA = 0.5     # for |Gamma| - 20.0
H_PHI = 1e-3        # finite-difference step for phi_L/phi_S curvature
H_KAPPA = 1e-3
H_GAMMA = 1e-3


def _penalty_structural(model, hvd, pip, sp):
    """Closed-form re-derivation of harvey_gas.py's persistence / score-
    amplitude / covariate-norm / static-validity penalty terms (the ones
    this study's H1 concerns) directly from the DECODED parameters -- i.e.
    excluding the path-dependent LAM_STATE/LAM_ETA explosion guards, which
    need the realised f_arr/eta_arr and are a separate (numerical-safety,
    not identification) concern. Mirrors harvey_gas.py's _run_filter
    exactly for the terms it covers."""
    LAM_LONG, LAM_SHORT, LAM_COV, LAM_STATIC, LAM_SCORE = 1e5, 1e5, 1e2, 1e5, 1e3
    pen = 0.0
    gamma_v, zeta_v = sp.get("gamma", 1.0), sp.get("zeta", 3.0)
    pen += LAM_STATIC * max(0.0, gamma_v - zeta_v + 1e-6) ** 2
    for s in sp:
        pen += LAM_STATIC * max(0.0, abs(sp[s]) - 20.0) ** 2
    rho = pip.get("rho", 0.0)
    pen += LAM_LONG * max(0.0, abs(rho) - 0.98) ** 2
    free_phi_L = model.include_long and model.fixed_phi_L is None and model.phi_L_max is None
    free_phi_S = model.fixed_phi_S is None and model.phi_S_max is None
    for j, hj in hvd.items():
        if free_phi_L:
            pen += LAM_LONG * max(0.0, hj["phi_L"] - 0.995) ** 2
            pen += LAM_LONG * max(0.0, -hj["phi_L"]) ** 2
        pen += LAM_SHORT * max(0.0, hj["phi_S"] - hj["phi_L"] + 1e-4) ** 2
        if free_phi_S:
            pen += LAM_SHORT * max(0.0, -hj["phi_S"]) ** 2
        pen += LAM_SCORE * max(0.0, abs(hj["kappa_L"]) - 2.0) ** 2
        pen += LAM_SCORE * max(0.0, abs(hj["kappa_S"]) - 2.0) ** 2
        pen += LAM_COV * max(0.0, np.linalg.norm(hj["Gamma_L"]) - 20.0) ** 2
        pen += LAM_COV * max(0.0, np.linalg.norm(hj["Gamma_S"]) - 20.0) ** 2
    return pen


def _boundary_distances(model, hj):
    d = {}
    if model.include_long:
        d["phi_L_minus_0995"] = hj["phi_L"] - 0.995
        d["phi_L_neg"] = -hj["phi_L"]
        d["kappa_L_minus_2"] = abs(hj["kappa_L"]) - 2.0
        d["GammaL_norm_minus_20"] = (np.linalg.norm(hj["Gamma_L"]) if len(hj["Gamma_L"]) else 0.0) - 20.0
        d["order_violation"] = hj["phi_S"] - hj["phi_L"] + 1e-4
    d["phi_S_neg"] = -hj["phi_S"]
    d["kappa_S_minus_2"] = abs(hj["kappa_S"]) - 2.0
    d["GammaS_norm_minus_20"] = (np.linalg.norm(hj["Gamma_S"]) if len(hj["Gamma_S"]) else 0.0) - 20.0
    return d


def _any_boundary_active(d):
    for k, v in d.items():
        if k.endswith("_neg"):
            tol = 0.005              # positivity: only flag a real sign violation
        elif "kappa" in k:
            tol = TOL_KAPPA
        elif "Gamma" in k:
            tol = TOL_GAMMA
        else:
            tol = TOL_PHI            # upper bound (0.995) / order violation
        if v > -tol:
            return True
    return False


def _curvature_2d(f, theta, i, j, hi, hj_):
    th = theta.copy()

    def ev(di, dj):
        t = th.copy(); t[i] += di; t[j] += dj
        return f(t)

    f00 = ev(0, 0)
    Hii = (ev(hi, 0) - 2 * f00 + ev(-hi, 0)) / hi ** 2
    Hjj = (ev(0, hj_) - 2 * f00 + ev(0, -hj_)) / hj_ ** 2
    Hij = (ev(hi, hj_) - ev(hi, -hj_) - ev(-hi, hj_) + ev(-hi, -hj_)) / (4 * hi * hj_)
    return np.array([[Hii, Hij], [Hij, Hjj]])


def _ridge_summary(H):
    if not np.all(np.isfinite(H)):
        return {"eig_lo": None, "eig_hi": None, "ratio": None, "ridge_eigvec": None}
    eigval, eigvec = np.linalg.eigh(H)
    order = np.argsort(np.abs(eigval))
    lo, hi = float(np.abs(eigval[order[0]])), float(np.abs(eigval[order[-1]]))
    ratio = float(hi / lo) if lo > 1e-8 else float("inf")
    vec = eigvec[:, order[0]].tolist()
    return {"eig_lo": lo, "eig_hi": hi, "ratio": ratio, "ridge_eigvec": vec}


def diagnose_one(loc, stage, cfg, branch):
    short = STATION_REGISTRY[loc].get("short", loc)
    sfx = "" if branch == "phi_only" else "__xi"
    cdir = D.OUT / short / stage / (cfg + sfx)
    rp, tp = cdir / "result.json", cdir / "theta.npy"
    if not (rp.exists() and tp.exists()):
        return None
    res = json.loads(rp.read_text())
    theta = np.load(tp)
    data = D._load_data(loc)
    y_tr = data["y_train"]

    model, fk, ok, mk = D._build_for_augment(res, data, cdir)
    Xl_tr, Xs_tr = fk["X_long"], fk["X_short"]
    names = model.parameter_names()

    paths = model.filter(theta, y_tr, Xl_tr, Xs_tr)
    if not paths:
        return {"loc": short, "stage": stage, "cfg": cfg, "branch": branch, "invalid": True}
    hvd = paths["harvey"]
    sp = paths["static"]
    _, pi_theta, _ = model._split_theta(theta)
    pip = model._decode_pi(pi_theta)

    pen_structural = _penalty_structural(model, hvd, pip, sp)

    # numerical gradient of the PENALIZED objective (same objective used at
    # fit time) -- recovers the full gradient VECTOR (result.json only kept
    # its inf-norm), so we can see WHICH parameter carries the large value.
    obj = lambda th: model._run_filter(th, y_tr, X_long=Xl_tr, X_short=Xs_tr, return_paths=False)
    g = numerical_gradient_checked(obj, theta)
    grad = np.asarray(g["grad"])
    argmax_i = int(np.argmax(np.abs(grad)))

    def loglik_of(th):
        p = model.filter(th, y_tr, Xl_tr, Xs_tr)
        return p["loglik"] if p else -1e12

    eff = paths["eff_start"]
    out_per_j = {}
    for j in model.harvey_target:
        hj = hvd[j]
        bd = _boundary_distances(model, hj)
        boundary_active = _any_boundary_active(bd)
        argmax_name = names[argmax_i]
        is_boundary_kind = re.match(
            rf"^(phi_L|phi_S|kappa_L|kappa_S|Gamma_L|Gamma_S)_{re.escape(j)}(_|$)", argmax_name
        ) is not None
        argmax_is_this_j_boundary_param = is_boundary_kind and boundary_active

        entry = {
            "phi_L": hj["phi_L"] if model.include_long else None,
            "phi_S": hj["phi_S"],
            "kappa_L": hj["kappa_L"] if model.include_long else None,
            "kappa_S": hj["kappa_S"],
            "GammaL_norm": float(np.linalg.norm(hj["Gamma_L"])) if model.include_long and len(hj["Gamma_L"]) else None,
            "GammaS_norm": float(np.linalg.norm(hj["Gamma_S"])) if len(hj["Gamma_S"]) else None,
            "boundary_distances": bd,
            "boundary_active": boundary_active,
            "argmax_grad_is_this_j_boundary_param": argmax_is_this_j_boundary_param,
        }

        # H2: (phi_L, phi_S) curvature ridge test on the PURE log-likelihood
        if model.include_long and (f"phi_L_{j}" in names) and (f"phi_S_{j}" in names):
            iL, iS = names.index(f"phi_L_{j}"), names.index(f"phi_S_{j}")
            H = _curvature_2d(loglik_of, theta, iL, iS, H_PHI, H_PHI)
            entry["h2_phiL_phiS_curvature"] = _ridge_summary(H)

        # H3: covariate <-> score correlation, + (kappa_S, largest Gamma_S) ridge test
        s_j = paths["s_arr"][:, model.tv_names.index(j)]
        corr = {}
        if model.n_short > 0:
            Xs_eff = Xs_tr[eff:eff + len(s_j)]
            for c, nm in enumerate(model.short_names):
                col = Xs_eff[:, c]
                if np.std(col) > 1e-10 and np.std(s_j) > 1e-10:
                    corr[nm] = float(np.corrcoef(col, s_j)[0, 1])
            if len(model.short_names) and f"kappa_S_{j}" in names:
                c_star = int(np.argmax(np.abs(hj["Gamma_S"]))) if len(hj["Gamma_S"]) else None
                gname = f"Gamma_S_{j}_{model.short_names[c_star]}" if c_star is not None else None
                if gname in names:
                    iK, iG = names.index(f"kappa_S_{j}"), names.index(gname)
                    H3 = _curvature_2d(loglik_of, theta, iK, iG, H_KAPPA, H_GAMMA)
                    entry["h3_kappaS_GammaS_curvature"] = _ridge_summary(H3)
                    entry["h3_kappaS_GammaS_which_cov"] = model.short_names[c_star]
        if model.include_long and model.n_long > 0:
            Xl_eff = Xl_tr[eff:eff + len(s_j)]
            for c, nm in enumerate(model.long_names):
                col = Xl_eff[:, c]
                if np.std(col) > 1e-10 and np.std(s_j) > 1e-10:
                    corr[nm] = float(np.corrcoef(col, s_j)[0, 1])
        entry["h3_corr_covariate_score"] = corr

        # H4: seasonal contribution vs long-run level (Stage 4 only)
        if model.n_harmonics > 0 and model.include_long and "Gamma_arr" in paths:
            seas = paths["Gamma_arr"][:, model.tv_names.index(j)]
            level = paths["L_arr"][:, model.tv_names.index(j)]
            if np.std(seas) > 1e-10 and np.std(level) > 1e-10:
                entry["h4_corr_seasonal_long"] = float(np.corrcoef(seas, level)[0, 1])

        out_per_j[j] = entry

    result = {
        "loc": short, "stage": stage, "cfg": cfg, "branch": branch,
        "loglik": paths["loglik"], "penalty_total": paths["penalty"],
        "penalty_structural": pen_structural,
        "grad_inf_norm": float(np.max(np.abs(grad))),
        "grad_argmax_name": names[argmax_i], "grad_argmax_value": float(grad[argmax_i]),
        "grad_converged": g["converged"],
        "stored_grad_inf_norm": res.get("grad_norm_inf_numeric"),
        "n_long": model.n_long, "n_short": model.n_short, "include_long": model.include_long,
        "n_harmonics": model.n_harmonics,
        "per_j": out_per_j,
    }
    return result


def _worker(arg):
    loc, stage, cfg, branch, force = arg
    short = STATION_REGISTRY[loc].get("short", loc)
    sfx = "" if branch == "phi_only" else "__xi"
    outdir = OUT / short / stage
    outdir.mkdir(parents=True, exist_ok=True)
    outp = outdir / f"{cfg}{sfx}.json"
    if outp.exists() and not force:
        return "skip", str(outp)
    try:
        r = diagnose_one(loc, stage, cfg, branch)
        if r is None:
            return "missing", str(outp)
        outp.write_text(json.dumps(r, indent=2, default=float))
        return "done", str(outp)
    except Exception:
        outp.with_suffix(".error.txt").write_text(traceback.format_exc())
        return "err", str(outp)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    tasks = [(loc, stage, cfg, br, args.force)
             for loc in D.LOCATIONS for stage in STAGES
             for cfg in D.HARVEY_COV for br in ("phi_only", "phi_xi")]
    print(f"{len(tasks)} configs to diagnose")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    tally = {"done": 0, "skip": 0, "err": 0, "missing": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            st, p = fut.result()
            tally[st] += 1
            if st == "err" or i % 20 == 0:
                print(f"  [{i}/{len(tasks)}] {st} {p}")
    print("diagnostics pass complete", tally)


if __name__ == "__main__":
    main()
