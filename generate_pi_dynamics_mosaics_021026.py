"""
2026-10-02: occurrence-residual QQ and ACF mosaics for the six pi-dynamics
alternatives of the joint ZA-GAS experiment (reports/newpi) ->
reports/pi_dynamics_gas11_021026/.

WHAT CHANGES RELATIVE TO generate_pi_dynamics_gas11_mosaics.py
    * Alternatives: A, B, C, D, A7, C7 from run_pi_dynamics_mixed_fisher.py,
      i.e. every alternative estimated JOINTLY with the GAS(1,1) magnitude
      model; D is the joint phi-linked dynamics (pi_t = logistic(lambda0 +
      lambda1 phi_{t|t-1}) with phi from the same filter), A7/C7 are A/C with
      y-lags {1,2,3,364,365,366,367}. Unit score scaling.
    * Layout: two pages per diagnostic, each a separate image with
      rows = the six locations and columns = three alternatives
          page 1: A, B, C        page 2: D, A7, C7

EXECUTION FLOW
    for each (location, alternative):
        theta  <- artifacts/pi_dynamics_mixed_fisher/<short>/<alt>/unit/estimated_parameters.csv
        y_fit  <- first 8 calendar years of the training series (same split as the fit)
        pi_t   <- model._run_filter(theta, y_fit, return_paths=True)["pi"]   (Numba filter, no refit)
        r_t    <- randomised Bernoulli quantile residuals of 1{y_t>0} given pi_t
                  (diagnostics.occurrence_pit, seed 42 -- as in the earlier mosaics)
    page images via diagnostics.plots.stage_{qq,acf}_grid_mosaic, with the
    earlier generator's figure settings and label tidying reused unchanged.

Output (PNG + PDF):
    pi_dynamics_qq_mosaic_page1 / _page2
    pi_dynamics_acf400_mosaic_page1 / _page2
    mosaic_file_index.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_pi_dynamics_gas11_mosaics as M
import run_pi_dynamics_mixed_fisher as RP
from data.station_loader import STATION_REGISTRY
from diagnostics.multilocation_mosaics import LOCATIONS
from diagnostics.occurrence_pit import bernoulli_quantile_residuals
from diagnostics.plots import stage_qq_grid_mosaic, stage_acf_grid_mosaic

ART = ROOT / "artifacts" / "pi_dynamics_mixed_fisher"
OUT = ROOT / "reports" / "pi_dynamics_gas11_021026"
SCALING = "unit"
SEED = 42
PAGES = [["A", "B", "C"], ["D", "A7", "C7"]]
LABELS = {a: f"Alternative {a}" for a in ["A", "B", "C", "D", "A7", "C7"]}


def insample_qr(location: str, alt: str):
    short = STATION_REGISTRY[location]["short"]
    p = ART / short / alt / SCALING / "estimated_parameters.csv"
    if not p.exists():
        print(f"  [SKIP] {short}/{alt}: no {SCALING} fit")
        return None
    theta = pd.read_csv(p)["value"].to_numpy(float)
    y_fit, _y_val, _cut = RP.load_split(location)
    model = RP.build_model(alt, SCALING)
    paths = model._run_filter(theta, y_fit, return_paths=True)
    if not paths:
        print(f"  [SKIP] {short}/{alt}: filter failed")
        return None
    eff = int(paths["eff_start"])
    pi = np.asarray(paths["pi"], float)
    wet = (np.asarray(y_fit)[eff:eff + len(pi)] > 0).astype(float)
    return bernoulli_quantile_residuals(pi, wet, rng=np.random.default_rng(SEED))


DISPLAY = {"bh": "Belo Horizonte", "cruzeiro": "Cruzeiro do Sul", "darwin": "Darwin",
           "garanhuns": "Garanhuns", "manaus": "Manaus", "salvador": "Salvador"}
ALTS = ["A", "B", "C", "D", "A7", "C7"]


def summary_table() -> None:
    """
    Summary of alternatives (validation block): the earlier
    summary_of_alternatives table, redone for all six joint-fit alternatives.
    Models estimated on the first 8 calendar years of the training series,
    scored one step ahead with no re-fit on the final 2 years
    (run_pi_dynamics_mixed_fisher.py, unit scaling); values read from each
    fit's metadata.json. Writes summary_of_alternatives.{csv,tex,pdf}.
    """
    import json
    import subprocess
    rows = []
    for station in LOCATIONS:
        short = STATION_REGISTRY[station]["short"]
        for a in ALTS:
            m = json.loads((ART / short / a / SCALING / "metadata.json").read_text())
            rows.append({"location": DISPLAY[short], "alternative": a,
                         "rmse": m["val_rmse"], "crps": m["val_crps"],
                         "mean_pi_y0": m["val_pi_mean_dry"], "mean_pi_ygt0": m["val_pi_mean_wet"],
                         "n_dry": m["val_n_dry"], "n_wet": m["val_n_wet"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary_of_alternatives.csv", index=False)
    body = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
            r"\caption{Summary of alternatives A, B, C, D, A7 and C7 by location. "
            r"Lower RMSE and CRPS are better.}",
            r"\label{tab:summary_of_alternatives}",
            r"\begin{tabular}{llrrrrrr}", r"\hline",
            r"Location & Alternative & RMSE & CRPS & \shortstack{Mean $\pi$\\$\mid\, y=0$} & "
            r"\shortstack{Mean $\pi$\\$\mid\, y>0$} & $N$ dry & $N$ wet \\", r"\hline"]
    for i, (loc, sub) in enumerate(df.groupby("location", sort=False)):
        if i:
            body.append(r"\hline")
        for _, r in sub.iterrows():
            body.append(f"{loc} & {r.alternative} & {r.rmse:.3f} & {r.crps:.3f} & {r.mean_pi_y0:.3f} & "
                        f"{r.mean_pi_ygt0:.3f} & {int(r.n_dry)} & {int(r.n_wet)} " + r"\\")
    body += [r"\hline", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(body)
    (OUT / "summary_of_alternatives.tex").write_text(tex, encoding="utf-8")
    doc = "\n".join([r"\documentclass[10pt,a4paper]{article}", r"\usepackage[margin=2cm]{geometry}",
                     r"\usepackage{amsmath,float}", r"\begin{document}", tex, r"\end{document}", ""])
    (OUT / "summary_of_alternatives_doc.tex").write_text(doc, encoding="utf-8")
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "summary_of_alternatives_doc.tex"],
                       cwd=OUT, capture_output=True, text=True)
    pdf = OUT / "summary_of_alternatives_doc.pdf"
    if pdf.exists():
        pdf.replace(OUT / "summary_of_alternatives.pdf")
    for ext in ("aux", "log"):
        (OUT / f"summary_of_alternatives_doc.{ext}").unlink(missing_ok=True)
    print(df.round(3).to_string(index=False))


def main() -> None:
    M.MOSAICS_DIR = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    locs = M.LOCATION_DISPLAY_NAMES
    grid = {a: {} for page in PAGES for a in page}
    for station in LOCATIONS:
        disp = STATION_REGISTRY[station]["display"]
        for a in grid:
            grid[a][disp] = insample_qr(station, a)
    country_kw = M.LOCATION_COUNTRIES if M.SHOW_COUNTRY_SUBTITLE else None
    written = []
    for k, page in enumerate(PAGES, start=1):
        fig = stage_qq_grid_mosaic(grid, LABELS, page, locs, suptitle="",
                                   target_print_width_in=M.TARGET_PRINT_WIDTH_IN,
                                   base_print_pt=M.BASE_PRINT_PT,
                                   location_names=M.LOCATION_NAMES, location_countries=country_kw)
        M._tidy_left_labels_and_legend(fig, r"$r_t$", "upper left")
        if not M.SHOW_XLABEL:
            M._clear_xlabels(fig)
        written.append(M._save_mosaic(fig, f"pi_dynamics_qq_mosaic_page{k}"))
        fig = stage_acf_grid_mosaic(grid, LABELS, page, locs, suptitle="", n_lags=M.ACF_N_LAGS,
                                    target_print_width_in=M.TARGET_PRINT_WIDTH_IN,
                                    base_print_pt=M.ACF_BASE_PRINT_PT,
                                    location_names=M.LOCATION_NAMES, location_countries=country_kw)
        M._tidy_left_labels_and_legend(fig, ("ACF" if M.SHOW_ACF_YLABEL else ""), "lower left")
        if not M.SHOW_XLABEL:
            M._clear_xlabels(fig)
        written.append(M._save_mosaic(fig, f"pi_dynamics_acf400_mosaic_page{k}"))
    pd.DataFrame({"file": sorted(p.name for p in written),
                  }).to_csv(OUT / "mosaic_file_index.csv", index=False)
    for p in written:
        print("  wrote", p.name, "(+ .pdf)")


if __name__ == "__main__":
    if "--table" in sys.argv:
        OUT.mkdir(parents=True, exist_ok=True)
        summary_table()
    else:
        main()
        summary_table()
