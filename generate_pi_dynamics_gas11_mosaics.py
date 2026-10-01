"""
Cross-location x cross-variant diagnostic mosaics for the pi-dynamics
GAS(1,1) experiment (2026-09-09).

=============================================================================
OVERVIEW
=============================================================================

generate_pi_dynamics_gas11_report.py produces, per location, one PIT
histogram and one 400-lag ACF figure PER occurrence-dynamics variant, and
the per-location report groups them by location only. This script instead
builds mosaics that span every location AND every variant in a grid:

    rows    = LOCATIONS_PER_IMAGE locations (3, in the order of
              LOCATION_DISPLAY_NAMES -- 6 total, split across two images)
    columns = the four occurrence-dynamics alternatives (A ... D)

A single 6-row grid was hard to read (2026-09-12 feedback), so each
diagnostic is split into two images of 3 rows each rather than one image
of 6:

    pi_dynamics_gas11_qq_mosaic_1 / _2      normal QQ plot of the
                                            randomised occurrence quantile
                                            residuals (locations 1-3 / 4-6)
    pi_dynamics_gas11_acf400_mosaic_1 / _2  ACF of those residuals at 400
                                            lags, +/-2/sqrt(n) null band
                                            (locations 1-3 / 4-6)

Per docs/REPORTING.md Sec 2-3 this script performs NO optimisation and NO
model fitting: every quantile-residual array was already computed and saved
by run_pi_dynamics_gas11_validation.py (paths.npz["qr"] -- the IN-SAMPLE
residuals over the 8-year fit block); this only reads those artifacts and
lays them out. Point ARTIFACTS_DIR (just below the imports) at the earlier
whole-series experiment to reproduce the previous mosaics instead.

=============================================================================
CODE WALKTHROUGH
=============================================================================

    main()
      |
    for each variant_key in VARIANT_ORDER:
        for each of the six locations:
            generate_pi_dynamics_gas11_report.load_variant(short, variant_key)
                -> {..., "_qr": ndarray}  or  None (missing/failed)
        collect -> qr_grid[variant_key][location_display]
      |
    for each half in LOCATION_HALVES (3 locations each):
        diagnostics.plots.stage_qq_grid_mosaic( qr_grid, VARIANT_LABELS,
            VARIANT_ORDER, half, ... )     # rows=loc (this half), cols=variant
    for each half in LOCATION_HALVES:
        diagnostics.plots.stage_acf_grid_mosaic( same )
      |
    save each as PNG (150 dpi) + PDF to reports/mosaics_new_12092026/,
    write mosaic_file_index.csv

The two grid builders are the generic rows=location / columns=<keyed axis>
mosaics already used for the Stage 1-3 multi-location report (their
"stages" axis is just an ordered list of dict keys plus a label map --
here it carries the four variant keys instead of stage numbers).

=============================================================================
USAGE / REGENERATING THE MOSAICS
=============================================================================

    python generate_pi_dynamics_gas11_mosaics.py

Re-reads ARTIFACTS_DIR and overwrites the eight image files below. Safe to
run any time; it never fits a model. If the underlying fits change, re-run
run_pi_dynamics_gas11_validation.py FIRST, then this. The earlier single
6-row images under reports/pi_dynamics_gas11/mosaics/ are left as-is (not
regenerated here, so they stay available as a record of what was there
before the split) -- MOSAICS_DIR below is the new, separate output folder.

Output (PNG for quick viewing, PDF vector for LaTeX):
    reports/mosaics_new_12092026/pi_dynamics_gas11_qq_mosaic_1.png / .pdf
    reports/mosaics_new_12092026/pi_dynamics_gas11_qq_mosaic_2.png / .pdf
    reports/mosaics_new_12092026/pi_dynamics_gas11_acf400_mosaic_1.png / .pdf
    reports/mosaics_new_12092026/pi_dynamics_gas11_acf400_mosaic_2.png / .pdf
    reports/mosaics_new_12092026/mosaic_file_index.csv

-----------------------------------------------------------------------------
FIGURE DESIGN -- every knob is a module constant just below the imports,
edit there and re-run, no other file is involved:

    SHOW_SUPTITLE      True/False  -- the figure-level title above the grid
    SUPTITLE_QQ / _ACF str         -- that title's text (only used if shown)
    SHOW_LEGEND       True/False   -- the "45 degrees line" / "+/-2/sqrt(n)
                                      null band" legend box on the top-left
                                      panel
    SHOW_COUNTRY_SUBTITLE True/False -- the "(Brazil)" / "(Australia)" line
                                      under each row's location name
    SHOW_XLABEL       True/False   -- the shared bottom-row x-axis label
                                      ("Theoretical N(0,1) quantiles" / "Lag
                                      (days)")
    SHOW_ACF_YLABEL   True/False   -- the "ACF" y-axis label (QQ's "$r_t$"
                                      y-axis label is untouched by this flag)
    VARIANT_LABELS    dict         -- the four column headers
    LOCATION_DISPLAY_NAMES / _NAMES / _COUNTRIES -- row order and row labels
    LOCATIONS_PER_IMAGE int        -- rows per split image (default 3, so 6
                                      locations -> 2 images; set to 6 for a
                                      single image, 1 for one-per-location)
    MOSAICS_DIR       Path         -- output folder
    ACF_N_LAGS        int          -- lags on the ACF x-axis
    DPI              int           -- PNG resolution
    TARGET_PRINT_WIDTH_IN, BASE_PRINT_PT, ACF_BASE_PRINT_PT -- font-size
                                      calibration (see diagnostics/plots.py's
                                      "Print-size calibration" note); larger
                                      TARGET_PRINT_WIDTH_IN => smaller text.

The panel contents themselves (what a QQ / ACF panel plots, reference
lines, the shaded null band) live in diagnostics/plots.py's _draw_qq_panel
/ _draw_acf_panel and are shared with the Stage 1-3 report -- not changed
here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.station_loader import STATION_REGISTRY, mosaic_name
from diagnostics.multilocation_mosaics import LOCATIONS
from diagnostics.plots import stage_qq_grid_mosaic, stage_acf_grid_mosaic

VARIANT_ORDER = ["existing_ar_seasonal", "short_lags_ar", "short_lags_noar", "phi_linked"]

# The mosaics use the IN-SAMPLE quantile residuals over the 8-year fit block
# from run_pi_dynamics_gas11_validation.py (paths.npz["qr"]). To point them
# back at the earlier whole-series in-sample experiment instead, set this to
# ROOT / "artifacts" / "pi_dynamics_gas11_experiment".
ARTIFACTS_DIR = ROOT / "artifacts" / "pi_dynamics_gas11_validation_experiment"

# --- Figure design knobs (see the "FIGURE DESIGN" note in the docstring) ---
SHOW_SUPTITLE = False   # figure-level title above the grid
SHOW_LEGEND   = False   # "45 degrees line" / "null band" legend on panel (0,0)
SHOW_COUNTRY_SUBTITLE = False   # "(Brazil)" / "(Australia)" under each row label
SHOW_XLABEL   = False   # shared bottom-row x-axis label (both mosaics)
SHOW_ACF_YLABEL = False  # the ACF mosaic's "ACF" y-axis label
SUPTITLE_QQ = (r"Occurrence quantile residuals $r_t=\Phi^{-1}(U_t)$: normal QQ plot, "
               r"all locations $\times$ all $\pi_t$ alternatives (GAS(1,1) magnitude)")
SUPTITLE_ACF = (r"Occurrence quantile-residual autocorrelation (400 lags, "
                r"$\pm 2/\sqrt{n}$ null band), all locations $\times$ all $\pi_t$ "
                r"alternatives (GAS(1,1) magnitude)")

# Column headers for the four occurrence-dynamics alternatives -- bare
# "Alternative X" labels only, no formula/lag-set description (see the
# report's Model Specification section for what each one is).
VARIANT_LABELS: Dict[str, str] = {
    "existing_ar_seasonal": "Alternative A",
    "short_lags_ar":        "Alternative B",
    "short_lags_noar":      "Alternative C",
    "phi_linked":           "Alternative D",
}

LOCATION_DISPLAY_NAMES = [STATION_REGISTRY[s]["display"] for s in LOCATIONS]
LOCATION_NAMES     = {STATION_REGISTRY[s]["display"]: mosaic_name(s) for s in LOCATIONS}
LOCATION_COUNTRIES = {STATION_REGISTRY[s]["display"]: f"({STATION_REGISTRY[s]['country']})" for s in LOCATIONS}

# A 6-row grid at 4 columns is too tall/dense to read comfortably (2026-09-12
# feedback), so each mosaic is split into two images of LOCATIONS_PER_IMAGE
# rows each, same column layout, filenames suffixed "_1"/"_2" in row order
# -- e.g. pi_dynamics_gas11_qq_mosaic_1.png (first 3 locations), _2.png (the
# last 3). Edit LOCATIONS_PER_IMAGE / LOCATION_DISPLAY_NAMES to change the
# split.
LOCATIONS_PER_IMAGE = 3
LOCATION_HALVES = [
    LOCATION_DISPLAY_NAMES[i:i + LOCATIONS_PER_IMAGE]
    for i in range(0, len(LOCATION_DISPLAY_NAMES), LOCATIONS_PER_IMAGE)
]

REPORT_DIR  = ROOT / "reports" / "pi_dynamics_gas11"
# Split mosaics land here (2026-09-12), not in REPORT_DIR/mosaics -- kept
# separate so the earlier single 6-row images stay put as a record of what
# was there before, instead of being silently overwritten.
MOSAICS_DIR = ROOT / "reports" / "mosaics_new_12092026"

ACF_N_LAGS = 400   # docs/REPORTING.md Sec 13
DPI        = 150   # matches the per-model figure resolution used elsewhere

# Landscape full-page target (these grids are 4 columns wide); the same
# print-size-calibration knobs the Stage 1-3 grids use.
TARGET_PRINT_WIDTH_IN = 9.5
BASE_PRINT_PT = {
    "suptitle":   18.0,
    "header":     13.0,   # per-column variant labels (two lines each)
    "row_label":  13.0,   # rotated location names on the left
    "axis_label": 12.0,
    "tick":       10.0,
    "legend":     11.0,
    "annotation":  9.0,
}

# stage_acf_grid_mosaic packs 4 columns into a very wide canvas (7.2in per
# column), so its print-scale multiplier is large; the ACF rows are only
# 4.4in tall, so the default row-label / column-header point sizes render
# tall enough to bleed into the neighbouring row. Shrink just those two for
# the ACF mosaic (the panels themselves and their tick text are unchanged).
ACF_BASE_PRINT_PT = {**BASE_PRINT_PT, "row_label": 8.0, "header": 9.0}


def _load_qr(short: str, variant_key: str) -> Optional[np.ndarray]:
    """In-sample (8-year fit-block) quantile residuals for one
    (location, variant), or None if the artifact is missing/failed."""
    d = ARTIFACTS_DIR / short / variant_key
    meta_path, npz_path = d / "metadata.json", d / "paths.npz"
    if not (meta_path.exists() and npz_path.exists()):
        return None
    if not json.loads(meta_path.read_text()).get("success"):
        return None
    return np.load(npz_path)["qr"]


def _load_qr_grid() -> Dict[str, Dict[str, Optional[np.ndarray]]]:
    """{variant_key: {location display name: in-sample fit-block quantile-
    residual array, or None where that (location, variant) artifact is
    missing/failed}}."""
    grid: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    for variant_key in VARIANT_ORDER:
        per_loc: Dict[str, Optional[np.ndarray]] = {}
        for station in LOCATIONS:
            cfg = STATION_REGISTRY[station]
            qr = _load_qr(cfg["short"], variant_key)
            per_loc[cfg["display"]] = qr
            if qr is None:
                print(f"  [SKIP] {cfg['display']} / {variant_key}: no successful artifact")
        grid[variant_key] = per_loc
    return grid


def _tidy_left_labels_and_legend(fig: plt.Figure, short_ylabel: str, legend_loc: str) -> None:
    """
    Two cosmetic fixes applied to the Figure returned by
    stage_{qq,acf}_grid_mosaic, so diagnostics/plots.py -- shared with the
    Stage 1-3 multi-location report -- is left untouched:

    1. Those builders hardcode a long y-axis label on the leftmost column
       ("Sample quantile residuals $r_t$" / "Autocorrelation"); at this
       figure's proportions that vertical text overlaps the rotated bold
       location name _row_label draws just outside the same edge. Replace
       it with `short_ylabel`.
    2. They also hardcode a legend on panel (0,0). If SHOW_LEGEND is False
       (default) it is removed entirely; otherwise it is moved to
       `legend_loc` so it does not sit on the data.
    """
    for ax in fig.axes:
        if ax.get_ylabel():                       # grid sets it on column 0 only
            ax.set_ylabel(short_ylabel, fontsize=BASE_PRINT_PT["axis_label"] *
                          (fig.get_size_inches()[0] / TARGET_PRINT_WIDTH_IN))
    leg = fig.axes[0].get_legend()
    if leg is None:
        return
    if not SHOW_LEGEND:
        leg.remove()
        return
    try:
        leg.set_loc(legend_loc)
    except AttributeError:                        # older matplotlib
        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.axes[0].legend(handles, labels, loc=legend_loc,
                           fontsize=leg.get_texts()[0].get_fontsize())


def _clear_xlabels(fig: plt.Figure) -> None:
    """Blank the shared bottom-row x-axis label ("Theoretical N(0,1)
    quantiles" / "Lag (days)"), set by stage_{qq,acf}_grid_mosaic on the
    last row only -- done here (not in diagnostics/plots.py) for the same
    reason as _tidy_left_labels_and_legend."""
    for ax in fig.axes:
        ax.set_xlabel("")


def _save_mosaic(fig: plt.Figure, stem: str) -> Path:
    """PNG (quick view) + PDF (vector, for LaTeX per REPORTING.md Sec 26.12)."""
    MOSAICS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = MOSAICS_DIR / f"{stem}.png"
    pdf_path = MOSAICS_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    MOSAICS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Mosaics will be written to: {MOSAICS_DIR}")

    qr_grid = _load_qr_grid()
    n_ok = sum(1 for v in VARIANT_ORDER for loc in LOCATION_DISPLAY_NAMES
               if qr_grid[v][loc] is not None)
    print(f"Loaded {n_ok}/{len(VARIANT_ORDER) * len(LOCATION_DISPLAY_NAMES)} "
          f"(location, variant) quantile-residual arrays.")

    generated = []
    country_kw = LOCATION_COUNTRIES if SHOW_COUNTRY_SUBTITLE else None

    for half_idx, locs in enumerate(LOCATION_HALVES, start=1):
        fig_qq = stage_qq_grid_mosaic(
            qr_grid, VARIANT_LABELS, VARIANT_ORDER, locs,
            suptitle=(SUPTITLE_QQ if SHOW_SUPTITLE else ""),
            target_print_width_in=TARGET_PRINT_WIDTH_IN, base_print_pt=BASE_PRINT_PT,
            location_names=LOCATION_NAMES, location_countries=country_kw,
        )
        _tidy_left_labels_and_legend(fig_qq, r"$r_t$", "upper left")
        if not SHOW_XLABEL:
            _clear_xlabels(fig_qq)
        generated.append(_save_mosaic(fig_qq, f"pi_dynamics_gas11_qq_mosaic_{half_idx}"))
        print(f"  wrote {generated[-1].name} (+ .pdf)  [{', '.join(locs)}]")

    for half_idx, locs in enumerate(LOCATION_HALVES, start=1):
        fig_acf = stage_acf_grid_mosaic(
            qr_grid, VARIANT_LABELS, VARIANT_ORDER, locs,
            suptitle=(SUPTITLE_ACF if SHOW_SUPTITLE else ""),
            n_lags=ACF_N_LAGS,
            target_print_width_in=TARGET_PRINT_WIDTH_IN, base_print_pt=ACF_BASE_PRINT_PT,
            location_names=LOCATION_NAMES, location_countries=country_kw,
        )
        _tidy_left_labels_and_legend(fig_acf, ("ACF" if SHOW_ACF_YLABEL else ""), "lower left")
        if not SHOW_XLABEL:
            _clear_xlabels(fig_acf)
        generated.append(_save_mosaic(fig_acf, f"pi_dynamics_gas11_acf400_mosaic_{half_idx}"))
        print(f"  wrote {generated[-1].name} (+ .pdf)  [{', '.join(locs)}]")

    index_df = pd.DataFrame({"file": sorted(p.name for p in generated)})
    index_df.to_csv(MOSAICS_DIR / "mosaic_file_index.csv", index=False)
    print(f"Done. {len(generated)} mosaics + mosaic_file_index.csv in {MOSAICS_DIR}")


if __name__ == "__main__":
    main()
