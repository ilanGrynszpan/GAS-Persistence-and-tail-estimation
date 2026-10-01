"""
One-off patch: recompute collinearity_diagnostics() for every already-saved
info.json under artifacts/seasonality_gas_experiment/, using the FIXED
condition-number calculation (restricted to the GAS-lag coefficient block,
not the full parameter vector -- see run_seasonality_gas_analysis.py's
collinearity_diagnostics docstring, updated 2026-09-01). Does not touch
residuals.npy or multistep.csv (no re-simulation needed).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_seasonality_gas_analysis import (
    collinearity_diagnostics, OUT_DIR, ARTIFACTS_DIR,
    resolve_stage1_id, stage2_id, stage_subdir, LOCATIONS, BRANCHES, LAGSETS,
)
from data.station_loader import STATION_REGISTRY

for location in LOCATIONS:
    cfg = STATION_REGISTRY[location]
    run_id, short = cfg["run_id"], cfg.get("short", cfg["run_id"])
    run_dir = ARTIFACTS_DIR / run_id
    winners_path = run_dir / "stage_winners.json"
    if not winners_path.exists():
        continue
    winners = json.loads(winners_path.read_text())

    for branch in BRANCHES:
        for stage in ("stage1", "stage2"):
            for lagset in LAGSETS:
                info_path = OUT_DIR / short / f"{stage}_{branch}_{lagset}" / "info.json"
                if not info_path.exists():
                    continue
                model_id = resolve_stage1_id(winners, branch, lagset) if stage == "stage1" else stage2_id(branch, lagset)
                model_dir = run_dir / stage_subdir(stage, branch) / model_id
                coll = collinearity_diagnostics(model_dir, "phi")
                info = json.loads(info_path.read_text())
                info.update(coll)
                info_path.write_text(json.dumps(info, indent=2, default=float))
                print(f"patched {short}/{stage}_{branch}_{lagset}: cond={coll['cond_number']}")

print("Done.")
