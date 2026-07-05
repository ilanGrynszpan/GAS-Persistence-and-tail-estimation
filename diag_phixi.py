"""
Quick diagnostic: reproduce the phi+xi seasonal ZeroDivisionError.
Run after the main pipeline finishes to avoid interfering with it.
"""
import sys, traceback
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
from data.loader import BHDataLoader
from distributions.gb2_log_link import GB2LogLink
from pi_dynamics.factory import make_pi_dynamics
from models.za_gas_model import ZAGASModel
from constants import GAS_SEASONAL_LAGS

PRECIP_DIR  = Path(r"C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output")
ERA5_DIR    = ROOT / "data" / "input" / "ERA5"
NINO34_PATH = ROOT / "data" / "processed" / "pacific" / "NINO34_daily.csv"

loader = BHDataLoader(precip_dir=PRECIP_DIR, era5_dir=ERA5_DIR, nino34_path=NINO34_PATH)
data   = loader.load_all()
y      = data["y_train"]

dist   = GB2LogLink()
pi_dyn = make_pi_dynamics("ar_logistic", seasonal="daily")

for scaling in ["unit", "diagonal_inverse_fisher", "inverse_fisher"]:
    model = ZAGASModel(
        distribution=dist,
        pi_dynamics=pi_dyn,
        seasonal="daily",
        gas_lags=GAS_SEASONAL_LAGS["daily"],
        scaling=scaling,
    )
    print(f"\n{'='*60}")
    print(f"phi+xi  seasonal  {scaling}")
    print(f"{'='*60}")
    try:
        result = model.fit(y, verbose=False)
        print(f"  validity={result['validity']}  loglik={result['loglik']:.4f}")
    except Exception as exc:
        print(f"  EXCEPTION: {exc}")
        traceback.print_exc()
