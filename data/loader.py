"""
Data loader for the Belo Horizonte precipitation modelling pipeline.

=============================================================================
DATA SOURCES
=============================================================================

Precipitation (y_t):
    C:/.../data/output/BELO HORIZONTE_train.csv
    C:/.../data/output/BELO HORIZONTE_test.csv

    Columns used:
        "Data Medicao"                      — daily date (YYYY-MM-DD)
        "PRECIPITACAO TOTAL, DIARIO(mm)"    — daily rainfall in mm (y_t)

    Train period: 2014-01-01 to 2023-12-31  (3 652 days)
    Test  period: 2024-01-01 to 2025-12-31  (  731 days)
    The same train/test date partition is applied to all covariates.

Dew point (X_short component, weather covariates):
    data/input/ERA5/humidity/BELO_HORIZONTE_ERA5_dewpoint_2m_daily_2006_2025.csv
    Columns: "date", "dewpoint_2m_c"

Temperature (X_short component, weather covariates):
    data/input/ERA5/temperature/BELO_HORIZONTE_ERA5_temperature_2m_daily_2006_2025.csv
    Columns: "date", "temperature_2m_c"

ENSO — Niño 3.4 SST index (X_long component, climate covariates):
    data/processed/pacific/NINO34_daily.csv
    Columns: "date", "nino34_sst"

=============================================================================
COVARIATE BLOCKS (per MODELS.md §15–18)
=============================================================================

Standard GAS covariates (X_t):
    "dewpoint_short"       — D_{t-1}, D_{t-2}, D_{t-3}
    "dewpoint_seasonal"    — D_{t-1}…D_{t-3}, D_{t-364}…D_{t-367}
    "dewtemp_short"        — above + T_{t-1}, T_{t-2}, T_{t-3}
    "dewtemp_seasonal"     — above + T_{t-364}…T_{t-367}

Harvey long component (X_long):
    "enso_90d"             — E^{90}_{t-1}  (90-day trailing mean, lag 1)
    "enso_90d_30d"         — E^{90}_{t-1}, E^{30}_{t-1}
    "enso_90d_30d_daily"   — E^{90}_{t-1}, E^{30}_{t-1}, E_{t-1}

Harvey short component (X_short):
    Set to the selected best weather block from Stage 2.
    Same construction as the standard GAS covariate block.

=============================================================================
STANDARDISATION
=============================================================================

All covariates are z-scored using TRAINING statistics only.
Test-period covariates are standardised with the same training mean/std.
This prevents any information leakage from the test period.

=============================================================================
USAGE
=============================================================================

    loader = BHDataLoader(
        precip_dir  = Path("C:/.../data/output"),
        era5_dir    = Path("data/input/ERA5"),
        nino34_path = Path("data/processed/pacific/NINO34_daily.csv"),
    )
    data = loader.load_all()

    y_train    = data["y_train"]          # (T_train,)
    y_test     = data["y_test"]           # (T_test,)
    dates_train = data["dates_train"]
    dates_test  = data["dates_test"]
    blocks_train = data["covariate_blocks_train"]   # {block_name: ndarray}
    blocks_test  = data["covariate_blocks_test"]    # {block_name: ndarray}

=============================================================================
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class BHDataLoader:
    """
    Data loader for the Belo Horizonte daily precipitation experiment.

    All paths are resolved relative to the given root directories.
    The loader enforces a strict train/test split based on the precipitation
    file partition; no date leakage between train and test statistics.
    """

    PRECIP_DATE_COL  = "Data Medicao"
    PRECIP_VALUE_COL = "PRECIPITACAO TOTAL, DIARIO(mm)"
    NINO34_DATE_COL  = "date"
    NINO34_VALUE_COL = "nino34_sst"
    ERA5_DATE_COL    = "date"

    def __init__(
        self,
        precip_dir:  Path | str,
        era5_dir:    Path | str,
        nino34_path: Path | str,
        station_name: str = "BELO_HORIZONTE",
    ):
        self.precip_dir   = Path(precip_dir)
        self.era5_dir     = Path(era5_dir)
        self.nino34_path  = Path(nino34_path)
        self.station_name = station_name

    # ─────────────────────────────────────────────────────────────────────────
    # Raw loaders
    # ─────────────────────────────────────────────────────────────────────────

    def _load_precipitation(self) -> Tuple[pd.Series, pd.Series]:
        """Load train and test precipitation, indexed by date."""
        train_path = self.precip_dir / "BELO HORIZONTE_train.csv"
        test_path  = self.precip_dir / "BELO HORIZONTE_test.csv"

        def _read(path: Path) -> pd.Series:
            df = pd.read_csv(path, parse_dates=[self.PRECIP_DATE_COL])
            df = df.set_index(self.PRECIP_DATE_COL).sort_index()
            series = df[self.PRECIP_VALUE_COL].rename("precip")
            # Negative values (sensor errors) set to 0
            series = series.clip(lower=0.0)
            return series

        return _read(train_path), _read(test_path)

    def _load_dewpoint(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Load ERA5 dew point, aligned to the given date index."""
        dp_dir  = self.era5_dir / "humidity"
        matches = sorted(dp_dir.glob(f"*{self.station_name}*dewpoint*"))
        if not matches:
            matches = sorted(dp_dir.glob("*BELO*dewpoint*"))
        if not matches:
            raise FileNotFoundError(f"No dew-point file for {self.station_name} in {dp_dir}")
        df = pd.read_csv(matches[0], parse_dates=[self.ERA5_DATE_COL])
        df = df.set_index(self.ERA5_DATE_COL).sort_index()
        # Select the dew-point column (skip lat/lon/station)
        numeric = [c for c in df.columns
                   if df[c].dtype != object
                   and not any(s in c.lower() for s in ("lat", "lon", "station"))]
        col = numeric[0] if numeric else df.columns[0]
        return df[col].rename("dewpoint").reindex(dates)

    def _load_temperature(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Load ERA5 temperature, aligned to the given date index."""
        t_dir   = self.era5_dir / "temperature"
        matches = sorted(t_dir.glob(f"*{self.station_name}*temperature*"))
        if not matches:
            matches = sorted(t_dir.glob("*BELO*temperature*"))
        if not matches:
            raise FileNotFoundError(f"No temperature file for {self.station_name} in {t_dir}")
        df = pd.read_csv(matches[0], parse_dates=[self.ERA5_DATE_COL])
        df = df.set_index(self.ERA5_DATE_COL).sort_index()
        numeric = [c for c in df.columns
                   if df[c].dtype != object
                   and not any(s in c.lower() for s in ("lat", "lon", "station"))]
        col = numeric[0] if numeric else df.columns[0]
        return df[col].rename("temperature").reindex(dates)

    def _load_nino34(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Load daily Niño 3.4 SST index, aligned to the given date index."""
        df = pd.read_csv(self.nino34_path, parse_dates=[self.NINO34_DATE_COL])
        df = df.set_index(self.NINO34_DATE_COL).sort_index()
        col = (
            self.NINO34_VALUE_COL
            if self.NINO34_VALUE_COL in df.columns
            else [c for c in df.columns if df[c].dtype != object][0]
        )
        return df[col].rename("nino34").reindex(dates)

    # ─────────────────────────────────────────────────────────────────────────
    # Standardisation (training stats only)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fit_scaler(df_train: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Compute per-column mean and std from the training block."""
        mu  = df_train.mean()
        std = df_train.std().replace(0, 1.0)
        return mu, std

    @staticmethod
    def _apply_scaler(df: pd.DataFrame, mu: pd.Series, std: pd.Series) -> np.ndarray:
        """Standardise df using pre-fitted (mu, std); fill NaN with 0."""
        scaled = (df - mu) / std
        return scaled.fillna(0.0).values.astype(np.float64)

    # ─────────────────────────────────────────────────────────────────────────
    # Covariate block builders
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _lag_block(series: pd.Series, lags: List[int]) -> pd.DataFrame:
        """Build a DataFrame of lagged values.  lag l → column 'name_lag{l}'."""
        base = series.name or "x"
        frames = {}
        for l in lags:
            col = f"{base}_lag{l}"
            frames[col] = series.shift(l)
        return pd.DataFrame(frames, index=series.index)

    @staticmethod
    def _rolling_lag(series: pd.Series, window: int, shift: int = 1) -> pd.Series:
        """Trailing rolling mean, shifted by `shift` to avoid leakage."""
        name = f"{series.name or 'x'}_roll{window}"
        return series.shift(shift).rolling(window=window, min_periods=1).mean().rename(name)

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def load_all(self) -> Dict:
        """
        Load, align, and standardise all data for the BH experiment.

        Returns
        -------
        dict with keys:
            y_train, y_test                          — precipitation arrays
            dates_train, dates_test                  — DatetimeIndex
            covariate_blocks_train                   — {block_name: ndarray}
            covariate_blocks_test                    — {block_name: ndarray}
            covariate_col_names                      — {block_name: [col, ...]}
            summary                                  — data summary dict
        """
        # ── Precipitation ───────────────────────────────────────────────────
        precip_train, precip_test = self._load_precipitation()
        dates_train = precip_train.index
        dates_test  = precip_test.index

        # All-dates index for building lag matrices without edge NaN
        all_dates = dates_train.union(dates_test).sort_values()

        y_train = precip_train.values.astype(np.float64)
        y_test  = precip_test.values.astype(np.float64)

        # ── Covariates on full date range ───────────────────────────────────
        dp   = self._load_dewpoint(all_dates)
        temp = self._load_temperature(all_dates)
        n34  = self._load_nino34(all_dates)

        # Short-lag blocks: lags [1,2,3]  (per MODELS.md §15.1, §16.1)
        # Seasonal-lag blocks: lags [1,2,3,364,365,366,367] (per §15.2, §16.2)
        dew_short    = self._lag_block(dp,   [1, 2, 3])
        dew_seasonal = self._lag_block(dp,   [1, 2, 3, 364, 365, 366, 367])
        tmp_short    = self._lag_block(temp, [1, 2, 3])
        tmp_seasonal = self._lag_block(temp, [1, 2, 3, 364, 365, 366, 367])

        # Combined dew+temp blocks
        dewtemp_short    = pd.concat([dew_short,    tmp_short],    axis=1)
        dewtemp_seasonal = pd.concat([dew_seasonal, tmp_seasonal], axis=1)

        # ENSO long-component blocks (per MODELS.md §18):
        #   90-day rolling mean,  shifted 1 day to avoid leakage
        #   30-day rolling mean,  shifted 1 day
        #   Daily lag-1 of NINO34
        e90    = self._rolling_lag(n34, window=90, shift=1).rename("enso_roll90")
        e30    = self._rolling_lag(n34, window=30, shift=1).rename("enso_roll30")
        e_lag1 = n34.shift(1).rename("enso_lag1")

        enso_90d         = e90.to_frame()
        enso_90d_30d     = pd.concat([e90, e30],            axis=1)
        enso_90d_30d_day = pd.concat([e90, e30, e_lag1],    axis=1)

        # ── Collect all raw blocks ──────────────────────────────────────────
        raw_blocks: Dict[str, pd.DataFrame] = {
            "dewpoint_short":      dew_short,
            "dewpoint_seasonal":   dew_seasonal,
            "dewtemp_short":       dewtemp_short,
            "dewtemp_seasonal":    dewtemp_seasonal,
            "enso_90d":            enso_90d,
            "enso_90d_30d":        enso_90d_30d,
            "enso_90d_30d_daily":  enso_90d_30d_day,
        }

        # ── Train/test split + standardise ──────────────────────────────────
        blocks_train: Dict[str, np.ndarray] = {}
        blocks_test:  Dict[str, np.ndarray] = {}
        col_names:    Dict[str, List[str]]  = {}

        for name, raw_df in raw_blocks.items():
            raw_train = raw_df.reindex(dates_train).fillna(0.0)
            raw_test  = raw_df.reindex(dates_test).fillna(0.0)
            col_names[name] = list(raw_train.columns)

            # Fit scaler on training data only
            mu, std = self._fit_scaler(raw_train)

            blocks_train[name] = self._apply_scaler(raw_train, mu, std)
            blocks_test[name]  = self._apply_scaler(raw_test,  mu, std)

        # ── Data summary ────────────────────────────────────────────────────
        n_train = len(y_train)
        n_test  = len(y_test)
        summary = {
            "station":          "BELO HORIZONTE",
            "train_start":      str(dates_train[0].date()),
            "train_end":        str(dates_train[-1].date()),
            "test_start":       str(dates_test[0].date()),
            "test_end":         str(dates_test[-1].date()),
            "n_train":          n_train,
            "n_test":           n_test,
            "wet_days_train":   int((y_train > 0).sum()),
            "dry_days_train":   int((y_train == 0).sum()),
            "wet_frac_train":   float((y_train > 0).mean()),
            "mean_precip_train": float(y_train[y_train > 0].mean()) if (y_train > 0).any() else 0.0,
            "max_precip_train": float(y_train.max()),
            "wet_days_test":    int((y_test > 0).sum()),
            "wet_frac_test":    float((y_test > 0).mean()),
            "covariate_blocks": list(raw_blocks.keys()),
        }

        return {
            "y_train":                 y_train,
            "y_test":                  y_test,
            "dates_train":             dates_train,
            "dates_test":              dates_test,
            "covariate_blocks_train":  blocks_train,
            "covariate_blocks_test":   blocks_test,
            "covariate_col_names":     col_names,
            "summary":                 summary,
        }
