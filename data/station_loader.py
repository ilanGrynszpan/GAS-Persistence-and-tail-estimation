"""
Generic data loader for all precipitation stations.

=============================================================================
SUPPORTED STATIONS
=============================================================================

Brazilian INMET stations (date column: "Data Medicao"):
    BELO HORIZONTE, MANAUS, CRUZEIRO DO SUL (ACRE),
    GARANHUNS (PERNAMBUCO), SALVADOR, SÃO PAULO

International stations (date column: "date"):
    DARWIN AIRPORT, TORONTO

=============================================================================
STATION REGISTRY
=============================================================================

Each station entry specifies:
    name        : exact file prefix used in data/output/{name}_train.csv
    era5_key    : ERA5 file prefix (e.g. "BELO_HORIZONTE")
    date_col    : precipitation date column ("Data Medicao" or "date")
    run_id      : artifact run identifier

=============================================================================
DATA SOURCES
=============================================================================

Precipitation (y_t):
    C:/.../data/output/{name}_train.csv   — training observations
    C:/.../data/output/{name}_test.csv    — test observations

ERA5 dew point:
    data/input/ERA5/humidity/{era5_key}_ERA5_dewpoint_2m_daily_2006_2025.csv
    Column: "dewpoint_2m_c"

ERA5 temperature:
    data/input/ERA5/temperature/{era5_key}_ERA5_temperature_2m_daily_2006_2025.csv
    Column: "temperature_2m_c"

ENSO — El Niño / La Niña / Neutral state (shared across all stations):
    data/processed/pacific/ENSO_clean.csv
    Columns: "date" (monthly), "ONI", "el_nino", "la_nina", "neutral"
    Coverage: 1950-01-01 onward

    NOTE (fixed 2026-07-08): the previous source,
    data/processed/pacific/NINO34_daily.csv ("nino34_sst"), is DAILY RAW
    Niño 3.4 sea-surface temperature (24-30 degC, no climatology removed) —
    not an anomaly. Applying the standard +/-0.5 degC ENSO threshold to it
    classified every single day as El Nino. ENSO_clean.csv's "ONI" column is
    the genuine NOAA Oceanic Nino Index (3-month running mean anomaly) with
    el_nino/la_nina/neutral already classified at +/-0.5 using the standard
    methodology. Because ONI is monthly, each month's value is broadcast to
    its calendar days and lagged by one full month (a month's ONI is not
    known until the month completes), avoiding both the data bug and any
    look-ahead leakage.

=============================================================================
COVARIATE BLOCKS  (identical construction to BHDataLoader)
=============================================================================

Standard GAS covariates (X_t):
    "dewpoint_short"       — D_{t-1}, D_{t-2}, D_{t-3}
    "dewpoint_seasonal"    — D_{t-1}…D_{t-3}, D_{t-364}…D_{t-367}
    "dewtemp_short"        — above + T_{t-1}, T_{t-2}, T_{t-3}
    "dewtemp_seasonal"     — above + T_{t-364}…T_{t-367}

Harvey long component (X_long) — El Nino / La Nina dummies (neutral is the
baseline, omitted category), per MODELS.md §18/§2a of the 2026-07-07 fix:
    "enso_dummy_current"   — el_nino_t, la_nina_t
    "enso_dummy_lag1"      — + el_nino_{t-1mo}, la_nina_{t-1mo}
    "enso_dummy_lag3"      — + el_nino_{t-3mo}, la_nina_{t-3mo}
(all already lagged 1 month relative to "today" to avoid look-ahead; the
"lag1"/"lag3" tier names refer to *additional* whole-month lags beyond that
mandatory 1-month safety lag)

=============================================================================
STANDARDISATION
=============================================================================

All covariates are z-scored using TRAINING statistics only.
Pre-ERA5 dates (before 2006) are filled with 0 after standardisation.
This is equivalent to imputing the training mean and does not inject
future information into the training period.

=============================================================================
USAGE
=============================================================================

    from data.station_loader import StationDataLoader, STATION_REGISTRY

    loader = StationDataLoader.from_registry("DARWIN AIRPORT",
                                              precip_dir=...,
                                              era5_dir=...,
                                              nino34_path=...)
    data = loader.load_all()

    # identical interface to BHDataLoader.load_all()
    y_train       = data["y_train"]
    y_test        = data["y_test"]
    dates_train   = data["dates_train"]
    covariate_blocks_train = data["covariate_blocks_train"]
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Station registry
# ─────────────────────────────────────────────────────────────────────────────

STATION_REGISTRY: Dict[str, Dict] = {
    "BELO HORIZONTE": {
        "era5_key": "BELO_HORIZONTE",
        "date_col": "Data Medicao",
        "run_id":   "run_20260701_bh",
        "short":    "bh",
        "display":  "Belo Horizonte",
    },
    "CRUZEIRO DO SUL (ACRE)": {
        "era5_key": "CRUZEIRO_DO_SUL",
        "date_col": "Data Medicao",
        "run_id":   "run_20260705_cruzeiro",
        "short":    "cruzeiro",
        "display":  "Cruzeiro do Sul",
    },
    "DARWIN AIRPORT": {
        "era5_key": "DARWIN_AIRPORT",
        "date_col": "date",
        "run_id":   "run_20260705_darwin",
        "short":    "darwin",
        "display":  "Darwin Airport",
    },
    "GARANHUNS (PERNAMBUCO)": {
        "era5_key": "GARANHUNS",
        "date_col": "Data Medicao",
        "run_id":   "run_20260705_garanhuns",
        "short":    "garanhuns",
        "display":  "Garanhuns",
    },
    "MANAUS": {
        "era5_key": "MANAUS",
        "date_col": "Data Medicao",
        "run_id":   "run_20260705_manaus",
        "short":    "manaus",
        "display":  "Manaus",
    },
    "SALVADOR": {
        "era5_key": "SALVADOR",
        "date_col": "Data Medicao",
        "run_id":   "run_20260705_salvador",
        "short":    "salvador",
        "display":  "Salvador",
    },
    "SÃO PAULO": {
        "era5_key": "SAO_PAULO",
        "date_col": "Data Medicao",
        "run_id":   "run_20260705_saopaulo",
        "short":    "saopaulo",
        "display":  "São Paulo",
    },
    "TORONTO": {
        "era5_key": "TORONTO",
        "date_col": "date",
        "run_id":   "run_20260705_toronto",
        "short":    "toronto",
        "display":  "Toronto",
    },
}

# Stations to exclude from multi-location runs
EXCLUDED_STATIONS = {"RIYADH OBS. (O.A.P."}


class StationDataLoader:
    """
    Generic data loader for any station in the STATION_REGISTRY.

    Handles two precipitation file formats:
      - Brazilian INMET: date_col="Data Medicao"
      - International:   date_col="date"

    ERA5 data (2006–2025) is aligned to the precipitation date range.
    Observations before 2006 receive ERA5 = 0 (training mean after z-scoring)
    — equivalent to imputing the training-period mean, causing no leakage.

    Returns the same data dictionary as BHDataLoader.load_all().
    """

    PRECIP_VALUE_COL = "PRECIPITACAO TOTAL, DIARIO(mm)"
    ENSO_DATE_COL    = "date"
    ENSO_COLS        = ["ONI", "el_nino", "la_nina", "neutral"]
    ERA5_DATE_COL    = "date"

    def __init__(
        self,
        station_name: str,
        era5_key:     str,
        date_col:     str,
        precip_dir:   Path | str,
        era5_dir:     Path | str,
        enso_path:    Path | str,
    ):
        self.station_name = station_name
        self.era5_key     = era5_key
        self.date_col     = date_col
        self.precip_dir   = Path(precip_dir)
        self.era5_dir     = Path(era5_dir)
        self.enso_path    = Path(enso_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Class method: construct from registry
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_registry(
        cls,
        station_name: str,
        precip_dir:   Path | str,
        era5_dir:     Path | str,
        enso_path:    Path | str,
    ) -> "StationDataLoader":
        """Create loader from the station registry."""
        if station_name not in STATION_REGISTRY:
            raise KeyError(
                f"Station '{station_name}' not in registry. "
                f"Available: {sorted(STATION_REGISTRY)}"
            )
        cfg = STATION_REGISTRY[station_name]
        return cls(
            station_name=station_name,
            era5_key=cfg["era5_key"],
            date_col=cfg["date_col"],
            precip_dir=precip_dir,
            era5_dir=era5_dir,
            enso_path=enso_path,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Raw loaders
    # ──────────────────────────────────────────────────────────────────────────

    def _load_precipitation(self) -> Tuple[pd.Series, pd.Series]:
        """Load train and test precipitation, indexed by date."""
        train_path = self.precip_dir / f"{self.station_name}_train.csv"
        test_path  = self.precip_dir / f"{self.station_name}_test.csv"

        def _read(path: Path) -> pd.Series:
            df = pd.read_csv(path, parse_dates=[self.date_col])
            df = df.set_index(self.date_col).sort_index()
            series = df[self.PRECIP_VALUE_COL].rename("precip")
            # Sensor errors / negative values → 0
            return series.clip(lower=0.0)

        return _read(train_path), _read(test_path)

    def _load_era5_variable(
        self, subdir: str, keyword: str, dates: pd.DatetimeIndex
    ) -> pd.Series:
        """Load one ERA5 variable (dew point or temperature), aligned to dates."""
        era5_subdir = self.era5_dir / subdir
        matches = sorted(era5_subdir.glob(f"*{self.era5_key}*{keyword}*"))
        if not matches:
            raise FileNotFoundError(
                f"No {keyword} ERA5 file for {self.era5_key} in {era5_subdir}"
            )
        df = pd.read_csv(matches[0], parse_dates=[self.ERA5_DATE_COL])
        df = df.set_index(self.ERA5_DATE_COL).sort_index()
        # Select numeric column that is not lat/lon/station
        numeric = [
            c for c in df.columns
            if df[c].dtype != object
            and not any(s in c.lower() for s in ("lat", "lon", "station"))
        ]
        col = numeric[0] if numeric else df.columns[0]
        return df[col].rename(keyword).reindex(dates)

    def _load_enso_daily(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Load monthly ENSO state (ONI anomaly + El Nino/La Nina/Neutral dummies)
        and broadcast it to daily resolution, lagged by one full calendar
        month to avoid look-ahead (a month's ONI is not known until the
        month completes).

        Returns a daily-indexed DataFrame with columns:
            oni                          — continuous ONI anomaly (for plots)
            el_nino_t, la_nina_t         — current-state dummies (tier 1)
            el_nino_lag1mo, la_nina_lag1mo — one additional month back (tier 2)
            el_nino_lag3mo, la_nina_lag3mo — three additional months back (tier 3)
            enso_label                   — "El Niño" / "La Niña" / "Neutral"
        """
        df = pd.read_csv(self.enso_path, parse_dates=[self.ENSO_DATE_COL])
        df = df.set_index(self.ENSO_DATE_COL).sort_index()
        base = df[self.ENSO_COLS]

        monthly = pd.DataFrame(index=base.index)
        # shift(1): a month's ONI/classification only usable the following month
        monthly["oni"]              = base["ONI"].shift(1)
        monthly["el_nino_t"]        = base["el_nino"].shift(1)
        monthly["la_nina_t"]        = base["la_nina"].shift(1)
        monthly["el_nino_lag1mo"]   = base["el_nino"].shift(2)
        monthly["la_nina_lag1mo"]   = base["la_nina"].shift(2)
        monthly["el_nino_lag3mo"]   = base["el_nino"].shift(4)
        monthly["la_nina_lag3mo"]   = base["la_nina"].shift(4)
        monthly["enso_label"] = np.select(
            [monthly["el_nino_t"] == 1, monthly["la_nina_t"] == 1],
            ["El Niño", "La Niña"],
            default="Neutral",
        )

        daily_grid = pd.date_range(
            monthly.index.min(), max(dates.max(), monthly.index.max()), freq="D"
        )
        daily = (
            monthly.reindex(monthly.index.union(daily_grid))
            .sort_index()
            .ffill()
            .reindex(daily_grid)
        )
        return daily.reindex(dates)

    # ──────────────────────────────────────────────────────────────────────────
    # Standardisation (training statistics only)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fit_scaler(df_train: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        mu  = df_train.mean()
        std = df_train.std().replace(0, 1.0)
        return mu, std

    @staticmethod
    def _apply_scaler(
        df: pd.DataFrame, mu: pd.Series, std: pd.Series
    ) -> np.ndarray:
        """Standardise df using pre-fitted (mu, std); fill NaN with 0."""
        scaled = (df - mu) / std
        return scaled.fillna(0.0).values.astype(np.float64)

    # ──────────────────────────────────────────────────────────────────────────
    # Covariate block builders
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _lag_block(series: pd.Series, lags: List[int]) -> pd.DataFrame:
        """DataFrame of lagged values.  lag l → column '{name}_lag{l}'."""
        base   = series.name or "x"
        frames = {f"{base}_lag{l}": series.shift(l) for l in lags}
        return pd.DataFrame(frames, index=series.index)

    @staticmethod
    def _rolling_lag(series: pd.Series, window: int, shift: int = 1) -> pd.Series:
        """Trailing rolling mean, shifted by `shift` to avoid look-ahead."""
        name = f"{series.name or 'x'}_roll{window}"
        return (
            series.shift(shift)
            .rolling(window=window, min_periods=1)
            .mean()
            .rename(name)
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def load_all(self) -> Dict:
        """
        Load, align, and standardise all data for this station.

        Returns
        -------
        dict with keys:
            y_train, y_test                          — precipitation arrays
            dates_train, dates_test                  — DatetimeIndex
            covariate_blocks_train                   — {block_name: ndarray}
            covariate_blocks_test                    — {block_name: ndarray}
            covariate_col_names                      — {block_name: [col, ...]}
            nino34_train, nino34_test                — ONI anomaly (unscaled, corrected)
            enso_label_train, enso_label_test         — "El Niño"/"La Niña"/"Neutral"
            summary                                  — data summary dict
        """
        # ── Precipitation ───────────────────────────────────────────────────
        precip_train, precip_test = self._load_precipitation()
        dates_train = precip_train.index
        dates_test  = precip_test.index

        # Build a continuous date index spanning both periods for lag alignment
        all_dates = dates_train.union(dates_test).sort_values()

        y_train = precip_train.values.astype(np.float64)
        y_test  = precip_test.values.astype(np.float64)

        # ── ERA5 covariates on full date range ──────────────────────────────
        dp   = self._load_era5_variable("humidity",    "dewpoint",    all_dates)
        temp = self._load_era5_variable("temperature", "temperature", all_dates)
        enso = self._load_enso_daily(all_dates)

        # Short-lag blocks: lags [1,2,3]  (per MODELS.md §15.1, §16.1)
        dew_short    = self._lag_block(dp,   [1, 2, 3])
        dew_seasonal = self._lag_block(dp,   [1, 2, 3, 364, 365, 366, 367])
        tmp_short    = self._lag_block(temp, [1, 2, 3])
        tmp_seasonal = self._lag_block(temp, [1, 2, 3, 364, 365, 366, 367])

        dewtemp_short    = pd.concat([dew_short,    tmp_short],    axis=1)
        dewtemp_seasonal = pd.concat([dew_seasonal, tmp_seasonal], axis=1)

        # ENSO long-component blocks: El Nino / La Nina dummies (neutral is
        # the omitted baseline), per MODELS.md §18 / the 2026-07-07 data fix.
        # Already 1-month-lagged inside _load_enso_daily; "lag1mo"/"lag3mo"
        # here add further whole-month lags on top of that safety lag.
        enso_dummy_current = enso[["el_nino_t", "la_nina_t"]]
        enso_dummy_lag1 = pd.concat(
            [enso_dummy_current, enso[["el_nino_lag1mo", "la_nina_lag1mo"]]], axis=1
        )
        enso_dummy_lag3 = pd.concat(
            [enso_dummy_lag1, enso[["el_nino_lag3mo", "la_nina_lag3mo"]]], axis=1
        )

        raw_blocks: Dict[str, pd.DataFrame] = {
            "dewpoint_short":      dew_short,
            "dewpoint_seasonal":   dew_seasonal,
            "dewtemp_short":       dewtemp_short,
            "dewtemp_seasonal":    dewtemp_seasonal,
            "enso_dummy_current":  enso_dummy_current,
            "enso_dummy_lag1":     enso_dummy_lag1,
            "enso_dummy_lag3":     enso_dummy_lag3,
        }

        # ── Train/test split + standardise ──────────────────────────────────
        blocks_train: Dict[str, np.ndarray] = {}
        blocks_test:  Dict[str, np.ndarray] = {}
        col_names:    Dict[str, List[str]]  = {}

        for name, raw_df in raw_blocks.items():
            raw_tr = raw_df.reindex(dates_train).fillna(0.0)
            raw_te = raw_df.reindex(dates_test).fillna(0.0)
            col_names[name] = list(raw_tr.columns)
            mu, std = self._fit_scaler(raw_tr)
            blocks_train[name] = self._apply_scaler(raw_tr, mu, std)
            blocks_test[name]  = self._apply_scaler(raw_te, mu, std)

        # Corrected (unscaled) ONI anomaly + precomputed El Nino/La Nina/Neutral
        # labels for ENSO diagnostics (replaces the previous raw-SST array).
        nino34_train = enso["oni"].reindex(dates_train).fillna(0.0).values.astype(np.float64)
        nino34_test  = enso["oni"].reindex(dates_test).fillna(0.0).values.astype(np.float64)
        enso_label_train = (
            enso["enso_label"].reindex(dates_train).fillna("Neutral").values
        )
        enso_label_test = (
            enso["enso_label"].reindex(dates_test).fillna("Neutral").values
        )

        # ── Data summary ────────────────────────────────────────────────────
        summary = {
            "station":           self.station_name,
            "train_start":       str(dates_train[0].date()),
            "train_end":         str(dates_train[-1].date()),
            "test_start":        str(dates_test[0].date()),
            "test_end":          str(dates_test[-1].date()),
            "n_train":           len(y_train),
            "n_test":            len(y_test),
            "wet_days_train":    int((y_train > 0).sum()),
            "dry_days_train":    int((y_train == 0).sum()),
            "wet_frac_train":    float((y_train > 0).mean()),
            "mean_precip_train": float(y_train[y_train > 0].mean()) if (y_train > 0).any() else 0.0,
            "max_precip_train":  float(y_train.max()),
            "wet_days_test":     int((y_test > 0).sum()),
            "wet_frac_test":     float((y_test > 0).mean()),
            "covariate_blocks":  list(raw_blocks.keys()),
        }

        return {
            "y_train":                y_train,
            "y_test":                 y_test,
            "dates_train":            dates_train,
            "dates_test":             dates_test,
            "covariate_blocks_train": blocks_train,
            "covariate_blocks_test":  blocks_test,
            "covariate_col_names":    col_names,
            "nino34_train":           nino34_train,
            "nino34_test":            nino34_test,
            "enso_label_train":       enso_label_train,
            "enso_label_test":        enso_label_test,
            "summary":                summary,
        }
