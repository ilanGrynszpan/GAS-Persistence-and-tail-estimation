"""
Covariate matrix construction for ZA-GAS models.

Handles:
  - ERA5 dew point and temperature (NetCDF or CSV, aligned to daily dates)
  - MJO RMM indices (CSV from BOM/NOAA)
  - Daily Niño 3.4 index (CSV)
  - Lag expansion
  - Rolling mean expansion
  - Interaction terms

All functions return pandas DataFrames with named columns, indexed by date.
The caller is responsible for alignment with the precipitation series dates.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Generic lag / rolling helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_lag_matrix(
    series: pd.Series,
    lags: Sequence[int],
    name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame of lagged values of a series.

    Parameters
    ----------
    series : pd.Series  (index = date)
    lags   : sequence of lag integers (0 = contemporaneous, 1 = lag-1, …)
    name   : base name for columns; defaults to series.name

    Returns
    -------
    pd.DataFrame with columns  f"{name}_lag{l}" for l in lags
    (l=0 gives the column "{name}_t" instead of "{name}_lag0")
    """
    base = name or (series.name or "x")
    frames = {}
    for l in lags:
        col = f"{base}_t" if l == 0 else f"{base}_lag{l}"
        frames[col] = series.shift(l)
    return pd.DataFrame(frames, index=series.index)


def build_rolling_mean_matrix(
    series: pd.Series,
    windows: Sequence[int],
    name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build trailing rolling-mean columns.

    Parameters
    ----------
    series  : pd.Series
    windows : list of window lengths (e.g. [30, 90])
    name    : base name for columns

    Returns
    -------
    pd.DataFrame with columns  f"{name}_roll{w}" for w in windows
    """
    base = name or (series.name or "x")
    frames = {}
    for w in windows:
        # min_periods=1 avoids NaN at start; shift(1) ensures trailing (no leakage)
        frames[f"{base}_roll{w}"] = series.shift(1).rolling(window=w, min_periods=1).mean()
    return pd.DataFrame(frames, index=series.index)


def build_interaction_column(
    col1: pd.Series,
    col2: pd.Series,
    name: Optional[str] = None,
) -> pd.Series:
    """Element-wise product of two (already standardized) series."""
    label = name or f"{col1.name}_x_{col2.name}"
    return (col1 * col2).rename(label)


# ──────────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────────

def _read_csv_with_date(path: Path, date_col: str, value_col: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    return df[value_col].rename(value_col)


def _resolve_era5_file(era5_dir: Path, keyword: str, station: Optional[str]) -> Path:
    """Find the station-specific ERA5 CSV by keyword (dewpoint/temperature)."""
    all_csv = sorted(era5_dir.glob("*.csv"))
    if not all_csv:
        raise FileNotFoundError(f"No CSV files found in {era5_dir}")
    kw = keyword.lower()
    st = station.lower() if station else None
    # Filter by both station name and keyword using plain string comparison
    if st:
        matches = [f for f in all_csv if st in f.name.lower() and kw in f.name.lower()]
        if matches:
            return matches[0]
    # Fallback: any file containing the keyword
    matches = [f for f in all_csv if kw in f.name.lower()]
    if matches:
        return matches[0]
    return all_csv[0]


def load_era5_dew_point(
    era5_dir: Union[str, Path],
    dates: pd.DatetimeIndex,
    date_col: str = "date",
    value_col: Optional[str] = None,
    station: Optional[str] = None,
) -> pd.Series:
    """
    Load ERA5 dew point temperature, aligned to `dates`.

    Parameters
    ----------
    era5_dir : directory containing ERA5 CSV files
    station  : station name prefix (e.g. "BELO_HORIZONTE") used to select
               the correct file when the directory holds multiple stations.
    """
    era5_dir = Path(era5_dir)
    csv_file = _resolve_era5_file(era5_dir, "dewpoint", station)
    df = pd.read_csv(csv_file, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    if value_col is None:
        # Skip lat/lon/station metadata columns
        skip = {"lat", "lon", "station"}
        numeric = [c for c in df.columns
                   if df[c].dtype != object
                   and not any(s in c.lower() for s in skip)]
        value_col = numeric[0] if numeric else df.columns[0]
    series = df[value_col].rename("dewpoint")
    return series.reindex(dates)


def load_era5_temperature(
    era5_dir: Union[str, Path],
    dates: pd.DatetimeIndex,
    date_col: str = "date",
    value_col: Optional[str] = None,
    station: Optional[str] = None,
) -> pd.Series:
    """
    Load ERA5 2m temperature, aligned to `dates`.

    Parameters
    ----------
    era5_dir : directory containing ERA5 CSV files
    station  : station name prefix used to select the correct file.
    """
    era5_dir = Path(era5_dir)
    csv_file = _resolve_era5_file(era5_dir, "temperature", station)
    df = pd.read_csv(csv_file, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    if value_col is None:
        skip = {"lat", "lon", "station"}
        numeric = [c for c in df.columns
                   if df[c].dtype != object
                   and not any(s in c.lower() for s in skip)]
        value_col = numeric[0] if numeric else df.columns[0]
    series = df[value_col].rename("temperature")
    return series.reindex(dates)


def load_mjo(
    path: Union[str, Path],
    dates: pd.DatetimeIndex,
    date_col: str = "date",
    rmm1_col: str = "RMM1",
    rmm2_col: str = "RMM2",
) -> pd.DataFrame:
    """
    Load MJO RMM1 and RMM2 indices, aligned to `dates`.

    Returns DataFrame with columns ["rmm1", "rmm2"].
    """
    path = Path(path)
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    out = pd.DataFrame({
        "rmm1": df[rmm1_col],
        "rmm2": df[rmm2_col],
    }, index=df.index)
    return out.reindex(dates)


def load_nino34(
    path: Union[str, Path],
    dates: pd.DatetimeIndex,
    date_col: str = "date",
    value_col: str = "nino34",
) -> pd.Series:
    """
    Load daily Niño 3.4 index, aligned to `dates`.

    Returns Series named "nino34".
    """
    path = Path(path)
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    if value_col not in df.columns:
        numeric_cols = [c for c in df.columns if df[c].dtype != object]
        value_col = numeric_cols[0] if numeric_cols else df.columns[0]
    series = df[value_col].rename("nino34")
    return series.reindex(dates)


# ──────────────────────────────────────────────────────────────────────────────
# Lag block registry  (covariate_block_name -> block spec)
# ──────────────────────────────────────────────────────────────────────────────

def _dew_block(series: pd.Series, lags: List[int], seasonal: bool = False) -> pd.DataFrame:
    short = build_lag_matrix(series, [0, 1, 2, 3], "dewpoint")
    if seasonal:
        seas = build_lag_matrix(series, [364, 365, 366, 367], "dewpoint")
        return pd.concat([short, seas], axis=1)
    return short[["dewpoint_t"] + [f"dewpoint_lag{l}" for l in lags if l > 0]]


def _temp_block(series: pd.Series, lags: List[int], seasonal: bool = False) -> pd.DataFrame:
    short = build_lag_matrix(series, [0, 1, 2, 3], "temperature")
    if seasonal:
        seas = build_lag_matrix(series, [364, 365, 366, 367], "temperature")
        return pd.concat([short, seas], axis=1)
    return short[["temperature_t"] + [f"temperature_lag{l}" for l in lags if l > 0]]


def _mjo_block(df_mjo: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """Build MJO lag block for given short lags (0..3, no seasonal)."""
    frames = []
    for l in lags:
        suffix = "_t" if l == 0 else f"_lag{l}"
        frames.append(df_mjo["rmm1"].shift(l).rename(f"rmm1{suffix}"))
        frames.append(df_mjo["rmm2"].shift(l).rename(f"rmm2{suffix}"))
    return pd.concat(frames, axis=1)


def build_covariate_blocks(
    dewpoint: Optional[pd.Series] = None,
    temperature: Optional[pd.Series] = None,
    mjo: Optional[pd.DataFrame] = None,
    nino34: Optional[pd.Series] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Build all lag blocks as specified in the modelling protocol.

    Returns
    -------
    dict mapping block_name -> pd.DataFrame of (unstandardized) covariates.
    Blocks with missing inputs are skipped.
    """
    blocks: Dict[str, pd.DataFrame] = {}

    # ── Dew point blocks ────────────────────────────────────────────────
    if dewpoint is not None:
        D = dewpoint.copy().rename("dewpoint")
        D_t_only = D.shift(0).rename("dewpoint_t")
        D_lag0 = build_lag_matrix(D, [0], "dewpoint")
        D_lag1 = build_lag_matrix(D, [0, 1], "dewpoint")
        D_short = build_lag_matrix(D, [0, 1, 2, 3], "dewpoint")
        D_seas_extra = build_lag_matrix(D, [364, 365, 366, 367], "dewpoint")
        D_seasonal = pd.concat([D_short, D_seas_extra], axis=1)

        blocks["dewpoint_t"] = D_lag0.rename(columns={"dewpoint_t": "dewpoint_t"})
        blocks["dewpoint_lag1"] = D_lag1
        blocks["dewpoint_short_lags"] = D_short
        blocks["dewpoint_seasonal_lags"] = D_seasonal

    # ── Temperature blocks ───────────────────────────────────────────────
    if temperature is not None:
        T = temperature.copy().rename("temperature")
        T_lag0 = build_lag_matrix(T, [0], "temperature")
        T_lag1 = build_lag_matrix(T, [0, 1], "temperature")
        T_short = build_lag_matrix(T, [0, 1, 2, 3], "temperature")
        T_seas_extra = build_lag_matrix(T, [364, 365, 366, 367], "temperature")
        T_seasonal = pd.concat([T_short, T_seas_extra], axis=1)

        blocks["temperature_t"] = T_lag0
        blocks["temperature_lag1"] = T_lag1
        blocks["temperature_short_lags"] = T_short
        blocks["temperature_seasonal_lags"] = T_seasonal

    # ── Dew point + Temperature joint blocks ─────────────────────────────
    if dewpoint is not None and temperature is not None:
        D_t = build_lag_matrix(dewpoint.rename("dewpoint"), [0], "dewpoint")
        T_t = build_lag_matrix(temperature.rename("temperature"), [0], "temperature")
        D_lag1 = build_lag_matrix(dewpoint.rename("dewpoint"), [0, 1], "dewpoint")
        T_lag1 = build_lag_matrix(temperature.rename("temperature"), [0, 1], "temperature")
        D_short = build_lag_matrix(dewpoint.rename("dewpoint"), [0, 1, 2, 3], "dewpoint")
        T_short = build_lag_matrix(temperature.rename("temperature"), [0, 1, 2, 3], "temperature")
        D_seas = pd.concat([D_short, build_lag_matrix(dewpoint.rename("dewpoint"), [364, 365, 366, 367], "dewpoint")], axis=1)
        T_seas = pd.concat([T_short, build_lag_matrix(temperature.rename("temperature"), [364, 365, 366, 367], "temperature")], axis=1)

        blocks["dewpoint_temperature_t"] = pd.concat([D_t, T_t], axis=1)
        blocks["dewpoint_temperature_lag1"] = pd.concat([D_lag1, T_lag1], axis=1)
        blocks["dewpoint_temperature_short_lags"] = pd.concat([D_short, T_short], axis=1)
        blocks["dewpoint_temperature_seasonal_lags"] = pd.concat([D_seas, T_seas], axis=1)

    # ── MJO blocks ────────────────────────────────────────────────────────
    if mjo is not None:
        blocks["mjo_t"] = _mjo_block(mjo, [0])
        blocks["mjo_lag1"] = _mjo_block(mjo, [0, 1])
        blocks["mjo_short_lags"] = _mjo_block(mjo, [0, 1, 2, 3])

    # ── Niño 3.4 blocks ───────────────────────────────────────────────────
    if nino34 is not None:
        N = nino34.copy().rename("nino34")
        N_t = N.shift(0).rename("nino34_t")
        N_lags = pd.concat([
            N.shift(0).rename("nino34_t"),
            N.shift(7).rename("nino34_lag7"),
            N.shift(30).rename("nino34_lag30"),
        ], axis=1)
        N_rolling = pd.concat([
            N.shift(0).rename("nino34_t"),
            N.shift(1).rolling(30, min_periods=1).mean().rename("nino34_roll30"),
            N.shift(1).rolling(90, min_periods=1).mean().rename("nino34_roll90"),
        ], axis=1)

        blocks["nino34_t"] = N_t.to_frame()
        blocks["nino34_lags"] = N_lags
        blocks["nino34_rolling"] = N_rolling

    # ── Interaction blocks ────────────────────────────────────────────────
    if dewpoint is not None and temperature is not None:
        D0 = dewpoint.shift(0)
        T0 = temperature.shift(0)
        blocks["interaction_dewpoint_temperature_t"] = build_interaction_column(
            D0.rename("dewpoint_t"), T0.rename("temperature_t"), "dew_x_temp"
        ).to_frame()

    if dewpoint is not None and mjo is not None:
        D0 = dewpoint.shift(0)
        R1 = mjo["rmm1"].shift(0)
        R2 = mjo["rmm2"].shift(0)
        blocks["interaction_dewpoint_mjo"] = pd.DataFrame({
            "dew_x_rmm1": (D0 * R1).values,
            "dew_x_rmm2": (D0 * R2).values,
        }, index=dewpoint.index)

    if dewpoint is not None and nino34 is not None:
        D0 = dewpoint.shift(0)
        N0 = nino34.shift(0)
        blocks["interaction_dewpoint_nino34"] = build_interaction_column(
            D0.rename("dewpoint_t"), N0.rename("nino34_t"), "dew_x_nino34"
        ).to_frame()

    if temperature is not None and nino34 is not None:
        T0 = temperature.shift(0)
        N0 = nino34.shift(0)
        blocks["interaction_temperature_nino34"] = build_interaction_column(
            T0.rename("temperature_t"), N0.rename("nino34_t"), "temp_x_nino34"
        ).to_frame()

    return blocks


def covariate_block_registry(
    dewpoint: Optional[pd.Series] = None,
    temperature: Optional[pd.Series] = None,
    mjo: Optional[pd.DataFrame] = None,
    nino34: Optional[pd.Series] = None,
) -> Dict[str, pd.DataFrame]:
    """Alias for build_covariate_blocks for backward compatibility."""
    return build_covariate_blocks(dewpoint, temperature, mjo, nino34)


def align_covariates(
    blocks: Dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    fill_value: float = 0.0,
) -> Dict[str, pd.DataFrame]:
    """
    Align all covariate blocks to a common DatetimeIndex.

    Missing dates are filled with fill_value (0.0 = mean-zero for standardized data).
    """
    aligned = {}
    for name, df in blocks.items():
        aligned[name] = df.reindex(dates).fillna(fill_value)
    return aligned


def df_to_array(df: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame to float32 numpy array, replacing NaN with 0."""
    return np.nan_to_num(df.values.astype(np.float64), nan=0.0)
