"""
LaTeX table generation for diagnostic results.

All functions return a plain string of LaTeX code that can be pasted
directly into a document or written to a .tex file.
"""

from __future__ import annotations
from typing import List
import numpy as np


def _fmt(x, decimals: int = 4) -> str:
    """Format a float for a LaTeX table cell."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{decimals}f}"


def _bool_cell(flag: bool) -> str:
    return r"\textbf{yes}" if flag else "no"


# ===========================================================================
# Information criteria table
# ===========================================================================

def latex_info_table(
    results: dict[str, dict],
    caption: str = "Model information criteria",
    label: str = "tab:info",
) -> str:
    """
    Produce a LaTeX table from a dict of  location -> info_table() results.

    Parameters
    ----------
    results : {location_name: {'loglik': ..., 'n_params': ..., 'aic': ..., 'bic': ...}}
    """
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{" + caption + "}",
        r"  \label{" + label + "}",
        r"  \begin{tabular}{lrrrr}",
        r"  \hline",
        r"  Location & $k$ & $n$ & AIC & BIC \\",
        r"  \hline",
    ]
    for loc, d in results.items():
        row = (
            f"  {loc} & {d['n_params']} & {d['n_obs']} & "
            f"{_fmt(d['aic'], 2)} & {_fmt(d['bic'], 2)} \\\\"
        )
        lines.append(row)
    lines += [
        r"  \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ===========================================================================
# Coverage tests table
# ===========================================================================

def latex_coverage_table(
    results: dict[str, List[dict]],
    caption: str = "Coverage test results",
    label: str = "tab:coverage",
) -> str:
    """
    LaTeX table for coverage_tests() results across multiple locations.

    Parameters
    ----------
    results : {location: list_of_dicts_from_coverage_tests()}
    """
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{" + caption + "}",
        r"  \label{" + label + "}",
        r"  \begin{tabular}{llrrrrrc}",
        r"  \hline",
        r"  Location & $\alpha$ & Viol. & Rate & "
        r"$LR_{uc}$ & $p_{uc}$ & $LR_{cc}$ & $p_{cc}$ \\",
        r"  \hline",
    ]
    for loc, rows in results.items():
        for r in rows:
            line = (
                f"  {loc} & {r['alpha']:.2f} & {r['violations']} & "
                f"{_fmt(r['violation_rate'], 3)} & "
                f"{_fmt(r['LR_uc'], 3)} & {_fmt(r['pvalue_uc'], 3)} & "
                f"{_fmt(r['LR_cc'], 3)} & {_fmt(r['pvalue_cc'], 3)} \\\\"
            )
            lines.append(line)
    lines += [
        r"  \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ===========================================================================
# Jarque-Bera table
# ===========================================================================

def latex_jb_table(
    results: dict[str, dict],
    caption: str = "Jarque-Bera normality test on quantile residuals",
    label: str = "tab:jb",
) -> str:
    """
    LaTeX table for jarque_bera() results.

    Parameters
    ----------
    results : {location: jarque_bera_result_dict}
              Each dict has keys stat, pvalue, skewness, kurtosis.
    """
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{" + caption + "}",
        r"  \label{" + label + "}",
        r"  \begin{tabular}{lrrrrc}",
        r"  \hline",
        r"  Location & Skewness & Kurtosis & JB stat & $p$-value & Reject ($5\%$) \\",
        r"  \hline",
    ]
    for loc, d in results.items():
        reject = "yes" if d["pvalue"] < 0.05 else "no"
        row = (
            f"  {loc} & {_fmt(d['skewness'], 3)} & {_fmt(d['kurtosis'], 3)} & "
            f"{_fmt(d['stat'], 2)} & {_fmt(d['pvalue'], 4)} & {reject} \\\\"
        )
        lines.append(row)
    lines += [
        r"  \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ===========================================================================
# Simulation metrics table
# ===========================================================================

def latex_simulation_table(
    results: dict[str, dict],
    caption: str = "Out-of-sample forecast evaluation",
    label: str = "tab:sim",
) -> str:
    """
    LaTeX table for simulation metric dicts.

    Each value dict should have keys: rmse, mad, crps  (and optionally
    rmse_is, mad_is, crps_is for in-sample).
    """
    has_is = any("rmse_is" in d for d in results.values())

    if has_is:
        header = (
            r"  Location & RMSE (IS) & MAD (IS) & CRPS (IS) & "
            r"RMSE (OOS) & MAD (OOS) & CRPS (OOS) \\"
        )
        col_spec = r"  \begin{tabular}{lrrrrrr}"
    else:
        header = r"  Location & RMSE & MAD & CRPS \\"
        col_spec = r"  \begin{tabular}{lrrr}"

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{" + caption + "}",
        r"  \label{" + label + "}",
        col_spec,
        r"  \hline",
        header,
        r"  \hline",
    ]
    for loc, d in results.items():
        if has_is:
            row = (
                f"  {loc} & {_fmt(d.get('rmse_is'))} & {_fmt(d.get('mad_is'))} & "
                f"{_fmt(d.get('crps_is'))} & {_fmt(d.get('rmse'))} & "
                f"{_fmt(d.get('mad'))} & {_fmt(d.get('crps'))} \\\\"
            )
        else:
            row = (
                f"  {loc} & {_fmt(d.get('rmse'))} & {_fmt(d.get('mad'))} & "
                f"{_fmt(d.get('crps'))} \\\\"
            )
        lines.append(row)
    lines += [
        r"  \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)
