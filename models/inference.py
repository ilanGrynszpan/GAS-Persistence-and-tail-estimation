"""
Frequentist inference for GAS model parameters.

Computes numerical Hessian at the MLE, inverts it to obtain
the asymptotic covariance matrix, and reports SE, z-stat, p-value, and CI.

If the Hessian is not positive definite (common with many parameters),
the function falls back to the diagonal of the BFGS inverse Hessian
approximation stored in the fit result, or reports NaN.
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime
from scipy.stats import norm as _norm


# ──────────────────────────────────────────────────────────────────────────────

def numerical_hessian(
    neg_loglik_fn,
    theta: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Central-difference numerical Hessian of neg_loglik_fn at theta.

    H_{ij} = ( f(e_i+e_j) - f(e_i-e_j) - f(-e_i+e_j) + f(-e_i-e_j) ) / (4*eps^2)

    Uses a simpler forward-difference approximation (faster):
        H_i = (grad(theta + eps*e_i) - grad(theta)) / eps

    Returns symmetric matrix.
    """
    n = len(theta)
    H = np.zeros((n, n))
    g0 = approx_fprime(theta, neg_loglik_fn, eps)
    for i in range(n):
        ei       = np.zeros(n)
        ei[i]    = eps
        gi       = approx_fprime(theta + ei, neg_loglik_fn, eps)
        H[i, :]  = (gi - g0) / eps
    return (H + H.T) / 2.0


def covariance_from_hessian(
    H: np.ndarray,
    fallback: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, bool]:
    """
    Invert Hessian to get asymptotic covariance matrix.

    Returns (cov_matrix, is_reliable).
    If H is not PD, attempts pseudo-inverse; sets is_reliable=False.
    """
    try:
        cov = np.linalg.inv(H)
        # Check PD: all eigenvalues positive
        eigs = np.linalg.eigvalsh(cov)
        if np.all(eigs > 0):
            return cov, True
    except np.linalg.LinAlgError:
        pass

    # Fallback: pseudo-inverse
    try:
        cov = np.linalg.pinv(H)
        return cov, False
    except np.linalg.LinAlgError:
        pass

    # Last resort: diagonal of BFGS H^{-1} (fallback array)
    if fallback is not None:
        fb = np.asarray(fallback)
        if fb.ndim == 2:
            cov = fb
        else:
            cov = np.diag(fb)
        return cov, False

    return np.full((len(H), len(H)), np.nan), False


def inference_table(
    theta: np.ndarray,
    param_names: List[str],
    cov: np.ndarray,
    is_reliable: bool,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """
    Compute SE, z-stat, p-value, CI for each parameter.

    Returns DataFrame with columns:
        parameter, estimate, SE, z_stat, p_value, ci_lower, ci_upper,
        sig_10pct, sig_5pct, sig_1pct, se_reliable
    """
    z_crit = _norm.ppf(1.0 - (1.0 - ci_level) / 2.0)
    diag   = np.diag(cov)
    se     = np.where(diag > 0, np.sqrt(np.abs(diag)), np.nan)
    z_stat = theta / se
    p_val  = 2.0 * (1.0 - _norm.cdf(np.abs(z_stat)))

    rows = []
    for i, name in enumerate(param_names):
        se_i  = float(se[i])
        est_i = float(theta[i])
        z_i   = float(z_stat[i])
        p_i   = float(p_val[i])
        rows.append({
            "parameter":   name,
            "estimate":    est_i,
            "SE":          se_i,
            "z_stat":      z_i,
            "p_value":     p_i,
            "ci_lower":    est_i - z_crit * se_i,
            "ci_upper":    est_i + z_crit * se_i,
            "sig_10pct":   p_i < 0.10 if np.isfinite(p_i) else False,
            "sig_5pct":    p_i < 0.05 if np.isfinite(p_i) else False,
            "sig_1pct":    p_i < 0.01 if np.isfinite(p_i) else False,
            "se_reliable": is_reliable,
        })
    return pd.DataFrame(rows)


def compute_inference(
    model,
    fit_result: dict,
    y_train: np.ndarray,
    extra_data: dict,
    eps: float = 1e-4,
    use_numerical_hessian: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, bool]:
    """
    Full inference pipeline: Hessian → covariance → inference table.

    Parameters
    ----------
    model          : CovZAGASModel or LongShortZAGASModel
    fit_result     : dict returned by model.fit()
    y_train        : training observations
    extra_data     : passed to the model's objective (X, X_long, X_short)
    eps            : finite-difference step size
    use_numerical_hessian : if False, use BFGS H^{-1} only

    Returns
    -------
    (inf_table, cov_matrix, is_reliable)
    """
    theta = fit_result["theta"]
    names = model.parameter_names()

    # Build the negative log-likelihood function for this dataset
    if hasattr(model, "_run_filter"):
        if "X_long" in extra_data and "X_short" in extra_data:
            def neg_ll(t):
                return model._run_filter(
                    t, y_train, extra_data["X_long"], extra_data["X_short"],
                    return_paths=False
                )
        elif "X" in extra_data:
            def neg_ll(t):
                return model._run_filter(t, y_train, extra_data["X"], return_paths=False)
        else:
            def neg_ll(t):
                return model._run_filter(t, y_train, return_paths=False)
    else:
        raise ValueError("model must have _run_filter method")

    cov      = None
    reliable = False

    if use_numerical_hessian:
        try:
            H = numerical_hessian(neg_ll, theta, eps=eps)
            cov, reliable = covariance_from_hessian(
                H, fallback=fit_result.get("hess_inv")
            )
        except Exception:
            cov = None

    if cov is None:
        # Fallback to BFGS Hessian inverse
        bfgs_hi = fit_result.get("hess_inv")
        if bfgs_hi is not None:
            cov, reliable = np.array(bfgs_hi), False
        else:
            cov = np.full((len(theta), len(theta)), np.nan)

    table = inference_table(theta, names, cov, reliable)
    return table, cov, reliable


def latex_inference_table(
    inf_table: pd.DataFrame,
    caption: str = "Parameter inference",
    label: str = "tab:inference",
    highlight_sig: bool = True,
) -> str:
    """
    Render inference table as LaTeX longtable (handles many rows).
    Significant coefficients (5%) are bold.
    """
    lines = [
        r"\begin{longtable}{lrrrrrr}",
        r"\caption{" + caption + r"} \label{" + label + r"} \\",
        r"\hline",
        r"Parameter & $\hat\beta$ & SE & $z$ & $p$ & CI$_{2.5\%}$ & CI$_{97.5\%}$ \\",
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        r"Parameter & $\hat\beta$ & SE & $z$ & $p$ & CI$_{2.5\%}$ & CI$_{97.5\%}$ \\",
        r"\hline",
        r"\endhead",
        r"\hline",
        r"\endfoot",
    ]

    def _fmt(x, d=4):
        if pd.isna(x) or not np.isfinite(float(x)):
            return "--"
        return f"{float(x):.{d}f}"

    def _star(row):
        if row["sig_1pct"]:   return "***"
        if row["sig_5pct"]:   return "**"
        if row["sig_10pct"]:  return "*"
        return ""

    for _, row in inf_table.iterrows():
        name  = str(row["parameter"]).replace("_", r"\_")
        est   = _fmt(row["estimate"])
        se    = _fmt(row["SE"])
        z     = _fmt(row["z_stat"])
        p     = _fmt(row["p_value"])
        cil   = _fmt(row["ci_lower"])
        ciu   = _fmt(row["ci_upper"])
        star  = _star(row)
        if highlight_sig and row["sig_5pct"] and np.isfinite(float(row.get("p_value", np.nan))):
            est = r"\textbf{" + est + "}"
        lines.append(f"  {name} & {est}{star} & {se} & {z} & {p} & {cil} & {ciu} \\\\")

    lines += [r"\end{longtable}"]
    note = (
        r"\noindent\small " + (
            "Standard errors from numerical Hessian."
            if inf_table["se_reliable"].all()
            else "Standard errors unreliable (Hessian not PD); interpret with caution."
        )
    )
    lines.append(note)
    lines.append(r"Significance: *** $p<0.01$, ** $p<0.05$, * $p<0.10$.")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Correlation / covariance matrix helpers
# ──────────────────────────────────────────────────────────────────────────────

def correlation_from_cov(cov: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix."""
    d = np.sqrt(np.abs(np.diag(cov)))
    d[d == 0] = np.nan
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def plot_cov_corr_heatmap(
    matrix: np.ndarray,
    param_names: List[str],
    title: str,
    kind: str = "corr",
    figsize: tuple = (10, 8),
) -> "plt.Figure":
    """
    Heatmap of a covariance or correlation matrix.

    kind : 'corr' → diverging RdBu centred at 0, clipped to [-1,1]
           'cov'  → sequential Blues
    """
    import matplotlib.pyplot as _plt
    import matplotlib.colors as _mc

    n = len(param_names)
    fig, ax = _plt.subplots(figsize=figsize)

    if kind == "corr":
        vmin, vmax = -1.0, 1.0
        cmap = "RdBu_r"
    else:
        v = np.nanmax(np.abs(matrix))
        vmin, vmax = -v, v
        cmap = "RdBu_r"

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    _plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [p.replace("omega_", "ω_").replace("alpha_", "α_")
              .replace("gamma_", "γ_").replace("beta_", "β_")
              for p in param_names]
    ax.set_xticklabels(short, rotation=90, fontsize=max(5, 8 - n // 10))
    ax.set_yticklabels(short, fontsize=max(5, 8 - n // 10))
    ax.set_title(title, fontsize=10, fontweight="bold")

    # Annotate cells when small enough
    if n <= 15:
        for i in range(n):
            for j in range(n):
                v = matrix[i, j]
                txt = f"{v:.2f}" if np.isfinite(v) else ""
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=6, color="black" if abs(v) < 0.7 else "white")

    fig.tight_layout()
    return fig


def latex_cov_corr_tables(
    cov: np.ndarray,
    corr: np.ndarray,
    param_names: List[str],
    caption_prefix: str = "",
    label_prefix: str = "tab",
    max_cols: int = 12,
) -> str:
    """
    LaTeX landscape longtable for covariance and correlation matrices.
    If the matrix is larger than max_cols, emit block-diagonal chunks.
    """
    n = len(param_names)

    def _f(x):
        try:
            v = float(x)
            return f"{v:.4f}" if np.isfinite(v) else "--"
        except Exception:
            return "--"

    def _matrix_table(mat, names, caption, label):
        col_spec = "l" + "r" * len(names)
        safe = [p.replace("_", r"\_") for p in names]
        header = "Parameter & " + " & ".join(safe) + r" \\"
        rows = []
        for i, rname in enumerate(names):
            rname_safe = rname.replace("_", r"\_")
            cells = [rname_safe] + [_f(mat[i, j]) for j in range(len(names))]
            rows.append(" & ".join(cells) + r" \\")
        return (
            r"\begin{landscape}" + "\n"
            r"\begin{longtable}{" + col_spec + "}\n"
            rf"\caption{{{caption}}} \label{{{label}}} \\" + "\n"
            r"\hline" + "\n" + header + "\n"
            r"\hline\endfirsthead" + "\n"
            r"\hline" + "\n" + header + "\n"
            r"\hline\endhead\hline\endfoot" + "\n"
            + "\n".join(rows) + "\n"
            r"\end{longtable}" + "\n"
            r"\end{landscape}" + "\n"
        )

    parts = []
    # Emit in chunks of max_cols
    for start in range(0, n, max_cols):
        end   = min(start + max_cols, n)
        chunk = list(range(start, end))
        sub_names = [param_names[i] for i in chunk]
        sub_cov   = cov[np.ix_(chunk, chunk)]
        sub_corr  = corr[np.ix_(chunk, chunk)]
        suffix    = f" (cols {start+1}–{end})" if n > max_cols else ""
        parts.append(_matrix_table(
            sub_cov, sub_names,
            f"{caption_prefix} covariance matrix{suffix}",
            f"{label_prefix}_cov_{start}",
        ))
        parts.append(_matrix_table(
            sub_corr, sub_names,
            f"{caption_prefix} correlation matrix{suffix}",
            f"{label_prefix}_corr_{start}",
        ))

    return "\n\n".join(parts)
