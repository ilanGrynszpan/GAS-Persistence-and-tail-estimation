"""
Shared parameter/estimate/CI/constraint table builder (2026-09-06).

=============================================================================
OVERVIEW
=============================================================================

One shared LaTeX table format, used across every stage (1-5) and every
model class in the final report, per direct specification:

    parameter | estimate | std error | 95% upper CI | 95% lower CI |
    upper constraint | lower constraint

Per-parameter classification (in priority order):
  1. Gradient-threshold gate: if |grad_i| >= GRAD_THRESHOLD (0.3, per direct
     instruction), the parameter's SE/CI cannot be trusted regardless of
     what the raw Hessian-based delta-method SE says.
       - If a boundary CONSTRAINT is plausibly active at/near this
         parameter's estimate (individual OR joint, see below): std error
         column reads "(boundary estimated)", CI columns read "-".
       - Otherwise: std error column reads "(high gradient)", CI columns
         read "-".
  2. Otherwise: ordinary delta-method SE and a symmetric 95% CI
     (estimate +/- 1.96*SE).

Constraints: each parameter's (lower, upper) constraint strings come from a
model-specific CONSTRAINT MAP (see build_constraint_map_* below), matching
the ACTUAL penalty/hard-guard code in models/gas_filter.py,
models/cov_gas_model.py, models/harvey_gas.py, and
models/threshold_weighted_xi_gas.py -- not invented. A constraint that
applies to a COMBINATION of parameters (e.g. sum_l|B_l|<0.98, or the
Harvey ordering 0<phi_S<phi_L<1) is tagged with a shared "group" key;
groups are assigned an escalating number of asterisks (*, **, ***, ...) in
first-appearance order within a table, and each distinct group gets its
own explanatory footnote below the table.

This module only builds the (rows, footnotes) content -- callers wrap it
in whatever LaTeX table environment / caption / label their report uses
(see save_constraint_table below for the default longtable wrapper).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

GRAD_THRESHOLD = 0.3


def _tex(s) -> str:
    return (str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#"))


#: GB2 distribution shape parameters (static, not GAS-dynamics or
#: zero-augmentation) -- EXCLUDED from the zero-in-CI flag by direct
#: instruction. Everything else reported in these tables (GAS recursion
#: coefficients, Harvey Long/Short parameters, seasonal parameters, the
#: occurrence/zero-augmentation dynamics rho/omega0/omega_y_*) is eligible.
GB2_DIST_PARAMS = {"gamma", "zeta", "xi", "phi"}


def _is_gb2_dist_param(name: str) -> bool:
    return name in GB2_DIST_PARAMS


def _fmt_num(v, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"{float(v):.{digits}f}"


@dataclass
class Constraint:
    lower: Optional[str] = None   # e.g. "0", "-1", None = unconstrained
    upper: Optional[str] = None   # e.g. "1", "0.98", "2", None = unconstrained
    group: Optional[str] = None   # shared key => joint/combination constraint
    group_desc: Optional[str] = None  # footnote text (same for every member of the group)
    # Unambiguous constraint SLACK: however many parameters and whatever
    # direction the actual penalty formula uses (e.g. 0.98 - sum_l|B_l|,
    # or (phi_L - phi_S) for an ordering phi_S<phi_L, or (zeta-gamma) for
    # a kurtosis gamma<zeta requirement), boundary_gap must always be
    # computed by the CALLER as "how much room is left before the
    # constraint activates" -- near zero (or negative) means pinned,
    # comfortably positive means interior. This sidesteps any ambiguity
    # about which direction/parameter the raw estimate should be compared
    # against, which a plain (value, bound) pair cannot express uniformly
    # for both "sum stays below a limit" and "gap between two things
    # stays positive" constraint shapes.
    boundary_gap: Optional[float] = None
    boundary_margin: float = 0.02   # "near" = gap <= this


ConstraintMap = Dict[str, Constraint]


def _is_near_boundary(estimate: float, c: Constraint) -> bool:
    """
    If boundary_gap is set (joint constraints, or any individual
    constraint the caller prefers to compute explicitly), proximity is
    simply gap <= margin. Otherwise, for a plain individual box
    constraint, proximity is checked directly against whichever of
    lower/upper is a literal finite bound (not a reference to another
    parameter's name).
    """
    if c.boundary_gap is not None:
        return c.boundary_gap <= c.boundary_margin
    for bound_str in (c.upper, c.lower):
        if bound_str is None:
            continue
        try:
            bound = float(bound_str)
        except ValueError:
            continue  # e.g. "phi_L_phi" -- a reference, not a literal number
        if abs(estimate - bound) <= c.boundary_margin:
            return True
    return False


def build_rows(
    names: Sequence[str],
    estimates: Sequence[float],
    gradients: Sequence[float],
    raw_variances: Sequence[Optional[float]],
    derivs: Sequence[float],
    constraints: ConstraintMap,
    pretty_name: Callable[[str], str],
    grad_threshold: float = GRAD_THRESHOLD,
) -> Tuple[List[dict], List[Tuple[str, str]]]:
    """
    Returns (rows, group_footnotes, zero_ci_used).

    rows: list of {parameter, estimate, se, ci_lo, ci_hi, lower, upper}
          (all already-formatted strings except parameter, which is the
          pretty LaTeX symbol).
    group_footnotes: [(marker, description)] in first-appearance order,
                      marker = "**", "***", "****", ... (a plain single "*"
                      is reserved for the zero-in-CI flag on the estimate
                      column -- see zero_ci_used below -- so combination-
                      constraint groups start at two asterisks).
    zero_ci_used: True if at least one row's estimate was flagged with a
                  single "*" (a correctly-inferred 95% CI, i.e. NOT a
                  boundary/high-gradient case, that contains zero, for a
                  GAS dynamics or zero-augmentation parameter -- NOT a
                  GB2 distribution parameter). Callers use this to decide
                  whether to print the "* estimated parameter cannot be
                  statistically rejected to equal zero" footnote.
    """
    group_marker: Dict[str, str] = {}
    footnotes: List[Tuple[str, str]] = []
    zero_ci_used = False

    def marker_for(group: Optional[str], desc: Optional[str]) -> str:
        if group is None:
            return ""
        if group not in group_marker:
            # Combination-constraint groups start at TWO asterisks -- a
            # single "*" is reserved for the zero-in-CI estimate flag
            # below, so the two marker families never collide.
            m = "*" * (len(group_marker) + 2)
            group_marker[group] = m
            footnotes.append((m, desc or ""))
        return group_marker[group]

    rows = []
    for name, est, g, v_raw, deriv in zip(names, estimates, gradients, raw_variances, derivs):
        c = constraints.get(name, Constraint())
        m = marker_for(c.group, c.group_desc)
        if m:
            # A joint/combination constraint: the cell carries ONLY the
            # marker -- the actual bound/relationship (which parameter(s)
            # it involves, what the numeric limit is) belongs in the
            # footnote, not repeated in every member's cell.
            lower_disp = m if c.lower is not None else "--"
            upper_disp = m if c.upper is not None else "--"
        else:
            lower_disp = c.lower if c.lower is not None else "--"
            upper_disp = c.upper if c.upper is not None else "--"

        # Boundary proximity is checked FIRST and independently of the
        # gradient: a parameter pinned against a soft penalty can converge
        # to a genuinely tiny raw gradient (the penalty's own gradient
        # exactly cancels the likelihood's there), but the naive
        # delta-method SE is still invalid at a constrained optimum --
        # only an interior, unconstrained optimum gives that SE its usual
        # meaning. Only when NOT near a boundary does the plain gradient
        # threshold decide between an ordinary SE/CI and "(high gradient)".
        near = _is_near_boundary(est, c)
        g_ok = np.isfinite(g) and abs(g) < grad_threshold
        if near:
            se_disp = "(boundary estimated)"
            ci_lo_disp = ci_hi_disp = "-"
        elif not g_ok:
            se_disp = "(high gradient)"
            ci_lo_disp = ci_hi_disp = "-"
        else:
            se = abs(deriv) * np.sqrt(v_raw) if (v_raw is not None and np.isfinite(v_raw) and v_raw > 0) else None
            if se is None:
                se_disp = "(raw SE invalid)"
                ci_lo_disp = ci_hi_disp = "-"
            else:
                se_disp = _fmt_num(se)
                ci_lo, ci_hi = est - 1.96 * se, est + 1.96 * se
                ci_lo_disp, ci_hi_disp = _fmt_num(ci_lo), _fmt_num(ci_hi)

        estimate_disp = _fmt_num(est)
        # Zero-in-CI flag: only meaningful when the CI was actually,
        # correctly inferred (not a boundary/high-gradient placeholder),
        # and only for GAS dynamics / zero-augmentation parameters -- a
        # GB2 distribution shape parameter (gamma/zeta/xi/phi) is excluded
        # by direct instruction. Uses a plain single "*", distinct from
        # the "**", "***", ... family reserved for combination constraints.
        if near_or_high := (near or not g_ok):
            pass  # CI columns are placeholders ("-"); no zero-in-CI check applies
        elif ci_lo_disp not in ("-", "--") and not _is_gb2_dist_param(name):
            try:
                if float(ci_lo_disp) <= 0.0 <= float(ci_hi_disp):
                    estimate_disp += "*"
                    zero_ci_used = True
            except ValueError:
                pass

        rows.append({
            "parameter": pretty_name(name),
            "estimate": estimate_disp,
            "se": se_disp,
            "ci_lo": ci_lo_disp,
            "ci_hi": ci_hi_disp,
            "lower": lower_disp,
            "upper": upper_disp,
            "_grad": g, "_gname": name,
        })
    return rows, footnotes, zero_ci_used


def pretty_name_generic(name: str) -> str:
    """
    One shared LaTeX-symbol mapping covering every model class in the
    final report (ZAGASModel/CovZAGASModel/HarveyZAGASModel/
    ThresholdWeightedXiModel). Falls back to a plain escaped name for
    anything unrecognized (covariate names, etc.) rather than guessing.
    """
    if name in ("gamma", "zeta", "xi", "phi"):
        return {"gamma": r"$\gamma$", "zeta": r"$\zeta$", "xi": r"$\xi$", "phi": r"$\varphi$"}[name]
    if name in ("omega0", "rho"):
        return {"omega0": r"$\omega_0$", "rho": r"$\rho$"}[name]
    m = re.match(r"^omega_y_(\d+)$", name)
    if m:
        return rf"$\omega_{{y,{m.group(1)}}}$"
    m = re.match(r"^(A|B)_(phi|xi)_(\d+)$", name)
    if m:
        block, j, lag = m.groups()
        return rf"${block}_{{{j},{lag}}}$"
    m = re.match(r"^(omega|f0|L0|S0|kappa_L|kappa_S|phi_L|phi_S|psi_gamma|kappa_gamma)_(phi|xi)$", name)
    if m:
        block, j = m.groups()
        sym = {"omega": r"\omega", "f0": "f_0", "L0": "L_0", "S0": "S_0",
               "kappa_L": r"\kappa_L", "kappa_S": r"\kappa_S", "kappa_gamma": r"\kappa_\gamma",
               "phi_L": r"\varphi_L", "phi_S": r"\varphi_S", "psi_gamma": r"\psi_\gamma"}[block]
        return f"${{{sym}}}_{{{j}}}$"
    m = re.match(r"^seas0_(phi|xi)_(\d+)$", name)
    if m:
        return rf"$\gamma_{{1|0,{m.group(2)}}}$"
    m = re.match(r"^seasstar0_(phi|xi)_(\d+)$", name)
    if m:
        return rf"$\gamma^{{*}}_{{1|0,{m.group(2)}}}$"
    m = re.match(r"^Gamma_(L|S)_(phi|xi)_(.+)$", name)
    if m:
        comp, j, cov = m.groups()
        return f"$c_{{{comp}}}$ ({_tex(cov)})"
    m = re.match(r"^gamma_(phi|xi)_(.+)$", name)  # CovZAGASModel covariate coefficients
    if m:
        j, cov = m.groups()
        return f"$c_{{{j}}}$ ({_tex(cov)})"
    if name in ("A_xi_1",):
        return r"$A_\xi$"
    if name in ("B_xi_1",):
        return r"$B_\xi$"
    if name in ("omega_xi", "f0_xi"):
        return {"omega_xi": r"$\omega_\xi$", "f0_xi": r"$f_{0,\xi}$"}[name]
    return _tex(name)


# ─────────────────────────────────────────────────────────────────────────────
# Model-specific constraint maps -- built directly from the ACTUAL penalty
# formulas in each model's own _run_filter (see file:line references in
# each docstring), not invented. Every function returns {param_name:
# Constraint}; callers merge dicts across static/pi/dynamic blocks.
# ─────────────────────────────────────────────────────────────────────────────

KURTOSIS_DESC = "Kurtosis-existence requirement ($\\gamma<\\zeta$), enforced jointly on the two static GB2 shape parameters (models/gas_filter.py: hard reject if $\\gamma\\ge\\zeta$, soft penalty below a 0.5 margin)."


def constraints_static(est: pd.Series) -> ConstraintMap:
    """gamma/zeta joint kurtosis constraint -- shared by every model class
    in this report (all wrap GASFilter's static-parameter handling)."""
    cm: ConstraintMap = {}
    if "gamma" in est.index and "zeta" in est.index:
        gap = float(est["zeta"] - est["gamma"])
        for p, other in (("gamma", "zeta"), ("zeta", "gamma")):
            cm[p] = Constraint(lower="0" if p == "gamma" else other, upper=other if p == "gamma" else None,
                                group="gz", group_desc=KURTOSIS_DESC, boundary_gap=gap, boundary_margin=0.5)
    return cm


def constraints_pi(est: pd.Series) -> ConstraintMap:
    """Occurrence dynamics: rho individually bounded |rho|<0.98
    (models/gas_filter.py / cov_gas_model.py / harvey_gas.py -- identical
    pi_dynamics module shared by all). omega0/omega_y_* unconstrained."""
    cm: ConstraintMap = {}
    if "rho" in est.index:
        cm["rho"] = Constraint(lower="-0.98", upper="0.98")
    return cm


def constraints_gas_dynamics(est: pd.Series, tv_names: Sequence[str], lags: Sequence[int]) -> ConstraintMap:
    """Standard GAS(1,1)-or-richer dynamics (ZAGASModel/CovZAGASModel):
    per tv_name j, sum_l|A_{j,l}|<2 and sum_l|B_{j,l}|<0.98, each a JOINT
    constraint across every lag of that same target (models/gas_filter.py
    lines ~726-733)."""
    cm: ConstraintMap = {}
    for j in tv_names:
        a_names = [f"A_{j}_{l}" for l in lags if f"A_{j}_{l}" in est.index]
        b_names = [f"B_{j}_{l}" for l in lags if f"B_{j}_{l}" in est.index]
        a_sum = float(np.sum(np.abs(est[a_names]))) if a_names else 0.0
        b_sum = float(np.sum(np.abs(est[b_names]))) if b_names else 0.0
        a_desc = rf"Joint score-amplitude bound $\sum_\ell|A_{{{j},\ell}}|<2$ across every lag of ${j}$'s own GAS recursion."
        b_desc = rf"Joint stationarity bound $\sum_\ell|B_{{{j},\ell}}|<0.98$ across every lag of ${j}$'s own GAS recursion."
        for n in a_names:
            cm[n] = Constraint(lower=None, upper="2 (sum, abs)", group=f"asum_{j}", group_desc=a_desc,
                                boundary_gap=2.0 - a_sum, boundary_margin=0.05)
        for n in b_names:
            cm[n] = Constraint(lower=None, upper="0.98 (sum, abs)", group=f"bsum_{j}", group_desc=b_desc,
                                boundary_gap=0.98 - b_sum, boundary_margin=0.02)
    return cm


def constraints_cov(est: pd.Series, tv_names: Sequence[str], cov_names: Sequence[str]) -> ConstraintMap:
    """CovZAGASModel's covariate-coefficient vector: joint L2-norm bound
    ||gamma_j||<20 per target j (models/cov_gas_model.py line ~508-509)."""
    cm: ConstraintMap = {}
    for j in tv_names:
        names_j = [f"gamma_{j}_{c}" for c in cov_names if f"gamma_{j}_{c}" in est.index]
        if not names_j:
            continue
        norm = float(np.linalg.norm(est[names_j]))
        desc = rf"Joint $L_2$-norm bound $\|c_{{{j}}}\|<20$ across every covariate coefficient for ${j}$."
        for n in names_j:
            cm[n] = Constraint(lower=None, upper="20 (L2 norm)", group=f"cov_{j}", group_desc=desc,
                                boundary_gap=20.0 - norm, boundary_margin=1.0)
    return cm


def constraints_harvey(est: pd.Series, harvey_target: Sequence[str], long_names: Sequence[str],
                        short_names: Sequence[str], restrict_persistence_upper: bool,
                        use_auxiliary_penalties: bool, n_harmonics: int,
                        kappa_gamma_bound: Optional[float], seasonal_damping_max: Optional[float]) -> ConstraintMap:
    """Harvey long-short (+ optional seasonal) dynamics
    (models/harvey_gas.py lines ~1105-1145). Per target j:
      phi_L: 0 < phi_L (< 0.995 if restrict_persistence_upper) -- individual.
      phi_S: 0 < phi_S < phi_L -- lower bound individual, upper bound is
             the JOINT ordering constraint with phi_L.
      kappa_L, kappa_S: |.|<2 individually, only if use_auxiliary_penalties.
      Gamma_L, Gamma_S: joint L2-norm<20 per target, only if
             use_auxiliary_penalties.
      kappa_gamma: unconstrained unless kappa_gamma_bound is set (then
             symmetric -bound<.<bound, individual).
      psi_gamma: 0<.<seasonal_damping_max (individual), only present when
             seasonal_damping_max is set.
      seas0/seasstar0: unconstrained (free initial conditions).
    """
    cm: ConstraintMap = {}
    ORDER_DESC = r"Persistence ordering $0<\varphi_S<\varphi_L$, enforced jointly across the Long and Short components for the same target."
    for j in harvey_target:
        phi_L_name, phi_S_name = f"phi_L_{j}", f"phi_S_{j}"
        if phi_L_name in est.index and phi_S_name in est.index:
            phi_L, phi_S = float(est[phi_L_name]), float(est[phi_S_name])
            upper_L = "0.995" if restrict_persistence_upper else None
            cm[phi_L_name] = Constraint(lower="0", upper=upper_L,
                                         boundary_gap=(0.995 - phi_L) if restrict_persistence_upper else None,
                                         boundary_margin=0.005)
            cm[phi_S_name] = Constraint(lower="0", upper=f"varphi_L_{j}", group=f"order_{j}", group_desc=ORDER_DESC,
                                         boundary_gap=phi_L - phi_S, boundary_margin=0.02)
        if use_auxiliary_penalties:
            for block in ("kappa_L", "kappa_S"):
                n = f"{block}_{j}"
                if n in est.index:
                    cm[n] = Constraint(lower="-2", upper="2", boundary_gap=2.0 - abs(float(est[n])), boundary_margin=0.05)
            for comp, names_list in (("L", long_names), ("S", short_names)):
                cov_names_j = [f"Gamma_{comp}_{j}_{c}" for c in names_list if f"Gamma_{comp}_{j}_{c}" in est.index]
                if not cov_names_j:
                    continue
                norm = float(np.linalg.norm(est[cov_names_j]))
                desc = rf"Joint $L_2$-norm bound $\|c_{{{comp}}}\|<20$ across every {('Long' if comp=='L' else 'Short')} covariate coefficient for ${j}$."
                for n in cov_names_j:
                    cm[n] = Constraint(lower=None, upper="20 (L2 norm)", group=f"cov{comp}_{j}", group_desc=desc,
                                        boundary_gap=20.0 - norm, boundary_margin=1.0)
        if n_harmonics > 0:
            kg_name = f"kappa_gamma_{j}"
            if kappa_gamma_bound is not None and kg_name in est.index:
                kg = float(est[kg_name])
                cm[kg_name] = Constraint(lower=f"-{kappa_gamma_bound:g}", upper=f"{kappa_gamma_bound:g}",
                                          boundary_gap=kappa_gamma_bound - abs(kg), boundary_margin=0.05)
            if seasonal_damping_max is not None:
                psi_name = f"psi_gamma_{j}"
                if psi_name in est.index:
                    psi = float(est[psi_name])
                    cm[psi_name] = Constraint(lower="0", upper=f"{seasonal_damping_max:g}",
                                               boundary_gap=seasonal_damping_max - psi, boundary_margin=0.005)
    return cm


def constraints_threshxi(est: pd.Series) -> ConstraintMap:
    """Stage 5 threshold-weighted xi (models/threshold_weighted_xi_gas.py
    lines ~199-201): |B_xi|<0.98 and |A_xi|<2, both individual (GAS_LAG=1,
    so there is only one lag -- no sum across multiple parameters)."""
    cm: ConstraintMap = {}
    if "A_xi_1" in est.index:
        a = float(est["A_xi_1"])
        cm["A_xi_1"] = Constraint(lower="-2", upper="2", boundary_gap=2.0 - abs(a), boundary_margin=0.05)
    if "B_xi_1" in est.index:
        b = float(est["B_xi_1"])
        cm["B_xi_1"] = Constraint(lower="-0.98", upper="0.98", boundary_gap=0.98 - abs(b), boundary_margin=0.02)
    return cm


def diagnose_high_gradient_rows(rows: List[dict], grad_threshold: float = GRAD_THRESHOLD,
                                 pretty_name: Optional[Callable[[str], str]] = None,
                                 corr_check: Optional[Callable[[str], Optional[Tuple[str, float]]]] = None) -> str:
    """
    Generic concentrated-vs-generalized high-gradient diagnosis (mirrors
    generate_harmonic_seasonal_report.py's diagnose_high_gradient, made
    model-agnostic). corr_check(param_name) -> (other_param_pretty, corr)
    for the single most-correlated other parameter, or None -- typically a
    closure over that model's own hess_inv, used to report WHY a high
    gradient occurred when it is plausibly a collinearity effect.
    """
    grads = {r["_gname"]: r["_grad"] for r in rows if np.isfinite(r["_grad"])}
    if not grads:
        return ""
    top_name, top_val = max(grads.items(), key=lambda kv: abs(kv[1]))
    top_abs = abs(top_val)
    if top_abs < grad_threshold:
        return ""
    n_large = sum(1 for v in grads.values() if abs(v) > 0.1 * top_abs)
    n_total = len(grads)
    frac = n_large / n_total
    pn = pretty_name or (lambda x: x)
    top5 = sorted(grads.items(), key=lambda kv: -abs(kv[1]))[:5]
    lines = ["Top gradients: " + ", ".join(f"{pn(p)}={v:.2f}" for p, v in top5) + "."]
    if frac <= 0.15:
        lines.append(f"CONCENTRATED in {n_large}/{n_total} parameter(s) -- consistent with a single boundary pin.")
    else:
        lines.append(f"GENERALIZED: {n_large}/{n_total} parameters ({100*frac:.0f}\\%) are within an order of "
                      f"magnitude of the worst -- consistent with a broader identification issue, not one pin.")
    if corr_check is not None:
        res = corr_check(top_name)
        if res is not None:
            other, corr = res
            lines.append(f"{pn(top_name)} is strongly correlated with {other} (implied corr.\\ from hess\\_inv "
                          f"$={corr:.2f}$), consistent with the gradient being driven by mutual collinearity "
                          f"rather than either parameter individually.")
    return " ".join(lines)


ZERO_CI_FOOTNOTE = "estimated parameter cannot be statistically rejected to equal zero"


def save_constraint_table(rows: List[dict], footnotes: List[Tuple[str, str]],
                           caption: str, label: str, diagnosis: str = "",
                           zero_ci_used: bool = False) -> str:
    header = r"Parameter & Estimate & Std.\ Error & 95\% CI (lower) & 95\% CI (upper) & Lower Constraint & Upper Constraint \\"
    body_lines = []
    for r in rows:
        body_lines.append(f"{r['parameter']} & {r['estimate']} & {r['se']} & {r['ci_lo']} & {r['ci_hi']} & "
                           f"{r['lower']} & {r['upper']} \\\\")
    body = "\n".join(body_lines)
    table = "\n".join([
        r"\begin{longtable}{lrrrrrr}",
        rf"\caption{{{caption}}}\label{{{label}}} \\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\bottomrule", r"\endfoot",
        body,
        r"\end{longtable}",
    ])
    parts = [table]
    fn_lines = []
    if zero_ci_used:
        fn_lines.append(rf"* {ZERO_CI_FOOTNOTE}")
    fn_lines += [rf"{m} {_tex(desc)}" for m, desc in footnotes]
    if fn_lines:
        parts.append(r"\smallskip\noindent\footnotesize " + " \\quad ".join(fn_lines))
    if diagnosis:
        parts.append(rf"\textit{{Gradient diagnosis:}} {diagnosis}")
    return "\n\n".join(parts)
