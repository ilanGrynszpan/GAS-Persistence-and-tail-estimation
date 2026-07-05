"""
Numba-compiled inner loop for the ZA-GAS GB2 filter.

Provides ~50-100x speedup over the equivalent pure-Python loop by:
  1. Compiling the filter loop to native code with @njit.
  2. Replacing scipy.special calls (gammaln, betaln, digamma) with
     math.lgamma and an asymptotic digamma approximation, both of which
     are available inside numba's nopython mode.

Two entry-point functions are exported:

    nb_filter_phi_only(...)  --  phi time-varying; xi/gamma/zeta static
    nb_filter_phi_xi(...)    --  phi and xi time-varying; gamma/zeta static

Both return (loglik, f_arr, s_arr, eta_arr, pi_arr) or indicate failure
via loglik = -1e12.

Penalty computation stays in Python (cheap, avoids numba complexity).
Scaling modes: 0 = unit, 1 = diagonal_inverse_fisher, 2 = inverse_fisher.
For phi-only, modes 1 and 2 are equivalent (1×1 matrix).

=============================================================================
FIRST-CALL COMPILATION
=============================================================================

Numba compiles specialised machine code on the first call for each unique
combination of argument dtypes and array shapes.  Expect 5-20 s on the
first call for each (n_gas_lags, n_pi_lags) combination; all subsequent
calls are fast.  The thesis uses at most 2-3 distinct lag combinations so
the compilation cost is paid at most 3 times.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit


# ─────────────────────────────────────────────────────────────────────────────
# Special-function building blocks (all @njit)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _log_sigmoid(x: float) -> float:
    """Numerically stable log σ(x) = log(1 / (1 + exp(-x)))."""
    if x >= 0.0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


@njit(cache=True)
def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


@njit(cache=True)
def _betaln(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


@njit(cache=True)
def _digamma(x: float) -> float:
    """
    Digamma ψ(x) approximation via recurrence + asymptotic expansion.

    Accurate to ~12 significant figures for x > 0.
    Algorithm: shift x > 6 using ψ(x) = ψ(x+1) - 1/x, then apply the
    asymptotic series (Abramowitz & Stegun 6.3.18).

    Guard: when a = exp(xi) underflows to 0.0 during BFGS line search,
    ψ(0+) → -∞.  Return -1e308 so the score evaluates to a finite
    (large-negative) number rather than raising ZeroDivisionError.
    """
    if x <= 0.0:
        return -1.0e308
    result = 0.0
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    result += (
        math.log(x)
        - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
    )
    return result


@njit(cache=True)
def _trigamma(x: float) -> float:
    """
    Trigamma ψ₁(x) = d²logΓ/dx² via recurrence + asymptotic expansion.

    Used for the Fisher information of xi.  Accurate to ~10 significant
    figures for x > 0.  Asymptotic: A&S 6.4.12.

    Guard: ψ₁(0+) → +∞.  Return 1e308 to avoid ZeroDivisionError when
    a = exp(xi) underflows to 0.0 during BFGS line search.
    """
    if x <= 0.0:
        return 1.0e308
    result = 0.0
    while x < 6.0:
        result += 1.0 / (x * x)
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    result += (
        inv
        + 0.5 * inv2
        + inv2 * inv / 6.0
        - inv2 * inv2 * inv / 30.0
        + inv2 * inv2 * inv2 * inv / 42.0
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GB2 log-pdf, score, Fisher information (all @njit, scalar inputs)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _gb2_logpdf(y: float, phi: float, xi: float,
                gamma: float, zeta: float) -> float:
    """GB2 log-pdf at observation y.

    Guard: when phi << -745, sigma = exp(phi) underflows to 0.0 in IEEE 754.
    In numba nopython mode y/0.0 raises ZeroDivisionError (unlike numpy which
    returns inf).  Return -inf so the filter's isfinite check triggers the
    -1e12 penalty path rather than crashing the optimizer.
    """
    sigma = math.exp(phi)
    if sigma <= 0.0:
        return -math.inf
    a     = math.exp(xi)
    b     = math.exp(zeta)
    p     = math.exp(-gamma)
    z     = (y / sigma) ** p
    ln_beta = _betaln(a, b)
    return (
        math.log(p)
        - math.log(sigma)
        + (a * p - 1.0) * math.log(y / sigma)
        - ln_beta
        - (a + b) * math.log(1.0 + z)
    )


@njit(cache=True)
def _gb2_score_phi(y: float, phi: float, xi: float,
                   gamma: float, zeta: float) -> float:
    """∂logf/∂phi = p*((a+b)*w - a)   where w = (y/σ)^p / (1+(y/σ)^p).

    Guard: sigma underflow → as phi→-∞ the limit is w→1, score→p*b.
    The filter exits via _gb2_logpdf returning -inf before reaching here,
    so this guard is defensive only.
    """
    sigma = math.exp(phi)
    if sigma <= 0.0:
        b = math.exp(zeta)
        p = math.exp(-gamma)
        return p * b   # limit as phi → -∞
    a     = math.exp(xi)
    b     = math.exp(zeta)
    p     = math.exp(-gamma)
    z     = (y / sigma) ** p
    w     = z / (1.0 + z)
    return p * ((a + b) * w - a)


@njit(cache=True)
def _gb2_score_xi(y: float, phi: float, xi: float,
                  gamma: float, zeta: float) -> float:
    """∂logf/∂xi = a*(p*log(y/σ) - ψ(a) + ψ(a+b) - log(1+z)).

    Guard: sigma underflow → leading terms p*log(y/σ) and log(1+z)≈p*log(y/σ)
    cancel, so the limit is 0.  Defensive guard only (filter exits earlier).
    """
    sigma = math.exp(phi)
    if sigma <= 0.0:
        return 0.0   # limit as phi → -∞
    a     = math.exp(xi)
    b     = math.exp(zeta)
    p     = math.exp(-gamma)
    z     = (y / sigma) ** p
    logy  = math.log(y / sigma)
    return a * (p * logy - _digamma(a) + _digamma(a + b) - math.log(1.0 + z))


@njit(cache=True)
def _gb2_fi_phi(xi: float, gamma: float, zeta: float) -> float:
    """I_{phi,phi} = p^2 * a * b / (a+b+1)."""
    a = math.exp(xi)
    b = math.exp(zeta)
    p = math.exp(-gamma)
    return p * p * a * b / (a + b + 1.0)


@njit(cache=True)
def _gb2_fi_xi(xi: float, zeta: float) -> float:
    """I_{xi,xi} = a^2 * [ψ₁(a) - ψ₁(a+b)]."""
    a = math.exp(xi)
    b = math.exp(zeta)
    return a * a * (_trigamma(a) - _trigamma(a + b))


@njit(cache=True)
def _gb2_fi_cross(xi: float, gamma: float, zeta: float) -> float:
    """I_{phi,xi} = +p * a * b / (a+b).

    Derived from Cov(∂ℓ/∂phi, ∂ℓ/∂xi) = sigma·a·(p/sigma)·b/(a+b)
    using u = (y/sigma)^p/(1+(y/sigma)^p) ~ Beta(a,b) and
    Cov(u, log u) = b/(a+b)^2.  Note denominator is (a+b), not (a+b+1).
    """
    a = math.exp(xi)
    b = math.exp(zeta)
    p = math.exp(-gamma)
    return p * a * b / (a + b)


# ─────────────────────────────────────────────────────────────────────────────
# Compiled filter: phi-only variant
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def nb_filter_phi_only(
    y:          np.ndarray,   # (T,)   observations
    f0:         float,        # initial state for phi
    omega:      float,        # GAS intercept for phi
    A:          np.ndarray,   # (n_gas_lags,)  score loadings
    B:          np.ndarray,   # (n_gas_lags,)  persistence
    lags_gas:   np.ndarray,   # (n_gas_lags,)  int64
    xi_s:       float,        # static xi
    gamma_s:    float,        # static gamma
    zeta_s:     float,        # static zeta
    pi_omega0:  float,        # pi intercept
    pi_rho:     float,        # pi AR coefficient
    pi_wy:      np.ndarray,   # (n_pi_lags,) pi y-lag coefficients
    lags_pi:    np.ndarray,   # (n_pi_lags,)  int64
    max_lag:    int,
    scaling:    int,          # 0=unit, 1=diag_FI (2=full_FI same for 1-param)
    lam_state:  float,        # penalty weight for large states
    state_lim:  float,        # STATE_LIMIT (15.0)
    lam_eta:    float,        # penalty weight for large eta
    eta_lim:    float,        # ETA_LIMIT (30.0)
) -> tuple:
    """
    GAS filter for phi-only ZA-GB2 model, compiled by numba.

    Returns (loglik, penalty, f_arr, s_arr, eta_arr, pi_arr).
    Returns loglik = -1e12 on numerical failure.
    """
    T       = len(y)
    n_lgas  = len(lags_gas)
    n_lpi   = len(lags_pi)

    f_arr   = np.zeros(T + 1)
    s_arr   = np.zeros(T)
    eta_arr = np.zeros(T)
    pi_arr  = np.zeros(T)

    for k in range(max_lag + 1):
        f_arr[k] = f0

    loglik  = 0.0
    penalty = 0.0

    for t in range(max_lag, T):
        phi_t = f_arr[t]

        # ── State penalty ────────────────────────────────────────────────────
        excess = abs(phi_t) - state_lim
        if excess > 0.0:
            penalty += lam_state * excess * excess

        # ── Pi dynamics ──────────────────────────────────────────────────────
        i = t - max_lag
        eta_prev = eta_arr[i - 1] if i > 0 else 0.0
        eta_t = pi_omega0 + pi_rho * eta_prev
        for k in range(n_lpi):
            t_lag = t - lags_pi[k]
            y_lag = y[t_lag] if t_lag >= 0 else 0.0
            eta_t += pi_wy[k] * y_lag
        eta_arr[i] = eta_t

        # ── Eta penalty ──────────────────────────────────────────────────────
        excess_eta = abs(eta_t) - eta_lim
        if excess_eta > 0.0:
            penalty += lam_eta * excess_eta * excess_eta

        pi_t = _sigmoid(eta_t)
        pi_arr[i] = pi_t

        # ── Log-likelihood ───────────────────────────────────────────────────
        if y[t] == 0.0:
            loglik += _log_sigmoid(-eta_t)   # log(1 - pi_t)
        else:
            ll_dist = _gb2_logpdf(y[t], phi_t, xi_s, gamma_s, zeta_s)
            if not math.isfinite(ll_dist):
                return -1e12, penalty, f_arr, s_arr, eta_arr, pi_arr
            ll_t = _log_sigmoid(eta_t) + ll_dist
            if not math.isfinite(ll_t):
                return -1e12, penalty, f_arr, s_arr, eta_arr, pi_arr
            loglik += ll_t

        # ── Score ────────────────────────────────────────────────────────────
        s_t = 0.0
        if y[t] > 0.0:
            s_raw = _gb2_score_phi(y[t], phi_t, xi_s, gamma_s, zeta_s)
            if scaling == 1:
                fi = _gb2_fi_phi(xi_s, gamma_s, zeta_s)
                if fi > 0.0 and math.isfinite(fi):
                    s_t = s_raw / fi
                    if not math.isfinite(s_t):
                        s_t = s_raw
                else:
                    s_t = s_raw
            else:
                s_t = s_raw
        s_arr[t] = s_t

        # ── GAS update ───────────────────────────────────────────────────────
        score_p = 0.0
        ar_p    = 0.0
        for k in range(n_lgas):
            l   = lags_gas[k]
            idx = t - l + 1
            s_past = s_arr[idx] if idx >= 0 else 0.0
            f_past = f_arr[idx] if idx >= 0 else f0
            score_p += A[k] * s_past
            ar_p    += B[k] * f_past
        f_next = omega + score_p + ar_p
        if not math.isfinite(f_next):
            return -1e12, penalty, f_arr, s_arr, eta_arr, pi_arr
        f_arr[t + 1] = f_next

    return loglik, penalty, f_arr, s_arr, eta_arr, pi_arr


# ─────────────────────────────────────────────────────────────────────────────
# Compiled filter: phi + xi variant
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def nb_filter_phi_xi(
    y:          np.ndarray,   # (T,)
    f0_phi:     float,
    f0_xi:      float,
    omega_phi:  float,
    omega_xi:   float,
    A_phi:      np.ndarray,   # (n_gas_lags,)
    A_xi:       np.ndarray,   # (n_gas_lags,)
    B_phi:      np.ndarray,   # (n_gas_lags,)
    B_xi:       np.ndarray,   # (n_gas_lags,)
    lags_gas:   np.ndarray,
    gamma_s:    float,
    zeta_s:     float,
    pi_omega0:  float,
    pi_rho:     float,
    pi_wy:      np.ndarray,
    lags_pi:    np.ndarray,
    max_lag:    int,
    scaling:    int,
    lam_state:  float,
    state_lim:  float,
    lam_eta:    float,
    eta_lim:    float,
) -> tuple:
    """GAS filter for phi+xi ZA-GB2 model, compiled by numba."""
    T      = len(y)
    n_lgas = len(lags_gas)
    n_lpi  = len(lags_pi)

    f_phi   = np.zeros(T + 1)
    f_xi    = np.zeros(T + 1)
    s_phi   = np.zeros(T)
    s_xi    = np.zeros(T)
    eta_arr = np.zeros(T)
    pi_arr  = np.zeros(T)

    for k in range(max_lag + 1):
        f_phi[k] = f0_phi
        f_xi[k]  = f0_xi

    loglik  = 0.0
    penalty = 0.0

    for t in range(max_lag, T):
        phi_t = f_phi[t]
        xi_t  = f_xi[t]

        # ── State penalties ──────────────────────────────────────────────────
        for fval in (phi_t, xi_t):
            ex = abs(fval) - state_lim
            if ex > 0.0:
                penalty += lam_state * ex * ex

        # ── Pi dynamics ──────────────────────────────────────────────────────
        i = t - max_lag
        eta_prev = eta_arr[i - 1] if i > 0 else 0.0
        eta_t = pi_omega0 + pi_rho * eta_prev
        for k in range(n_lpi):
            t_lag = t - lags_pi[k]
            y_lag = y[t_lag] if t_lag >= 0 else 0.0
            eta_t += pi_wy[k] * y_lag
        eta_arr[i] = eta_t

        ex_eta = abs(eta_t) - eta_lim
        if ex_eta > 0.0:
            penalty += lam_eta * ex_eta * ex_eta

        pi_t = _sigmoid(eta_t)
        pi_arr[i] = pi_t

        # ── Log-likelihood ───────────────────────────────────────────────────
        if y[t] == 0.0:
            loglik += _log_sigmoid(-eta_t)
        else:
            ll_dist = _gb2_logpdf(y[t], phi_t, xi_t, gamma_s, zeta_s)
            if not math.isfinite(ll_dist):
                return -1e12, penalty, f_phi, f_xi, s_phi, s_xi, eta_arr, pi_arr
            ll_t = _log_sigmoid(eta_t) + ll_dist
            if not math.isfinite(ll_t):
                return -1e12, penalty, f_phi, f_xi, s_phi, s_xi, eta_arr, pi_arr
            loglik += ll_t

        # ── Score + scaling ──────────────────────────────────────────────────
        sp_phi = 0.0
        sp_xi  = 0.0
        if y[t] > 0.0:
            raw_phi = _gb2_score_phi(y[t], phi_t, xi_t, gamma_s, zeta_s)
            raw_xi  = _gb2_score_xi( y[t], phi_t, xi_t, gamma_s, zeta_s)

            if scaling == 0:  # unit
                sp_phi = raw_phi
                sp_xi  = raw_xi

            elif scaling == 1:  # diagonal inverse FI
                fi_p = _gb2_fi_phi(xi_t, gamma_s, zeta_s)
                fi_x = _gb2_fi_xi( xi_t, zeta_s)
                if fi_p > 0.0 and math.isfinite(fi_p):
                    sp_phi = raw_phi / fi_p
                else:
                    sp_phi = raw_phi
                if fi_x > 0.0 and math.isfinite(fi_x):
                    sp_xi = raw_xi / fi_x
                else:
                    sp_xi = raw_xi

            else:  # scaling == 2: full inverse FI (2x2)
                fi_pp = _gb2_fi_phi(xi_t, gamma_s, zeta_s)
                fi_xx = _gb2_fi_xi( xi_t, zeta_s)
                fi_px = _gb2_fi_cross(xi_t, gamma_s, zeta_s)
                det   = fi_pp * fi_xx - fi_px * fi_px
                if det > 1e-12 and math.isfinite(det):
                    # I^{-1} * grad
                    sp_phi = ( fi_xx * raw_phi - fi_px * raw_xi) / det
                    sp_xi  = (-fi_px * raw_phi + fi_pp * raw_xi) / det
                    if not (math.isfinite(sp_phi) and math.isfinite(sp_xi)):
                        # diagonal fallback
                        sp_phi = raw_phi / fi_pp if fi_pp > 0 else raw_phi
                        sp_xi  = raw_xi  / fi_xx if fi_xx > 0 else raw_xi
                else:
                    # unit fallback
                    sp_phi = raw_phi
                    sp_xi  = raw_xi

        s_phi[t] = sp_phi
        s_xi[t]  = sp_xi

        # ── GAS update ───────────────────────────────────────────────────────
        score_p_phi = ar_p_phi = 0.0
        score_p_xi  = ar_p_xi  = 0.0
        for k in range(n_lgas):
            l   = lags_gas[k]
            idx = t - l + 1
            sp = s_phi[idx] if idx >= 0 else 0.0
            sx = s_xi[idx]  if idx >= 0 else 0.0
            fp = f_phi[idx] if idx >= 0 else f0_phi
            fx = f_xi[idx]  if idx >= 0 else f0_xi
            score_p_phi += A_phi[k] * sp
            score_p_xi  += A_xi[k]  * sx
            ar_p_phi    += B_phi[k] * fp
            ar_p_xi     += B_xi[k]  * fx

        fn_phi = omega_phi + score_p_phi + ar_p_phi
        fn_xi  = omega_xi  + score_p_xi  + ar_p_xi
        if not (math.isfinite(fn_phi) and math.isfinite(fn_xi)):
            return -1e12, penalty, f_phi, f_xi, s_phi, s_xi, eta_arr, pi_arr
        f_phi[t + 1] = fn_phi
        f_xi[t + 1]  = fn_xi

    return loglik, penalty, f_phi, f_xi, s_phi, s_xi, eta_arr, pi_arr


# ─────────────────────────────────────────────────────────────────────────────
# Compiled filters: covariate-augmented variants
# ─────────────────────────────────────────────────────────────────────────────
# These are identical to the base filters above except the GAS update adds a
# precomputed covariate contribution C_j[t] = X_t @ gamma_j for each TV
# parameter j.  C_phi and C_xi are computed in Python via NumPy matrix-vector
# products (X_safe @ gamma_arr) before entering nopython mode — this avoids
# passing 2-D matrices into numba and keeps the JIT function simple.
#
# Mathematical identity (for the phi+xi case):
#   f_phi[t+1] = omega_phi + sum_l A_phi_l * s_phi[t-l+1]
#                           + sum_l B_phi_l * f_phi[t-l+1]
#                           + C_phi[t]           ← new term
#   f_xi[t+1]  = omega_xi  + sum_l A_xi_l  * s_xi[t-l+1]
#                           + sum_l B_xi_l  * f_xi[t-l+1]
#                           + C_xi[t]            ← new term
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def nb_filter_phi_only_cov(
    y:          np.ndarray,
    f0:         float,
    omega:      float,
    A:          np.ndarray,
    B:          np.ndarray,
    lags_gas:   np.ndarray,
    xi_s:       float,
    gamma_s:    float,
    zeta_s:     float,
    pi_omega0:  float,
    pi_rho:     float,
    pi_wy:      np.ndarray,
    lags_pi:    np.ndarray,
    max_lag:    int,
    scaling:    int,
    lam_state:  float,
    state_lim:  float,
    lam_eta:    float,
    eta_lim:    float,
    C_phi:      np.ndarray,   # (T,)  precomputed X @ gamma_phi
) -> tuple:
    """
    GAS filter for phi-only ZA-GB2 model with exogenous covariates.

    Identical to nb_filter_phi_only except the GAS update adds C_phi[t].
    Returns (loglik, penalty, f_arr, s_arr, eta_arr, pi_arr).
    """
    T      = len(y)
    n_lgas = len(lags_gas)
    n_lpi  = len(lags_pi)

    f_arr   = np.zeros(T + 1)
    s_arr   = np.zeros(T)
    eta_arr = np.zeros(T)
    pi_arr  = np.zeros(T)

    for k in range(max_lag + 1):
        f_arr[k] = f0

    loglik  = 0.0
    penalty = 0.0

    for t in range(max_lag, T):
        phi_t = f_arr[t]

        excess = abs(phi_t) - state_lim
        if excess > 0.0:
            penalty += lam_state * excess * excess

        i = t - max_lag
        eta_prev = eta_arr[i - 1] if i > 0 else 0.0
        eta_t = pi_omega0 + pi_rho * eta_prev
        for k in range(n_lpi):
            t_lag = t - lags_pi[k]
            y_lag = y[t_lag] if t_lag >= 0 else 0.0
            eta_t += pi_wy[k] * y_lag
        eta_arr[i] = eta_t

        excess_eta = abs(eta_t) - eta_lim
        if excess_eta > 0.0:
            penalty += lam_eta * excess_eta * excess_eta

        pi_t = _sigmoid(eta_t)
        pi_arr[i] = pi_t

        if y[t] == 0.0:
            loglik += _log_sigmoid(-eta_t)
        else:
            ll_dist = _gb2_logpdf(y[t], phi_t, xi_s, gamma_s, zeta_s)
            if not math.isfinite(ll_dist):
                return -1e12, penalty, f_arr, s_arr, eta_arr, pi_arr
            ll_t = _log_sigmoid(eta_t) + ll_dist
            if not math.isfinite(ll_t):
                return -1e12, penalty, f_arr, s_arr, eta_arr, pi_arr
            loglik += ll_t

        s_t = 0.0
        if y[t] > 0.0:
            s_raw = _gb2_score_phi(y[t], phi_t, xi_s, gamma_s, zeta_s)
            if scaling == 1:
                fi = _gb2_fi_phi(xi_s, gamma_s, zeta_s)
                if fi > 0.0 and math.isfinite(fi):
                    s_t = s_raw / fi
                    if not math.isfinite(s_t):
                        s_t = s_raw
                else:
                    s_t = s_raw
            else:
                s_t = s_raw
        s_arr[t] = s_t

        score_p = 0.0
        ar_p    = 0.0
        for k in range(n_lgas):
            l   = lags_gas[k]
            idx = t - l + 1
            s_past = s_arr[idx] if idx >= 0 else 0.0
            f_past = f_arr[idx] if idx >= 0 else f0
            score_p += A[k] * s_past
            ar_p    += B[k] * f_past
        f_next = omega + score_p + ar_p + C_phi[t]   # covariate term added
        if not math.isfinite(f_next):
            return -1e12, penalty, f_arr, s_arr, eta_arr, pi_arr
        f_arr[t + 1] = f_next

    return loglik, penalty, f_arr, s_arr, eta_arr, pi_arr


@njit(cache=True)
def nb_filter_phi_xi_cov(
    y:          np.ndarray,
    f0_phi:     float,
    f0_xi:      float,
    omega_phi:  float,
    omega_xi:   float,
    A_phi:      np.ndarray,
    A_xi:       np.ndarray,
    B_phi:      np.ndarray,
    B_xi:       np.ndarray,
    lags_gas:   np.ndarray,
    gamma_s:    float,
    zeta_s:     float,
    pi_omega0:  float,
    pi_rho:     float,
    pi_wy:      np.ndarray,
    lags_pi:    np.ndarray,
    max_lag:    int,
    scaling:    int,
    lam_state:  float,
    state_lim:  float,
    lam_eta:    float,
    eta_lim:    float,
    C_phi:      np.ndarray,   # (T,)  precomputed X @ gamma_phi
    C_xi:       np.ndarray,   # (T,)  precomputed X @ gamma_xi
) -> tuple:
    """
    GAS filter for phi+xi ZA-GB2 model with exogenous covariates.

    Identical to nb_filter_phi_xi except the GAS update adds C_phi[t] / C_xi[t].
    Returns (loglik, penalty, f_phi, f_xi, s_phi, s_xi, eta_arr, pi_arr).
    """
    T      = len(y)
    n_lgas = len(lags_gas)
    n_lpi  = len(lags_pi)

    f_phi   = np.zeros(T + 1)
    f_xi    = np.zeros(T + 1)
    s_phi   = np.zeros(T)
    s_xi    = np.zeros(T)
    eta_arr = np.zeros(T)
    pi_arr  = np.zeros(T)

    for k in range(max_lag + 1):
        f_phi[k] = f0_phi
        f_xi[k]  = f0_xi

    loglik  = 0.0
    penalty = 0.0

    for t in range(max_lag, T):
        phi_t = f_phi[t]
        xi_t  = f_xi[t]

        for fval in (phi_t, xi_t):
            ex = abs(fval) - state_lim
            if ex > 0.0:
                penalty += lam_state * ex * ex

        i = t - max_lag
        eta_prev = eta_arr[i - 1] if i > 0 else 0.0
        eta_t = pi_omega0 + pi_rho * eta_prev
        for k in range(n_lpi):
            t_lag = t - lags_pi[k]
            y_lag = y[t_lag] if t_lag >= 0 else 0.0
            eta_t += pi_wy[k] * y_lag
        eta_arr[i] = eta_t

        ex_eta = abs(eta_t) - eta_lim
        if ex_eta > 0.0:
            penalty += lam_eta * ex_eta * ex_eta

        pi_t = _sigmoid(eta_t)
        pi_arr[i] = pi_t

        if y[t] == 0.0:
            loglik += _log_sigmoid(-eta_t)
        else:
            ll_dist = _gb2_logpdf(y[t], phi_t, xi_t, gamma_s, zeta_s)
            if not math.isfinite(ll_dist):
                return -1e12, penalty, f_phi, f_xi, s_phi, s_xi, eta_arr, pi_arr
            ll_t = _log_sigmoid(eta_t) + ll_dist
            if not math.isfinite(ll_t):
                return -1e12, penalty, f_phi, f_xi, s_phi, s_xi, eta_arr, pi_arr
            loglik += ll_t

        sp_phi = 0.0
        sp_xi  = 0.0
        if y[t] > 0.0:
            raw_phi = _gb2_score_phi(y[t], phi_t, xi_t, gamma_s, zeta_s)
            raw_xi  = _gb2_score_xi( y[t], phi_t, xi_t, gamma_s, zeta_s)

            if scaling == 0:
                sp_phi = raw_phi
                sp_xi  = raw_xi

            elif scaling == 1:
                fi_p = _gb2_fi_phi(xi_t, gamma_s, zeta_s)
                fi_x = _gb2_fi_xi( xi_t, zeta_s)
                if fi_p > 0.0 and math.isfinite(fi_p):
                    sp_phi = raw_phi / fi_p
                else:
                    sp_phi = raw_phi
                if fi_x > 0.0 and math.isfinite(fi_x):
                    sp_xi = raw_xi / fi_x
                else:
                    sp_xi = raw_xi

            else:
                fi_pp = _gb2_fi_phi(xi_t, gamma_s, zeta_s)
                fi_xx = _gb2_fi_xi( xi_t, zeta_s)
                fi_px = _gb2_fi_cross(xi_t, gamma_s, zeta_s)
                det   = fi_pp * fi_xx - fi_px * fi_px
                if det > 1e-12 and math.isfinite(det):
                    sp_phi = ( fi_xx * raw_phi - fi_px * raw_xi) / det
                    sp_xi  = (-fi_px * raw_phi + fi_pp * raw_xi) / det
                    if not (math.isfinite(sp_phi) and math.isfinite(sp_xi)):
                        sp_phi = raw_phi / fi_pp if fi_pp > 0 else raw_phi
                        sp_xi  = raw_xi  / fi_xx if fi_xx > 0 else raw_xi
                else:
                    sp_phi = raw_phi
                    sp_xi  = raw_xi

        s_phi[t] = sp_phi
        s_xi[t]  = sp_xi

        score_p_phi = ar_p_phi = 0.0
        score_p_xi  = ar_p_xi  = 0.0
        for k in range(n_lgas):
            l   = lags_gas[k]
            idx = t - l + 1
            sp = s_phi[idx] if idx >= 0 else 0.0
            sx = s_xi[idx]  if idx >= 0 else 0.0
            fp = f_phi[idx] if idx >= 0 else f0_phi
            fx = f_xi[idx]  if idx >= 0 else f0_xi
            score_p_phi += A_phi[k] * sp
            score_p_xi  += A_xi[k]  * sx
            ar_p_phi    += B_phi[k] * fp
            ar_p_xi     += B_xi[k]  * fx

        fn_phi = omega_phi + score_p_phi + ar_p_phi + C_phi[t]   # covariate term
        fn_xi  = omega_xi  + score_p_xi  + ar_p_xi  + C_xi[t]    # covariate term
        if not (math.isfinite(fn_phi) and math.isfinite(fn_xi)):
            return -1e12, penalty, f_phi, f_xi, s_phi, s_xi, eta_arr, pi_arr
        f_phi[t + 1] = fn_phi
        f_xi[t + 1]  = fn_xi

    return loglik, penalty, f_phi, f_xi, s_phi, s_xi, eta_arr, pi_arr


# ─────────────────────────────────────────────────────────────────────────────
# Harvey long-short filter  (φ-only and φ+ξ variants)
#
# State equations (for each TV parameter j ∈ {φ, ξ}):
#
#   f_{j,t}   = ω_j + L_{j,t} + S_{j,t}
#
#   L_{j,t+1} = B_L_j · L_{j,t}  +  A_L_j · s_{j,t}  +  CL_j[t]
#   S_{j,t+1} = B_S_j · S_{j,t}  +  A_S_j · s_{j,t}  +  CS_j[t]
#
# where  CL_j[t] = X_long[t]  @ gamma_L_j   (precomputed in Python)
#        CS_j[t] = X_short[t] @ gamma_S_j   (precomputed in Python)
#
# Penalties for B_L/B_S stationarity and score amplitude are computed in
# Python before calling these functions (pre_penalty).  Only per-step
# state-magnitude and eta-magnitude penalties are added here.
#
# eta_arr and pi_arr are returned i-indexed (length T - max_lag).
# L_arr, S_arr are t-indexed (length T+1); s_arr is t-indexed (length T).
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def nb_harvey_phi_only_filter(
    y:         np.ndarray,   # (T,)  observations
    omega:     float,        # constant added to L + S at each step
    L0:        float,        # initial long component
    A_L:       float,        # long score loading
    B_L:       float,        # long AR coefficient
    S0:        float,        # initial short component
    A_S:       float,        # short score loading
    B_S:       float,        # short AR coefficient
    xi_s:      float,        # static ξ = log a
    gamma_s:   float,        # static γ = −log p
    zeta_s:    float,        # static ζ = log b
    pi_omega0: float,
    pi_rho:    float,
    pi_wy:     np.ndarray,   # (n_lpi,)
    lags_pi:   np.ndarray,   # (n_lpi,)
    max_lag:   int,
    scaling:   int,
    lam_state: float,
    state_lim: float,
    lam_eta:   float,
    eta_lim:   float,
    CL:        np.ndarray,   # (T,) X_long  @ gamma_L  (zeros if n_long==0)
    CS:        np.ndarray,   # (T,) X_short @ gamma_S  (zeros if n_short==0)
) -> tuple:
    """Harvey filter for φ-only ZA-GB2. Returns (loglik, penalty, L, S, s, eta, pi)."""
    T     = len(y)
    n_eff = T - max_lag
    n_lpi = len(lags_pi)

    L_arr   = np.zeros(T + 1)
    S_arr   = np.zeros(T + 1)
    s_arr   = np.zeros(T)
    eta_arr = np.zeros(n_eff)
    pi_arr  = np.zeros(n_eff)

    L_arr[0] = L0
    S_arr[0] = S0

    loglik  = 0.0
    penalty = 0.0

    for t in range(max_lag, T):
        i     = t - max_lag
        phi_t = omega + L_arr[t] + S_arr[t]

        excess = abs(phi_t) - state_lim
        if excess > 0.0:
            penalty += lam_state * excess * excess

        eta_prev = eta_arr[i - 1] if i > 0 else 0.0
        eta_t    = pi_omega0 + pi_rho * eta_prev
        for k in range(n_lpi):
            t_lag = t - lags_pi[k]
            y_lag = y[t_lag] if t_lag >= 0 else 0.0
            eta_t += pi_wy[k] * y_lag
        eta_arr[i] = eta_t

        ex_eta = abs(eta_t) - eta_lim
        if ex_eta > 0.0:
            penalty += lam_eta * ex_eta * ex_eta

        pi_arr[i] = _sigmoid(eta_t)

        if y[t] == 0.0:
            loglik += _log_sigmoid(-eta_t)
        else:
            ll_dist = _gb2_logpdf(y[t], phi_t, xi_s, gamma_s, zeta_s)
            if not math.isfinite(ll_dist):
                return -1e12, penalty, L_arr, S_arr, s_arr, eta_arr, pi_arr
            ll_t = _log_sigmoid(eta_t) + ll_dist
            if not math.isfinite(ll_t):
                return -1e12, penalty, L_arr, S_arr, s_arr, eta_arr, pi_arr
            loglik += ll_t

        s_t = 0.0
        if y[t] > 0.0:
            s_raw = _gb2_score_phi(y[t], phi_t, xi_s, gamma_s, zeta_s)
            if scaling == 1:
                fi = _gb2_fi_phi(xi_s, gamma_s, zeta_s)
                if fi > 0.0 and math.isfinite(fi):
                    s_t = s_raw / fi
                    if not math.isfinite(s_t):
                        s_t = s_raw
                else:
                    s_t = s_raw
            else:
                s_t = s_raw
        s_arr[t] = s_t

        L_next = B_L * L_arr[t] + A_L * s_t + CL[t]
        S_next = B_S * S_arr[t] + A_S * s_t + CS[t]
        if not (math.isfinite(L_next) and math.isfinite(S_next)):
            return -1e12, penalty, L_arr, S_arr, s_arr, eta_arr, pi_arr
        L_arr[t + 1] = L_next
        S_arr[t + 1] = S_next

    return loglik, penalty, L_arr, S_arr, s_arr, eta_arr, pi_arr


@njit(cache=True)
def nb_harvey_phi_xi_filter(
    y:          np.ndarray,
    omega_phi:  float,
    omega_xi:   float,
    L0_phi:     float,
    S0_phi:     float,
    A_L_phi:    float,
    B_L_phi:    float,
    A_S_phi:    float,
    B_S_phi:    float,
    L0_xi:      float,
    S0_xi:      float,
    A_L_xi:     float,
    B_L_xi:     float,
    A_S_xi:     float,
    B_S_xi:     float,
    gamma_s:    float,
    zeta_s:     float,
    pi_omega0:  float,
    pi_rho:     float,
    pi_wy:      np.ndarray,
    lags_pi:    np.ndarray,
    max_lag:    int,
    scaling:    int,
    lam_state:  float,
    state_lim:  float,
    lam_eta:    float,
    eta_lim:    float,
    CL_phi:     np.ndarray,   # (T,) X_long  @ gamma_L_phi
    CS_phi:     np.ndarray,   # (T,) X_short @ gamma_S_phi
    CL_xi:      np.ndarray,   # (T,) X_long  @ gamma_L_xi
    CS_xi:      np.ndarray,   # (T,) X_short @ gamma_S_xi
) -> tuple:
    """Harvey filter for φ+ξ ZA-GB2. Returns (loglik, penalty, Lphi, Sphi, Lxi, Sxi, sphi, sxi, eta, pi)."""
    T     = len(y)
    n_eff = T - max_lag
    n_lpi = len(lags_pi)

    L_phi   = np.zeros(T + 1)
    S_phi   = np.zeros(T + 1)
    L_xi    = np.zeros(T + 1)
    S_xi    = np.zeros(T + 1)
    s_phi   = np.zeros(T)
    s_xi    = np.zeros(T)
    eta_arr = np.zeros(n_eff)
    pi_arr  = np.zeros(n_eff)

    L_phi[0] = L0_phi;  S_phi[0] = S0_phi
    L_xi[0]  = L0_xi;   S_xi[0]  = S0_xi

    loglik  = 0.0
    penalty = 0.0

    for t in range(max_lag, T):
        i     = t - max_lag
        phi_t = omega_phi + L_phi[t] + S_phi[t]
        xi_t  = omega_xi  + L_xi[t]  + S_xi[t]

        for fval in (phi_t, xi_t):
            ex = abs(fval) - state_lim
            if ex > 0.0:
                penalty += lam_state * ex * ex

        eta_prev = eta_arr[i - 1] if i > 0 else 0.0
        eta_t    = pi_omega0 + pi_rho * eta_prev
        for k in range(n_lpi):
            t_lag = t - lags_pi[k]
            y_lag = y[t_lag] if t_lag >= 0 else 0.0
            eta_t += pi_wy[k] * y_lag
        eta_arr[i] = eta_t

        ex_eta = abs(eta_t) - eta_lim
        if ex_eta > 0.0:
            penalty += lam_eta * ex_eta * ex_eta

        pi_arr[i] = _sigmoid(eta_t)

        if y[t] == 0.0:
            loglik += _log_sigmoid(-eta_t)
        else:
            ll_dist = _gb2_logpdf(y[t], phi_t, xi_t, gamma_s, zeta_s)
            if not math.isfinite(ll_dist):
                return -1e12, penalty, L_phi, S_phi, L_xi, S_xi, s_phi, s_xi, eta_arr, pi_arr
            ll_t = _log_sigmoid(eta_t) + ll_dist
            if not math.isfinite(ll_t):
                return -1e12, penalty, L_phi, S_phi, L_xi, S_xi, s_phi, s_xi, eta_arr, pi_arr
            loglik += ll_t

        sp_phi = 0.0
        sp_xi  = 0.0
        if y[t] > 0.0:
            raw_phi = _gb2_score_phi(y[t], phi_t, xi_t, gamma_s, zeta_s)
            raw_xi  = _gb2_score_xi( y[t], phi_t, xi_t, gamma_s, zeta_s)

            if scaling == 0:
                sp_phi = raw_phi
                sp_xi  = raw_xi
            elif scaling == 1:
                fi_p = _gb2_fi_phi(xi_t, gamma_s, zeta_s)
                fi_x = _gb2_fi_xi( xi_t, zeta_s)
                sp_phi = raw_phi / fi_p if (fi_p > 0.0 and math.isfinite(fi_p)) else raw_phi
                sp_xi  = raw_xi  / fi_x if (fi_x > 0.0 and math.isfinite(fi_x)) else raw_xi
                if not math.isfinite(sp_phi): sp_phi = raw_phi
                if not math.isfinite(sp_xi):  sp_xi  = raw_xi
            else:
                fi_pp = _gb2_fi_phi(xi_t, gamma_s, zeta_s)
                fi_xx = _gb2_fi_xi( xi_t, zeta_s)
                fi_px = _gb2_fi_cross(xi_t, gamma_s, zeta_s)
                det   = fi_pp * fi_xx - fi_px * fi_px
                if det > 1e-12 and math.isfinite(det):
                    sp_phi = ( fi_xx * raw_phi - fi_px * raw_xi) / det
                    sp_xi  = (-fi_px * raw_phi + fi_pp * raw_xi) / det
                    if not (math.isfinite(sp_phi) and math.isfinite(sp_xi)):
                        sp_phi = raw_phi / fi_pp if fi_pp > 0 else raw_phi
                        sp_xi  = raw_xi  / fi_xx if fi_xx > 0 else raw_xi
                else:
                    sp_phi = raw_phi
                    sp_xi  = raw_xi

        s_phi[t] = sp_phi
        s_xi[t]  = sp_xi

        L_phi_next = B_L_phi * L_phi[t] + A_L_phi * sp_phi + CL_phi[t]
        S_phi_next = B_S_phi * S_phi[t] + A_S_phi * sp_phi + CS_phi[t]
        L_xi_next  = B_L_xi  * L_xi[t]  + A_L_xi  * sp_xi  + CL_xi[t]
        S_xi_next  = B_S_xi  * S_xi[t]  + A_S_xi  * sp_xi  + CS_xi[t]

        if not (math.isfinite(L_phi_next) and math.isfinite(S_phi_next)
                and math.isfinite(L_xi_next) and math.isfinite(S_xi_next)):
            return -1e12, penalty, L_phi, S_phi, L_xi, S_xi, s_phi, s_xi, eta_arr, pi_arr

        L_phi[t + 1] = L_phi_next
        S_phi[t + 1] = S_phi_next
        L_xi[t + 1]  = L_xi_next
        S_xi[t + 1]  = S_xi_next

    return loglik, penalty, L_phi, S_phi, L_xi, S_xi, s_phi, s_xi, eta_arr, pi_arr


# ─────────────────────────────────────────────────────────────────────────────
# Warm up: trigger compilation for common lag combinations at import time
# ─────────────────────────────────────────────────────────────────────────────

def _warmup():
    """
    Pre-compile for the two most common lag combos (short [1,2,3] and
    seasonal [1,2,3,364-367]).  Called once at module import.  Each call
    takes 5-20 s the first time; subsequent imports read from the disk
    cache (cache=True above).
    """
    import warnings
    warnings.filterwarnings("ignore")

    y_dummy = np.zeros(400, dtype=np.float64)
    y_dummy[::3] = 1.0   # some wet days

    for gas_lags in [
        np.array([1, 2, 3], dtype=np.int64),
        np.array([1, 2, 3, 364, 365, 366, 367], dtype=np.int64),
    ]:
        pi_lags = np.array([1, 2, 3, 364, 365, 366, 367], dtype=np.int64)
        n_gl    = len(gas_lags)
        n_pl    = len(pi_lags)
        ml      = int(max(gas_lags.max(), pi_lags.max()))

        # phi-only
        nb_filter_phi_only(
            y_dummy,
            f0=0.0, omega=0.0,
            A=np.full(n_gl, 0.01),
            B=np.full(n_gl, 0.3),
            lags_gas=gas_lags,
            xi_s=0.0, gamma_s=0.2, zeta_s=1.5,
            pi_omega0=0.0, pi_rho=0.3,
            pi_wy=np.zeros(n_pl),
            lags_pi=pi_lags,
            max_lag=ml,
            scaling=0,
            lam_state=1e3, state_lim=15.0,
            lam_eta=1e2,   eta_lim=30.0,
        )

        # phi+xi
        nb_filter_phi_xi(
            y_dummy,
            f0_phi=0.0, f0_xi=0.0,
            omega_phi=0.0, omega_xi=0.0,
            A_phi=np.full(n_gl, 0.01), A_xi=np.full(n_gl, 0.01),
            B_phi=np.full(n_gl, 0.3),  B_xi=np.full(n_gl, 0.3),
            lags_gas=gas_lags,
            gamma_s=0.2, zeta_s=1.5,
            pi_omega0=0.0, pi_rho=0.3,
            pi_wy=np.zeros(n_pl),
            lags_pi=pi_lags,
            max_lag=ml,
            scaling=0,
            lam_state=1e3, state_lim=15.0,
            lam_eta=1e2,   eta_lim=30.0,
        )

        # phi-only with covariates
        C_dummy = np.zeros(len(y_dummy))
        nb_filter_phi_only_cov(
            y_dummy,
            f0=0.0, omega=0.0,
            A=np.full(n_gl, 0.01),
            B=np.full(n_gl, 0.3),
            lags_gas=gas_lags,
            xi_s=0.0, gamma_s=0.2, zeta_s=1.5,
            pi_omega0=0.0, pi_rho=0.3,
            pi_wy=np.zeros(n_pl),
            lags_pi=pi_lags,
            max_lag=ml,
            scaling=0,
            lam_state=1e3, state_lim=15.0,
            lam_eta=1e2,   eta_lim=30.0,
            C_phi=C_dummy,
        )

        # phi+xi with covariates
        nb_filter_phi_xi_cov(
            y_dummy,
            f0_phi=0.0, f0_xi=0.0,
            omega_phi=0.0, omega_xi=0.0,
            A_phi=np.full(n_gl, 0.01), A_xi=np.full(n_gl, 0.01),
            B_phi=np.full(n_gl, 0.3),  B_xi=np.full(n_gl, 0.3),
            lags_gas=gas_lags,
            gamma_s=0.2, zeta_s=1.5,
            pi_omega0=0.0, pi_rho=0.3,
            pi_wy=np.zeros(n_pl),
            lags_pi=pi_lags,
            max_lag=ml,
            scaling=0,
            lam_state=1e3, state_lim=15.0,
            lam_eta=1e2,   eta_lim=30.0,
            C_phi=C_dummy,
            C_xi=C_dummy,
        )

    # Harvey phi-only (no GAS lag array — lag structure is scalar B_L/B_S)
    nb_harvey_phi_only_filter(
        y_dummy,
        omega=0.0, L0=0.0, A_L=0.01, B_L=0.95, S0=0.0, A_S=0.05, B_S=0.5,
        xi_s=0.0, gamma_s=0.2, zeta_s=1.5,
        pi_omega0=0.0, pi_rho=0.3,
        pi_wy=np.zeros(n_pl),
        lags_pi=pi_lags,
        max_lag=ml,
        scaling=1,
        lam_state=1e3, state_lim=15.0,
        lam_eta=1e2,   eta_lim=30.0,
        CL=C_dummy, CS=C_dummy,
    )

    # Harvey phi+xi
    nb_harvey_phi_xi_filter(
        y_dummy,
        omega_phi=0.0, omega_xi=0.0,
        L0_phi=0.0, S0_phi=0.0, A_L_phi=0.01, B_L_phi=0.95, A_S_phi=0.05, B_S_phi=0.5,
        L0_xi=0.0,  S0_xi=0.0,  A_L_xi=0.01,  B_L_xi=0.95,  A_S_xi=0.05,  B_S_xi=0.5,
        gamma_s=0.2, zeta_s=1.5,
        pi_omega0=0.0, pi_rho=0.3,
        pi_wy=np.zeros(n_pl),
        lags_pi=pi_lags,
        max_lag=ml,
        scaling=1,
        lam_state=1e3, state_lim=15.0,
        lam_eta=1e2,   eta_lim=30.0,
        CL_phi=C_dummy, CS_phi=C_dummy,
        CL_xi=C_dummy,  CS_xi=C_dummy,
    )


_warmup()
