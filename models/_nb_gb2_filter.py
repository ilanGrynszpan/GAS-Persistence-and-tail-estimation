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
Scaling modes: 0 = unit, 1 = diagonal_inverse_fisher, 2 = inverse_fisher,
3 = mixed_inverse_fisher (phi-only ZA kernel only; Fisher information of the
zero-adjusted mixed density). For one time-varying parameter, modes 1 and 2
are equivalent (1x1 matrix); before 2026-10-01 the one-parameter kernels
checked only `scaling == 1`, so mode 2 silently ran as unit scaling.

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
def _pi_link_terms(eta_t: float, pi_link: int):
    """
    Occurrence link: map the latent eta_t to (pi_t, log pi_t, log(1 - pi_t)).

    pi_link == 0 : LOGISTIC (default, unchanged historical behaviour)
        pi = sigma(eta) = 1 / (1 + exp(-eta))
    pi_link == 1 : RECIPROCAL  (2026-09-10 request)
        pi = 1 / (1 + eta)      -- eta acts like the "dry odds" (1-pi)/pi.
        Requires eta > -1 for a valid probability; the value is clamped
        into [1e-12, 1 - 1e-12] as a numerical safeguard (CLAUDE.md:
        "use numerical safeguards rather than allowing invalid
        evaluations"), so an eta that momentarily leaves the valid region
        yields a finite, penalised-by-fit log-likelihood rather than a
        NaN. eta itself is separately bounded by the eta-magnitude penalty
        that every caller already applies.
    """
    if pi_link == 0:
        return _sigmoid(eta_t), _log_sigmoid(eta_t), _log_sigmoid(-eta_t)
    denom = 1.0 + eta_t
    if denom < 1e-12:
        denom = 1e-12
    pi_t = 1.0 / denom
    if pi_t < 1e-12:
        pi_t = 1e-12
    elif pi_t > 1.0 - 1e-12:
        pi_t = 1.0 - 1e-12
    return pi_t, math.log(pi_t), math.log(1.0 - pi_t)


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
    a = exp(xi) underflows to 0.0 during BFGS line search. The x <= 0.0
    check alone is not sufficient: for x positive but smaller than
    ~2.2e-162, x*x itself underflows to exactly 0.0 (float64's smallest
    positive value is ~4.9e-324), so 1.0/(x*x) still raises even though
    x > 0.0. Observed in practice (2026-08-31) once
    HarveyZAGASModel(use_auxiliary_penalties=False) removed the |zeta|<20
    static-magnitude clamp: an unbounded BFGS trial step can push zeta_s
    (or xi_t) far enough that a=exp(xi) or a+b lands in this underflow
    range. Since trigamma diverges to +inf as x->0+ regardless, treating
    "too small to square accurately" the same as "x<=0.0" is mathematically
    consistent, not merely a numerical patch.
    """
    if x <= 0.0:
        return 1.0e308
    result = 0.0
    while x < 6.0:
        xx = x * x
        if xx <= 0.0:
            return 1.0e308
        result += 1.0 / xx
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

    Guard: if both a and b underflow to 0.0 (xi, zeta << -745), the
    denominator is 0/0 → ZeroDivisionError in Numba's Python-compatible
    float division.  The limit as a,b→0⁺ is 0, so return 0.0.
    """
    a = math.exp(xi)
    b = math.exp(zeta)
    denom = a + b
    if denom <= 0.0:
        return 0.0
    p = math.exp(-gamma)
    return p * a * b / denom


# ─────────────────────────────────────────────────────────────────────────────
# Scaled score of the zero-adjusted (mixed) density  (2026-10-01)
# ─────────────────────────────────────────────────────────────────────────────
# f(y) = (1 - pi_t) 1{y=0} + pi_t g(y; phi_t, xi_t) 1{y>0}, logistic link,
# eta_t = ... + lambda_1 phi_{t|t-1}  (lambda_1 = pi_lam_phi, 0 unless the
# occurrence probability is phi-linked). With w = 1{y>0}:
#
#   score    s_phi = w dlog g/dphi + lambda_1 (w - pi_t),   s_xi = w dlog g/dxi
#   Fisher   M = [[pi I_pp + lambda_1^2 pi (1-pi),  pi I_px],
#                 [pi I_px,                         pi I_xx]]
#   (I_.. = GB2 Fisher information, log scale; cross terms with the
#    occurrence part vanish because E[dlog g] = 0.)
#
# scaling codes:
#   0  unit                                  s
#   1  GB2 diagonal inverse FI (historical)  w * dlog g / diag(I)     (lambda_1 must be 0)
#   2  GB2 full inverse FI     (historical)  w * I^{-1} dlog g        (lambda_1 must be 0)
#   3  mixed full inverse FI                 M^{-1} s
#   4  mixed diagonal inverse FI             s / diag(M)
# Codes 1-2 reproduce the pre-2026-10-01 kernels exactly (fallbacks:
# full -> diagonal -> unit). For one time-varying parameter 1==2 and 3==4.

@njit(cache=True)
def _scaled_score_1(y_t: float, phi_t: float, xi_s: float, gamma_s: float, zeta_s: float,
                    pi_t: float, lam: float, scaling: int) -> float:
    wet = 1.0 if y_t > 0.0 else 0.0
    s_mag = _gb2_score_phi(y_t, phi_t, xi_s, gamma_s, zeta_s) if y_t > 0.0 else 0.0
    s_raw = s_mag + lam * (wet - pi_t)
    if scaling == 0:
        return s_raw
    fi = _gb2_fi_phi(xi_s, gamma_s, zeta_s)
    if scaling == 1 or scaling == 2:
        if fi > 0.0 and math.isfinite(fi):
            out = s_mag / fi
            return out if math.isfinite(out) else s_mag
        return s_mag
    m = pi_t * fi + lam * lam * pi_t * (1.0 - pi_t)
    if m > 0.0 and math.isfinite(m):
        out = s_raw / m
        return out if math.isfinite(out) else s_raw
    return s_raw


@njit(cache=True)
def _scaled_score_2(y_t: float, phi_t: float, xi_t: float, gamma_s: float, zeta_s: float,
                    pi_t: float, lam: float, scaling: int):
    wet = 1.0 if y_t > 0.0 else 0.0
    raw_phi = 0.0
    raw_xi = 0.0
    if y_t > 0.0:
        raw_phi = _gb2_score_phi(y_t, phi_t, xi_t, gamma_s, zeta_s)
        raw_xi = _gb2_score_xi(y_t, phi_t, xi_t, gamma_s, zeta_s)
    s_phi = raw_phi + lam * (wet - pi_t)
    s_xi = raw_xi
    if scaling == 0:
        return s_phi, s_xi
    fi_pp = _gb2_fi_phi(xi_t, gamma_s, zeta_s)
    fi_xx = _gb2_fi_xi(xi_t, zeta_s)
    fi_px = _gb2_fi_cross(xi_t, gamma_s, zeta_s)
    if scaling == 1 or scaling == 2:
        if y_t <= 0.0:
            return 0.0, 0.0
    else:
        fi_pp = pi_t * fi_pp + lam * lam * pi_t * (1.0 - pi_t)
        fi_xx = pi_t * fi_xx
        fi_px = pi_t * fi_px
        raw_phi = s_phi
        raw_xi = s_xi
    if scaling == 1 or scaling == 4:
        sp = raw_phi / fi_pp if (fi_pp > 0.0 and math.isfinite(fi_pp)) else raw_phi
        sx = raw_xi / fi_xx if (fi_xx > 0.0 and math.isfinite(fi_xx)) else raw_xi
        if not math.isfinite(sp):
            sp = raw_phi
        if not math.isfinite(sx):
            sx = raw_xi
        return sp, sx
    det = fi_pp * fi_xx - fi_px * fi_px
    if det > 1e-12 and math.isfinite(det):
        sp = (fi_xx * raw_phi - fi_px * raw_xi) / det
        sx = (-fi_px * raw_phi + fi_pp * raw_xi) / det
        if not (math.isfinite(sp) and math.isfinite(sx)):
            sp = raw_phi / fi_pp if fi_pp > 0 else raw_phi
            sx = raw_xi / fi_xx if fi_xx > 0 else raw_xi
        return sp, sx
    return raw_phi, raw_xi


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
    pi_link:    int,        # 0=logistic (default), 1=reciprocal pi=1/(1+eta)
    max_lag:    int,
    scaling:    int,          # see _scaled_score_1: 0 unit, 1/2 GB2 inverse FI, 3/4 mixed inverse FI
    lam_state:  float,        # penalty weight for large states
    state_lim:  float,        # STATE_LIMIT (15.0)
    lam_eta:    float,        # penalty weight for large eta
    eta_lim:    float,        # ETA_LIMIT (30.0)
    pi_lam_phi: float = 0.0,  # lambda_1: eta_t += lambda_1 * phi_{t|t-1} (0 = pi does not depend on phi)
) -> tuple:
    """
    GAS filter for phi-only ZA-GB2 model, compiled by numba.

    Occurrence logit:
        eta_t = omega0 + rho*eta_{t-1} + sum_l w_l y_{t-l} + lambda_1 * phi_{t|t-1}
    (lambda_1 = pi_lam_phi; 0 for every pi dynamics except the joint
    phi-linked one, which uses omega0=lambda_0, rho=0 and no y-lags).

    Score of the zero-adjusted (mixed) density w.r.t. the log-scale phi,
    with I_t = 1{y_t > 0}:
        s_t = I_t * dlog g/dphi + lambda_1 * (I_t - pi_t)
    (the second term is the occurrence part, present only when pi depends
    on phi; with logistic link d pi/d phi = lambda_1 pi (1 - pi)).

    Scaling:
        0      unit                      s_t
        1, 2   GB2 inverse FI            I_t * dlog g/dphi / I_g   (1 param: diag == full)
        3      mixed inverse FI          s_t / [pi_t I_g + lambda_1^2 pi_t (1 - pi_t)]
    where I_g = p^2 a b / (a+b+1) is the GB2 Fisher information for phi.
    Modes 1/2 are only defined for lambda_1 = 0 (ZAGASModel enforces this).

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
        eta_t += pi_lam_phi * phi_t
        eta_arr[i] = eta_t

        # ── Eta penalty ──────────────────────────────────────────────────────
        excess_eta = abs(eta_t) - eta_lim
        if excess_eta > 0.0:
            penalty += lam_eta * excess_eta * excess_eta

        pi_t, _lpi_t, _l1mpi_t = _pi_link_terms(eta_t, pi_link)
        pi_arr[i] = pi_t

        # ── Log-likelihood ───────────────────────────────────────────────────
        if y[t] == 0.0:
            loglik += _l1mpi_t   # log(1 - pi_t)
        else:
            ll_dist = _gb2_logpdf(y[t], phi_t, xi_s, gamma_s, zeta_s)
            if not math.isfinite(ll_dist):
                return -1e12, penalty, f_arr, s_arr, eta_arr, pi_arr
            ll_t = _lpi_t + ll_dist
            if not math.isfinite(ll_t):
                return -1e12, penalty, f_arr, s_arr, eta_arr, pi_arr
            loglik += ll_t

        s_t = _scaled_score_1(y[t], phi_t, xi_s, gamma_s, zeta_s, pi_t, pi_lam_phi, scaling)
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
    pi_link:    int,        # 0=logistic (default), 1=reciprocal pi=1/(1+eta)
    max_lag:    int,
    scaling:    int,
    lam_state:  float,
    state_lim:  float,
    lam_eta:    float,
    eta_lim:    float,
    pi_lam_phi: float = 0.0,  # lambda_1: eta_t += lambda_1 * phi_{t|t-1}
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
        eta_t += pi_lam_phi * phi_t
        eta_arr[i] = eta_t

        ex_eta = abs(eta_t) - eta_lim
        if ex_eta > 0.0:
            penalty += lam_eta * ex_eta * ex_eta

        pi_t, _lpi_t, _l1mpi_t = _pi_link_terms(eta_t, pi_link)
        pi_arr[i] = pi_t

        # ── Log-likelihood ───────────────────────────────────────────────────
        if y[t] == 0.0:
            loglik += _l1mpi_t
        else:
            ll_dist = _gb2_logpdf(y[t], phi_t, xi_t, gamma_s, zeta_s)
            if not math.isfinite(ll_dist):
                return -1e12, penalty, f_phi, f_xi, s_phi, s_xi, eta_arr, pi_arr
            ll_t = _lpi_t + ll_dist
            if not math.isfinite(ll_t):
                return -1e12, penalty, f_phi, f_xi, s_phi, s_xi, eta_arr, pi_arr
            loglik += ll_t

        # ── Score + scaling ──────────────────────────────────────────────────
        sp_phi, sp_xi = _scaled_score_2(y[t], phi_t, xi_t, gamma_s, zeta_s, pi_t, pi_lam_phi, scaling)
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
    pi_link:    int,        # 0=logistic (default), 1=reciprocal pi=1/(1+eta)
    max_lag:    int,
    scaling:    int,
    lam_state:  float,
    state_lim:  float,
    lam_eta:    float,
    eta_lim:    float,
    C_phi:      np.ndarray,   # (T,)  precomputed X @ gamma_phi
    pi_lam_phi: float = 0.0,  # lambda_1: eta_t += lambda_1 * phi_{t|t-1}
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
        eta_t += pi_lam_phi * phi_t
        eta_arr[i] = eta_t

        excess_eta = abs(eta_t) - eta_lim
        if excess_eta > 0.0:
            penalty += lam_eta * excess_eta * excess_eta

        pi_t, _lpi_t, _l1mpi_t = _pi_link_terms(eta_t, pi_link)
        pi_arr[i] = pi_t

        if y[t] == 0.0:
            loglik += _l1mpi_t
        else:
            ll_dist = _gb2_logpdf(y[t], phi_t, xi_s, gamma_s, zeta_s)
            if not math.isfinite(ll_dist):
                return -1e12, penalty, f_arr, s_arr, eta_arr, pi_arr
            ll_t = _lpi_t + ll_dist
            if not math.isfinite(ll_t):
                return -1e12, penalty, f_arr, s_arr, eta_arr, pi_arr
            loglik += ll_t

        s_t = _scaled_score_1(y[t], phi_t, xi_s, gamma_s, zeta_s, pi_t, pi_lam_phi, scaling)
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
    pi_link:    int,        # 0=logistic (default), 1=reciprocal pi=1/(1+eta)
    max_lag:    int,
    scaling:    int,
    lam_state:  float,
    state_lim:  float,
    lam_eta:    float,
    eta_lim:    float,
    C_phi:      np.ndarray,   # (T,)  precomputed X @ gamma_phi
    C_xi:       np.ndarray,   # (T,)  precomputed X @ gamma_xi
    pi_lam_phi: float = 0.0,  # lambda_1: eta_t += lambda_1 * phi_{t|t-1}
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
        eta_t += pi_lam_phi * phi_t
        eta_arr[i] = eta_t

        ex_eta = abs(eta_t) - eta_lim
        if ex_eta > 0.0:
            penalty += lam_eta * ex_eta * ex_eta

        pi_t, _lpi_t, _l1mpi_t = _pi_link_terms(eta_t, pi_link)
        pi_arr[i] = pi_t

        if y[t] == 0.0:
            loglik += _l1mpi_t
        else:
            ll_dist = _gb2_logpdf(y[t], phi_t, xi_t, gamma_s, zeta_s)
            if not math.isfinite(ll_dist):
                return -1e12, penalty, f_phi, f_xi, s_phi, s_xi, eta_arr, pi_arr
            ll_t = _lpi_t + ll_dist
            if not math.isfinite(ll_t):
                return -1e12, penalty, f_phi, f_xi, s_phi, s_xi, eta_arr, pi_arr
            loglik += ll_t

        sp_phi, sp_xi = _scaled_score_2(y[t], phi_t, xi_t, gamma_s, zeta_s, pi_t, pi_lam_phi, scaling)
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
    pi_link:    int,        # 0=logistic (default), 1=reciprocal pi=1/(1+eta)
    max_lag:   int,
    scaling:   int,
    lam_state: float,
    state_lim: float,
    lam_eta:   float,
    eta_lim:   float,
    CL:        np.ndarray,   # (T,) X_long  @ gamma_L  (zeros if n_long==0)
    CS:        np.ndarray,   # (T,) X_short @ gamma_S  (zeros if n_short==0)
    n_harmonics: int,        # 0 reproduces the exact previous behaviour
    cos_h:     np.ndarray,   # (n_harmonics,) cos(2*pi*h/S), h=1..n_harmonics (empty if 0)
    sin_h:     np.ndarray,   # (n_harmonics,) sin(2*pi*h/S)
    kappa_gamma: float,      # seasonal score loading (models/harvey_gas.py's kappa_gamma_j)
    seas0:     np.ndarray,   # (n_harmonics,) gamma_{1|0,h} -- free initial condition, empty if 0
    seasstar0: np.ndarray,   # (n_harmonics,) gamma*_{1|0,h}
    psi_gamma: float,        # rotation damping, 1.0 = undamped/pure rotation (models/harvey_gas.py's seasonal_damping_max)
    pi_lam_phi: float = 0.0,  # lambda_1: eta_t += lambda_1 * phi_{t|t-1}
) -> tuple:
    """
    Harvey filter for φ-only ZA-GB2, with models/harvey_gas.py's optional
    trigonometric seasonal component (see that module's docstring for the
    full derivation -- this is the numba-accelerated mirror of its Python
    fallback's identical recursion, verified bit-for-bit against it before
    use). seas_h/seas*_h are seeded from seas0/seasstar0 (free parameters)
    at the loop's first executed iteration and are purely local loop state
    here, never returned.

    Returns (loglik, penalty, L, S, s, eta, pi, Gamma_arr) -- Gamma_arr is
    the per-t total seasonal contribution sum_h(seas_h), all zeros when
    n_harmonics=0.
    """
    T     = len(y)
    n_eff = T - max_lag
    n_lpi = len(lags_pi)
    m     = n_harmonics

    L_arr   = np.zeros(T + 1)
    S_arr   = np.zeros(T + 1)
    s_arr   = np.zeros(T)
    eta_arr = np.zeros(n_eff)
    pi_arr  = np.zeros(n_eff)
    Gamma_arr = np.zeros(T)

    L_arr[0] = L0
    S_arr[0] = S0
    Gm     = seas0.copy()
    GmStar = seasstar0.copy()

    loglik  = 0.0
    penalty = 0.0

    for t in range(max_lag, T):
        i     = t - max_lag
        gamma_t = 0.0
        if m > 0:
            gamma_t = np.sum(Gm)
            Gamma_arr[t] = gamma_t
        phi_t = omega + L_arr[t] + S_arr[t] + gamma_t

        excess = abs(phi_t) - state_lim
        if excess > 0.0:
            penalty += lam_state * excess * excess

        eta_prev = eta_arr[i - 1] if i > 0 else 0.0
        eta_t    = pi_omega0 + pi_rho * eta_prev
        for k in range(n_lpi):
            t_lag = t - lags_pi[k]
            y_lag = y[t_lag] if t_lag >= 0 else 0.0
            eta_t += pi_wy[k] * y_lag
        eta_t += pi_lam_phi * phi_t
        eta_arr[i] = eta_t

        ex_eta = abs(eta_t) - eta_lim
        if ex_eta > 0.0:
            penalty += lam_eta * ex_eta * ex_eta

        pi_t, _lpi_t, _l1mpi_t = _pi_link_terms(eta_t, pi_link)
        pi_arr[i] = pi_t

        if y[t] == 0.0:
            loglik += _l1mpi_t
        else:
            ll_dist = _gb2_logpdf(y[t], phi_t, xi_s, gamma_s, zeta_s)
            if not math.isfinite(ll_dist):
                return -1e12, penalty, L_arr, S_arr, s_arr, eta_arr, pi_arr, Gamma_arr
            ll_t = _lpi_t + ll_dist
            if not math.isfinite(ll_t):
                return -1e12, penalty, L_arr, S_arr, s_arr, eta_arr, pi_arr, Gamma_arr
            loglik += ll_t

        s_t = _scaled_score_1(y[t], phi_t, xi_s, gamma_s, zeta_s, pi_t, pi_lam_phi, scaling)
        s_arr[t] = s_t

        L_next = B_L * L_arr[t] + A_L * s_t + CL[t]
        S_next = B_S * S_arr[t] + A_S * s_t + CS[t]
        if not (math.isfinite(L_next) and math.isfinite(S_next)):
            return -1e12, penalty, L_arr, S_arr, s_arr, eta_arr, pi_arr, Gamma_arr
        L_arr[t + 1] = L_next
        S_arr[t + 1] = S_next

        if m > 0:
            # Trigonometric seasonal (models/harvey_gas.py's module
            # docstring): SAME scalar score s_t drives every harmonic
            # identically, only the rotation angle differs. psi_gamma
            # damps the whole rotation each step (1.0 = undamped/pure
            # rotation, the originally-specified model).
            Gm_next     = psi_gamma * ( Gm * cos_h + GmStar * sin_h) + kappa_gamma * s_t
            GmStar_next = psi_gamma * (-Gm * sin_h + GmStar * cos_h) + kappa_gamma * s_t
            ok = True
            for hh in range(m):
                if not (math.isfinite(Gm_next[hh]) and math.isfinite(GmStar_next[hh])):
                    ok = False
            if not ok:
                return -1e12, penalty, L_arr, S_arr, s_arr, eta_arr, pi_arr, Gamma_arr
            Gm     = Gm_next
            GmStar = GmStar_next

    return loglik, penalty, L_arr, S_arr, s_arr, eta_arr, pi_arr, Gamma_arr


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
    pi_link:    int,        # 0=logistic (default), 1=reciprocal pi=1/(1+eta)
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
    n_harmonics: int,         # 0 reproduces the exact previous behaviour
    cos_h:      np.ndarray,   # (n_harmonics,) cos(2*pi*h/S)
    sin_h:      np.ndarray,   # (n_harmonics,) sin(2*pi*h/S)
    kappa_gamma_phi: float,
    kappa_gamma_xi:  float,
    seas0_phi:     np.ndarray,  # (n_harmonics,) gamma_{1|0,h} for phi -- free init, empty if 0
    seasstar0_phi: np.ndarray,
    seas0_xi:      np.ndarray,  # same, for xi
    seasstar0_xi:  np.ndarray,
    psi_gamma_phi: float,       # rotation damping for phi, 1.0 = undamped/pure rotation
    psi_gamma_xi:  float,       # rotation damping for xi
    pi_lam_phi: float = 0.0,  # lambda_1: eta_t += lambda_1 * phi_{t|t-1}
) -> tuple:
    """
    Harvey filter for φ+ξ ZA-GB2, with the same optional trigonometric
    seasonal component as nb_harvey_phi_only_filter (see that function's
    docstring) -- one independent seasonal state pair per dynamic
    parameter (phi, xi), each with its own kappa_gamma and its own free
    initial condition.

    Returns (loglik, penalty, Lphi, Sphi, Lxi, Sxi, sphi, sxi, eta, pi,
    Gamma_arr_phi, Gamma_arr_xi).
    """
    T     = len(y)
    n_eff = T - max_lag
    n_lpi = len(lags_pi)
    m     = n_harmonics

    L_phi   = np.zeros(T + 1)
    S_phi   = np.zeros(T + 1)
    L_xi    = np.zeros(T + 1)
    S_xi    = np.zeros(T + 1)
    s_phi   = np.zeros(T)
    s_xi    = np.zeros(T)
    eta_arr = np.zeros(n_eff)
    pi_arr  = np.zeros(n_eff)
    Gamma_arr_phi = np.zeros(T)
    Gamma_arr_xi  = np.zeros(T)

    L_phi[0] = L0_phi;  S_phi[0] = S0_phi
    L_xi[0]  = L0_xi;   S_xi[0]  = S0_xi
    Gm_phi     = seas0_phi.copy(); GmStar_phi = seasstar0_phi.copy()
    Gm_xi      = seas0_xi.copy();  GmStar_xi  = seasstar0_xi.copy()

    loglik  = 0.0
    penalty = 0.0

    for t in range(max_lag, T):
        i     = t - max_lag
        gamma_t_phi = 0.0
        gamma_t_xi  = 0.0
        if m > 0:
            gamma_t_phi = np.sum(Gm_phi)
            gamma_t_xi  = np.sum(Gm_xi)
            Gamma_arr_phi[t] = gamma_t_phi
            Gamma_arr_xi[t]  = gamma_t_xi
        phi_t = omega_phi + L_phi[t] + S_phi[t] + gamma_t_phi
        xi_t  = omega_xi  + L_xi[t]  + S_xi[t]  + gamma_t_xi

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
        eta_t += pi_lam_phi * phi_t
        eta_arr[i] = eta_t

        ex_eta = abs(eta_t) - eta_lim
        if ex_eta > 0.0:
            penalty += lam_eta * ex_eta * ex_eta

        pi_t, _lpi_t, _l1mpi_t = _pi_link_terms(eta_t, pi_link)
        pi_arr[i] = pi_t

        if y[t] == 0.0:
            loglik += _l1mpi_t
        else:
            ll_dist = _gb2_logpdf(y[t], phi_t, xi_t, gamma_s, zeta_s)
            if not math.isfinite(ll_dist):
                return -1e12, penalty, L_phi, S_phi, L_xi, S_xi, s_phi, s_xi, eta_arr, pi_arr, Gamma_arr_phi, Gamma_arr_xi
            ll_t = _lpi_t + ll_dist
            if not math.isfinite(ll_t):
                return -1e12, penalty, L_phi, S_phi, L_xi, S_xi, s_phi, s_xi, eta_arr, pi_arr, Gamma_arr_phi, Gamma_arr_xi
            loglik += ll_t

        sp_phi, sp_xi = _scaled_score_2(y[t], phi_t, xi_t, gamma_s, zeta_s, pi_t, pi_lam_phi, scaling)
        s_phi[t] = sp_phi
        s_xi[t]  = sp_xi

        L_phi_next = B_L_phi * L_phi[t] + A_L_phi * sp_phi + CL_phi[t]
        S_phi_next = B_S_phi * S_phi[t] + A_S_phi * sp_phi + CS_phi[t]
        L_xi_next  = B_L_xi  * L_xi[t]  + A_L_xi  * sp_xi  + CL_xi[t]
        S_xi_next  = B_S_xi  * S_xi[t]  + A_S_xi  * sp_xi  + CS_xi[t]

        if not (math.isfinite(L_phi_next) and math.isfinite(S_phi_next)
                and math.isfinite(L_xi_next) and math.isfinite(S_xi_next)):
            return -1e12, penalty, L_phi, S_phi, L_xi, S_xi, s_phi, s_xi, eta_arr, pi_arr, Gamma_arr_phi, Gamma_arr_xi

        L_phi[t + 1] = L_phi_next
        S_phi[t + 1] = S_phi_next
        L_xi[t + 1]  = L_xi_next
        S_xi[t + 1]  = S_xi_next

        if m > 0:
            Gm_phi_next     = psi_gamma_phi * ( Gm_phi * cos_h + GmStar_phi * sin_h) + kappa_gamma_phi * sp_phi
            GmStar_phi_next = psi_gamma_phi * (-Gm_phi * sin_h + GmStar_phi * cos_h) + kappa_gamma_phi * sp_phi
            Gm_xi_next      = psi_gamma_xi  * ( Gm_xi  * cos_h + GmStar_xi  * sin_h) + kappa_gamma_xi  * sp_xi
            GmStar_xi_next  = psi_gamma_xi  * (-Gm_xi  * sin_h + GmStar_xi  * cos_h) + kappa_gamma_xi  * sp_xi
            ok = True
            for hh in range(m):
                if not (math.isfinite(Gm_phi_next[hh]) and math.isfinite(GmStar_phi_next[hh])
                        and math.isfinite(Gm_xi_next[hh]) and math.isfinite(GmStar_xi_next[hh])):
                    ok = False
            if not ok:
                return -1e12, penalty, L_phi, S_phi, L_xi, S_xi, s_phi, s_xi, eta_arr, pi_arr, Gamma_arr_phi, Gamma_arr_xi
            Gm_phi, GmStar_phi = Gm_phi_next, GmStar_phi_next
            Gm_xi,  GmStar_xi  = Gm_xi_next,  GmStar_xi_next

    return loglik, penalty, L_phi, S_phi, L_xi, S_xi, s_phi, s_xi, eta_arr, pi_arr, Gamma_arr_phi, Gamma_arr_xi


@njit(cache=True)
def nb_gas1phi_harveyxi_filter(
    y:          np.ndarray,
    omega_phi:  float,   # plain GAS(1,1) for phi: phi_{t+1} = omega + A*s_t + B*phi_t
    G0_phi:     float,
    A_phi:      float,
    B_phi:      float,
    omega_xi:   float,   # Harvey long+short (or structural, when the long
    L0_xi:      float,   # inputs below are all zero) for xi, same as
    S0_xi:      float,   # nb_harvey_phi_xi_filter's xi block.
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
    pi_link:    int,
    max_lag:    int,
    scaling:    int,
    lam_state:  float,
    state_lim:  float,
    lam_eta:    float,
    eta_lim:    float,
    CL_xi:      np.ndarray,   # (T,) X_long  @ gamma_L_xi
    CS_xi:      np.ndarray,   # (T,) X_short @ gamma_S_xi
    n_harmonics: int,
    cos_h:      np.ndarray,
    sin_h:      np.ndarray,
    kappa_gamma_xi: float,
    seas0_xi:      np.ndarray,
    seasstar0_xi:  np.ndarray,
    psi_gamma_xi:  float,
) -> tuple:
    """
    Hybrid filter (2026-09-11, direct user request): phi follows a PLAIN
    GAS(1,1) recursion (no covariates, no long/short, no seasonal harmonic
    -- HarveyZAGASModel's gas1_target), xi keeps the full Harvey treatment
    (long+short, or structural when the long-block inputs are zeroed,
    optionally with the trigonometric seasonal harmonic) -- identical xi
    block to nb_harvey_phi_xi_filter, just with phi's L/S pair replaced by
    a single plain-GAS state G_phi. Static/decoded-parameter penalties
    (persistence limits, score-amplitude, covariate-norm) are computed in
    Python BEFORE this kernel is called (see _run_filter's `penalty`
    argument to _nb_harvey_dispatch) -- this kernel only accumulates the
    PATH-DEPENDENT penalties (state/eta explosion), exactly like every
    other nb_harvey_* kernel.

    Returns (loglik, penalty, G_phi, L_xi, S_xi, s_phi, s_xi, eta, pi, Gamma_arr_xi).
    """
    T     = len(y)
    n_eff = T - max_lag
    n_lpi = len(lags_pi)
    m     = n_harmonics

    G_phi   = np.zeros(T + 1)
    L_xi    = np.zeros(T + 1)
    S_xi    = np.zeros(T + 1)
    s_phi   = np.zeros(T)
    s_xi    = np.zeros(T)
    eta_arr = np.zeros(n_eff)
    pi_arr  = np.zeros(n_eff)
    Gamma_arr_xi = np.zeros(T)

    G_phi[0] = G0_phi
    L_xi[0]  = L0_xi
    S_xi[0]  = S0_xi
    Gm_xi     = seas0_xi.copy()
    GmStar_xi = seasstar0_xi.copy()

    loglik  = 0.0
    penalty = 0.0

    for t in range(max_lag, T):
        i = t - max_lag
        gamma_t_xi = 0.0
        if m > 0:
            gamma_t_xi = np.sum(Gm_xi)
            Gamma_arr_xi[t] = gamma_t_xi
        phi_t = G_phi[t]
        xi_t  = omega_xi + L_xi[t] + S_xi[t] + gamma_t_xi

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

        pi_t, _lpi_t, _l1mpi_t = _pi_link_terms(eta_t, pi_link)
        pi_arr[i] = pi_t

        if y[t] == 0.0:
            loglik += _l1mpi_t
        else:
            ll_dist = _gb2_logpdf(y[t], phi_t, xi_t, gamma_s, zeta_s)
            if not math.isfinite(ll_dist):
                return -1e12, penalty, G_phi, L_xi, S_xi, s_phi, s_xi, eta_arr, pi_arr, Gamma_arr_xi
            ll_t = _lpi_t + ll_dist
            if not math.isfinite(ll_t):
                return -1e12, penalty, G_phi, L_xi, S_xi, s_phi, s_xi, eta_arr, pi_arr, Gamma_arr_xi
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

        G_phi_next = omega_phi + A_phi * sp_phi + B_phi * G_phi[t]
        L_xi_next  = B_L_xi * L_xi[t] + A_L_xi * sp_xi + CL_xi[t]
        S_xi_next  = B_S_xi * S_xi[t] + A_S_xi * sp_xi + CS_xi[t]

        if not (math.isfinite(G_phi_next) and math.isfinite(L_xi_next) and math.isfinite(S_xi_next)):
            return -1e12, penalty, G_phi, L_xi, S_xi, s_phi, s_xi, eta_arr, pi_arr, Gamma_arr_xi

        G_phi[t + 1] = G_phi_next
        L_xi[t + 1]  = L_xi_next
        S_xi[t + 1]  = S_xi_next

        if m > 0:
            Gm_xi_next     = psi_gamma_xi * ( Gm_xi * cos_h + GmStar_xi * sin_h) + kappa_gamma_xi * sp_xi
            GmStar_xi_next = psi_gamma_xi * (-Gm_xi * sin_h + GmStar_xi * cos_h) + kappa_gamma_xi * sp_xi
            ok = True
            for hh in range(m):
                if not (math.isfinite(Gm_xi_next[hh]) and math.isfinite(GmStar_xi_next[hh])):
                    ok = False
            if not ok:
                return -1e12, penalty, G_phi, L_xi, S_xi, s_phi, s_xi, eta_arr, pi_arr, Gamma_arr_xi
            Gm_xi, GmStar_xi = Gm_xi_next, GmStar_xi_next

    return loglik, penalty, G_phi, L_xi, S_xi, s_phi, s_xi, eta_arr, pi_arr, Gamma_arr_xi


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
            lags_pi=pi_lags, pi_link=0,
            max_lag=ml,
            scaling=0,
            lam_state=1e3, state_lim=15.0,
            lam_eta=1e2,   eta_lim=30.0,
            pi_lam_phi=0.0,
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
            lags_pi=pi_lags, pi_link=0,
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
            lags_pi=pi_lags, pi_link=0,
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
            lags_pi=pi_lags, pi_link=0,
            max_lag=ml,
            scaling=0,
            lam_state=1e3, state_lim=15.0,
            lam_eta=1e2,   eta_lim=30.0,
            C_phi=C_dummy,
            C_xi=C_dummy,
        )

    # Harvey phi-only (no GAS lag array — lag structure is scalar B_L/B_S)
    # -- both the no-seasonal (n_harmonics=0) and seasonal (n_harmonics=3,
    # a representative small m) signatures are pre-compiled here so a
    # real harmonic-seasonal fit's FIRST call is not the one paying numba's
    # 5-20s JIT compilation cost.
    no_h = (0, np.zeros(0), np.zeros(0))
    h3   = (3, np.cos(2 * np.pi * np.arange(1, 4) / 365.25), np.sin(2 * np.pi * np.arange(1, 4) / 365.25))
    for n_harmonics, cos_h, sin_h in (no_h, h3):
        nb_harvey_phi_only_filter(
            y_dummy,
            omega=0.0, L0=0.0, A_L=0.01, B_L=0.95, S0=0.0, A_S=0.05, B_S=0.5,
            xi_s=0.0, gamma_s=0.2, zeta_s=1.5,
            pi_omega0=0.0, pi_rho=0.3,
            pi_wy=np.zeros(n_pl),
            lags_pi=pi_lags, pi_link=0,
            max_lag=ml,
            scaling=1,
            lam_state=1e3, state_lim=15.0,
            lam_eta=1e2,   eta_lim=30.0,
            CL=C_dummy, CS=C_dummy,
            n_harmonics=n_harmonics, cos_h=cos_h, sin_h=sin_h, kappa_gamma=0.05,
            seas0=np.zeros(n_harmonics), seasstar0=np.zeros(n_harmonics), psi_gamma=0.999,
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
            lags_pi=pi_lags, pi_link=0,
            max_lag=ml,
            scaling=1,
            lam_state=1e3, state_lim=15.0,
            lam_eta=1e2,   eta_lim=30.0,
            CL_phi=C_dummy, CS_phi=C_dummy,
            CL_xi=C_dummy,  CS_xi=C_dummy,
            n_harmonics=n_harmonics, cos_h=cos_h, sin_h=sin_h,
            kappa_gamma_phi=0.05, kappa_gamma_xi=0.05,
            seas0_phi=np.zeros(n_harmonics), seasstar0_phi=np.zeros(n_harmonics),
            seas0_xi=np.zeros(n_harmonics), seasstar0_xi=np.zeros(n_harmonics),
            psi_gamma_phi=0.999, psi_gamma_xi=0.999,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Regime-sensitive xi-only filter (models.regime_gas.RegimeXiOnlyModel)
# ─────────────────────────────────────────────────────────────────────────────
#
# phi_t/pi_t are frozen, precomputed arrays (not part of theta) -- only xi's
# own GAS(1,1) recursion is estimated here, with a threshold-sensitive score
# loading A_eff = A + A_ext*1(y_t > threshold_c). This mirrors
# RegimeXiOnlyModel._run_filter exactly (2026-09-06: that method previously
# ran a pure-Python loop -- ~50x slower, per the class's own historical
# comment -- solely because nobody had extended this module's existing
# numba-safe GB2 score/Fisher building blocks to include the regime branch;
# the branch itself is a plain data-dependent indicator, fully numba-safe).
# Penalty terms (persistence/score/covariate-norm bounds, which depend only
# on theta, and the state-excess bound, which is a cheap vectorized numpy
# op on the returned xi_arr) are computed in Python by the caller, exactly
# as this module's convention already is for every other filter here.

@njit(cache=True)
def nb_regime_xi_only_filter(
    y:            np.ndarray,   # (T,) full series
    phi_full:     np.ndarray,   # (T,) frozen, aligned phi path
    pi_full:      np.ndarray,   # (T,) frozen, aligned pi path
    X:            np.ndarray,   # (T, n_cov) xi covariates (nan-safe)
    gamma_v:      float,
    zeta_v:       float,
    eff:          int,          # recursion start index
    f0:           float,
    omega:        float,
    A:            float,
    B:            float,
    A_ext:        float,
    gamma_cov:    np.ndarray,   # (n_cov,)
    threshold_c:  float,
    scaling:      int,          # 0 = unit, 1 = diagonal_inverse_fisher
) -> tuple:
    """
    Returns (loglik, xi_arr, s_arr). loglik = -1e12 on numerical failure
    (matching every other filter in this module's failure convention).
    """
    T      = len(y)
    n_cov  = X.shape[1]
    xi_arr = np.zeros(T + 1)
    s_arr  = np.zeros(T)
    for k in range(eff + 1):
        xi_arr[k] = f0

    loglik = 0.0

    for t in range(eff, T):
        phi_t = phi_full[t]
        if not math.isfinite(phi_t):
            return -1e12, xi_arr, s_arr
        xi_t = xi_arr[t]

        pi_t = pi_full[t]
        if pi_t < 1e-12:
            pi_t = 1e-12
        elif pi_t > 1.0 - 1e-12:
            pi_t = 1.0 - 1e-12

        if y[t] == 0.0:
            ll_t = math.log(1.0 - pi_t)
        else:
            ll_dist = _gb2_logpdf(y[t], phi_t, xi_t, gamma_v, zeta_v)
            if not math.isfinite(ll_dist):
                return -1e12, xi_arr, s_arr
            ll_t = math.log(pi_t) + ll_dist
        if not math.isfinite(ll_t):
            return -1e12, xi_arr, s_arr
        loglik += ll_t

        s_t = 0.0
        if y[t] > 0.0:
            s_raw = _gb2_score_xi(y[t], phi_t, xi_t, gamma_v, zeta_v)
            if scaling == 1 or scaling == 2:  # one parameter: diagonal == full inverse FI
                fi = _gb2_fi_xi(xi_t, zeta_v)
                if fi > 1e-8 and math.isfinite(fi):
                    s_t = s_raw / fi
                    if not math.isfinite(s_t):
                        s_t = s_raw
                else:
                    s_t = s_raw
            else:
                s_t = s_raw
            if not math.isfinite(s_t):
                return -1e12, xi_arr, s_arr
        s_arr[t] = s_t

        regime_t = 1.0 if y[t] > threshold_c else 0.0
        A_eff = A + A_ext * regime_t
        cov_term = 0.0
        for k in range(n_cov):
            cov_term += gamma_cov[k] * X[t, k]
        xi_next = omega + A_eff * s_t + B * xi_t + cov_term
        if not math.isfinite(xi_next):
            return -1e12, xi_arr, s_arr
        xi_arr[t + 1] = xi_next

    return loglik, xi_arr, s_arr


_warmup()
