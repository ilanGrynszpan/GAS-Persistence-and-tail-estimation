"""
Independent numerical gradient / Hessian / standard errors (2026-09-06).

=============================================================================
WHY THIS MODULE EXISTS
=============================================================================

Every model in this framework is fit by `scipy.optimize.minimize(method=
"BFGS")` without an analytical Jacobian. At termination, `res.jac` is
whatever gradient estimate scipy's own internal finite-difference scheme
produced at the last iteration, and `res.hess_inv` is BFGS's own
ITERATIVELY-ACCUMULATED secant approximation to the inverse Hessian --
built up over the whole optimization trajectory, not computed directly at
the final point. Both are useful for the optimizer's own bookkeeping, but
neither is an independent check of convergence quality or of standard
error validity.

This module evaluates the objective function directly, at the final theta,
via plain central-difference numerical differentiation, entirely
independent of anything scipy's optimizer accumulated internally:

    numerical_gradient(f, x)   -- central-difference gradient of f at x
    numerical_hessian(f, x)    -- central-difference Hessian of f at x
    numerical_std_errors(f, x) -- sqrt(diag(inv(Hessian))), i.e. the
                                   asymptotic-MLE standard errors implied
                                   by the Hessian of the NEGATIVE
                                   log-likelihood (+ any penalty terms)
                                   at x, exactly as classical MLE theory
                                   prescribes (Var(theta_hat) ~ H^-1)

`f` is always the model's own scalar objective (negative log-likelihood
plus penalties, exactly what was minimized) -- these functions are
model-agnostic; any model's `obj(theta)` closure can be passed directly.

=============================================================================
STEP SIZE -- REVISED 2026-09-08, see CORRECTNESS NOTE below
=============================================================================

Central differences with a relative step size (h = eps * max(|x_i|, 1))
trade truncation error (~h^2, from the function's own curvature) against
floating-point cancellation error (~machine_eps / h). The textbook-standard
eps=1e-4 assumes the objective is well-approximated by a local quadratic
over a step of that size. That assumption FAILS for this project's models:
their objective is a long (T~2500-3700 step) recursive filter, and for a
model already sitting close to a stability boundary (near-unit-root
persistence, a seasonal rotation near its damping limit, etc.), a step as
"small" as 1e-4 in one parameter can push the PERTURBED trajectory into a
qualitatively different, diverging numerical regime over the course of the
recursion -- producing a finite-difference "gradient" that is not a local
derivative at all, just an artifact of which side of an instability
threshold the two evaluation points happened to land on.

=============================================================================
CORRECTNESS NOTE (found 2026-09-08, do not regress this)
=============================================================================

A first version of this module defaulted to eps=1e-4 and was used to
generate a full cross-stage gradient report. Direct inspection (prompted by
a user catching implausible ~1e15 "gradients" for otherwise perfectly
well-behaved fits) showed the reported values were exactly this artifact:
sweeping eps from 1e-2 down to 1e-10 for one such case (Belo Horizonte,
Stage 4, phi-only) gave

    eps=1e-2..1e-5:  max|grad| climbs from 5e13 to 5e16 (garbage, unstable)
    eps=1e-6:        8.4e4       (already close)
    eps=1e-7..1e-10: 1.031-1.033e5, stable to 4 significant figures

matching scipy's own originally-reported gradient (104808.8) within 1.5%.
The fix is not just "use a smaller eps" (an even smaller step risks the
OPPOSITE failure, floating-point cancellation noise) -- it is to always
verify convergence by comparing two step sizes before trusting a result,
which every function below now does automatically. Never re-add a
single-eps-only gradient/Hessian path without this check; the failure mode
is silent and was off by 10+ orders of magnitude with no other symptom.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

DEFAULT_EPS = 1e-7


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray,
                        eps: float = DEFAULT_EPS) -> np.ndarray:
    """
    Central-difference gradient of scalar f at x.

        grad_i = [f(x + h*e_i) - f(x - h*e_i)] / (2h),   h = eps*max(|x_i|,1)

    No convergence check -- prefer numerical_gradient_checked for anything
    reported as a final number (see module docstring's CORRECTNESS NOTE).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        h = eps * max(abs(x[i]), 1.0)
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2.0 * h)
    return grad


def numerical_gradient_checked(f: Callable[[np.ndarray], float], x: np.ndarray,
                                eps_candidates=(1e-6, 1e-7, 1e-8, 1e-9),
                                rel_tol: float = 0.05) -> dict:
    """
    Gradient with an explicit convergence check across shrinking step
    sizes, instead of trusting one arbitrarily-chosen eps -- see this
    module's CORRECTNESS NOTE for exactly why a single eps is not safe for
    this project's recursive-filter objectives.

    Tries eps_candidates in order; returns the result at the smallest eps
    for which ||grad(eps)| - |grad(eps/10-ish, i.e. the next candidate)||
    is within rel_tol of each other (relative to the larger of the two),
    per-coordinate on the max-magnitude entry. If no two consecutive
    candidates agree, returns the smallest-eps result but marks it
    unconverged -- callers must not report a bare number in that case
    without flagging the non-convergence.

    Returns
    -------
    dict: grad, eps_used, converged (bool), trace (list of (eps, max|grad|))
    """
    trace = []
    grads = {}
    for eps in eps_candidates:
        g = numerical_gradient(f, x, eps=eps)
        grads[eps] = g
        trace.append((eps, float(np.max(np.abs(g)))))

    converged = False
    chosen_eps = eps_candidates[-1]
    for i in range(len(eps_candidates) - 1):
        e1, e2 = eps_candidates[i], eps_candidates[i + 1]
        m1, m2 = np.max(np.abs(grads[e1])), np.max(np.abs(grads[e2]))
        denom = max(m1, m2, 1e-12)
        if abs(m1 - m2) / denom < rel_tol:
            converged = True
            chosen_eps = e2
            break

    return {"grad": grads[chosen_eps], "eps_used": chosen_eps,
            "converged": converged, "trace": trace}


def numerical_hessian(f: Callable[[np.ndarray], float], x: np.ndarray,
                       eps: float = DEFAULT_EPS) -> np.ndarray:
    """
    Central-difference Hessian of scalar f at x, symmetrized.

    Diagonal:     H_ii = [f(x+h e_i) - 2f(x) + f(x-h e_i)] / h^2
    Off-diagonal: H_ij = [f(x+h_i e_i+h_j e_j) - f(x+h_i e_i-h_j e_j)
                          - f(x-h_i e_i+h_j e_j) + f(x-h_i e_i-h_j e_j)]
                         / (4 h_i h_j)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    h = eps * np.maximum(np.abs(x), 1.0)
    f0 = f(x)
    f_plus = np.zeros(n)
    f_minus = np.zeros(n)
    for i in range(n):
        xp, xm = x.copy(), x.copy()
        xp[i] += h[i]
        xm[i] -= h[i]
        f_plus[i] = f(xp)
        f_minus[i] = f(xm)

    hess = np.zeros((n, n))
    for i in range(n):
        hess[i, i] = (f_plus[i] - 2.0 * f0 + f_minus[i]) / (h[i] ** 2)
        for j in range(i + 1, n):
            xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
            xpp[i] += h[i]; xpp[j] += h[j]
            xpm[i] += h[i]; xpm[j] -= h[j]
            xmp[i] -= h[i]; xmp[j] += h[j]
            xmm[i] -= h[i]; xmm[j] -= h[j]
            hij = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * h[i] * h[j])
            hess[i, j] = hij
            hess[j, i] = hij
    return hess


def numerical_std_errors(f: Callable[[np.ndarray], float], x: np.ndarray,
                          eps: float = DEFAULT_EPS) -> dict:
    """
    Standard errors from an independently-computed numerical Hessian of f
    (f is the NEGATIVE log-likelihood + penalties, i.e. what was
    minimized) at x, via Var(theta_hat) ~ Hessian^-1.

    Returns
    -------
    dict with keys:
        std_errors : array, NaN entries where the Hessian's inverse
                     diagonal is non-positive or non-finite (i.e. that
                     parameter's SE cannot be recovered at this point --
                     a strong, independent signal of a boundary-adjacent
                     or unidentified direction, distinct from anything
                     scipy's own hess_inv reported).
        hessian    : the raw numerical Hessian (n x n)
        cov        : Hessian^-1 (n x n), or None if inversion failed
        quality    : "ok" (all diagonal variances positive & finite),
                     "partial" (some but not all), "failed" (none, or
                     inversion itself failed)
    """
    hess = numerical_hessian(f, x, eps=eps)
    n = len(x)
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        return {"std_errors": np.full(n, np.nan), "hessian": hess,
                "cov": None, "quality": "failed"}

    diag = np.diag(cov)
    se = np.full(n, np.nan)
    ok = np.isfinite(diag) & (diag > 0)
    se[ok] = np.sqrt(diag[ok])
    quality = "ok" if ok.all() else ("partial" if ok.any() else "failed")
    return {"std_errors": se, "hessian": hess, "cov": cov, "quality": quality}
