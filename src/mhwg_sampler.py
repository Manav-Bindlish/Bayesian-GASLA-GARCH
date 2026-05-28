import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from gasla_core import normalising_constant, log_pdf_std


@dataclass
class HyperParams:
    mu0: float
    sigma_mu: float
    a_beta: float
    b_beta: float
    mu_alpha: float
    sigma_alpha: float
    a_lambda: float
    b_lambda: float


@dataclass
class MCMCResult:
    draws: np.ndarray
    accept_rate: np.ndarray
    final_tau: np.ndarray
    hyper: HyperParams
    n_iter: int
    n_burnin: int
    n_thin: int
    data: np.ndarray
    param_names: list = field(default_factory=lambda: ["mu", "beta", "alpha", "lambda"])


def _default_hyper(y: np.ndarray) -> HyperParams:
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med)))
    return HyperParams(
        mu0=med,
        sigma_mu=10.0 * float(np.std(y)),
        a_beta=2.0,
        b_beta=max(mad, 1e-3),
        mu_alpha=0.0,
        sigma_alpha=10.0,
        a_lambda=2.0,
        b_lambda=1.0,
    )


def _log_prior(mu, beta, alpha, lam, h: HyperParams) -> float:
    if beta <= 0.0 or lam <= 0.0:
        return -np.inf
    lp = -0.5 * ((mu - h.mu0) / h.sigma_mu) ** 2
    lp += -(h.a_beta + 1.0) * np.log(beta) - h.b_beta / beta
    lp += -0.5 * ((alpha - h.mu_alpha) / h.sigma_alpha) ** 2
    lp += (h.a_lambda - 1.0) * np.log(lam) - h.b_lambda * lam
    return lp


def _log_likelihood(y: np.ndarray, mu: float, beta: float, alpha: float, lam: float) -> float:
    if beta <= 0.0 or lam <= 0.0:
        return -np.inf
    C = normalising_constant(alpha, lam)
    if C <= 0.0:
        return -np.inf
    n = len(y)
    z = (y - mu) / beta
    ll = -n * np.log(beta)
    C1 = 1.0 + lam
    for zi in z:
        w = (1.0 - alpha * zi) ** 2 + 1.0
        if zi < 0.0:
            ll += np.log(w) - np.log(2.0 * C) + C1 * zi
        else:
            inner = 2.0 - np.exp(-lam * zi)
            if inner <= 0.0:
                return -np.inf
            ll += np.log(w) - np.log(2.0 * C) - zi + np.log(inner)
    return ll


def _log_likelihood_vec(y: np.ndarray, mu: float, beta: float, alpha: float, lam: float) -> float:
    if beta <= 0.0 or lam <= 0.0:
        return -np.inf
    C = normalising_constant(alpha, lam)
    if C <= 0.0:
        return -np.inf
    n = len(y)
    z = (y - mu) / beta
    ll = -n * np.log(beta)
    C1 = 1.0 + lam
    neg = z < 0.0
    pos = ~neg
    if neg.any():
        zn = z[neg]
        wn = (1.0 - alpha * zn) ** 2 + 1.0
        ll += np.sum(np.log(wn) - np.log(2.0 * C) + C1 * zn)
    if pos.any():
        zp = z[pos]
        wp = (1.0 - alpha * zp) ** 2 + 1.0
        inner = 2.0 - np.exp(-lam * zp)
        if np.any(inner <= 0.0):
            return -np.inf
        ll += np.sum(np.log(wp) - np.log(2.0 * C) - zp + np.log(inner))
    if not np.isfinite(ll):
        return -np.inf
    return ll


def _log_posterior(theta: np.ndarray, y: np.ndarray, h: HyperParams) -> float:
    mu, beta, alpha, lam = theta
    ll = _log_likelihood_vec(y, mu, beta, alpha, lam)
    if not np.isfinite(ll):
        return -np.inf
    lp = _log_prior(mu, beta, alpha, lam, h)
    if not np.isfinite(lp):
        return -np.inf
    return ll + lp


_TAU_MIN = 1e-4
_TAU_MAX = 10.0
_POS_FLOOR = 1e-9


def run_mhwg(
    y: np.ndarray,
    n_iter: int = 50_000,
    n_burnin: int = 10_000,
    n_thin: int = 5,
    init: Optional[np.ndarray] = None,
    hyper: Optional[HyperParams] = None,
    tau_init: Optional[np.ndarray] = None,
    adapt: bool = True,
    adapt_end: Optional[int] = None,
    target_ar: float = 0.44,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> MCMCResult:
    rng = np.random.default_rng(seed)

    if hyper is None:
        hyper = _default_hyper(y)

    if init is None:
        med = float(np.median(y))
        mad = float(np.median(np.abs(y - med)))
        init = np.array([med, max(mad, 1e-2), 0.5, 1.0])

    if tau_init is None:
        tau_init = np.array([0.5, 0.3, 0.5, 0.3])

    if adapt_end is None:
        adapt_end = n_burnin

    n_params = 4
    n_store = (n_iter - n_burnin) // n_thin
    draws = np.empty((n_store, n_params))

    theta = init.copy().astype(float)
    lp_curr = _log_posterior(theta, y, hyper)
    tau = tau_init.copy().astype(float)
    accept_count = np.zeros(n_params)

    store_idx = 0

    for t in range(1, n_iter + 1):

        # --- mu: symmetric Normal random-walk ---
        mu_prop = theta[0] + rng.normal(0.0, tau[0])
        theta_prop = theta.copy()
        theta_prop[0] = mu_prop
        lp_prop = _log_posterior(theta_prop, y, hyper)
        log_ar = lp_prop - lp_curr
        if np.log(rng.uniform()) < log_ar:
            theta = theta_prop
            lp_curr = lp_prop
            accept_count[0] += 1

        # --- beta: log-Normal random-walk, Jacobian correction ---
        beta_curr = max(theta[1], _POS_FLOOR)
        log_beta_curr = np.log(beta_curr)
        log_beta_prop = log_beta_curr + rng.normal(0.0, tau[1])
        beta_prop = np.exp(log_beta_prop)
        theta_prop = theta.copy()
        theta_prop[1] = beta_prop
        lp_prop = _log_posterior(theta_prop, y, hyper)
        log_ar = lp_prop - lp_curr + log_beta_curr - log_beta_prop
        if np.log(rng.uniform()) < log_ar:
            theta = theta_prop
            lp_curr = lp_prop
            accept_count[1] += 1

        # --- alpha: symmetric Normal random-walk ---
        alpha_prop = theta[2] + rng.normal(0.0, tau[2])
        theta_prop = theta.copy()
        theta_prop[2] = alpha_prop
        lp_prop = _log_posterior(theta_prop, y, hyper)
        log_ar = lp_prop - lp_curr
        if np.log(rng.uniform()) < log_ar:
            theta = theta_prop
            lp_curr = lp_prop
            accept_count[2] += 1

        # --- lambda: log-Normal random-walk, Jacobian correction ---
        lam_curr = max(theta[3], _POS_FLOOR)
        log_lam_curr = np.log(lam_curr)
        log_lam_prop = log_lam_curr + rng.normal(0.0, tau[3])
        lam_prop = np.exp(log_lam_prop)
        theta_prop = theta.copy()
        theta_prop[3] = lam_prop
        lp_prop = _log_posterior(theta_prop, y, hyper)
        log_ar = lp_prop - lp_curr + log_lam_curr - log_lam_prop
        if np.log(rng.uniform()) < log_ar:
            theta = theta_prop
            lp_curr = lp_prop
            accept_count[3] += 1

        # --- Robbins-Monro adaptive tuning ---
        if adapt and t <= adapt_end:
            gamma = t ** (-0.6)
            emp_ar = accept_count / t
            log_tau = np.log(tau) + gamma * (emp_ar - target_ar)
            tau = np.clip(np.exp(log_tau), _TAU_MIN, _TAU_MAX)

        # --- store post-burnin thinned draws ---
        if t > n_burnin and (t - n_burnin) % n_thin == 0:
            draws[store_idx] = theta
            store_idx += 1

        if verbose and t % max(1, n_iter // 10) == 0:
            ar = accept_count / t
            print(
                f"Iter {t:6d}/{n_iter} | AR: mu={ar[0]:.3f} beta={ar[1]:.3f} "
                f"alpha={ar[2]:.3f} lam={ar[3]:.3f} | "
                f"theta: {theta[0]:.3f} {theta[1]:.3f} {theta[2]:.3f} {theta[3]:.3f}"
            )

    final_ar = accept_count / n_iter

    if verbose:
        print(f"\n=== MCMC Complete | AR: mu={final_ar[0]:.3f} beta={final_ar[1]:.3f} "
              f"alpha={final_ar[2]:.3f} lam={final_ar[3]:.3f} ===")

    return MCMCResult(
        draws=draws[:store_idx],
        accept_rate=final_ar,
        final_tau=tau,
        hyper=hyper,
        n_iter=n_iter,
        n_burnin=n_burnin,
        n_thin=n_thin,
        data=y,
    )
