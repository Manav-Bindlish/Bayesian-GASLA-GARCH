import numpy as np
from scipy.optimize import minimize
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from gasla_core import normalising_constant, log_pdf_std, moments_std


def garch_variance_recursion(
    eps: np.ndarray, omega: float, a_g: float, b_g: float
) -> np.ndarray:
    T = len(eps)
    sigma2 = np.empty(T)
    denom = max(1.0 - a_g - b_g, 1e-8)
    sigma2[0] = omega / denom
    for t in range(1, T):
        sigma2[t] = omega + a_g * eps[t - 1] ** 2 + b_g * sigma2[t - 1]
    return sigma2


def _gasla_garch_loglik_natural(
    r: np.ndarray,
    mu: float,
    omega: float,
    a_g: float,
    b_g: float,
    alpha: float,
    lam: float,
) -> float:
    if omega <= 0.0 or a_g < 0.0 or b_g < 0.0 or lam <= 0.0:
        return -np.inf
    if a_g + b_g >= 1.0:
        return -np.inf

    C = normalising_constant(alpha, lam)
    if C <= 0.0:
        return -np.inf

    mu_z, tau_z = moments_std(alpha, lam)
    if tau_z <= 0.0 or not np.isfinite(tau_z):
        return -np.inf

    eps = r - mu
    sigma2 = garch_variance_recursion(eps, omega, a_g, b_g)
    if np.any(sigma2 <= 0.0):
        return -np.inf

    sigma = np.sqrt(sigma2)
    z_t = eps / sigma

    C1 = 1.0 + lam
    ll = 0.0
    log_tau_z = np.log(tau_z)
    log_C = np.log(C)
    log_2C = np.log(2.0 * C)

    for i in range(len(r)):
        u = mu_z + tau_z * z_t[i]
        w = (1.0 - alpha * u) ** 2 + 1.0
        if w <= 0.0:
            return -np.inf
        if u < 0.0:
            lf = np.log(w) - log_2C + C1 * u
        else:
            inner = 2.0 - np.exp(-lam * u)
            if inner <= 0.0:
                return -np.inf
            lf = np.log(w) - log_2C - u + np.log(inner)
        ll += lf - log_tau_z - np.log(sigma[i])

    return ll if np.isfinite(ll) else -np.inf


def _pack(mu, omega, a_g, b_g, alpha, lam):
    s = a_g + b_g
    if s >= 1.0:
        s = 0.999
    v1 = np.log(a_g / (s + 1e-12))
    v2 = np.log(b_g / (s + 1e-12))
    return np.array([mu, np.log(omega), np.log(s / (1.0 - s)), v1, alpha, np.log(lam)])


def _unpack(psi):
    mu = psi[0]
    omega = np.exp(psi[1])
    s = 1.0 / (1.0 + np.exp(-psi[2]))
    s = min(s, 0.9999)
    e_v1 = np.exp(psi[3])
    e_v2 = np.exp(psi[4]) if len(psi) > 5 else 1.0
    denom = e_v1 + e_v2 + 1e-12
    a_g = s * e_v1 / denom
    b_g = s * e_v2 / denom
    alpha = psi[5] if len(psi) > 5 else psi[4]
    lam = np.exp(psi[6] if len(psi) > 6 else psi[5])
    return mu, omega, a_g, b_g, alpha, lam


def _unpack6(psi):
    mu = psi[0]
    omega = np.exp(psi[1])
    s = 1.0 / (1.0 + np.exp(-psi[2]))
    s = min(s, 0.9999)
    e_v1 = np.exp(psi[3])
    e_v2 = 1.0
    denom = e_v1 + e_v2 + 1e-12
    a_g = s * e_v1 / denom
    b_g = s * e_v2 / denom
    alpha = psi[4]
    lam = np.exp(psi[5])
    return mu, omega, a_g, b_g, alpha, lam


def _neg_ll_gasla(psi, r):
    mu, omega, a_g, b_g, alpha, lam = _unpack6(psi)
    return -_gasla_garch_loglik_natural(r, mu, omega, a_g, b_g, alpha, lam)


def _normal_garch_loglik(r, mu, omega, a_g, b_g):
    if omega <= 0.0 or a_g < 0.0 or b_g < 0.0 or a_g + b_g >= 1.0:
        return -np.inf
    eps = r - mu
    sigma2 = garch_variance_recursion(eps, omega, a_g, b_g)
    if np.any(sigma2 <= 0.0):
        return -np.inf
    ll = -0.5 * np.sum(np.log(2.0 * np.pi * sigma2) + eps ** 2 / sigma2)
    return ll if np.isfinite(ll) else -np.inf


def _unpack_normal(psi):
    mu = psi[0]
    omega = np.exp(psi[1])
    s = 1.0 / (1.0 + np.exp(-psi[2]))
    s = min(s, 0.9999)
    e_v = np.exp(psi[3])
    a_g = s * e_v / (e_v + 1.0 + 1e-12)
    b_g = s * 1.0 / (e_v + 1.0 + 1e-12)
    return mu, omega, a_g, b_g


def _neg_ll_normal(psi, r):
    mu, omega, a_g, b_g = _unpack_normal(psi)
    return -_normal_garch_loglik(r, mu, omega, a_g, b_g)


def fit_gasla_garch(r: np.ndarray, verbose: bool = True) -> dict:
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med))) + 1e-4
    psi0 = np.array([med, np.log(mad ** 2), np.log(0.85 / 0.15), np.log(0.3), 0.5, np.log(1.0)])

    res = minimize(
        _neg_ll_gasla,
        psi0,
        args=(r,),
        method="Nelder-Mead",
        options={"maxiter": 20_000, "xatol": 1e-6, "fatol": 1e-6, "adaptive": True},
    )

    mu, omega, a_g, b_g, alpha, lam = _unpack6(res.x)
    ll = -res.fun
    n_params = 6
    n = len(r)
    aic = -2.0 * ll + 2.0 * n_params
    bic = -2.0 * ll + n_params * np.log(n)

    if verbose:
        print(f"\n=== GASLa-GARCH(1,1) Fit ===")
        print(f"  mu={mu:.4f}  omega={omega:.6f}  alpha_G={a_g:.4f}  beta_G={b_g:.4f}")
        print(f"  alpha={alpha:.4f}  lambda={lam:.4f}")
        print(f"  alpha_G + beta_G = {a_g + b_g:.4f}  [stationarity: {'OK' if a_g+b_g < 1 else 'VIOLATED'}]")
        print(f"  LogLik={ll:.4f}  AIC={aic:.4f}  BIC={bic:.4f}")

    return dict(mu=mu, omega=omega, alpha_G=a_g, beta_G=b_g, alpha=alpha, lam=lam,
                loglik=ll, aic=aic, bic=bic, n_params=n_params, converged=res.success)


def fit_normal_garch(r: np.ndarray, verbose: bool = True) -> dict:
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med))) + 1e-4
    psi0 = np.array([med, np.log(mad ** 2), np.log(0.85 / 0.15), np.log(0.3)])

    res = minimize(
        _neg_ll_normal,
        psi0,
        args=(r,),
        method="Nelder-Mead",
        options={"maxiter": 20_000, "xatol": 1e-6, "fatol": 1e-6, "adaptive": True},
    )

    mu, omega, a_g, b_g = _unpack_normal(res.x)
    ll = -res.fun
    n_params = 4
    n = len(r)
    aic = -2.0 * ll + 2.0 * n_params
    bic = -2.0 * ll + n_params * np.log(n)

    if verbose:
        print(f"\n=== Normal-GARCH(1,1) Fit ===")
        print(f"  mu={mu:.4f}  omega={omega:.6f}  alpha_G={a_g:.4f}  beta_G={b_g:.4f}")
        print(f"  alpha_G + beta_G = {a_g + b_g:.4f}  [stationarity: {'OK' if a_g+b_g < 1 else 'VIOLATED'}]")
        print(f"  LogLik={ll:.4f}  AIC={aic:.4f}  BIC={bic:.4f}")

    return dict(mu=mu, omega=omega, alpha_G=a_g, beta_G=b_g,
                loglik=ll, aic=aic, bic=bic, n_params=n_params, converged=res.success)


def model_comparison_table(gasla_fit: dict, normal_fit: dict) -> None:
    print("\n" + "=" * 60)
    print("MODEL COMPARISON: GASLa-GARCH vs Normal-GARCH")
    print("=" * 60)
    print(f"{'Metric':<20} {'Normal-GARCH':>16} {'GASLa-GARCH':>16}")
    print("-" * 54)
    for key in ("loglik", "aic", "bic"):
        label = key.upper()
        better = "GASLa" if (
            (key == "loglik" and gasla_fit[key] > normal_fit[key]) or
            (key != "loglik" and gasla_fit[key] < normal_fit[key])
        ) else "Normal"
        print(f"{label:<20} {normal_fit[key]:16.4f} {gasla_fit[key]:16.4f}  <- {better} wins")
    delta_aic = normal_fit["aic"] - gasla_fit["aic"]
    delta_bic = normal_fit["bic"] - gasla_fit["bic"]
    print(f"\n  ΔAIC (Normal - GASLa) = {delta_aic:+.4f}")
    print(f"  ΔBIC (Normal - GASLa) = {delta_bic:+.4f}")
    if delta_aic > 0 and delta_bic > 0:
        print("  => GASLa-GARCH is PREFERRED by both AIC and BIC")
    elif delta_aic > 0:
        print("  => GASLa-GARCH preferred by AIC")
    elif delta_bic > 0:
        print("  => GASLa-GARCH preferred by BIC")
    else:
        print("  => Normal-GARCH preferred (unexpected for bimodal data)")
    print("=" * 60)
