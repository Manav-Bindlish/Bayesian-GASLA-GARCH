import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generators import rgasla
from gasla_core import moments_std
from mhwg_sampler import run_mhwg, HyperParams
from estimators import compute_bayes_estimates, compute_hpd, posterior_summary
from garch_model import fit_gasla_garch, fit_normal_garch, model_comparison_table

SEED = 2025
RNG = np.random.default_rng(SEED)


def simulate_bimodal_returns(n: int = 2000, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    regime = rng.binomial(1, 0.70, size=n)
    u_calm = rng.uniform(-0.5, 0.5, size=n)
    u_stress = rng.uniform(-0.5, 0.5, size=n)
    lap_calm = 0.002 - 0.08 * np.sign(u_calm) * np.log1p(-2.0 * np.abs(u_calm))
    lap_stress = -0.05 - 0.25 * np.sign(u_stress) * np.log1p(-2.0 * np.abs(u_stress))
    returns = np.where(regime == 1, lap_calm, lap_stress)
    returns += rng.normal(0.0, 0.02, size=n)
    return returns


def simulate_gasla_returns(n: int = 500, seed: int = SEED) -> np.ndarray:
    return rgasla(n, mu=0.0, beta=1.0, alpha=2.0, lam=1.5, seed=seed)


def print_header(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def run_bayesian_analysis(y: np.ndarray, label: str) -> dict:
    print_header(f"Bayesian MHwG Analysis — {label}")

    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med)))
    hyper = HyperParams(
        mu0=med,
        sigma_mu=10.0 * float(np.std(y)),
        a_beta=2.0,
        b_beta=max(mad, 1e-3),
        mu_alpha=0.0,
        sigma_alpha=10.0,
        a_lambda=2.0,
        b_lambda=1.0,
    )

    result = run_mhwg(
        y,
        n_iter=30_000,
        n_burnin=8_000,
        n_thin=5,
        hyper=hyper,
        seed=SEED,
        verbose=True,
    )

    print(f"\nPosterior draws retained: {result.draws.shape[0]}")
    print(f"Acceptance rates: {dict(zip(result.param_names, result.accept_rate.round(3)))}")
    print(f"Final proposal SDs: {dict(zip(result.param_names, result.final_tau.round(4)))}")

    est = compute_bayes_estimates(result.draws, c_linex=0.5, q_gel=0.5)
    print(f"\n--- Bayes Estimates (c_LINEX={est.c_linex}, q_GEL={est.q_gel}) ---")
    print(est.summary())

    hpd = compute_hpd(result.draws, level=0.95)
    print(f"\n--- 95% HPD Credible Intervals (Chen-Shao) ---")
    print(hpd.summary())

    summ = posterior_summary(result.draws, result.param_names)
    print(f"\n--- Posterior Quantile Summary ---")
    print(f"{'Param':<8} {'Mean':>10} {'SD':>10} {'Q2.5':>10} {'Median':>10} {'Q97.5':>10}")
    print("-" * 55)
    for i, p in enumerate(summ["param_names"]):
        print(
            f"{p:<8} {summ['mean'][i]:10.4f} {summ['sd'][i]:10.4f} "
            f"{summ['q2_5'][i]:10.4f} {summ['median'][i]:10.4f} {summ['q97_5'][i]:10.4f}"
        )

    return dict(result=result, estimates=est, hpd=hpd)


def run_garch_analysis(r: np.ndarray) -> None:
    print_header("GASLa-GARCH(1,1) vs Normal-GARCH(1,1)")
    print(f"Series length: {len(r)}")
    print(f"Mean: {r.mean():.4f}  Std: {r.std():.4f}  "
          f"Skew: {float(((r - r.mean())**3).mean() / r.std()**3):.4f}  "
          f"ExKurt: {float(((r - r.mean())**4).mean() / r.std()**4) - 3:.4f}")

    gasla_fit = fit_gasla_garch(r, verbose=True)
    normal_fit = fit_normal_garch(r, verbose=True)
    model_comparison_table(gasla_fit, normal_fit)


def main() -> None:
    print_header("PHASE 1 — Bayesian GASLa Estimation (Simulated Data)")
    y_gasla = simulate_gasla_returns(n=300, seed=SEED)
    print(f"Simulated {len(y_gasla)} GASLa(0, 1, alpha=2.0, lambda=1.5) observations")
    bayes_out = run_bayesian_analysis(y_gasla, label="GASLa Simulated Returns")

    print_header("PHASE 2 — GARCH Model Comparison (Bimodal Returns)")
    r_bimodal = simulate_bimodal_returns(n=2000, seed=SEED)
    print(f"Simulated {len(r_bimodal)} bimodal return observations (two-regime Laplace mixture)")
    run_garch_analysis(r_bimodal)

    print_header("PHASE 3 — GARCH Model Comparison (GASLa-Generated Returns)")
    r_gasla = rgasla(1500, mu=0.0, beta=0.5, alpha=1.5, lam=1.2, seed=SEED + 1)
    run_garch_analysis(r_gasla)

    print_header("ANALYSIS COMPLETE")
    print("All phases executed successfully.")
    print("Key outputs:")
    print("  - MHwG posterior draws with SEL / LINEX / GEL Bayes estimators")
    print("  - 95% HPD credible intervals via Chen-Shao algorithm")
    print("  - AIC / BIC model comparison: GASLa-GARCH vs Normal-GARCH")


if __name__ == "__main__":
    main()
