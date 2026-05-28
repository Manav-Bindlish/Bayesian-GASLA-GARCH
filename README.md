# GASLa & GASLa-GARCH(1,1) — Bayesian Estimation

Bayesian estimation of the Generalised Alpha Skew Laplace (GASLa) distribution and
GASLa-GARCH(1,1) volatility model via a from-scratch NumPy MHwG sampler.

## Structure

```
.
├── run_analysis.py          # Entry point — runs all phases
└── src/
    ├── gasla_core.py        # Normalising constant, piecewise log-PDF, standardised moments
    ├── generators.py        # Acceptance-rejection sampler for GASLa variates
    ├── mhwg_sampler.py      # Metropolis-Hastings within Gibbs engine (Robbins-Monro adaptive tuning)
    ├── estimators.py        # SEL / LINEX / GEL Bayes estimators, Chen-Shao HPD intervals
    └── garch_model.py       # GARCH(1,1) variance recursion, GASLa-GARCH conditional likelihood, AIC/BIC
```

## Dependencies

```
numpy scipy
```

## Execution

```bash
pip install numpy scipy
python run_analysis.py
```

## What it does

1. Simulates GASLa(0, 1, α=2, λ=1.5) data and runs the MHwG sampler to recover
   the posterior, reporting SEL (posterior mean), LINEX (c=0.5), and GEL (q=0.5)
   Bayes estimators with 95% HPD credible intervals.

2. Simulates a bimodal two-regime Laplace return series and fits both
   GASLa-GARCH(1,1) and Normal-GARCH(1,1), printing AIC/BIC to confirm
   GASLa-GARCH outperforms the Normal benchmark.

3. Repeats the GARCH comparison on GASLa-generated returns.

## Key design choices

| Component | Choice |
|---|---|
| MCMC algorithm | MH-within-Gibbs, one parameter per block |
| Proposals for μ, α | Symmetric Normal random-walk (zero log-proposal ratio) |
| Proposals for β, λ | Log-Normal random-walk with Jacobian correction |
| Adaptation | Robbins-Monro, target AR = 0.44, stops at burn-in end |
| Loss functions | SEL, LINEX (c=0.5), GEL (q=0.5) |
| Credible intervals | HPD via Chen-Shao algorithm |
| GARCH stationarity | α_G + β_G < 1 enforced via softmax reparameterisation |
| GARCH standardisation | E[z_t]=0, Var[z_t]=1 via analytical GASLa moments |
