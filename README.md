# GASLa–GARCH Bayesian Estimation

This repository implements the GASLa distribution, MH-within-Gibbs Bayesian inference, and GASLa-GARCH(1,1) modeling with diagnostics and VaR/ES forecasts.

## Folder Structure

```
research/
├─ run_analysis.py
├─ requirements.txt
├─ README.md
└─ src/
   ├─ gasla_core.py
   ├─ generators.py
   ├─ mhwg_sampler.py
   ├─ estimators.py
   ├─ garch_model.py
   └─ mcmc_diagnostics.py
```

## Run From Scratch (No Environment Installed)

```cmd
cd C:\Users\Admin\Desktop\research
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_analysis.py
```

## What You’ll See

- AIC/BIC comparison (GASLa-GARCH vs Normal-GARCH)
- Bayes estimates (SEL/LINEX/GEL)
- MCMC diagnostics (ESS, R-hat, ACF)
- Multi-horizon VaR/ES forecasts
- Diagnostic plots saved to `outputs/plots`

## Equations Used (Exactly What the Code Uses)

### 1) GASLa Normalizing Constant

$$
C(\alpha,\lambda)=2\left(1+\alpha^2-\frac{\alpha\lambda(2+\lambda)}{(1+\lambda)^2}\right)
$$

### 2) GASLa Base PDF (Piecewise)

Let $z=(y-\mu)/\beta$, $C_1=1+\lambda$.

$$
f(z;\alpha,\lambda)=\begin{cases}
\dfrac{(1-\alpha z)^2+1}{2C(\alpha,\lambda)}\,e^{C_1 z}, & z<0\\
\dfrac{(1-\alpha z)^2+1}{2C(\alpha,\lambda)}\,e^{-z}\left(2-e^{-\lambda z}\right), & z\ge 0
\end{cases}
$$

### 3) GASLa Log-Likelihood (Location-Scale)

Let $\mathcal{I}^- = \{i:z_i<0\}$ and $\mathcal{I}^+ = \{i:z_i\ge 0\}$.

$$
L(\theta\mid\mathbf{y})=\frac{1}{(2\beta C)^n}\prod_{i\in\mathcal{I}^-}\Bigl[(1-\alpha z_i)^2+1\Bigr]e^{C_1 z_i}
\prod_{i\in\mathcal{I}^+}\Bigl[(1-\alpha z_i)^2+1\Bigr]e^{-z_i}\left(2-e^{-\lambda z_i}\right)
$$

$$
\ell(\theta\mid\mathbf{y})=-n\log 2-n\log\beta-n\log C(\alpha,\lambda)+\sum_{i=1}^n\log\Bigl[(1-\alpha z_i)^2+1\Bigr]
+C_1\sum_{i\in\mathcal{I}^-}z_i-\sum_{i\in\mathcal{I}^+}z_i+\sum_{i\in\mathcal{I}^+}\log\left(2-e^{-\lambda z_i}\right)
$$

### 4) Standardization Moments

$$
\mu_z(\alpha,\lambda)=\frac{1}{C}\left[2\left(1-\frac{1}{C_1^2}\right)-4\alpha+6\alpha^2\left(1-\frac{1}{C_1^4}\right)\right]
$$

$$
\nu_z(\alpha,\lambda)=E[X^2]=\frac{1}{C}\left[4-12\alpha\left(1-\frac{1}{C_1^4}\right)+24\alpha^2\right]
$$

$$
\tau_z^2(\alpha,\lambda)=\nu_z(\alpha,\lambda)-\mu_z(\alpha,\lambda)^2
$$

Standardized innovation:

$$
z_t=\frac{X_t-\mu_z(\alpha,\lambda)}{\tau_z(\alpha,\lambda)},\quad X_t\sim\text{GASLa}(\alpha,\lambda)
$$

### 5) Priors

$$
\pi(\theta)=\pi_\mu(\mu)\pi_\beta(\beta)\pi_\alpha(\alpha)\pi_\lambda(\lambda)
$$

Normal priors (for $\mu$ and $\alpha$):

$$
\mu\sim\mathcal{N}(\mu_0,\sigma_\mu^2),\quad \alpha\sim\mathcal{N}(\mu_\alpha,\sigma_\alpha^2)
$$

Inverse-Gamma prior (for $\beta$):

$$
\pi_\beta(\beta)=\frac{b_\beta^{a_\beta}}{\Gamma(a_\beta)}\beta^{-a_\beta-1}\exp\left(-\frac{b_\beta}{\beta}\right)
$$

Gamma prior (for $\lambda$):

$$
\pi_\lambda(\lambda)=\frac{b_\lambda^{a_\lambda}}{\Gamma(a_\lambda)}\lambda^{a_\lambda-1}e^{-b_\lambda\lambda}
$$

### 6) Posterior

$$
\pi(\theta\mid\mathbf{y})\propto L(\theta\mid\mathbf{y})\,\pi(\theta)
$$

### 7) MH-within-Gibbs Proposals

Normal random-walk (for $\mu$ and $\alpha$):

$$
\mu^*=\mu+\epsilon_\mu,\quad \alpha^*=\alpha+\epsilon_\alpha,\quad \epsilon\sim\mathcal{N}(0,\tau^2)
$$

Log-normal random-walk (for $\beta$ and $\lambda$):

$$
\log\beta^*=\log\beta+\epsilon_\beta,\quad \log\lambda^*=\log\lambda+\epsilon_\lambda
$$

Jacobian correction:

$$
\log q(\theta\mid\theta^*)-\log q(\theta^*\mid\theta)=\log\beta-\log\beta^*,\quad \log\lambda-\log\lambda^*
$$

### 8) Robbins–Monro Adaptation

$$
\log\tau_k^{(t+1)}=\log\tau_k^{(t)}+\gamma_t\left(\bar{a}_k^{(t)}-a^*\right),\quad \gamma_t=t^{-0.6},\ a^*=0.44
$$

### 9) Bayes Estimators

Squared Error Loss (SEL):

$$
\hat{\theta}_{SEL}=E[\theta\mid\mathbf{y}]
$$

LINEX (with $c=0.5$):

$$
\hat{\theta}_{LINEX}=-\frac{1}{c}\log\left(E\left[e^{-c\theta}\mid\mathbf{y}\right]\right)
$$

General Entropy Loss (GEL, $q=0.5$):

$$
\hat{\theta}_{GEL}=\left(E\left[\theta^{-q}\mid\mathbf{y}\right]\right)^{-1/q}
$$

### 10) Chen–Shao HPD Interval

Given sorted draws $\{\theta_{(1)},\ldots,\theta_{(M)}\}$, the HPD interval of level $\alpha$ is

$$
\left[\theta_{(j^*)},\theta_{(j^*+m)}\right],\quad j^*=\arg\min_j\left(\theta_{(j+m)}-\theta_{(j)}\right),\ m=\lfloor\alpha M\rfloor
$$

### 11) GARCH(1,1)

$$
r_t=\mu+\sigma_t z_t,\quad z_t\sim\text{GASLa}_0(\alpha,\lambda)
$$

$$
\sigma_t^2=\omega+\alpha_G\varepsilon_{t-1}^2+\beta_G\sigma_{t-1}^2
$$

Stationarity:

$$
\alpha_G+\beta_G<1
$$

### 12) VaR Forecast

$$
\text{VaR}_{p,t}^{(h)}=-\left[\hat{\mu}(h)+\hat{\sigma}_{t,h}\cdot Q_p^{z}\right]
$$
# GASLa–GARCH Bayesian Estimation

This repo implements the GASLa distribution, MH-within-Gibbs Bayesian inference, and GASLa-GARCH(1,1) modeling with diagnostics and VaR/ES forecasts.

## Folder Structure

```
research/
├─ run_analysis.py
├─ requirements.txt
├─ README.md
└─ src/
   ├─ gasla_core.py
   ├─ generators.py
   ├─ mhwg_sampler.py
   ├─ estimators.py
   ├─ garch_model.py
   └─ mcmc_diagnostics.py
```

## Run From Scratch (No Environment Installed)

```cmd
cd C:\Users\Admin\Desktop\research
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_analysis.py
```

### What You’ll See

- AIC/BIC comparison (GASLa-GARCH vs Normal-GARCH)
- Bayes estimates (SEL/LINEX/GEL)
- MCMC diagnostics (ESS, R-hat, ACF)
- Multi-horizon VaR/ES forecasts
- Diagnostic plots saved to `outputs/plots`

## Equations Used (Exactly What the Code Uses)

### 1) GASLa Normalizing Constant

$$
C(\alpha,\lambda)=2\left(1+\alpha^2-\frac{\alpha\lambda(2+\lambda)}{(1+\lambda)^2}\right)
$$

### 2) GASLa Base PDF (Piecewise)

Let $z=(y-\mu)/\beta$, $C_1=1+\lambda$.

$$
f(z;\alpha,\lambda)=\begin{cases}
\dfrac{(1-\alpha z)^2+1}{2C(\alpha,\lambda)}\,e^{C_1 z}, & z<0\\
\dfrac{(1-\alpha z)^2+1}{2C(\alpha,\lambda)}\,e^{-z}\left(2-e^{-\lambda z}\right), & z\ge 0
\end{cases}
$$

### 3) GASLa Log-Likelihood (Location-Scale)

Let $\mathcal{I}^- = \{i:z_i<0\}$ and $\mathcal{I}^+ = \{i:z_i\ge 0\}$.

$$
L(\theta\mid\mathbf{y})=\frac{1}{(2\beta C)^n}\prod_{i\in\mathcal{I}^-}\Bigl[(1-\alpha z_i)^2+1\Bigr]e^{C_1 z_i}
\prod_{i\in\mathcal{I}^+}\Bigl[(1-\alpha z_i)^2+1\Bigr]e^{-z_i}\left(2-e^{-\lambda z_i}\right)
$$

$$
\ell(\theta\mid\mathbf{y})=-n\log 2-n\log\beta-n\log C(\alpha,\lambda)+\sum_{i=1}^n\log\Bigl[(1-\alpha z_i)^2+1\Bigr]
+C_1\sum_{i\in\mathcal{I}^-}z_i-\sum_{i\in\mathcal{I}^+}z_i+\sum_{i\in\mathcal{I}^+}\log\left(2-e^{-\lambda z_i}\right)
$$

### 4) Standardization Moments

$$
\mu_z(\alpha,\lambda)=\frac{1}{C}\left[2\left(1-\frac{1}{C_1^2}\right)-4\alpha+6\alpha^2\left(1-\frac{1}{C_1^4}\right)\right]
$$

$$
\nu_z(\alpha,\lambda)=E[X^2]=\frac{1}{C}\left[4-12\alpha\left(1-\frac{1}{C_1^4}\right)+24\alpha^2\right]
$$

$$
\tau_z^2(\alpha,\lambda)=\nu_z(\alpha,\lambda)-\mu_z(\alpha,\lambda)^2
$$

Standardized innovation:

$$
z_t=\frac{X_t-\mu_z(\alpha,\lambda)}{\tau_z(\alpha,\lambda)},\quad X_t\sim\text{GASLa}(\alpha,\lambda)
$$

### 5) Priors

$$
\pi(\theta)=\pi_\mu(\mu)\pi_\beta(\beta)\pi_\alpha(\alpha)\pi_\lambda(\lambda)
$$

Normal priors (for $\mu$ and $\alpha$):

$$
\mu\sim\mathcal{N}(\mu_0,\sigma_\mu^2),\quad \alpha\sim\mathcal{N}(\mu_\alpha,\sigma_\alpha^2)
$$

Inverse-Gamma prior (for $\beta$):

$$
\pi_\beta(\beta)=\frac{b_\beta^{a_\beta}}{\Gamma(a_\beta)}\beta^{-a_\beta-1}\exp\left(-\frac{b_\beta}{\beta}\right)
$$

Gamma prior (for $\lambda$):

$$
\pi_\lambda(\lambda)=\frac{b_\lambda^{a_\lambda}}{\Gamma(a_\lambda)}\lambda^{a_\lambda-1}e^{-b_\lambda\lambda}
$$

### 6) Posterior

$$
\pi(\theta\mid\mathbf{y})\propto L(\theta\mid\mathbf{y})\,\pi(\theta)
$$

### 7) MH-within-Gibbs Proposals

Normal random-walk (for $\mu$ and $\alpha$):

$$
\mu^*=\mu+\epsilon_\mu,\quad \alpha^*=\alpha+\epsilon_\alpha,\quad \epsilon\sim\mathcal{N}(0,\tau^2)
$$

Log-normal random-walk (for $\beta$ and $\lambda$):

$$
\log\beta^*=\log\beta+\epsilon_\beta,\quad \log\lambda^*=\log\lambda+\epsilon_\lambda
$$

Jacobian correction:

$$
\log q(\theta\mid\theta^*)-\log q(\theta^*\mid\theta)=\log\beta-\log\beta^*,\quad \log\lambda-\log\lambda^*
$$

### 8) Robbins–Monro Adaptation

$$
\log\tau_k^{(t+1)}=\log\tau_k^{(t)}+\gamma_t\left(\bar{a}_k^{(t)}-a^*\right),\quad \gamma_t=t^{-0.6},\ a^*=0.44
$$

### 9) Bayes Estimators

Squared Error Loss (SEL):

$$
\hat{\theta}_{SEL}=E[\theta\mid\mathbf{y}]
$$

LINEX (with $c=0.5$):

$$
\hat{\theta}_{LINEX}=-\frac{1}{c}\log\left(E\left[e^{-c\theta}\mid\mathbf{y}\right]\right)
$$

General Entropy Loss (GEL, $q=0.5$):

$$
\hat{\theta}_{GEL}=\left(E\left[\theta^{-q}\mid\mathbf{y}\right]\right)^{-1/q}
$$

### 10) Chen–Shao HPD Interval

Given sorted draws $\{\theta_{(1)},\ldots,\theta_{(M)}\}$, the HPD interval of level $\alpha$ is

$$
\left[\theta_{(j^*)},\theta_{(j^*+m)}\right],\quad j^*=\arg\min_j\left(\theta_{(j+m)}-\theta_{(j)}\right),\ m=\lfloor\alpha M\rfloor
$$

### 11) GARCH(1,1)

$$
r_t=\mu+\sigma_t z_t,\quad z_t\sim\text{GASLa}_0(\alpha,\lambda)
$$

$$
\sigma_t^2=\omega+\alpha_G\varepsilon_{t-1}^2+\beta_G\sigma_{t-1}^2
$$

Stationarity:

$$
\alpha_G+\beta_G<1
$$

### 12) VaR Forecast

$$
\text{VaR}_{p,t}^{(h)}=-\left[\hat{\mu}(h)+\hat{\sigma}_{t,h}\cdot Q_p^{z}\right]
$$
# GASLa-GARCH Bayesian Estimation (Phases 1–3)

This repository implements the GASLa distribution, MH-within-Gibbs Bayesian inference, and GASLa-GARCH(1,1) volatility modeling. It includes simulation, estimation, diagnostics, and forecast risk measures (VaR/ES).

## File Structure

```
research/
├─ run_analysis.py
├─ requirements.txt
├─ README.md
└─ src/
	├─ gasla_core.py
	├─ generators.py
	├─ mhwg_sampler.py
	├─ estimators.py
	├─ garch_model.py
	└─ mcmc_diagnostics.py
```

## Run From Scratch (No Existing Virtual Environment)

```cmd
cd C:\Users\Admin\Desktop\research
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_analysis.py
```

## Outputs

The analysis will:
- Simulate a bimodal GASLa-GARCH return series
- Run the MHwG sampler on standardized innovations
- Report SEL/LINEX/GEL Bayes estimates and Chen–Shao HPD intervals
- Print AIC/BIC comparisons versus Normal-GARCH
- Save trace/ACF/QQ diagnostic plots to `outputs/plots`
- Print multi-horizon VaR/ES forecasts

## Equations (From the Research Document)

### GASLa Normalizing Constant

$$
C(\alpha,\lambda)=2\left(1+\alpha^2-\frac{\alpha\lambda(2+\lambda)}{(1+\lambda)^2}\right)
$$

### GASLa Base PDF (Piecewise)

Let $z=(y-\mu)/\beta$, $C_1=1+\lambda$.

$$
f(z;\alpha,\lambda)=\begin{cases}
\dfrac{\bigl(1-\alpha z\bigr)^2+1}{2C(\alpha,\lambda)}\,e^{C_1 z}, & z<0\\
\dfrac{\bigl(1-\alpha z\bigr)^2+1}{2C(\alpha,\lambda)}\,e^{-z}\left(2-e^{-\lambda z}\right), & z\ge 0
\end{cases}
$$

### GASLa Log-Likelihood (Location-Scale)

Let $\mathcal{I}^- = \{i:z_i<0\}$ and $\mathcal{I}^+ = \{i:z_i\ge 0\}$.

$$
L(\theta\mid\mathbf{y})=\frac{1}{(2\beta C)^n}\prod_{i\in\mathcal{I}^-}\Bigl[(1-\alpha z_i)^2+1\Bigr]e^{C_1 z_i}
\prod_{i\in\mathcal{I}^+}\Bigl[(1-\alpha z_i)^2+1\Bigr]e^{-z_i}\left(2-e^{-\lambda z_i}\right)
$$

$$
\ell(\theta\mid\mathbf{y})=-n\log 2-n\log\beta-n\log C(\alpha,\lambda)+\sum_{i=1}^n\log\Bigl[(1-\alpha z_i)^2+1\Bigr]
 + C_1\sum_{i\in\mathcal{I}^-}z_i-\sum_{i\in\mathcal{I}^+}z_i+\sum_{i\in\mathcal{I}^+}\log\left(2-e^{-\lambda z_i}\right)
$$

### Standardization Moments

$$
\mu_z(\alpha,\lambda)=\frac{1}{C}\left[2\left(1-\frac{1}{C_1^2}\right)-4\alpha+6\alpha^2\left(1-\frac{1}{C_1^4}\right)\right]
$$

$$
\nu_z(\alpha,\lambda)=E[X^2]=\frac{1}{C}\left[4-12\alpha\left(1-\frac{1}{C_1^4}\right)+24\alpha^2\right]
$$

$$
	au_z^2(\alpha,\lambda)=\nu_z(\alpha,\lambda)-\mu_z(\alpha,\lambda)^2
$$

Standardized innovation:

$$
z_t=\frac{X_t-\mu_z(\alpha,\lambda)}{\tau_z(\alpha,\lambda)},\quad X_t\sim\text{GASLa}(\alpha,\lambda)
$$

### Priors

$$
\pi(\theta)=\pi_\mu(\mu)\pi_\beta(\beta)\pi_\alpha(\alpha)\pi_\lambda(\lambda)
$$

Normal priors (for $\mu$ and $\alpha$):

$$
\mu\sim\mathcal{N}(\mu_0,\sigma_\mu^2),\quad \alpha\sim\mathcal{N}(\mu_\alpha,\sigma_\alpha^2)
$$

Inverse-Gamma prior (for $\beta$):

$$
\pi_\beta(\beta)=\frac{b_\beta^{a_\beta}}{\Gamma(a_\beta)}\beta^{-a_\beta-1}\exp\left(-\frac{b_\beta}{\beta}\right)
$$

Gamma prior (for $\lambda$):

$$
\pi_\lambda(\lambda)=\frac{b_\lambda^{a_\lambda}}{\Gamma(a_\lambda)}\lambda^{a_\lambda-1}e^{-b_\lambda\lambda}
$$

### Posterior

$$
\pi(\theta\mid\mathbf{y})\propto L(\theta\mid\mathbf{y})\,\pi(\theta)
$$

### MH-within-Gibbs Proposals

Normal random-walk proposals for $\mu$ and $\alpha$:

$$
\mu^*=\mu+\epsilon_\mu,\quad \alpha^*=\alpha+\epsilon_\alpha,\quad \epsilon\sim\mathcal{N}(0,\tau^2)
$$

Log-normal random-walk proposals for $\beta$ and $\lambda$:

$$
\log\beta^*=\log\beta+\epsilon_\beta,\quad \log\lambda^*=\log\lambda+\epsilon_\lambda
$$

Jacobian correction:

$$
\log q(\theta\mid\theta^*)-\log q(\theta^*\mid\theta)=\log\beta-\log\beta^*,\quad \log\lambda-\log\lambda^*
$$

### Robbins–Monro Adaptation

$$
\log\tau_k^{(t+1)}=\log\tau_k^{(t)}+\gamma_t\left(\bar{a}_k^{(t)}-a^*\right),\quad \gamma_t=t^{-0.6},\ a^*=0.44
$$

### Bayes Estimators

Squared Error Loss (SEL):

$$
\hat{\theta}_{SEL}=E[\theta\mid\mathbf{y}]
$$

LINEX (with $c=0.5$):

$$
\hat{\theta}_{LINEX}=-\frac{1}{c}\log\left(E\left[e^{-c\theta}\mid\mathbf{y}\right]\right)
$$

General Entropy Loss (GEL, $q=0.5$):

$$
\hat{\theta}_{GEL}=\left(E\left[\theta^{-q}\mid\mathbf{y}\right]\right)^{-1/q}
$$

### Chen–Shao HPD Interval

Given sorted draws $\{\theta_{(1)},\ldots,\theta_{(M)}\}$, the HPD interval of level $\alpha$ is

$$
\left[\theta_{(j^*)},\theta_{(j^*+m)}\right],\quad j^*=\arg\min_j\left(\theta_{(j+m)}-\theta_{(j)}\right),\ m=\lfloor\alpha M\rfloor
$$

### GARCH(1,1)

$$
r_t=\mu+\sigma_t z_t,\quad z_t\sim\text{GASLa}_0(\alpha,\lambda)
$$

$$
\sigma_t^2=\omega+\alpha_G\varepsilon_{t-1}^2+\beta_G\sigma_{t-1}^2
$$

Stationarity:

$$
\alpha_G+\beta_G<1
$$

### VaR Forecast

$$
	ext{VaR}_{p,t}^{(h)}=-\left[\hat{\mu}(h)+\hat{\sigma}_{t,h}\cdot Q_p^{z}\right]
$$
