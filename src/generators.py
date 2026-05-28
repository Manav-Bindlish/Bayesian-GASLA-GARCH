import numpy as np
from numpy.random import Generator


def _rlaplace(rng: Generator, n: int, mu: float = 0.0, beta: float = 1.0) -> np.ndarray:
    u = rng.uniform(-0.5, 0.5, size=n)
    return mu - beta * np.sign(u) * np.log1p(-2.0 * np.abs(u))


def _rasla(rng: Generator, n: int, alpha: float) -> np.ndarray:
    M_env = 2.0
    samples = np.empty(n)
    accepted = 0
    while accepted < n:
        batch = max(int((n - accepted) * M_env * 1.5), 1)
        x = _rlaplace(rng, batch)
        u = rng.uniform(0.0, 1.0, size=batch)
        weight = ((1.0 - alpha * x) ** 2 + 1.0) / (2.0 * (1.0 + alpha**2))
        mask = u < (weight / M_env)
        accepted_batch = x[mask]
        take = min(len(accepted_batch), n - accepted)
        samples[accepted : accepted + take] = accepted_batch[:take]
        accepted += take
    return samples


def rgasla(
    n: int,
    mu: float,
    beta: float,
    alpha: float,
    lam: float,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = np.empty(n)
    accepted = 0
    while accepted < n:
        batch = max(int((n - accepted) * 3), 1)
        W = _rasla(rng, batch, alpha)
        Z = _rlaplace(rng, batch)
        mask = lam * W > Z
        accepted_W = W[mask]
        take = min(len(accepted_W), n - accepted)
        samples[accepted : accepted + take] = mu + beta * accepted_W[:take]
        accepted += take
    return samples


def rgasla_std(
    n: int,
    alpha: float,
    lam: float,
    seed: int | None = None,
) -> np.ndarray:
    from gasla_core import moments_std

    mu_z, tau_z = moments_std(alpha, lam)
    raw = rgasla(n, mu=0.0, beta=1.0, alpha=alpha, lam=lam, seed=seed)
    return (raw - mu_z) / tau_z
