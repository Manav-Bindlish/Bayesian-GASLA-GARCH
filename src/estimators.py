import numpy as np
from dataclasses import dataclass


@dataclass
class BayesEstimates:
    param_names: list
    sel: np.ndarray
    linex: np.ndarray
    gel: np.ndarray
    post_mean: np.ndarray
    post_sd: np.ndarray
    c_linex: float
    q_gel: float

    def summary(self) -> str:
        lines = [
            f"{'Param':<8} {'SEL':>10} {'LINEX':>10} {'GEL':>10} {'Post_SD':>10}",
            "-" * 52,
        ]
        for i, p in enumerate(self.param_names):
            gel_str = f"{self.gel[i]:10.6f}" if np.isfinite(self.gel[i]) else "        N/A"
            lines.append(
                f"{p:<8} {self.sel[i]:10.6f} {self.linex[i]:10.6f} {gel_str} {self.post_sd[i]:10.6f}"
            )
        return "\n".join(lines)


@dataclass
class HPDIntervals:
    param_names: list
    lower: np.ndarray
    upper: np.ndarray
    width: np.ndarray
    level: float

    def summary(self) -> str:
        lines = [
            f"{'Param':<8} {'Lower':>12} {'Upper':>12} {'Width':>12}",
            "-" * 48,
        ]
        for i, p in enumerate(self.param_names):
            lines.append(
                f"{p:<8} {self.lower[i]:12.6f} {self.upper[i]:12.6f} {self.width[i]:12.6f}"
            )
        return "\n".join(lines)


def sel_estimator(draws: np.ndarray) -> np.ndarray:
    return np.mean(draws, axis=0)


def linex_estimator(draws: np.ndarray, c: float = 0.5) -> np.ndarray:
    M = draws.shape[0]
    mgf = np.mean(np.exp(-c * draws), axis=0)
    mgf = np.where(mgf <= 0.0, np.finfo(float).tiny, mgf)
    return -(1.0 / c) * np.log(mgf)


def gel_estimator(draws: np.ndarray, q: float = 0.5, positive_mask: np.ndarray | None = None) -> np.ndarray:
    n_params = draws.shape[1]
    result = np.full(n_params, np.nan)
    mask = positive_mask if positive_mask is not None else np.ones(n_params, dtype=bool)
    for k in range(n_params):
        if mask[k]:
            col = draws[:, k]
            if np.all(col > 0.0):
                neg_q_moment = np.mean(col ** (-q))
                if neg_q_moment > 0.0:
                    result[k] = neg_q_moment ** (-1.0 / q)
    return result


def compute_bayes_estimates(
    draws: np.ndarray,
    param_names: list | None = None,
    c_linex: float = 0.5,
    q_gel: float = 0.5,
) -> BayesEstimates:
    if param_names is None:
        param_names = ["mu", "beta", "alpha", "lambda"]

    n_params = draws.shape[1]
    positive_mask = np.array([p in ("beta", "lambda") for p in param_names])

    sel = sel_estimator(draws)
    linex = linex_estimator(draws, c=c_linex)

    gel_raw = gel_estimator(draws, q=q_gel, positive_mask=positive_mask)
    gel = np.where(positive_mask, gel_raw, sel)

    post_sd = np.std(draws, axis=0, ddof=1)

    return BayesEstimates(
        param_names=param_names,
        sel=sel,
        linex=linex,
        gel=gel,
        post_mean=sel,
        post_sd=post_sd,
        c_linex=c_linex,
        q_gel=q_gel,
    )


def chen_shao_hpd(draws: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M, n_params = draws.shape
    n_keep = int(np.floor(level * M))
    lower = np.empty(n_params)
    upper = np.empty(n_params)
    width = np.empty(n_params)
    for k in range(n_params):
        sorted_col = np.sort(draws[:, k])
        widths = sorted_col[n_keep:] - sorted_col[: M - n_keep]
        idx = int(np.argmin(widths))
        lower[k] = sorted_col[idx]
        upper[k] = sorted_col[idx + n_keep]
        width[k] = widths[idx]
    return lower, upper, width


def compute_hpd(
    draws: np.ndarray,
    param_names: list | None = None,
    level: float = 0.95,
) -> HPDIntervals:
    if param_names is None:
        param_names = ["mu", "beta", "alpha", "lambda"]
    lower, upper, width = chen_shao_hpd(draws, level=level)
    return HPDIntervals(
        param_names=param_names,
        lower=lower,
        upper=upper,
        width=width,
        level=level,
    )


def gelman_rubin(chain1: np.ndarray, chain2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    M = min(len(chain1), len(chain2))
    c1 = chain1[:M]
    c2 = chain2[:M]
    W = 0.5 * (np.var(c1, axis=0, ddof=1) + np.var(c2, axis=0, ddof=1))
    grand_mean = 0.5 * (np.mean(c1, axis=0) + np.mean(c2, axis=0))
    B = M * (
        (np.mean(c1, axis=0) - grand_mean) ** 2
        + (np.mean(c2, axis=0) - grand_mean) ** 2
    )
    var_hat = ((M - 1) / M) * W + (1.0 / M) * B
    R_hat = np.sqrt(var_hat / np.where(W > 0, W, np.finfo(float).tiny))
    acf_sum = np.ones(c1.shape[1] if c1.ndim > 1 else 1)
    if c1.ndim == 1:
        c1 = c1[:, None]
    for k in range(c1.shape[1]):
        acf_vals = np.correlate(c1[:, k] - c1[:, k].mean(), c1[:, k] - c1[:, k].mean(), mode="full")
        acf_vals = acf_vals[M - 1 :] / acf_vals[M - 1]
        pos_lags = acf_vals[1:] > 0.05
        acf_sum[k] = 1.0 + 2.0 * np.sum(acf_vals[1:][pos_lags])
    ess = M / acf_sum
    return R_hat, ess


def posterior_summary(draws: np.ndarray, param_names: list | None = None) -> dict:
    if param_names is None:
        param_names = ["mu", "beta", "alpha", "lambda"]
    q = np.quantile(draws, [0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
    return {
        "param_names": param_names,
        "mean": np.mean(draws, axis=0),
        "sd": np.std(draws, axis=0, ddof=1),
        "q2_5": q[0],
        "q25": q[1],
        "median": q[2],
        "q75": q[3],
        "q97_5": q[4],
    }
