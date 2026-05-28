import numpy as np


def normalising_constant(alpha: float, lam: float) -> float:
    C1 = 1.0 + lam
    return 2.0 * (1.0 + alpha**2 - alpha * lam * (2.0 + lam) / C1**2)


def log_pdf_std(z: float, alpha: float, lam: float) -> float:
    C = normalising_constant(alpha, lam)
    if C <= 0.0:
        return -np.inf

    C1 = 1.0 + lam
    w = (1.0 - alpha * z) ** 2 + 1.0

    if z < 0.0:
        return np.log(w) - np.log(2.0 * C) + C1 * z
    else:
        inner = 2.0 - np.exp(-lam * z)
        if inner <= 0.0:
            return -np.inf
        return np.log(w) - np.log(2.0 * C) - z + np.log(inner)


def log_pdf_std_vec(z: np.ndarray, alpha: float, lam: float) -> np.ndarray:
    return np.array([log_pdf_std(float(zi), alpha, lam) for zi in z])


def moments_std(alpha: float, lam: float) -> tuple[float, float]:
    C = normalising_constant(alpha, lam)
    C1 = 1.0 + lam

    mu_z = (1.0 / C) * (
        2.0 * (1.0 - 1.0 / C1**2)
        - 4.0 * alpha
        + 6.0 * alpha**2 * (1.0 - 1.0 / C1**4)
    )

    E_X2 = (1.0 / C) * (
        4.0
        - 12.0 * alpha * (1.0 - 1.0 / C1**4)
        + 24.0 * alpha**2
    )

    var_z = E_X2 - mu_z**2
    if var_z <= 0.0 or not np.isfinite(var_z):
        return 0.0, 1.0

    return mu_z, np.sqrt(var_z)


def log_pdf_standardised(z: float, alpha: float, lam: float) -> float:
    mu_z, tau_z = moments_std(alpha, lam)
    u = mu_z + tau_z * z
    log_fu = log_pdf_std(u, alpha, lam)
    if not np.isfinite(log_fu):
        return -np.inf
    return log_fu + np.log(tau_z)


def log_pdf_standardised_vec(z: np.ndarray, alpha: float, lam: float) -> np.ndarray:
    return np.array([log_pdf_standardised(float(zi), alpha, lam) for zi in z])
