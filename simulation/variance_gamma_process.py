import numpy as np

def simulate_vg_paths(S0, r, sigma, theta, nu, T, steps, n_paths):
    """
    Simule des trajectoires sous processus Variance Gamma.

    Args:
        S0 (float): prix initial
        r (float): taux sans risque
        sigma (float): vol VG
        theta (float): drift VG
        nu (float): paramètre de variance gamma
        T (float): horizon
        steps (int): pas de temps
        n_paths (int): nombre de chemins

    Returns:
        np.ndarray: matrice (n_paths, steps + 1)
    """
    dt = T / steps
    omega = (1 / nu) * np.log(1 - theta * nu - 0.5 * sigma**2 * nu)
    drift = r + omega  # <- attention : c’est +omega ici

    S = np.zeros((n_paths, steps + 1))
    S[:, 0] = S0

    for t in range(1, steps + 1):
        G = np.random.gamma(dt / nu, nu, size=n_paths)
        Z = np.random.normal(0, 1, size=n_paths)
        dX = theta * G + sigma * np.sqrt(G) * Z
        S[:, t] = S[:, t - 1] * np.exp((drift - 0.5 * sigma**2) * dt + dX)

    return S


def phi_vg(u, T, r, sigma, theta, nu):
    omega = (1 / nu) * np.log(1 - theta * nu - 0.5 * sigma**2 * nu)
    return np.exp(
        1j * u * (r + omega) * T -
        (T / nu) * np.log(1 - 1j * theta * nu * u + 0.5 * sigma**2 * nu * u**2)
    )



