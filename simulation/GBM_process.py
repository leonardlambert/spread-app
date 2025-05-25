import numpy as np

def simulate_gbm_paths(S0, r, sigma, T, steps, n_paths):
    """
    Simule des trajectoires GBM (Geometric Brownian Motion)
    sous mesure risque-neutre.

    Args:
        S0 (float): Prix initial du sous-jacent
        r (float): Taux sans risque
        sigma (float): Volatilité
        T (float): Horizon (en années)
        steps (int): Nombre de pas
        n_paths (int): Nombre de trajectoires à simuler

    Returns:
        np.ndarray: Matrice (n_paths, steps + 1) des trajectoires simulées
    """
    dt = T / steps
    t = np.linspace(0, T, steps + 1)
    S = np.zeros((n_paths, steps + 1))
    S[:, 0] = S0

    # bruit brownien simulé
    Z = np.random.normal(0, 1, (n_paths, steps))
    increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    S[:, 1:] = S0 * np.exp(log_paths)

    return S
