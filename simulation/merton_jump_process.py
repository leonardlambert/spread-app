import numpy as np

def simulate_mjd_paths(S0, r, sigma, lamb, mu_j, sigma_j, T, steps, n_paths):
    dt = T / steps
    S = np.zeros((n_paths, steps + 1))
    S[:, 0] = S0

    for t in range(1, steps + 1):
        Z = np.random.normal(0, 1, size=n_paths)
        N = np.random.poisson(lamb * dt, size=n_paths)
        J = np.random.normal(mu_j, sigma_j, size=n_paths)
        jumps = N * J

        dS = (r - 0.5 * sigma**2 - lamb * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)) * dt + sigma * np.sqrt(dt) * Z + jumps
        S[:, t] = S[:, t - 1] * np.exp(dS)

    return S

def phi_merton(u, T, r, sigma, lamb, mu_j, sigma_j):

    kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    drift = r - lamb * kappa

    diffusion_term = 1j * u * drift * T - 0.5 * sigma**2 * u**2 * T
    jump_term = lamb * T * (np.exp(1j * u * mu_j - 0.5 * sigma_j**2 * u**2) - 1)

    return np.exp(diffusion_term + jump_term)
