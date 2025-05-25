import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes_price(S, K, T, r, sigma, option_type):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def compute_greeks(S, K, T, r, sigma, option_type):
    if T == 0:
        return 0, 0, 0, 0, 0
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -norm.cdf(-d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    return delta, gamma, theta, vega, rho

def implied_volatility(price, S, K, T, r, option_type):
    try:
        def objective(sigma):
            return black_scholes_price(S, K, T, r, sigma, option_type) - price

        return brentq(objective, 1e-6, 3.0, maxiter=500, xtol=1e-6)
    except Exception:
        print(f"⚠️ IV failed for price={price}, K={K}, S={S}, T={T}")
        return None

def phi_bsm(u, T, r, sigma):
    return np.exp(1j * u * (r - 0.5 * sigma**2) * T - 0.5 * sigma**2 * u**2 * T)