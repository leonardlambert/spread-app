import numpy as np
from numpy.fft import fft
from scipy.stats import norm
from simulation.variance_gamma_process import phi_vg
from simulation.merton_jump_process import phi_merton
from simulation.BSM import phi_bsm

def price_fft_option(K, S0, T, r, char_func, args=(), alpha=0.5, N=2**12, lamb=0.01, call=True):
    eta = 2 * np.pi / (N * lamb)
    beta = np.log(S0) - (N / 2) * lamb

    u = np.arange(N) * eta
    k = beta + np.arange(N) * lamb
    strikes = np.exp(k)

    phi = char_func(u - 1j * (alpha + 1), T, r, *args)
    numerator = np.exp(-r * T) * phi * np.exp(1j * u * (np.log(S0) - beta))
    denominator = alpha ** 2 + alpha - u ** 2 + 1j * (2 * alpha + 1) * u
    integrand = numerator / denominator * eta
    integrand[0] *= 0.5

    fft_vals = np.fft.fft(integrand).real
    prices = fft_vals * lamb / np.pi
    prices /= lamb ** 2

    price_interp = np.interp(K, strikes, prices)
    if not call:
        return price_interp - (S0 - K * np.exp(-r * T))
    return price_interp
