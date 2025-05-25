import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import json
import os
import time


from datetime import datetime, timezone
from simulation.variance_gamma_process import simulate_vg_paths, phi_vg
from simulation.merton_jump_process import simulate_mjd_paths,phi_merton
from simulation.GBM_process import simulate_gbm_paths
from simulation.BSM import black_scholes_price, compute_greeks, implied_volatility
from simulation.FFT_pricer import price_fft_option



BOOK_FILE = "option_book.json"

def load_book():
    if os.path.exists(BOOK_FILE):
        with open(BOOK_FILE, "r") as f:
            return json.load(f)
    return []

def save_book(book):
    with open(BOOK_FILE, "w") as f:
        json.dump(book, f, indent=2)

if "book" not in st.session_state:
    st.session_state.book = load_book()


st.set_page_config(layout="wide")
st.title("Option Spread Analyzer")

tabs = st.tabs(["SPREAD DESIGN", "SPREAD BOOK", "MONTE CARLO"])

with tabs[0]:
    S0 = st.number_input("Spot", value=100.0)
    T = st.number_input("Time to Maturity", value=1)
    r = st.number_input("Risk-Free Rate", value=0.05)
    sigma = st.number_input("Volatility", value=0.2)



    st.subheader("Define spread")
    num_legs = st.number_input("Number of legs", min_value=1, max_value=10, value=2, step=1)
    legs = []
    ready_to_plot = True


    for i in range(num_legs):
        col1, col2, col3 = st.columns(3)
        with col1:
            option_type = st.selectbox(f"Type (leg {i+1})", ["call", "put"], key=f"type_{i}")
        with col2:
            strike = st.number_input(f"Strike (leg {i+1})", key=f"strike_{i}")
        with col3:
            position = st.selectbox(f"Position (leg {i+1})", [1, -1], format_func=lambda x: "Long" if x == 1 else "Short", key=f"pos_{i}")
        if strike == 0:
            ready_to_plot = False
        legs.append({"type": option_type, "strike": strike, "position": position})

    if st.button("Add to book"):
        st.session_state.book.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "S0": S0,
            "T": T,
            "r": r,
            "sigma": sigma,
            "legs": legs
        })
        save_book(st.session_state.book)
        st.success("Strategy added to book.")

    if ready_to_plot:

        S_range = np.linspace(0.01, S0 * 2, 500)
        fig, axs = plt.subplots(3, 2, figsize=(14, 10))

        spread_value_today = np.zeros_like(S_range)
        payoff = np.zeros_like(S_range)
        delta_total = np.zeros_like(S_range)
        gamma_total = np.zeros_like(S_range)
        theta_total = np.zeros_like(S_range)
        vega_total = np.zeros_like(S_range)
        rho_total = np.zeros_like(S_range)
        net_premium = 0.0

        for leg in legs:
            K = leg["strike"]
            pos = leg["position"]
            opt_type = leg["type"]

            leg_price = black_scholes_price(S0, K, T, r, sigma, opt_type)
            net_premium += pos * leg_price

            if opt_type == 'call':
                payoff_leg = np.maximum(S_range - K, 0)
            else:
                payoff_leg = np.maximum(K - S_range, 0)

            payoff += pos * payoff_leg

            for i, S in enumerate(S_range):
                delta, gamma, theta, vega, rho = compute_greeks(S, K, T, r, sigma, opt_type)
                price_at_S = black_scholes_price(S, K, T, r, sigma, opt_type)

                spread_value_today[i] += pos * price_at_S
                delta_total[i] += pos * delta
                gamma_total[i] += pos * gamma
                theta_total[i] += pos * theta
                vega_total[i] += pos * vega
                rho_total[i] += pos * rho

        profit = payoff - net_premium
        spread_value_today -= net_premium



        from scipy.interpolate import interp1d

        # Trouver les indices où le profit change de signe
        sign_changes = np.where(np.diff(np.sign(profit)) != 0)[0]

        # Interpolation pour chaque break-even point
        for idx in sign_changes:
            p1, p2 = profit[idx], profit[idx + 1]
            s1, s2 = S_range[idx], S_range[idx + 1]

            # Interpolation linéaire pour break-even exact
            if p2 != p1:  # éviter division par zéro
                S_be = s1 - p1 * (s2 - s1) / (p2 - p1)
                axs[0, 0].axvline(S_be, color='red', linestyle=':', linewidth=1.5)
                axs[0, 0].text(S_be, 0, f'{S_be:.2f}', color='red', ha='left', va='bottom')

        axs[0, 0].plot(S_range, spread_value_today, label='BSM Value', linestyle='-')
        axs[0, 0].plot(S_range, profit, label='Profit', linestyle='--')

        axs[0, 1].plot(S_range, delta_total, label="delta")
        axs[1, 0].plot(S_range, gamma_total, label="gamma")
        axs[1, 1].plot(S_range, theta_total, label="theta")
        axs[2, 0].plot(S_range, vega_total, label="vega")
        axs[2, 1].plot(S_range, rho_total, label="rho")

        axs[0, 0].axhline(0, color='black', linewidth=0.5)
        axs[0, 0].set_title("Spread Value / Profit / Payoff")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        for ax, title in zip([axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]],
                             ["Delta", "Gamma", "Theta", "Vega", "Rho"]):
            ax.set_title(title)
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.info("Enter a valid strike for each leg to show analysis.")

with tabs[1]:
    st.subheader("Book Overview")

    if st.button("Reset book"):
        st.session_state.book = []
        save_book([])
        st.warning("Book cleared.")

    # Total Greeks + Rows
    total_delta = total_gamma = total_theta = total_vega = total_rho = 0
    rows = []

    for strat in st.session_state.book:
        d_total = g_total = t_total = v_total = r_total = 0
        for leg in strat['legs']:
            d, g, t, v, r_ = compute_greeks(
                strat['S0'], leg['strike'], strat['T'], strat['r'], strat['sigma'], leg['type']
            )
            d_total += leg['position'] * d
            g_total += leg['position'] * g
            t_total += leg['position'] * t
            v_total += leg['position'] * v
            r_total += leg['position'] * r_

        total_delta += d_total
        total_gamma += g_total
        total_theta += t_total
        total_vega += v_total
        total_rho += r_total

        legs_str = "\n".join(
            [f"Long {leg['type']} @ {leg['strike']}" for leg in strat['legs'] if leg['position'] == 1] +
            [f"Short {leg['type']} @ {leg['strike']}" for leg in strat['legs'] if leg['position'] == -1]
        )

        greeks_str = (
            f"Delta: {d_total:.2f}\nGamma: {g_total:.2f}\nTheta: {t_total:.2f}\n"
            f"Vega: {v_total:.2f}\nRho: {r_total:.2f}"
        )

        params_str = (
            f"Spot: {strat['S0']}\nTTM: {strat['T']}\nr: {strat['r']}\nvol: {strat['sigma']}"
        )

        rows.append({
            "Date": strat.get("timestamp", "N/A"),
            "Market Params": params_str,
            "Legs": legs_str,
            "Greeks": greeks_str,
        })

    if rows:
        st.markdown("### Total Book Exposure")
        st.markdown(
            f"Delta: `{total_delta:.2f}` | Gamma: `{total_gamma:.2f}` | "
            f"Theta: `{total_theta:.2f}` | Vega: `{total_vega:.2f}` | Rho: `{total_rho:.2f}`"
        )
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No saved strategies in the book.")

with tabs[2]:
    st.subheader("Monte Carlo Simulation on Book")
    st.markdown("Simulate total book value using different asset price processes.")

    n_paths = st.number_input("Number of simulations", min_value=100, max_value=100000, value=5000, step=1000)
    T_horizon = st.number_input("Time horizon (years)", min_value=0.01, max_value=5.0, value=1.0)
    steps = st.number_input("Steps per path", min_value=10, max_value=365, value=50)

    processes_selected2 = st.multiselect(
        "Select price process(es)",
        ["GBM", "Variance Gamma", "Merton Jump Diffusion"],
        default=["GBM"]
    )

    if st.button("Run Simulation"):
        if not st.session_state.book:
            st.warning("Book is empty.")
        else:
            np.random.seed()
            S0 = st.session_state.book[0]['S0']
            sigma = st.session_state.book[0]['sigma']
            r = st.session_state.book[0]['r']

            for process_name in processes_selected2:
                if process_name == "GBM":
                    spot_paths = simulate_gbm_paths(S0, r, sigma, T_horizon, steps, n_paths)

                elif process_name == "Variance Gamma":
                    spot_paths = simulate_vg_paths(S0, r, sigma, theta=-0.1, nu=0.2, T=T_horizon, steps=steps,
                                                   n_paths=n_paths)

                elif process_name == "Merton Jump Diffusion":
                    spot_paths = simulate_mjd_paths(S0, r, sigma, lamb=0.1, mu_j=-0.05, sigma_j=0.2, T=T_horizon,
                                                    steps=steps, n_paths=n_paths)

                final_values = []
                for path_idx in range(int(n_paths)):
                    S_T = spot_paths[path_idx, -1]
                    book_value = 0
                    for strat in st.session_state.book:
                        for leg in strat["legs"]:
                            K = leg["strike"]
                            pos = leg["position"]
                            opt_type = leg["type"]
                            payoff = max(S_T - K, 0) if opt_type == "call" else max(K - S_T, 0)
                            book_value += pos * payoff
                    discounted = book_value * np.exp(-r * T_horizon)
                    final_values.append(discounted)

                initial_book_value = sum(
                    leg["position"] * black_scholes_price(S0, leg["strike"], strat["T"], strat["r"], sigma, leg["type"])
                    for strat in st.session_state.book for leg in strat["legs"]
                ) or 1e-8

                pnl_percent = [(v - initial_book_value) / initial_book_value * 100 for v in final_values]

                mean_val = np.mean(pnl_percent)
                std_val = np.std(pnl_percent)

                st.markdown(f"**{process_name} — Mean %PnL:** {mean_val:.2f}%")
                st.markdown(f"**Std deviation:** {std_val:.2f}%")

                fig, ax = plt.subplots()
                ax.hist(pnl_percent, bins=50, color='skyblue', edgecolor='black')
                ax.axvline(0, color='black', linestyle='--')
                ax.set_title(f"{process_name} — Discounted %PnL")
                ax.set_xlabel("%PnL")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)















