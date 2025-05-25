# Option Spread Analysis Tool

A tool to design, visualize, and simulate multi-leg option spread strategies.

---

## Features

1. **Spread construction and visualization**  
   - Build option spreads with any number of legs  
   - Visualize payoff, BSM value, and Greek profiles

2. **Spread book management**  
   - Strategies saved locally in a `JSON` file  
   - Aggregated Greeks computed for the full book

3. **Monte Carlo simulation**  
   - Estimate expected return and variance of the book  
   - Select process: GBM, Variance Gamma, or Merton Jump  
   - Customize number of paths, steps, and time horizon

---

## How to Run

- Make sure you have **Python 3.8+** installed
- Install dependencies with:

```bash
pip install -r requirements.txt
