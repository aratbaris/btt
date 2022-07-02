"""Simulation models for randomly generated historical returns."""
import numpy as np
import pandas as pd


def simulate_returns(stock_returns, run, seed=101):
    num_days = len(stock_returns)
    mu = stock_returns.mean()
    sigma = stock_returns.std()
    np.random.seed(seed)
    all_simulations = []
    for _ in range(run):
        simulated_returns = np.random.normal(
            loc=mu, scale=sigma, size=num_days)
        all_simulations.append(simulated_returns)
    return pd.DataFrame(all_simulations)


def get_simulated_best_returns(all_simulations, all_positions):
    simulated_best_returns = (
        (all_simulations @ all_positions.values) / len(all_positions) * 252).max(axis=1)
    return simulated_best_returns
