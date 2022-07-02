"""Backtesting optimized techical analysis trading strategies."""

import pandas_datareader.data as pdr
from strategy import rsi, sma, tsi
from backtest import returns, visuals
from simulations import montecarlo

aapl = pdr.DataReader('AAPL', 'yahoo', start='2018-07-12', end='2022-06-30')
aapl = rsi.get_mult_rsi_pos(aapl, num_rsi=20)
aapl = sma.get_mult_sma_pos(aapl, num_sma=20)
aapl = tsi.get_mult_tsi_pos(aapl, num_tsi=20)
aapl.dropna(inplace=True)
aapl['detrend_ret'] = returns.get_detrended_returns(aapl['Close'])
aapl.dropna(inplace=True)


sims = montecarlo.simulate_returns(aapl.detrend_ret, 500)


for strat in ['rsi', 'sma', 'tsi']:
    columns = [strat in col for col in aapl.columns]
    all_positions = aapl.loc[:, columns]
    cumret = aapl.detrend_ret @ all_positions
    yrl_rets = cumret / len(all_positions) * 252
    sims_best_returns = montecarlo.get_simulated_best_returns(sims, all_positions)
    print(strat)
    visuals.plot_simulation_test(sims_best_returns,  max(yrl_rets))
