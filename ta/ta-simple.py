"""Backtesting simple techical analysis trading strategies."""

import pandas_datareader.data as pdr
from strategy import rsi, sma, tsi
from backtest import returns, visuals
from simulations import montecarlo

aapl = pdr.DataReader('AAPL', 'yahoo', start='2018-07-12', end='2022-06-30')
aapl['rsi_pos'] = rsi.get_positions(aapl['Close'], window=14).values
aapl['sma_pos'] = sma.get_positions(aapl['Close'], window=14).values
aapl['tsi_pos'] = tsi.get_positions(aapl['Close'], tsi_window=14).values

aapl.dropna(inplace=True)
aapl['detrended_ret'] = returns.get_detrended_returns(aapl['Close'])
aapl.dropna(inplace=True)

sims = montecarlo.simulate_returns(aapl.detrended_ret, 500)

for strat_pos in ['rsi_pos', 'sma_pos', 'tsi_pos']:
    strat_ret = (aapl[strat_pos] * aapl['detrended_ret']).mean()*252
    sims_cumret = sims @ aapl[strat_pos].values
    sims_rets = sims_cumret / len(aapl[strat_pos]) * 252
    print(strat_pos)
    visuals.plot_simulation_test(sims_rets, strat_ret)
