"""Backtesting optimized technical analysis trading strategies."""

import data.finmodelprep as fin
from models.strategy.SMA import SMA
from models.strategy.RSI import RSI
from models.strategy.TSI import TSI
import backtest.visualization as viz
import backtest.stockreturn as ret
import simulations.MonteCarlo as simu

import pandas as pd

# GET FREE API KEY https://site.financialmodelingprep.com/developer/docs/
api_key = ''
stock_name = 'AAPL'
num_days = 1000
aapl = fin.get_stock_data_from_finmodelprep(api_key, stock_name, num_days)


sma = SMA(aapl['close'])
sma_all_positions = sma.get_all_positions(num_sma=20)

rsi = RSI(aapl['close'])
rsi_all_positions = rsi.get_all_positions(num_rsi=20)

tsi = TSI(aapl['close'])
tsi_all_positions = tsi.get_all_positions(num_tsi=20)

common_index = pd.concat(
    [sma_all_positions,
     rsi_all_positions,
     tsi_all_positions], axis=1).dropna().index

aapl_detrended_returns = ret.get_detrended_returns(
    aapl['close']).loc[common_index]

mc = simu.MonteCarlo()
mc.simulate_returns(aapl_detrended_returns, 500)

names = ['SMA', 'RSI', 'TSI']
for name, all_positions in zip(names,
                               [tsi_all_positions,
                                rsi_all_positions,
                                sma_all_positions]):

    all_positions = all_positions.loc[common_index]
    best_return = max(aapl_detrended_returns.dot(
        all_positions) / len(all_positions) * 252)
    simulated_best_returns = mc.get_simulated_best_returns(all_positions)
    viz.plot_simulation_test(simulated_best_returns, best_return)
    viz.print_p_upper(simulated_best_returns, best_return)
    print(name)
