"""Backtesting simple techical analysis trading strategies."""

import data.finmodelprep as fin
from models.strategy.SMA import SMA
from models.strategy.RSI import RSI
from models.strategy.TSI import TSI
import backtest.visualization as viz
import backtest.stockreturn as ret
import simulations.MonteCarlo as simu

import pandas as pd

# GET FREE API KEY https://site.financialmodelingprep.com/developer/docs/
api_key = 'b449bf5a0bb38fe12889fd79f02662c2'
stock_name = 'AAPL'
num_days = 1000
aapl = fin.get_stock_data_from_finmodelprep(api_key, stock_name, num_days)


sma = SMA(aapl['close'])
sma_signals = sma.calculate_signals()

rsi = RSI(aapl['close'])
rsi_signals = rsi.get_signals()

tsi = TSI(aapl['close'])
tsi_signals = tsi.get_signals()

all_signals = pd.concat(
    [sma_signals, rsi_signals, tsi_signals], axis=1).dropna()

detrended_returns = ret.get_detrended_returns(
    aapl['close']).reset_index().loc[all_signals.index]['return']

mc = simu.MonteCarlo()
aapl_simulations = mc.simulate_returns(detrended_returns, 500)
avg_yrl_simulation_rets = aapl_simulations.mean()*252

names = ['SMA', 'RSI', 'TSI']
for name, i in zip(names, all_signals):
    daily_trade_returns = all_signals[i].shift(1).values * detrended_returns
    yrl_trade_return = daily_trade_returns.mean()*252
    viz.plot_simulation_test(avg_yrl_simulation_rets, yrl_trade_return)
    viz.print_p_upper(avg_yrl_simulation_rets, yrl_trade_return)
    print(name)
