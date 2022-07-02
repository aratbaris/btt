"""Backtesting ML strategies."""

import pandas_datareader.data as pdr
import simulations.montecarlo as mc
import backtest.returns as ret
from ml import ml_funcs
import pandas as pd
pd.options.mode.chained_assignment = None


aapl = pdr.DataReader('AAPL', 'yahoo', start='2019-09-10', end='2022-06-25')
aapl = ml_funcs.make_indicators(aapl)
aapl.dropna(inplace=True)
aapl_train, aapl_test = ml_funcs.split_train_test(aapl, train_size=0.7)
for df in [aapl_train, aapl_test]:
    ml_funcs.make_binary_ret_targets(df)
    df['detrended_ret'] = ret.get_detrended_returns(df.Close)
    df.dropna(inplace=True)
feature_cols = ['sma_diff', 'bb_oscilator', 'obv', 'atr']
X_train, X_test, y_train, y_test = ml_funcs.split_feature_target(
    aapl_train, aapl_test, feature_cols, 'target')


sims = mc.simulate_returns(aapl_test.detrended_ret, 1000)


knn_all_preds = ml_funcs.make_mult_knn(50, X_train, y_train, X_test, y_test)
tree_all_preds = ml_funcs.make_mult_tree(50, X_train, y_train, X_test, y_test)


ml_funcs.test_mult_strategy('KNN', knn_all_preds, aapl_test.detrended_ret, sims)
ml_funcs.test_mult_strategy('TREE', tree_all_preds, aapl_test.detrended_ret, sims)
