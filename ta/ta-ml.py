"""Backtesting ML strategies."""

import pandas_datareader.data as pdr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from simulations import montecarlo
from backtest import visuals, returns
import ml.ml_funcs as mlprep
import pandas as pd
pd.options.mode.chained_assignment = None


aapl = pdr.DataReader('AAPL', 'yahoo', start='2019-09-10', end='2022-06-25')
aapl = mlprep.make_indicators(aapl)
aapl.dropna(inplace=True)
aapl_train, aapl_test = mlprep.split_train_test(aapl, train_size=0.7)
for df in [aapl_train, aapl_test]:
    mlprep.make_binary_ret_targets(df)
    df['detrended_ret'] = returns.get_detrended_returns(df.Close)
    df.dropna(inplace=True)
feature_cols = ['sma_diff', 'bb_oscilator', 'obv', 'atr']
X_train, X_test, y_train, y_test = mlprep.split_feature_target(
    aapl_train, aapl_test, feature_cols, 'target')


sims = montecarlo.simulate_returns(aapl_test.detrended_ret, 1000)


knn_model = KNeighborsClassifier()
tree_model = DecisionTreeClassifier()
name = ['KNN', 'TREE']
for name, model in zip(name, [knn_model, tree_model]):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    dly_rets = aapl_test.detrended_ret * preds
    yrl_ret = dly_rets.mean()*252
    sims_cumret = sims @ preds
    sims_ret = sims_cumret / len(preds) * 252
    print(name, 'Accuracy: ', model.score(X_test, y_test))
    visuals.plot_simulation_test(sims_ret, yrl_ret)
