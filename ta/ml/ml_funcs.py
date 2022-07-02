"""ML prep helper func."""


import ml.indicator_funcs as ind
import backtest.returns as ret

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import backtest.visuals as viz


def split_train_test(df, train_size):
    split_idx = int(len(df)*train_size)
    df_train = df[:split_idx]
    df_test = df[split_idx:]
    return df_train, df_test


def split_feature_target(df_train, df_test, feature_cols, target_col):
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    y_train = df_train[target_col]
    y_test = df_test[target_col]
    return X_train, X_test, y_train, y_test


def make_indicators(df):

    df['sma_diff'] = ind.calculate_normalized_sma_diff(df.Close, 14, 50)
    df['bb_oscilator'] = ind.calculate_bollinger_band_oscilator(df.Close)
    df['obv'] = ind.calculate_on_balance_volume(df.Close, df.Volume)
    df['atr'] = ind.calculate_avg_true_range(df.Close, df.High, df.Low)

    return df


def make_binary_ret_targets(df):
    df['bin_rets'] = ret.get_bin_rets(df.Close)
    df['target'] = df.bin_rets.shift(-1)
    df.dropna(inplace=True)
    return df


def make_mult_knn(max_n, X_train, y_train, X_test, y_test):
    all_preds, train_acc, test_acc = [], [], []
    for k in range(max_n):
        mdl = KNeighborsClassifier(n_neighbors=k+1)
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        all_preds.append(preds)
        train_acc.append(mdl.score(X_train, y_train))
        test_acc.append(mdl.score(X_test, y_test))
    all_preds = pd.DataFrame(all_preds).T
    return all_preds


def make_mult_tree(depths, X_train, y_train, X_test, y_test):
    all_preds, train_acc, test_acc = [], [], []
    for k in range(depths):
        mdl = DecisionTreeClassifier(max_depth=k+1)
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        all_preds.append(preds)
        train_acc.append(mdl.score(X_train, y_train))
        test_acc.append(mdl.score(X_test, y_test))
    all_preds = pd.DataFrame(all_preds).T
    return all_preds


def test_mult_strategy(name, all_preds, stock_returns, simulations):
    cumret = all_preds.T @ stock_returns.values
    highest_ret = max(cumret / len(stock_returns) * 252)
    sims_cumret = (simulations @ all_preds)  # daily cumilative
    sims_best_ret = (sims_cumret / len(stock_returns) * 252).max(axis=1)
    print(f'Highest backteting profit from {name} Strategy Test ')
    viz.plot_simulation_test(sims_best_ret, highest_ret)


def plot_knn_opt(neighbors, test_scores, train_scores):

    plt.title('k-NN: Varying Number of Neighbors')
    plt.plot(neighbors, test_scores, label='Testing Accuracy')
    plt.plot(neighbors, train_scores, label='Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()
