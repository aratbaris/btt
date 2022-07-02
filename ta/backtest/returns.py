"""Return calculation functions for backtesting."""
import numpy as np


def get_returns(price):  # log returns

    logret = np.log(price / price.shift(1)).rename('return')

    return logret


def get_detrended_returns(price):  # log returns
    logret = get_returns(price)
    avg_return = logret.mean()
    detrended_logret = logret - avg_return
    return detrended_logret


def get_bin_rets(prices):  # Closing price Pandas Series with Date Index

    diff = prices - prices.shift(1)
    diff_binary = []
    for i in range(len(diff)):
        if diff[i] > 0:
            diff_binary.append(1)
        elif diff[i] < 0:
            diff_binary.append(0)
        else:
            diff_binary.append(np.nan)

    return diff_binary
