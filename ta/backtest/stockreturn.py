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
