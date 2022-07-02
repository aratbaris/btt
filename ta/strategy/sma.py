"""Simple Moving Average trading strategy."""

import numpy as np
import pandas as pd


def calc_sma(prices, window):
    return prices.rolling(window).mean()


def calc_signals(prices, sma):
    signals = []
    for day_price, day_sma in zip(prices, sma):
        if day_price < day_sma:
            signals.append(0)
        elif day_price >= day_sma:
            signals.append(1)
        else:
            signals.append(np.nan)
    return pd.Series(signals)


def get_positions(prices, window):
    sma = calc_sma(prices, window)
    signals = calc_signals(prices, sma)
    return signals.shift(1)


def get_mult_sma_pos(df, num_sma):

    window_range = np.linspace(2, num_sma*2, num_sma, dtype=int)

    for window in window_range:
        df[str(window) + '_sma'] = get_positions(df.Close, window).values

    return df
