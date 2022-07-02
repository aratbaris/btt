"""Relative Strength Index trading strategy."""
import pandas as pd
import numpy as np


def calc_rsi(price, window=14):
    dly_movement_amount = price - price.shift(1)

    dly_up_movement = []
    dly_down_movement = []
    for movement_amount in dly_movement_amount:
        if movement_amount > 0:
            dly_up_movement.append(movement_amount)
            dly_down_movement.append(0)
        elif movement_amount <= 0:
            dly_up_movement.append(0)
            dly_down_movement.append(movement_amount)
        else:
            dly_down_movement.append(np.nan)
            dly_up_movement.append(np.nan)
    rolling_up_movemnt = pd.Series(dly_up_movement).rolling(window).mean()
    rolling_down_movement_absolute = pd.Series(
        dly_down_movement).abs().rolling(window).mean()

    relative_strenght = rolling_up_movemnt / rolling_down_movement_absolute
    rsi = 100 - (100 / (1 + relative_strenght))

    return rsi


def calc_signals(rsi_ind):
    rsi_signals = []
    for rsi in rsi_ind:
        if rsi < 70:
            rsi_signals.append(1)
        elif rsi >= 70:
            rsi_signals.append(0)
        else:
            rsi_signals.append(rsi)

    return pd.Series(rsi_signals)


def get_positions(price, window=14):
    rsi = calc_rsi(price, window)
    signals = calc_signals(rsi)
    return signals.shift(1)


def get_mult_rsi_pos(df, num_rsi):

    window_range = np.linspace(2, num_rsi*2, num_rsi, dtype=int)

    for window in window_range:
        df[str(window)+'_rsi'] = get_positions(df.Close, window).values

    return df
