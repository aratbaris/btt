"""True Strenght Index indicator trading strategy."""
import numpy as np
import pandas as pd


def calculate_dly_diff(price):
    return price - price.shift(1)


def calculate_exp_mov_avg(val, window):
    return val.ewm(span=window).mean()


def calculate_double_exp_mov_avg(diff, first_window, second_window):
    first_mov_avg = calculate_exp_mov_avg(diff,  first_window)
    second_mov_avg = calculate_exp_mov_avg(first_mov_avg, second_window)
    return second_mov_avg


def calculate_tsi(price, first_window=25, second_window=13):
    dly_diff = calculate_dly_diff(price)
    dly_abs_diff = abs(dly_diff)
    diff_double_ema = calculate_double_exp_mov_avg(
        dly_diff, first_window, second_window)
    absolute_diff_double_ema = calculate_double_exp_mov_avg(
        dly_abs_diff, first_window, second_window)
    tsi = diff_double_ema / absolute_diff_double_ema * 100
    return tsi


def calculate_signals(tsi, tsi_window=14):
    tsi_ema = calculate_exp_mov_avg(tsi, tsi_window)
    signals = []
    for price, ema in zip(tsi, tsi_ema):
        if price < ema:
            signals.append(0)
        elif price >= ema:
            signals.append(1)
        else:
            signals.append(np.nan)
    return pd.Series(signals)


def get_positions(price, first_window=25, second_window=13, tsi_window=14):
    tsi = calculate_tsi(price, first_window, second_window)
    signals = calculate_signals(tsi, tsi_window)
    return signals.shift(1)


def get_mult_tsi_pos(df, num_tsi, first_window=25, second_window=13):
    # mult tsi changes only tsi window
    window_range = np.linspace(2, num_tsi*2, num_tsi, dtype=int)

    for window in window_range:
        df[str(window)+'_tsi'] = get_positions(
            df.Close, first_window, second_window, tsi_window=window).values

    return df
