"""True Strenght Index indicator trading strategy."""
import numpy as np
import pandas as pd


class TSI:

    def __init__(self, price):
        self.price = price

    def calculate_dly_diff(self):
        return self.price - self.price.shift(1)

    def calculate_exp_mov_avg(self, val, window):
        return val.ewm(span=window).mean()

    def calculate_double_exp_mov_avg(self, diff, first_window, second_window):
        first_degree_mov_avg = self.calculate_exp_mov_avg(diff,  first_window)
        second_degree_mov_avg = self.calculate_exp_mov_avg(
            first_degree_mov_avg, second_window)
        return second_degree_mov_avg

    def calculate_tsi(self, first_window=25, second_window=13):
        dly_diff = self.calculate_dly_diff()
        dly_abs_diff = abs(dly_diff)
        diff_double_ema = self.calculate_double_exp_mov_avg(
            dly_diff, first_window, second_window)
        absolute_diff_double_ema = self.calculate_double_exp_mov_avg(
            dly_abs_diff, first_window, second_window)
        self.tsi = diff_double_ema / absolute_diff_double_ema * 100
        return self.tsi

    def get_signals(self, tsi_window=14):
        tsi_price = self.calculate_tsi()
        self.tsi_ema = self.calculate_exp_mov_avg(tsi_price, tsi_window)
        self.signals = []
        for price, ema in zip(tsi_price, self.tsi_ema):
            if price < ema:
                self.signals.append(0)
            elif price >= ema:
                self.signals.append(1)
            else:
                self.signals.append(np.nan)
        return pd.Series(self.signals)

    def get_all_positions(self, num_tsi):
        self.num_tsi = num_tsi
        self.window_range = np.linspace(
            2, self.num_tsi*2, self.num_tsi, dtype=int)
        self.all_signals = []
        self.all_positions = []
        for window in self.window_range:
            signals = self.get_signals(tsi_window=window)
            self.all_signals.append(signals)
            self.all_positions.append(signals.shift(1))
        self.all_positions = pd.DataFrame(
            self.all_positions).T.set_index(self.price.index)
        return self.all_positions
