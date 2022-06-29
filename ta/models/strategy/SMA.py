"""Simple Moving Average trading strategy."""

import numpy as np
import pandas as pd


class SMA:

    def __init__(self, price):  # price : Pandas Series
        self.price = price

    def calculate_signals(self, window=14):
        self.indicator = self.price.rolling(window).mean()
        self.signals = []
        for price, sma in zip(self.price, self.indicator):
            if price < sma:
                self.signals.append(0)
            elif price >= sma:
                self.signals.append(1)
            else:
                self.signals.append(np.nan)
        return pd.Series(self.signals)

    def get_all_positions(self, num_sma):
        self.num_sma = num_sma
        self.window_range = np.linspace(
            2, self.num_sma*2, self.num_sma, dtype=int)
        self.all_signals = []
        self.all_positions = []
        for window in self.window_range:
            signals = self.calculate_signals(window=window)
            self.all_signals.append(signals)
            self.all_positions.append(signals.shift(1))
        self.all_positions = pd.DataFrame(
            self.all_positions).T.set_index(self.price.index)
        return self.all_positions
