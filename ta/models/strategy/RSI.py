"""Relative Strength Index trading strategy."""

import numpy as np
import pandas as pd


class RSI:

    def __init__(self, price):
        self.price = price

    def calculate_indicator(self, window=14):
        dly_movement_amount = self.price - self.price.shift(1)

        dly_up_movement = []
        dly_down_movement = []
        for movement_amount in dly_movement_amount:
            if movement_amount > 0:
                dly_up_movement.append(movement_amount)
                dly_down_movement.append(0)
            elif movement_amount < 0:
                dly_up_movement.append(0)
                dly_down_movement.append(movement_amount)
            else:
                dly_down_movement.append(np.nan)
                dly_up_movement.append(np.nan)
        rolling_up_movemnt = pd.Series(dly_up_movement).rolling(window).mean()
        rolling_down_movement_absolute = pd.Series(
            dly_down_movement).abs().rolling(window).mean()

        relative_strenght = rolling_up_movemnt / rolling_down_movement_absolute
        self.relative_strenght_index = 100 - (100 / (1 + relative_strenght))

        return self.relative_strenght_index

    def get_signals(self, window=14):
        rsi_ind = self.calculate_indicator(window)
        rsi_signals = []
        for rsi in rsi_ind:
            if rsi < 70:
                rsi_signals.append(1)
            elif rsi >= 70:
                rsi_signals.append(0)
            else:
                rsi_signals.append(rsi)

        return pd.Series(rsi_signals)

    def get_all_positions(self, num_rsi):
        self.num_rsi = num_rsi
        self.window_range = np.linspace(
            2, self.num_rsi*2, self.num_rsi, dtype=int)
        self.all_signals = []
        self.all_positions = []
        for window in self.window_range:
            signals = self.get_signals(window=window)
            self.all_signals.append(signals)
            self.all_positions.append(signals.shift(1))
        self.all_positions = pd.DataFrame(
            self.all_positions).T.set_index(self.price.index)
        return self.all_positions
