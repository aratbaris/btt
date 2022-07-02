from ta.trend import SMAIndicator, CCIIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, AwesomeOscillatorIndicator
from ta.volume import OnBalanceVolumeIndicator, EaseOfMovementIndicator


def calculate_normalized_sma_diff(close_price, window1=14, window2=50):
    sma14 = SMAIndicator(close_price, 14)
    sma14_ind = sma14.sma_indicator()
    sma50 = SMAIndicator(close_price, 50)
    sma50_ind = sma50.sma_indicator()
    sma_diff = (sma14_ind - sma50_ind)/sma14_ind
    return sma_diff


def calculate_bollinger_band_oscilator(close_price, ma=5, std=1):
    bb = BollingerBands(close_price)
    mid_b = bb.bollinger_mavg()
    bb_oscilator = (close_price - mid_b)
    return bb_oscilator


def calculate_on_balance_volume(close_price, volume):
    obv = OnBalanceVolumeIndicator(close_price, volume)
    obv = obv.on_balance_volume()
    return obv


def calculate_avg_true_range(close_price, high_price, low_price):
    atr = AverageTrueRange(high_price, low_price, close_price, window=14)
    atr = atr.average_true_range()
    return atr
