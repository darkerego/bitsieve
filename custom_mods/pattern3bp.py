import datetime
import time

import numpy as np
import pandas as pd
import talib as ta
import binance
#from modules import tsi


class ThreeBarPlay:
    def __init__(self, lookback_periods=26, tolerance=0.1, high_low_datapoints=5):
        self.lookback_periods = lookback_periods
        self.tolerance = tolerance
        self.high_low_datapoints = high_low_datapoints

        self.binance = binance.client.Client(None, None)

    def parse_candles(self, candles):
        open_time = [int(entry[0]) for entry in candles]
        _open = [float(entry[1]) for entry in candles]
        high = [float(entry[2]) for entry in candles]
        low = [float(entry[3]) for entry in candles]
        _close = [float(entry[4]) for entry in candles]
        base_volume = [float(entry[5]) for entry in candles]
        close_time = [float(entry[6]) for entry in candles]
        quote_volume = [float(entry[7]) for entry in candles]
        currency = ['USD' for x in range(len(open_time))]
        new_time = [datetime.datetime.fromtimestamp(t / 1000) for t in open_time]

        return pd.DataFrame(data=[_open, high, low, _close, base_volume, currency], index=new_time,
                            columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Currency'])

    def aggravate(self, symbol, interval):
        """
        [
          [
            1499040000000,      // Open time
            "0.01634790",       // Open
            "0.80000000",       // High
            "0.01575800",       // Low
            "0.01577100",       // Close
            "148976.11427815",  // Volume
            1499644799999,      // Close time
            "2434.19055334",    // Quote asset volume
            308,                // Number of trades
            "1756.87402397",    // Taker buy base asset volume
            "28.46694368",      // Taker buy quote asset volume
            "17928899.62484339" // Ignore.
          ]
        ]"""

        def get(symbol, interval):
            candles = False
            for i in range(3):
                try:
                    candles = self.binance.get_klines(symbol=symbol, interval=interval)
                except Exception as fuck:
                    print(f'Error: {fuck}')
                    time.sleep(0.25)
                else:
                    break
            return candles

        candles = get(symbol, interval)
        if not candles:
            return False
        return self.parse_candles(candles)

    def mean(self, df):
        return sum(df) / len(df)

    def average_candle(self, df: pd.DataFrame):
        moves = []
        for _ in range(len(df.open[0])):
            co = df.open.array[0][_]
            cc = df.high.array[0][_]
            cr = cc - co
            moves.append(cr)
        return self.mean(moves)

    def resistance(self, candle: pd.DataFrame):
        high_array = candle.High
        cx = sorted(high_array)
        highs = cx[-self.high_low_datapoints:]
        resistance = sum(highs) / len(highs)
        return resistance

    def support(self, candle: pd.DataFrame):
        low_array = candle.Low
        cx = sorted(low_array)
        lows = cx[:self.high_low_datapoints]
        resistance = sum(lows) / len(lows)
        return resistance

    def get_trend(self, candles, lookback=26):
        """
        determine trend - up or down
        """
        ma = ta.MA(candles.Close)
        for _ in range(0, lookback):



    def search(self, candles, market, period):
        def candle_range(h, o):
            return h - o

        print(f'Searching market {market} ... period {period} ... ')
        ac = self.average_candle(candles)
        last_two_o = candles.open[0][-2:]
        last_two_h = candles.high[0][-2:]
        current_support = self.support(candles)
        current_resistance = self.resistance(candles)

        last_c = last_two_o[-1:]
        last_h = last_two_h[-1:]
        last_2c = last_two_o[-2:]
        last_2h = last_two_h[-2:]
        last_range = candle_range(last_h, last_c)
        if last_range >= ac * (2 - self.tolerance):
            print('Current candle is a wide range igniting bar')
