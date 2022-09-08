#!/usr/bin/env python3
import argparse
import json
import sys
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_ta
import requests
import talib
from binance.client import Client

from indicator_framework import binance_ws_framework
from indicator_framework import mode_map
from indicator_framework.binance_ws_framework import TickerQue
from indicator_framework.binance_ws_framework import klines
from indicator_framework.panda_talib import PandasIndicators
from utils import colorprint, program_state, sql_lib
from utils.current_mode import CurrentMode

tickers = {}

CurrentMode_object = CurrentMode()
cp = colorprint.NewColorPrint()

from itertools import islice
from utils.watchlist import que

try:
    from main import _process
except ImportError:
    cp.red('Error importing program state!')
    _process = program_state.Process()
    _process.run = True

    no_proc = True
else:
    no_proc = True


class Instrument:
    """
    Sort of a lost cause in translation
    """

    def __init__(self, symbol, debug=True, verbose=True):
        self.symbol = symbol
        self.buys = []
        self.sells = []
        self.sars = []
        self.long = 0
        self.short = 0
        self.side = None
        self.signal_at = 0.0
        self.signal_ts = None
        self.has_signal = False
        self.total_score = 0

        self.uuids = []
        self.open = False
        self.closing = False
        self.sent = False
        self.current = False
        self.count = 0
        self.signal_open_time = None
        self.signal_close_time = None
        self.debug = debug
        self.last_price = 0.0
        self.verbose = verbose
        # self._process = TheSARsAreAllAligning(debug=self.debug, verbose=self.verbose)
        self._process = Strategy(self.symbol, periods=[60, 300, 900, 3600], min_score=10, quiet=False)

    def debug_print(self, text):
        if self.debug:
            cp.debug(text)


# def human_time():
#    return [str(datetime.now().today()), time.time()]


class CurrentSig:
    current_analysis = {
        'signal': {'signal': 'NEUTRAL', 'status': None, 'instrument': None, 'open_time': 0.0, 'Entry': 0.0, 'Exit': 0.0,
                   'closed_at': 0.0}}
    sql = sql_lib.SQLLiteConnection()

    def __get__(self):
        return self.current_analysis

    def __update__(self, data):
        self.current_analysis = data

    def __record__(self, datadict):
        self.sql.append(value=datadict, table='signals')


class Strategy:
    """
    Simple Framework For Analyzing FTX Candle Data
    """

    def __init__(self, market, periods=None, min_score=10, quiet=False):
        self.total_score = 0
        if periods is None:
            self.periods = [60, 300, 900, 3600, 14400]
        self.market = market
        self.periods = periods
        self.min_score = min_score
        self.indicators = ['sar', 'macd', 'ema_cross', 'bop']
        self.quiet = quiet
        self.cp = colorprint.NewColorPrint()
        self.sig = CurrentSig()
        self.pandas_ind = PandasIndicators()
        self.api = Client(None, None)
        self.debug = True
        self.buys = []
        self.sells = []
        self.long = 0
        self.short = 0
        self.count = 0
        self.adx = 0
        self.score = 0

    def reset(self):
        self.buys = []
        self.sells = []
        self.buys = []
        self.sells = []
        self.count = 0
        self.adx = 0
        self.score = 0
        self.count = 0
        self.adx = 0
        self.score = 0

    def _print(self, data):
        if self.quiet:
            pass
        else:
            print(data)

    def spot_ticker(self, market):
        ret = requests.get(f'https://ftx.com/api/markets/{market}').json()
        return ret['result']['price']

    def _future_ticker(self, market):
        ret = requests.get(f'https://ftx.com/api/futures/{market}').json()
        return ret['result']['mark']

    def weighted_std(self, values, weights):
        sum_of_weights = np.sum(weights)
        weighted_average = np.sum(values * weights) / sum_of_weights
        n = len(weights)
        numerator = np.sum(n * weights * (values - weighted_average) ** 2.0)
        denominator = (n - 1) * sum_of_weights
        weighted_std = np.sqrt(numerator / denominator)
        return weighted_std

    def calcweightedavg(self, data, weights):
        import pandas as pd
        import numpy as np
        # X is the dataset, as a Pandas' DataFrame
        mean = np.ma.average(data, axis=0,
                             weights=weights)  # Computing the weighted sample mean (fast, efficient and precise)

        # Convert to a Pandas' Series (it's just aesthetic and more
        # ergonomic; no difference in computed values)
        mean = pd.Series(mean, index=list(data.keys()))
        xm = data - mean  # xm = X diff to mean
        xm = xm.fillna(
            0)  # fill NaN with 0 (because anyway a variance of 0 is just void, but at least it keeps the other covariance's values computed correctly))
        sigma2 = 1. / (weights.sum() - 1) * xm.mul(weights, axis=0).T.dot(
            xm)  # Compute the unbiased weighted sample covariance

    def average(self, numbers):
        total = sum(numbers)
        total = float(total)
        total /= len(numbers)
        return total

    def variance(self, data, ddof=0):
        n = len(data)
        mean = sum(data) / n
        return sum((x - mean) ** 2 for x in data) / (n - ddof)

    def stdev(self, data):
        import math
        var = self.variance(data)
        std_dev = math.sqrt(var)
        return std_dev

    def moving_average(self, df, n):
        """Calculate the moving average for the given data.

        :param df: pandas.DataFrame
        :param n:
        :return: pandas.DataFrame
        """
        MA = talib.MA(df, n, matype=0)
        return MA[-1]

    def balance_of_power(self, open, high, low, close):
        bop = self.pandas_ind.bop(open, high, low, close)
        return bop

    def adx_ta(self, high, low, close, timeperiod=14):
        adx = pandas_ta.adx(high, low, close)
        return adx

    def exponential_moving_average(self, df, n):
        """

                :param df: pandas.DataFrame
                :param n:
                :return: pandas.DataFrame
                """
        EMA = talib.EMA(df, timeperiod=n)
        return EMA[-1]

    def generate_sar(self, high_array, low_array, acceleration=0.05, maximum=0.2):
        sar = talib.SAR(high_array, low_array, acceleration=acceleration, maximum=maximum)
        ticker = (self.future_ticker(self.market))
        sar = (sar[-3])
        if sar < ticker:
            # under candle, is long
            return 1, ticker, sar
        if sar > ticker:
            # above candle, is short
            return -1, ticker, sar
        return False, False, False

    def generate_macd(self, close_array, new_time):
        macd, macdsignal, macdhist = talib.MACD(close_array, fastperiod=12, slowperiod=26, signalperiod=9)

        crosses = []
        macdabove = False
        for i in range(len(macd)):
            if np.isnan(macd[i]) or np.isnan(macdsignal[i]):
                pass
            else:
                if macd[i] > macdsignal[i]:
                    if macdabove == False:
                        macdabove = True
                        cross = [new_time[i], macd[i], 'go']
                        crosses.append(cross)
                else:
                    if macdabove == True:
                        macdabove = False
                        cross = [new_time[i], macd[i], 'ro']
                        crosses.append(cross)
        if macdabove:
            return crosses[-1:], 1
        else:
            return crosses[-1:], -1

    def debug_print(self, text):
        """
        Idk why not just use logger
        :param text: print this
        :return:
        """
        if self.debug:
            cp.debug(text)

    def time_stamp(self):
        return time.time()

    def future_ticker(self, instrument):
        """
        Websocket ticker
        :param instrument: market to query
        :return: last price
        """
        """for x in self.api.futures_ticker():
            if x['symbol'] == instrument:
                return x['lastPrice']"""
        t = TickerQue.tickers.get(instrument)
        if t is not None:
            return t.get('price')
        return 0.0

    def get_klines(self, trading_pair, interval):
        """
        The last task is switching this over to the ws.
        :param trading_pair:
        :param interval:
        :return:
        """
        return self.api.get_klines(symbol=trading_pair, interval=interval)

    def parse_klines(self, klines):
        """
        [[1627646400000, '18.31000000', '35.61000000', '18.31000000', '23.90000000', '4483043.91000000', 1627689599999, '111672903.49790000', 229202, '2148029.00000000', '53861253.08460000', '0']]


        :param klines:
        :return:
        """
        open_time = [int(entry[0]) for entry in klines]
        low = [float(entry[1]) for entry in klines]
        mid = [float(entry[2]) for entry in klines]
        high = [float(entry[3]) for entry in klines]
        close = [float(entry[4]) for entry in klines]
        base_volume = [float(entry[5]) for entry in klines]
        close_time = [float(entry[6]) for entry in klines]
        quote_volume = [float(entry[7]) for entry in klines]

        # self.debug_print(close)
        volume_array = np.asarray(base_volume)
        close_array = np.asarray(close)
        high_array = np.asarray(high)
        low_array = np.asarray(low)
        new_time = [datetime.fromtimestamp(time / 1000) for time in open_time]
        if ((open_time, low, mid, high, close, close_array, high_array, low_array, new_time, volume_array)):
            return open_time, low, mid, high, close, close_array, high_array, low_array, new_time, volume_array
        return False

    def parse_candle(self, candle, kline_que_name, series=False):
        open_time = candle.get(kline_que_name).get('kline').get('open_time')
        open = candle.get(kline_que_name).get('kline').get('open')
        low = candle.get(kline_que_name).get('kline').get('low')
        high = candle.get(kline_que_name).get('kline').get('high')
        close = candle.get(kline_que_name).get('kline').get('close')
        base_volume = candle.get(kline_que_name).get('kline').get('base_volume')
        quote_volume = candle.get(kline_que_name).get('kline').get('quote_volume')
        volume_array = np.asarray(base_volume)
        open_array = np.asarray(open)
        close_array = np.asarray(close)
        high_array = np.asarray(high)
        low_array = np.asarray(low)
        open_array_ = pd.Series(open)
        high_array_ = pd.Series(high_array)
        low_array_ = pd.Series(low_array)
        close_array_ = pd.Series(close_array)
        new_time = [datetime.fromtimestamp(time / 1000) for time in open_time]
        #if not series:
        #    return open_time, open, low, high, close, open_array, close_array, high_array, low_array, new_time, volume_array
        #else:
        #    return open_time, open, low, high, close, open_array_, close_array_, high_array_, low_array_, new_time, volume_array
        if series:
            return open_array_, close_array_, high_array_, low_array_
        else:
            return open_array, close_array, high_array, low_array, volume_array, new_time

    def aggregate_klines(self, trading_pair, interval, closed=True, series=True):

        """
        {'XRPUSDT_1m': {'kline': {'symbol': 'XRPUSDT', 'time': '1m', 'open': '0.7587', 'close': '0.7597',
        'high': '0.7598', 'low': '0.7586', 'baseVol': '375112.0', 'quoteVol': '284793.31995', 'closed': False}}}
        :param trading_pair:
        :param interval:
        :param closed:
        :return:
        """
        success = False
        kline_que_name = f'{trading_pair}_{interval}'
        c = 0
        # print(klines.keys())

        if klines.get(kline_que_name):
            success = True
        else:
            print(f'Not present .. {kline_que_name}')
        if closed:
            if success:
                candle = klines[kline_que_name].__next__(closed=True)
                if candle:
                    cp.debug(candle)
                    if not series:
                        # open_time, open, low, high, close, close_array, high_array, low_array, new_time, volume_array = self.parse_candle(
                        #    candle, kline_que_name)
                        return self.parse_candle(candle, kline_que_name, series=False)
                        # else:
                        #    open_time, open, low, high, close, close_array, high_array, low_array, new_time, volume_array_ = self.parse_candle(candle, kline_que_name, series=True)
                        # return self.aggravate(symbol=trading_pair, interval=interval, series=True)
                    else:
                        return self.parse_candle(candle, kline_que_name, series=True)
            else:
                print('No candle data ...')
                _kline_que_name = f'{trading_pair}_{interval}'
                if _kline_que_name:
                    return self.aggravate(symbol=self.market, interval=interval)
                return False

        else:
            candle = klines[kline_que_name].__next__(closed=False)
            if success:
                if candle:
                    # open_time, open, low, high, close, open_array close_array, high_array, low_array, new_time, volume_array = self.parse_candle(candle, kline_que_name)
                    # return open_time, open, low, high, close, close_array, high_array, low_array, new_time, volume_array
                    if not series:
                        return self.parse_candle(candle, kline_que_name, series=False)
                    return self.parse_candle(candle, kline_que_name, series=True)
            else:
                print('No candle data ...')
                # open_time, open_time, open, low, high, close, close_array, high_array, low_array, new_time, volume_array = self.aggravate(
                #    symbol=trading_pair, interval=interval)
                # return open_time, open, low, high, close, close_array, high_array, low_array, new_time, volume_array
                if not series:
                    return self.parse_candle(candle, kline_que_name, series=False)
                return self.parse_candle(candle, kline_que_name, series=True)
            return False

    def aggravate(self, symbol, interval, series=False):
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
        ]


        TODO: deprecate rest calls
        :param symbol:
        :param interval:
        :return:
        """
        self.debug_print('Making rest call ... ')
        c = 0
        for i in range(0, 9):
            c += 1
            try:
                klines_ = self.api.get_klines(symbol=symbol, interval=interval, limit=1500)
            except Exception as fuck:
                if fuck:
                    cp.red(f'[error]: {fuck}')
                if c == 10:
                    return False
                time.sleep(c)
            else:
                break
        open_time = [int(entry[0]) for entry in klines_]
        open = [float(entry[1]) for entry in klines_]
        high = [float(entry[2]) for entry in klines_]
        low = [float(entry[3]) for entry in klines_]
        close = [float(entry[4]) for entry in klines_]
        base_volume = [float(entry[5]) for entry in klines_]
        close_time = [float(entry[6]) for entry in klines_]
        quote_volume = [float(entry[7]) for entry in klines_]
        # self.debug_print(close)
        volume_array = np.asarray(base_volume)
        open_time_array = np.asarray(open_time)
        open_array = np.asarray(open)
        close_array = np.asarray(close)
        high_array = np.asarray(high)
        low_array = np.asarray(low)
        _open_array_ = pd.Series(low)
        _high_array_ = pd.Series(high)
        _low_array_ = pd.Series(low)
        _close_array_ = pd.Series(close)
        new_time = [datetime.fromtimestamp(time / 1000) for time in open_time_array]
        """if series:
           return open_time, open, low, high, close, _close_array_, _high_array_, _low_array_, _open_array_, new_time, volume_array

        return open_time, high, open, low, close, close_array, high_array, low_array, open_array, new_time, volume_array"""

        if series:
            return _open_array_, _close_array_, _high_array_, _low_array_,
        return open_array, close_array, high_array, low_array, volume_array, new_time

    def get_ohlcv(self, symbol, period=None, series=False):
        """
        Periods in number of seconds:
        15s, 1m,  5m,  15m,  1h,   4h,   1d
        15, 60, 300, 900, 3600, 14400, 86400
        0.01736111111111111 %, 0.06944444444444445 %  0.3472222222222222 % 1.0416666666666665% 4.166666666666666% 16.666666666666664 % 77%
        """
        _close_array = []
        _high_array = []
        _low_array = []
        _open_time_array = []
        _open_array = []

        _volume_array = []

        # candles = api.ftx_api.fetchOHLCV(symbol=symbol, timeframe=period)
        if period is not None:
            # candles = api.ftx_api.fetchOHLCV(symbol=symbol, timeframe=period)
            _candles = requests.get(f'https://ftx.com/api/markets/{symbol}/candles?resolution={period}')
            # print(_candles.json())
            for c in _candles.json()['result']:
                _close_array.append(c['close'])
                _open_array.append(c['open'])
                _high_array.append(c['high'])
                _low_array.append(c['low'])
                _volume_array.append(['volume'])
                _open_time_array.append(c['time'])
            high_array = np.asarray(_high_array)
            low_array = np.asarray(_low_array)
            close_array = np.asarray(_close_array)
            volume_array = np.asarray(_volume_array)
            open_array = np.asarray(_open_array)
            open_array_ = pd.Series(_open_array)
            high_array_ = pd.Series(_high_array)
            low_array_ = pd.Series(_low_array)
            close_array_ = pd.Series(_close_array)

            new_time = [datetime.fromtimestamp(time / 1000) for time in _open_time_array]
            if series:
                return open_array_, high_array_, low_array_, close_array_
            return open_array, close_array, high_array, low_array, volume_array, new_time

    def get_change(self, current, previous):
        if current == previous:
            return 0
        try:
            return (abs(current - previous) / previous) * 100.0
        except ZeroDivisionError:
            return float('inf')

    def score_ta(self, period_string, period):
        """
        Turn FTX intervals into human friendly time periods
        """
        # _period = str(_period).rstrip(string.ascii_lowercase)[1]
        """to_human_format = _period / 60
        if float(to_human_format) >= 60:
            to_human_format = to_human_format / 60
            period_str = f'{to_human_format}h'
        elif float(to_human_format) < 1:
            period_str = f'{to_human_format}m'
        elif float(to_human_format) < 60 > 1:
            period_str = f'{to_human_format}m'
        else:
            return False"""
        self._print(f'Analysis on {period_string}')

        long_score = 0
        short_score = 0
        """open_array, close_array, high_array, low_array, volume_array, new_time = self.aggregate_klines(trading_pair=self.market,
                                                                                               interval=period_string)
        open_array_, close_array_, high_array_, low_array_ =self.aggregate_klines(
            self.market, period_string, closed=True, series=True)"""

        # bop = self.balance_of_power()
        """
         if not series:
            return open_time, open, low, high, close, open_array, close_array, high_array, low_array, new_time, volume_array
        else:
            return open_time, open, low, high, close, open_array_, close_array_, high_array_, low_array_, new_time, volume_array
        """

        open_array, close_array, high_array, low_array, volume_array, new_time = self.aggregate_klines(
            self.market, period_string, closed=True, series=False)
        # bop = self.balance_of_power()
        # macdret, rogo = self.generate_macd(close_array, new_time)

        macdret, rogo = self.generate_macd(close_array, new_time)
        open_array_, high_array_, low_array_, close_array_ = self.aggregate_klines(
            self.market, period_string, closed=True, series=True)
        bop_ret = talib.BOP(low_array_, high_array_, low_array_, close_array_)
        bop_ret = pd.DataFrame.from_dict(bop_ret[:1]).values.tolist()[0][0]
        # self.cp.white(f'BOP:{bop_ret}')
        adx_val = self.adx_ta(high_array_, low_array_, close_array_)
        # adx_len = len(adx_val)
        # print(adx_len)
        # print(pd.DataFrame.from_dict(adx_val[:1500]).values.tolist())
        alen = len(adx_val)
        tlen = len(adx_val.values.tolist())
        print(alen, tlen)
        adx_val = pd.DataFrame.from_dict(adx_val[:alen]).values.tolist()[tlen][0]
        self.cp.purple(f'ADX: {adx_val}')

        # print('MACD:', macdret, rogo)

        if bop_ret > 0:
            self.long += 1
        elif bop_ret < 0:
            self.short += 1
        if rogo == 1:
            self.long += 1
        elif rogo == -1:
            self.short += 1
        sar = self.generate_sar(high_array, low_array)
        if not sar:
            return 0, 0, 0
        if sar[0] == 1:
            self.long += 1
        elif sar[0] == -1:
            self.short += 1
        s = self.stdev(close_array)
        # self._print(('Standard Devation:', s))
        ema_long = self.exponential_moving_average(close_array, n=26)
        ema_short = self.exponential_moving_average(close_array, n=9)
        if ema_short > ema_long:
            self.long += 1
        if ema_short < ema_long:
            self.short += 1
        self.cp.red(f'Analysis for {period_string}:, {long_score}, {short_score}, {adx_val}')
        return long_score, short_score, adx_val

    def calculate_score(self):
        analysis = ''
        total_score = 0
        weighted_score = 0
        highest_score = 0
        period = 0
        p = 0
        _, p_list = CurrentMode_object.get()

        for period_str in p_list:
            if period_str == '1m':
                period = 60
            elif period_str == '5m':
                period = 300

            elif period_str == '15m':
                period = 900
            elif period_str == '30m':
                period = 1800
            elif period_str == '1h':
                period = 3600
            elif period_str == '2h':
                period = 7200
            elif period_str == '4h':
                period = 14400
            elif period_str == '6h':
                period = 21600
            elif period_str == '8h':
                period = 28800
            elif period_str == '12h':
                period = 43200
            elif period_str == '1d':
                period = 86400
            # period_str = self.market +'_'+ period_str

            p += 1
            print(p)
            # _period = period[0]

            highest_score += (period * len(self.indicators))

            # for i in self.periods:
            #    i = i[0]

            try:
                long_score, short_score, adx = self.score_ta(period_string=period_str, period=period)
            except TypeError as err:
                print('err', err)
                return False, False
            except Exception as fucker:
                print(repr(fucker))
            else:
                # long_score, short_score, adx = self.score_ta(period_str, period)
                if p <= 2:
                    print('Period', p)
                    if adx <= 20:
                        return 'NEUTRAL', 0

                if long_score > short_score:
                    self.total_score += 1
                elif short_score > long_score:
                    self.total_score -= 1
                self._print(('Non weighed score:', total_score))

            weighted = (self.total_score * period)
            weighted_score += weighted
            self.cp.yellow(f'Weighted Score: {weighted} for {period}, total: {weighted_score}')
        weighted_score = weighted_score / len(p_list)
        score_pct = 1 / (highest_score / weighted_score) * 100
        self.cp.red(f'Weighted Score: {weighted_score} / Highest Possible: {highest_score} ')

        self.cp.random_color(data='Calculating weighted score!', static_set='bright')
        if weighted_score > 0:
            cp.green('LONG')
            analysis = 'LONG'
            self.cp.green(f'Finished: score: {analysis}, Percent: {score_pct}')

        elif weighted_score < 0:
            cp.red('SHORT')
            analysis = 'SHORT'
            self.cp.red(f'Finished: score: {analysis}, Percent: {score_pct}')

        elif weighted_score == 0:
            analysis = 'NEUTRAL'
            self.cp.yellow(f'Finished: score: {analysis}')
            score_pct = 0
        self.forward_tester(signal=analysis)
        return analysis, score_pct

    def forward_tester(self, signal):
        """
        {'signal': {'signal': 'NEUTRAL', 'status': None, 'instrument': None, 'open_time': 0.0, 'Entry': 0.0, 'Exit': 0.0, 'closed_at': 0.0}}
        """
        ts = str(datetime.utcnow())
        current_tick = self.future_ticker(self.market)
        last_sig = self.sig.get()

        if last_sig.get('signal').get('signal') != signal:
            self._print(('Signal Closed!'))
            ot = self.sig.get().get('signal').get('open_time')
            entry = self.sig.get().get('signal').get('Entry')

            self._print((f"Signal {last_sig.get('signal').get('signal')} closed, {signal} now!"))
            self.sig.__record__(
                {'signal': signal, 'status': 'closed', 'instrument': self.market, 'open_time': ot, 'Entry': entry,
                 'Exit': current_tick,
                 'closed_at': ts})
            self.sig.__update__({'signal': {'signal': signal, 'status': 'open', 'instrument': self.market,
                                            'open_time': ts, 'Entry': current_tick, 'Exit': 0.0,
                                            'closed_at': 0.0}})
            self._print((self.sig.get()))
        else:
            self._print(('Signal Open!'))
            current = current_tick
            last = self.sig.get().get('signal').get('Entry')
            diff = self.get_change(current, last)

            if last_sig.get('signal').get('signal') == 'LONG':
                if current < last:
                    diff = diff * -1
            if last_sig.get('signal').get('signal') == 'SHORT':
                if current > last:
                    diff = diff * -1
            self.cp.white(f'Current Ticker: {current}, Price at open: {last}, PNL if closed now: {diff} %')
            self._print((self.sig.get()))
            pass

    def run(self):

        """try:
            signal_, score = strategy.calculate_score()
        except Exception as fuck:
            cp.alert(f"{['errored out']}, {fuck}")
        else:
            print('Updating ....')"""
        signal_, score = self.calculate_score()
        print(signal_, score)
        if signal_ != 'NEUTRAL' or score != 0 and signal_ is not None:
            print(signal_, score, self.market)
            msg = f'📉 [🔺] {self.market}: ENTER {signal_} @ ${self.future_ticker(self.market)}, {time.time()}! '
            json_msg = json.dumps(
                {"Enter": signal_, "instrument": self.market, "entry": self.future_ticker(self.market),
                 "time": time.time()})
            if msg and json_msg:
                print(msg, json_msg)


def human_time():
    return str('{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()))


def unique_sorted(values):
    "Return a sorted list of the given values, without duplicates."
    values = sorted(values)
    if not values:
        return []
    consecutive_pairs = zip(values, islice(values, 1, len(values)))
    result = [a for (a, b) in consecutive_pairs if a != b]
    result.append(values[-1])
    return result


class TheSARsAreAllAligning:
    """
    Sar spike/dip detector - constantly check the sar on multi time frames. If they all align,
    send a trade signal.
    """

    def __init__(self, debug=True, verbose=True):
        self.old_tfs = []
        self.instruments = {}
        self.debug = debug
        self.verbose = verbose
        self.api = Client()
        self.tfs = []

    def debug_print(self, text):
        """
        Idk why not just use logger
        :param text: print this
        :return:
        """
        if self.debug and text is not None:
            cp.debug(text)

    def time_stamp(self):
        return time.time()

    def future_ticker(self, instrument):
        """
        Websocket ticker
        :param instrument: market to query
        :return: last price
        """
        """for x in self.api.futures_ticker():
            if x['symbol'] == instrument:
                return x['lastPrice']"""
        t = TickerQue.tickers.get(instrument)
        if t is not None:
            return t.get('price')
        return 0.0

    def get_klines(self, trading_pair, interval):
        """
        The last task is switching this over to the ws.
        :param trading_pair:
        :param interval:
        :return:
        """
        return self.api.get_klines(symbol=trading_pair, interval=interval)

    def parse_klines(self, klines):
        """
        [[1627646400000, '18.31000000', '35.61000000', '18.31000000', '23.90000000', '4483043.91000000', 1627689599999, '111672903.49790000', 229202, '2148029.00000000', '53861253.08460000', '0']]


        :param klines:
        :return:
        """
        open_time = [int(entry[0]) for entry in klines]
        low = [float(entry[1]) for entry in klines]
        mid = [float(entry[2]) for entry in klines]
        high = [float(entry[3]) for entry in klines]
        close = [float(entry[4]) for entry in klines]
        base_volume = [float(entry[5]) for entry in klines]
        close_time = [float(entry[6]) for entry in klines]
        quote_volume = [float(entry[7]) for entry in klines]

        # self.debug_print(close)
        volume_array = np.asarray(base_volume)
        close_array = np.asarray(close)
        high_array = np.asarray(high)
        low_array = np.asarray(low)
        new_time = [datetime.fromtimestamp(time / 1000) for time in open_time]
        return open_time, low, mid, high, close, close_array, high_array, low_array, new_time, volume_array

    def parse_candle(self, candle, kline_que_name):
        open_time = candle.get(kline_que_name).get('kline').get('open_time')
        low = candle.get(kline_que_name).get('kline').get('low')
        mid = candle.get(kline_que_name).get('kline').get('mid')
        high = candle.get(kline_que_name).get('kline').get('high')
        close = candle.get(kline_que_name).get('kline').get('close')
        base_volume = candle.get(kline_que_name).get('kline').get('base_volume')
        quote_volume = candle.get(kline_que_name).get('kline').get('quote_volume')
        volume_array = np.asarray(base_volume)
        close_array = np.asarray(close)
        high_array = np.asarray(high)
        low_array = np.asarray(low)
        new_time = [datetime.fromtimestamp(time / 1000) for time in open_time]
        return open_time, low, mid, high, close, close_array, high_array, low_array, new_time, volume_array

    def aggregate_klines(self, trading_pair, interval, closed=True):

        """
        {'XRPUSDT_1m': {'kline': {'symbol': 'XRPUSDT', 'time': '1m', 'open': '0.7587', 'close': '0.7597',
        'high': '0.7598', 'low': '0.7586', 'baseVol': '375112.0', 'quoteVol': '284793.31995', 'closed': False}}}
        :param trading_pair:
        :param interval:
        :param closed:
        :return:
        """
        success = False
        kline_que_name = f'{trading_pair}_{interval}'
        c = 0
        # print(klines.keys())

        if klines.get(kline_que_name):
            success = True
        else:
            print(f'Not present .. {kline_que_name}')
        if closed:
            if success:
                candle = klines[kline_que_name].__next__(closed=True)
                if candle:
                    cp.debug(candle)
                    open_time, low, mid, high, close, close_array, high_array, low_array, new_time, volume_array = self.parse_candle(
                        candle, kline_que_name)
                    return open_time, low, mid, high, close, close_array, high_array, low_array, new_time, volume_array
                else:
                    print('No candle data ...')
                    open_time, low, mid, high, close, close_array, high_array, low_array, new_time, volume_array = self.aggravate(
                        symbol=trading_pair, interval=interval)
                    return open_time, low, mid, high, close, close_array, high_array, low_array, new_time, volume_array

            else:
                if success:
                    candle = klines[kline_que_name].__next__(closed=False)
                    if candle:
                        open_time, low, mid, high, close, close_array, high_array, low_array, new_time, volume_array = self.parse_candle(
                            candle, kline_que_name)
                        return open_time, low, mid, high, close, close_array, high_array, low_array, new_time, volume_array
                return False

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
        ]


        TODO: deprecate rest calls
        :param symbol:
        :param interval:
        :return:
        """
        self.debug_print('Making rest call ... ')
        c = 0
        for i in range(0, 9):
            c += 1
            try:
                klines = self.api.get_klines(symbol=symbol, interval=interval)
            except Exception as fuck:
                cp.red(f'[error]: {fuck}')
                if c == 10:
                    return False
                time.sleep(c)
            else:
                break
        open_time = [int(entry[0]) for entry in klines]
        low = [float(entry[1]) for entry in klines]
        mid = [float(entry[2]) for entry in klines]
        high = [float(entry[3]) for entry in klines]
        close = [float(entry[4]) for entry in klines]
        base_volume = [float(entry[5]) for entry in klines]
        close_time = [float(entry[6]) for entry in klines]
        quote_volume = [float(entry[7]) for entry in klines]
        # self.debug_print(close)
        volume_array = np.asarray(base_volume)
        close_array = np.asarray(close)
        high_array = np.asarray(high)
        low_array = np.asarray(low)
        new_time = [datetime.fromtimestamp(time / 1000) for time in open_time]
        return open_time, low, mid, high, close, close_array, high_array, low_array, new_time, volume_array

    def get_other_indicators(self, instrument, period):
        long_score = 0
        short_score = 0
        open_time, low, mid, high, close, close_array, high_array, low_array, new_time, volume_array = self.aggregate_klines(
            trading_pair=instrument, interval=period)
        macdret, rogo = self.generate_macd(close_array, new_time)
        ema_long = self.exponential_moving_average(close_array, n=26)
        ema_short = self.exponential_moving_average(close_array, n=9)
        if rogo == 1:
            long_score += 1
        elif rogo == -1:
            short_score += 1

        if ema_short > ema_long:
            long_score += 1
        if ema_short < ema_long:
            short_score += 1

    def exponential_moving_average(self, df, n):
        """

        :param df: pandas.DataFrame
        :param n:
        :return: pandas.DataFrame
        """
        EMA = talib.EMA(df, timeperiod=n)
        return EMA[-1]

    def _generate_sar(self, high_array, low_array, acceleration=0.05, maximum=0.2):
        sar = talib.SAR(high_array, low_array, acceleration=acceleration, maximum=maximum)
        ticker = (self.future_ticker(self.market))
        sar = (sar[-3])
        if sar < ticker:
            # under candle, is long
            return 1, ticker, sar
        if sar > ticker:
            # above candle, is short
            return -1, ticker, sar

    def generate_macd(self, close_array, new_time):
        macd, macdsignal, macdhist = talib.MACD(close_array, fastperiod=12, slowperiod=26, signalperiod=9)

        crosses = []
        macdabove = False
        for i in range(len(macd)):
            if np.isnan(macd[i]) or np.isnan(macdsignal[i]):
                pass
            else:
                if macd[i] > macdsignal[i]:
                    if macdabove == False:
                        macdabove = True
                        cross = [new_time[i], macd[i], 'go']
                        crosses.append(cross)
                else:
                    if macdabove == True:
                        macdabove = False
                        cross = [new_time[i], macd[i], 'ro']
                        crosses.append(cross)
        if macdabove:
            return crosses[-1:], 1
        else:
            return crosses[-1:], -1

    def generate_sar(self, high_array, low_array, acceleration=0.02, maximum=0.2, open_candle=False):
        """
        Use talib's parabolic sar function to return current psar value
        :param high_array: as array
        :param low_array:
        :param acceleration: acceleration factor
        :param maximum: acc max
        :return:
        """
        if open_candle:
            sar = talib.stream.SAR(high_array, low_array, acceleration=acceleration, maximum=maximum)
        else:
            sar = talib.SAR(high_array, low_array, acceleration=acceleration, maximum=maximum)
        return sar

    def calc_sar(self, sar, symbol, tf):
        """
        Determine if sar reads under or above the candle
        :param sar:
        :param symbol:
        :return: tuple
        """
        if self.verbose:
            print(symbol, tf)
        ticker = float(self.future_ticker(symbol))
        if ticker == 0:
            return False
        # print(sar)
        sar = sar[-3]
        # print(sar)
        sar = float(sar)

        if sar < ticker:
            # under candle, is long
            return 1, ticker, sar
        if sar > ticker:
            # above candle, is short
            return -1, ticker, sar

    def get_sar(self, symbol, period=None):
        """
        Grab kline data for multiple timeframes #TODO: aiohttp
        :param symbol:
        :param period:
        :return:
        """
        # self.debug_print('Making an api call....')
        # open_time, low, mid, high, close, close_array, high_array, low_array, new_time = self.aggravate(symbol=symbol,
        #                                                                                                interval=period)
        open_time, low, mid, high, close, close_array, high_array, low_array, new_time, volume_array = self.aggregate_klines(
            trading_pair=symbol, interval=period, closed=True)
        high_array = np.asarray(high_array)
        low_array = np.asarray(low_array)
        sar = self.generate_sar(high_array, low_array)
        try:
            s, x, _sar_ = self.calc_sar(sar, symbol, period)
        except TypeError:
            pass
        else:
            if _sar_:
                print(s, x, _sar_)
                return s, x, _sar_
        return False

    def sar_remover(self, field, item, instrument):
        """
        Remove historical sar values as they flip
        :param field:
        :param item:
        :return:
        """
        if field == 'sars':
            for s in instrument.sars:
                if s[0] == item:
                    instrument.sars.remove(s)
        elif field == 'buys':
            for b in instrument.buys:
                if b[0] == item:
                    instrument.buys.remove(b)
        elif field == 'sells':
            for ss in instrument.sells:
                if ss[0] == item:
                    instrument.sells.remove(ss)

    def sar_scalper(self, instrument, p_list=None):
        """
        main logic
        :param p_list: time periods to calculate
        :param instrument: what we're trading
        :return:1337 trade signals
        """
        recalc = False
        force_recalc = False
        instrument.sars = unique_sorted(instrument.sars)
        self.old_tfs = self.tfs
        _, p_list = CurrentMode_object.get()
        self.tfs = p_list
        p_list.sort()
        p_list = sorted(set(p_list))
        count = len(p_list)
        instrument_string = instrument.symbol
        current_ticker = float(self.future_ticker(instrument_string))
        # print(p_list)
        print(f'Processing {instrument.symbol}')

        if self.tfs != self.old_tfs:
            force_recalc = True
        for i in p_list:
            # instrument.sent = False
            # print(i)
            self.debug_print(f'Processing {i}')
            if len(instrument.sars):
                for sar in instrument.sars:
                    if sar[0] == i:
                        self.debug_print(f'Have {sar[0]}')
                        if instrument.buys.__contains__(i):
                            self.debug_print(f'{i} is long')
                            t = self.future_ticker(instrument=instrument_string)
                            if float(t) == 0:
                                recalc = True
                            else:
                                if float(sar[1]) <= float(t):
                                    pass
                                else:
                                    recalc = True
                                    instrument.buys.remove(i)
                                    self.sar_remover('sars', sar, instrument)
                                    instrument.long -= 1
                        if instrument.sells.__contains__(i):
                            self.debug_print(f'{i} is short')
                            t = self.future_ticker(instrument=instrument_string)
                            if float(t) == 0:
                                recalc = True
                            else:
                                if float(sar[1]) >= float(t):
                                    pass
                                else:
                                    recalc = True
                                    instrument.sells.remove(i)
                                    self.sar_remover('sars', sar, instrument)
                                    instrument.short -= 1

            else:
                recalc = True
                cp.yellow(f'Recalculating for {instrument.symbol}, tf: {i}')

            if recalc:
                self.debug_print(f'Getting {i}')
                try:
                    s, t, sr = self.get_sar(symbol=instrument_string, period=i)
                except TypeError:
                    return False
                else:
                    self.debug_print(f'recalc {i} {s} {t} {sr}')

                if s == 1:
                    self.debug_print('buy')
                    instrument.long += 1
                    instrument.buys.append(i)
                    instrument.sars.append([i, sr])
                elif s == -1:
                    self.debug_print('sell')
                    instrument.short += 1
                    instrument.sells.append(i)
                    instrument.sars.append([i, sr])

        if current_ticker == 0:
            return
        if force_recalc:
            instrument.short = 0
            instrument.long = 0
            instrument.sars = []
            instrument.buys = []
            instrument.sells = []
            if instrument.open:
                instrument.open = False
                instrument.signal_at = 0.0
                instrument.signal_ts = None
                instrument.has_signal = False
        print(instrument_string, count, instrument.long, instrument.short, instrument.sars, current_ticker)
        ts = self.time_stamp()
        print(f'{instrument.symbol}, {instrument.long}, {instrument.short}')
        if instrument.long + instrument.short > count or instrument.long > count or instrument.short > count or \
                (instrument.long < 0 and instrument.short < 0):
            instrument.long = 0
            instrument.short = 0
            instrument.buys = []
            instrument.sell = []
            instrument.sars = []
        instrument.last_price = current_ticker
        if len(instrument.sars) == count:

            if instrument.long == count:  # count == long
                # instrument.sent = True
                instrument.open = True
                instrument.closing = False
                if not instrument.has_signal:
                    instrument.has_signal = True
                    instrument.signal_at = current_ticker
                    instrument.signal_open_time = human_time()
                    instrument.signal_ts = time.time()
                    instrument.side = 'LONG'
                msg = f'📉 [🔺] {instrument_string}: ENTER LONG @ ${instrument.signal_at}, {instrument.signal_open_time}! '
                json_msg = json.dumps({"Enter": "long", "instrument": instrument_string, "entry": instrument.signal_at,
                                       "time": instrument.signal_open_time})
                cp.green(msg)
                instrument.has_signal = True
                return msg, json_msg

            elif instrument.short == count:
                instrument.open = True
                instrument.closing = False
                # instrument.sent = True

                if not instrument.has_signal:
                    instrument.has_signal = True
                    instrument.signal_ts = time.time()
                    instrument.signal_at = current_ticker
                    instrument.signal_open_time = human_time()
                    instrument.side = 'Short'
                instrument.signal_at = current_ticker
                msg = f'📈 [🔻] {instrument_string}: ENTER SHORT @ ${instrument.signal_at}, {instrument.signal_open_time}! '
                # json_msg = json.dumps(f'{"signal": "short", "instrument": f"{instrument_string}", "entry": "{instrument.signal_at}", "time": {instrument.signal_open_time}}')
                json_msg = json.dumps({"Enter": "short", "instrument": instrument_string, "entry": instrument.signal_at,
                                       "time": instrument.signal_open_time})
                cp.red(msg)
                instrument.has_signal = True
                return msg, json_msg
            # elif (count > instrument.short >= 0) and (instrument.short >= instrument.long) or instrument.short <= instrument.long :
            # elif instrument.long != count and instrument.short != count:
            else:
                cp.blue('Nothing yet....')

                if instrument.open:
                    cp.red(f'Closing signal {instrument.symbol}')
                    instrument.closing = True
                    instrument.signal_at = 0.0
                    instrument.signal_ts = None
                    instrument.has_signal = False
                    instrument.sent = False
                    instrument.open = False
                    instrument.short = 0
                    instrument.long = 0
                    instrument.sars = []
                    instrument.buys = []
                    instrument.sells = []
            cp.purple(
                f'Status: {instrument.symbol} Long: {instrument.long}, short: {instrument.short}, {instrument.sars}')

            # self.debug_print('THE END')
            return False, False


def cli_args():
    """
    If running from the cli
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--symbol', dest='symbol', type=str, default=None, help='Future Market to Query')
    parser.add_argument('-l', '--list', dest='symbol_list_file', type=str, default=None, help='Iterate over list'
                                                                                              'of symbols.')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true', help='Debug mode')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Increase verbosity')
    parser.add_argument('-m', '--mode', dest='mode', default='trend', type=str,
                        choices=['scalp', 'trend', 'precise', 'alerts'],
                        help='Which '
                             'mode of operation to use. Scalp checks 30m, 15m, 5m, and 1m. Trend checks 4h, 1h, 30m, 15m, '
                             'and 5m, while precise mode checks 12h, 6h, 4h, 2h, 1h, 30m and 15m')
    return parser.parse_args()


class BlitzKreig:
    """
    Start the insane shit
    """
    inst_list = []
    inst_str = []
    plist = []
    signals = {}

    def __init__(self):
        self.p_list = []
        self.current = {}
        self.args = None
        self.inst_list = []
        self.mode_map = mode_map.TimePeriods()
        self.current_mode_list = []

    def configure(self, args=None):

        cp.purple(f'[~] Using mode {args.mode}...')
        self.args = args
        null, p_list = self.mode_map.choose_mode(mode=args.mode)
        self.p_list = p_list
        CurrentMode_object.update(args.mode, tfs=p_list)

        if args.symbol:
            self.inst_list.append(Instrument(symbol=args.symbol))
        if args.symbol_list_file:
            with open(args.symbol_list_file, 'r') as f:
                f = f.readlines()
            try:
                for _ in f:
                    _ = symbol = _.strip('\r\n')
                    if args.verbose:
                        cp.debug(f'List Processing {_}')
                    self.inst_list.append(Instrument(symbol=symbol))
                    self.inst_str.append(symbol)
            except Exception as fuck:
                cp.red(f'[error]: {fuck}')
        try:
            if args.symbol_list:
                # cp.debug(f'{ic.instruments}')
                for sy in args.symbol_list:
                    self.inst_list.append(Instrument(symbol=sy))
                    self.inst_str.append(sy)

        except AttributeError:
            pass
        return self.inst_str

    def run_bot(self):
        """
        AbraKadabra
        :param instruments:
        :param periods:
        :return:
        """
        insts_ = []
        que.append(f'Signal Server restarted, running in {self.args.mode} mode ...')
        for i in self.inst_list:
            self.current[i.symbol] = False
            if not insts_.__contains__(i.symbol):
                insts_.append(i)
            else:
                insts_.remove(i)
        while True:
            if no_proc:
                pass
            else:
                status = _process.state()
                print(status)
                if not status:
                    que.append_signal('User requested signal stream stops. Shutting down')
                    return

            # self.current_mode_list = current_mode_list
            try:

                for i in insts_:
                    # print(i.symbol)
                    _null, current_mode_list = CurrentMode_object.get()
                    print('MODE DEBUB', _null, current_mode_list)
                    try:
                        ret = i.process_.run()
                    except Exception as f:
                        cp.debug(f)
            except KeyboardInterrupt:
                print('\nCaught Signal, Exiting with Grace ...')
                sys.exit(0)

    def run_bot_(self):
        """
        AbraKadabra
        :param instruments:
        :param periods:
        :return:
        """
        cp.red('Starting....')
        insts_ = []
        que.append(f'Signal Server restarted, running in {self.args.mode} mode ...')
        for i in self.inst_list:
            self.current[i.symbol] = False
            if not insts_.__contains__(i.symbol):
                insts_.append(i)
            else:
                insts_.remove(i)
        while True:
            if no_proc:
                pass
            else:
                status = _process.state()
                print(status)
                if not status:
                    que.append_signal('User requested signal stream stops. Shutting down')
                    return

            # self.current_mode_list = current_mode_list

            try:

                for i in insts_:
                    print(i.symbol)
                    _null, current_mode_list = CurrentMode_object.get()
                    print('MODE DEBUB', _null, current_mode_list)
                    try:
                        ret, json_ret = i._process.run()
                    except Exception as fuck:
                        pass


            except KeyboardInterrupt:
                print('\nCaught Signal, Exiting with Grace ...')
                sys.exit(0)

    def start_websocket(self, verbose=True):
        while True:
            try:
                binance_ws_framework.interactive(markets=self.inst_str)
                print('Started the stream...')
            except Exception as fuck:
                cp.red((repr(fuck)))
            else:
                print('Restarting...')


class InteractiveArgs:
    symbol = None
    symbol_list_file = None
    symbol_list = []
    debug = False
    verbose = False
    mode = 'scalp'

    # def args(self):
    #    return [{'symbol': self.symbol, 'symbol_list_file': self.symbol_list_file, 'symbol_list': self.symbol_list, 'debug': self.debug,
    #            'verbose': self.verbose, 'mode': self.mode}]


def interactive_call(args):
    """
    For importing to another program
    :return:
    """
    tws = BlitzKreig()
    insts = tws.configure(args)

    t = threading.Thread(target=tws.start_websocket, args=(insts,))
    t.start()
    tws.run_bot_()


def main():
    args = cli_args()
    print(args)
    if args.symbol or args.symbol_list_file:
        tws = BlitzKreig()
        insts = tws.configure(args)

        t = threading.Thread(target=tws.start_websocket, args=(insts,))
        t.start()
        tws = threading.Thread(target=tws.run_bot_(), args=(insts,))
        tws.start()
        print('Data aggregator is demonized.')


if __name__ == '__main__':
    main()
else:
    print('Use the interactive_call, and InteractiveParser to generate signals for consumption in another program.')
