#!/usr/bin/env python3.8
import argparse
import datetime
import json
import logging
import math
import random
import threading
import time
from time import sleep

import numpy as np
import pandas as pd
import pandas_ta
# import ta
import requests
import talib
from binance import client

from trade_engine import indicators
from trade_engine import market_enumeration
from trade_engine.pandas_talib import PandasIndicators
from utils import sql_lib
from utils.colorprinter import ColorPrint
from utils.mqtt_skel_ import MqSkel
from utils.init_websockets import klines, TickerQue, run, state

# from colored import fg, attr, bg
# from pandas import DataFrame
# from ta import momentum

bin = client.Client(None, None)

cp = ColorPrint()

signal_que = {}
period_map = [(60, '1m'), (180, '3m'), (300, '5m'), (900, '15m'), (1800, '30m'), (3600, '1h'), (7200, '2h'),
              (14400, '4h'), (21600, '6h'), (43200, '12h'), (86400, '1d'), (259200, '3d'), (604800, '1w'),
              (2592000, '1M')]


class Settings:
    min_score = 30
    min_adx = 20
    roc_periods = []
    volume_spike_periods = []
    volume_spike_multiplier = 1
    debug = False
    ema_windows = [(26, 9), (200, 50)]
    vsd_window = 26
    adx_window = 14
    pattern_detect = []
    min_pattern_rating = 50


"""logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        # logging.StreamHandler()
    ]
)"""


def lopic_log(data):
    with open('execution.log', 'w') as f:
        f.write(str(data + '\n'))


def period_mapper(p_seconds=None, p_string=None):
    for x in range(len(period_map)):
        # print(x)
        p = period_map.__getitem__(x)
        if p_seconds:
            # print(p)
            if p[0] == p_seconds:
                return p[1]
        if p_string:
            if p[1] == p_string:
                # print(p[0])
                return p[0]


class BadReversal:
    pass


class CurrentSig:

    def __init__(self, instrument, signal, score, entry=None, _exit=None, adx_avg=0, periods=[], events=[]):
        self.Lowest_pnl = None
        self.logger = logging.getLogger(__name__)
        self.sql = sql_lib.SQLLiteConnection(dbname='kline.sqlite')
        self.Instrument = instrument
        self.Signal = signal
        self.Score = score
        self.Open_time = 0
        self.Closed_at = 0
        self.Exit = 0
        self.Entry = entry
        self.Status = 'open'
        self.periods = periods
        self.Pnl = 0.0
        self.adx_avg = adx_avg
        self.Highest_PNL = 0.0
        self.Lowest_PNL = 0.0
        self.Live_score = 0.0
        self.Current_analysis = {}
        self.ROC = {}
        self.PNL = 0.0
        self.Events = events
        self.open(signal, score, entry)

    def open(self, signal, score, entry):
        """
        Open a new or reversed signal
        """
        self.logger.info(f'Opening new signal {self.Instrument}')
        self.Signal = signal
        self.Score = score
        self.Open_time = datetime.datetime.now()
        self.Entry = entry
        self.Highest_PNL = 0.0
        self.Lowest_PNL = 0.0
        self.Status = 'open'
        self.Current_analysis = {'Signal': self.Signal, 'Status': self.Status, 'Instrument': self.Instrument,
                                 'Score': self.Score, 'Live_score': self.Live_score, 'Mean_Adx': self.adx_avg,
                                 'Periods': self.periods, 'Open_time': self.Open_time, 'Entry': self.Entry,
                                 'Exit': self.Exit, 'Closed_at': self.Closed_at, 'PNL': self.Pnl,
                                 'Highest_PNL': self.Highest_PNL, 'Lowest_PNL': self.Lowest_PNL, 'Events': self.Events}

        self.publish(topic=f'/signals')
        self.publish(topic=f'/stream')
        # self.record_sql(self.Current_analysis)

    def close(self, exit_price, pnl, status='closed'):
        """
        Close a signal
        """
        # sig = signal_que.get(instrument)
        self.logger.info(f'Closing signal: {self.Instrument}')
        self.Closed_at = datetime.datetime.now()
        self.Status = f'{status}'
        self.Exit = exit_price
        self.PNL = pnl
        self.Current_analysis.update({'Exit': exit_price})
        self.Current_analysis.update({'Status': 'closed'})
        self.Current_analysis.update({'Closed_at': datetime.datetime.now()})
        self.Current_analysis.update({'PNL': f'{pnl}%'})
        # self.record_sql(self.Current_analysis)
        if self.publish(topic=f'/signals') and self.publish(topic=f'/stream'):
            return True

    def reverse(self, entry_price, score, pnl):
        """
        Reverse a signal (long to short, short to long) instead of going neutral
        """
        # print(f'Reversing {self.Current_analysis}')
        self.logger.info(f'Reversing signal: {self.Instrument}')
        self.close(exit_price=entry_price, pnl=pnl, status='reversed')

        if self.Signal == 'LONG':
            self.Signal = 'SHORT'
        elif self.Signal == 'SHORT':
            self.Signal = 'LONG'
        else:
            raise BadReversal
        # self.open(signal=sig, score=score, entry=entry_price)
        return True

    def get(self):
        return self.Current_analysis

    def update_fields(self, field, data):
        self.Current_analysis.update({field: data})

    def update_pnl(self, pnl, score, adx_avg, events):
        self.Current_analysis.update({"PNL": pnl})
        self.Current_analysis.update({'Live_score': score})
        self.Current_analysis.update({'Mean_Adx': adx_avg})
        self.Current_analysis.update({'Events': events})
        if pnl > self.Highest_PNL:
            self.Highest_PNL = pnl
            self.Current_analysis.update({'Highest_PNL': self.Highest_PNL})
        if pnl < self.Lowest_PNL:
            self.Lowest_PNL = pnl
            self.Current_analysis.update({'Lowest_PNL': self.Lowest_PNL})
        # self.fix_decimals()
        self.publish(topic=f'/stream')
        # self.publish(topic=f'/stream')

    def record_sql(self, datadict):
        self.sql.append(value=datadict, table='signals')

    def fix_decimals(self):
        anal = self.Current_analysis.copy()
        anal.update({'Signal': self.Signal,
                     'Status': self.Status,
                     'Instrument': self.Instrument,
                     'Score': round(float(self.Score), 2),
                     'Live_score': round(float(self.Live_score), 2),
                     'Mean_Adx': round(float(self.adx_avg), 2),
                     'Periods': self.periods,
                     'Events': self.Events,
                     'Open_time': self.Open_time,
                     'Entry': round(float(self.Entry), 4),
                     'Exit': round(float(self.Exit), 4),
                     'Closed_at': self.Closed_at,
                     'PNL': {round(float(self.Pnl), 8)},
                     'Highest_PNL': round(float(self.Highest_PNL), 8),
                     'Lowest_PNL': round(float(self.Lowest_PNL), 8)})
        self.Current_analysis = anal

    def publish(self, topic):
        # anal = self.fix_decimals()
        if s.min_score != 0:
            if self.Score < 0:
                score = self.Score * -1
            else:
                score = self.Score
        if s.min_score != 0:

            if score >= s.min_score and self.adx_avg > s.min_adx:
                # fix decimal rounding

                # print('Publishing ..')
                mq.mqPublish(payload=str(json.dumps(self.Current_analysis, default=str)), topic=topic)
        else:
            mq.mqPublish(payload=str(json.dumps(self.Current_analysis, default=str)), topic=topic)


class Strategy:
    """
    Simple Framework For Analyzing FTX Candle Data
    """

    def __init__(self, markets=None, periods=None, min_score=10, quiet=True):
        if periods is None:
            periods = [15, 60, 300, 900, 3600, 14400, 86400]
        if markets is None:
            self.markets = market_enumeration.InteractiveArgs().interactive_calL()
        self.logger = logging.getLogger(__name__)
        self.periods = periods
        self.min_score = min_score
        self.indicators = ['sar', 'macd', 'ema_cross', 'rsi']
        self.quiet = quiet
        self.cp = ColorPrint(quiet=self.quiet)
        # self.sig = CurrentSig()
        self.pandas_ind = PandasIndicators()
        # self.sql = sql_lib.SQLLiteConnection()

    def forward_tester(self, signal, market, score, adx_avg, events=[]):
        """
        {'signal': {'signal': 'NEUTRAL', 'status': None, 'instrument': None, 'open_time': 0.0, 'Entry': 0.0, 'Exit': 0.0, 'closed_at': 0.0}}
        """

        ts = str(datetime.datetime.utcnow())
        current_tick = self.future_ticker(instrument=market)
        # print(signal, market, score)
        if signal_que.get(market):
            # print(f'Have signal {market} in que ..., status: {signal}')
            last_sig = signal_que.get(market).get('object')
            status = last_sig.Status
            current = current_tick
            last = last_sig.Entry
            diff = self.get_change(current, last)

            # print(f'Forward Tester: {market}')
            if last_sig.Signal == 'LONG':
                if float(current) < float(last):
                    diff = diff * -1
            if last_sig.Signal == 'SHORT':
                if float(current) > float(last):
                    diff = diff * -1
            if last_sig.Signal == signal:
                self.cp.white(f'[~] Current Ticker: {current}, Price at open: {last}, PNL if closed now: {diff} %')
                last_sig.update_pnl(diff, score, adx_avg, events)
                # print('Signal Open!')
            else:
                # print('Signal Changed!')
                if signal == 'NEUTRAL':
                    # print('Closing ....')
                    if last_sig.close(exit_price=current_tick, pnl=diff):
                        sig = CurrentSig(instrument=market, signal=signal, score=score,
                                         entry=self.future_ticker(instrument=market),
                                         events=events)
                        signal_que.update({market: {'object': sig, 'json': sig.Current_analysis}})

                else:
                    self.logger.info(f'Reversing signal on {market} to {signal}')
                    if last_sig.reverse(entry_price=current_tick, score=score, pnl=diff):
                        sig = CurrentSig(instrument=market, signal=signal, score=score,
                                         entry=self.future_ticker(instrument=market), periods=self.periods,
                                         events=events)
                        signal_que.update({market: {'object': sig, 'json': sig.Current_analysis}})

        else:
            # if signal != 'NEUTRAL':
            print('Initialize new signal ... ')
            sig = CurrentSig(instrument=market, signal=signal, score=score, adx_avg=adx_avg,
                             entry=self.future_ticker(instrument=market), periods=self.periods,
                             events=events)
            signal_que.update({market: {'object': sig, 'json': sig.Current_analysis}})

            return

    def _print(self, data):
        if self.quiet:
            pass
        else:
            print(data)

    def spot_ticker(self, market):
        ret = requests.get(f'https://ftx.com/api/markets/{market}').json()
        return ret['result']['price']

    def binance_ticker_rest_fallback(self, instrument):
        for _ in bin.futures_symbol_ticker():
            if _.get('symbol') == instrument:
                return _

    def future_ticker(self, instrument):
        if args.debug or args.verbosity > - 3:
            print(f'Getting ticker for {instrument}')
        """
        Websocket ticker
        :param instrument: market to query
        :return: last price
        """
        t = TickerQue.tickers.get(instrument)
        self._print(t)
        if t is None:
            _ret = self.binance_ticker_rest_fallback(instrument=instrument)
            if _ret is not None:
                return float(_ret.get('price'))
        else:
            return float(t.get('price'))

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

    def roc(self, close):
        return talib.stream_ROCP(close)

    def balance_of_power(self, open, high, low, close):
        bop = self.pandas_ind.bop(open, high, low, close)
        return bop

    def adx(self, high, low, close, window=14):
        adx = pandas_ta.adx(high, low, close, length=window)
        return adx

    def volume_spike_detect(self, volume_array, symbol, interval, window=26):
        # TODO: open candles
        kline_que_name = f'{symbol}_{interval}'
        #volume_ma =
        volume_ma = self.moving_average(df=volume_array, n=window)

        current_val = volume_array[-1]
        if current_val > volume_ma * 2:
            mult = divmod(current_val, volume_ma)[0]
            cp.alert(f'[~] Volume spike {mult}x average on {kline_que_name}')
            return mult
        return False

    def exponential_moving_average(self, df, n):
        """
        :param df: pandas.DataFrame
        :param n:
        :return: pandas.DataFrame
        """
        EMA = talib.EMA(df, timeperiod=n)
        # print(EMA)
        return EMA[-1]


    def generate_sar(self, high_array, low_array, acceleration=np.asarray([0.02]), maximum=np.asarray([0.2]),
                     market='BTCUSDT'):
        # print(high_array, low_array)
        sar = talib.SAR(high_array, low_array, acceleration, maximum)

        ticker = (self.future_ticker(market))
        # print(ticker)
        # print(sar)
        if not ticker:
            self.logger.error(f'Error getting for market {market}')
            return 0, 0, 0
        else:
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

    def generate_rsi(self, close_array, window=14):
        rsi = indicators.relative_strength_index(close_array, n=window)
        if rsi:
            return rsi

    def parse_candle(self, candle, kline_que_name, series):
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
        new_time = [datetime.datetime.fromtimestamp(time / 1000) for time in open_time]

        return open_array_, close_array_, high_array_, low_array_, open_array, close_array, high_array, low_array, \
               volume_array, new_time

    def query_ohlc(self, symbol, period, closed=True):
        """
        {'kline': {'symbol': 'BTCUSDT', 'event_time': 0, 'open_time': 1643673600000, 'time': '1d', 'open':
        '38466.90000000', 'close': 38681.43, 'high': 39265.2, 'low': 38000.0, 'baseVol': 29764.92053,
        'quoteVol': 1148591296.6248717, 'closed': True, 'close_time': 1643759999999.0}}

        """
        kline_que_name = symbol + '_' + str(period)
        success = False
        # kline_que_name = f'{symbol}_{period}'
        c = 0
        # print(klines.keys())

        if klines.get(kline_que_name):
            success = True
        else:
            # print(f'Not present .. {kline_que_name}')

            return self.aggravate(symbol, period)

        if success:
            if not closed:
                return klines.get(kline_que_name).get(closed=False)

            # print(klines)
            klines_ = klines[kline_que_name].get(closed=True)
            if not len(klines):
                return

            # print('candle', klines_.get('kline'))
            # klines_ = json.dumps(klines_)
            # print('kline:', klines_)
            open_time = [float(entry.get('kline').get('open_time')) for entry in klines_]
            open = [float(entry.get('kline').get('open')) for entry in klines_]
            low = [float(entry.get('kline').get('low')) for entry in klines_]
            # mid = candle.get(kline_que_name).get('kline').get('mid') for entry in klines]
            high = [float(entry.get('kline').get('high')) for entry in klines_]
            close = [float(entry.get('kline').get('close')) for entry in klines_]
            base_volume = [float(entry.get('kline').get('baseVol')) for entry in klines_]
            # quote_volume = candle.get(kline_que_name).get('kline').get('quoteVol')
            # volume_array = np.asarray(base_volume)
            # close_array = np.asarray(close)
            # high_array = np.asarray(high)
            # low_array = np.asarray(low)
            new_time = [datetime.datetime.fromtimestamp(time / 1000) for time in open_time]

            volume_array = np.asarray(base_volume)
            open_array = np.asarray(open)
            close_array = np.asarray(close)
            high_array = np.asarray(high)
            low_array = np.asarray(low)
            open_array_ = pd.Series(open, dtype='float64')
            high_array_ = pd.Series(high_array, dtype='float64')
            low_array_ = pd.Series(low_array, dtype='float64')
            close_array_ = pd.Series(close_array, dtype='float64')

            # print(open_array_, close_array_, high_array_, low_array_, open_array, close_array, high_array_, low_array_, volume_array, new_time)

            return open_array_, close_array_, high_array_, low_array_, open_array, close_array, high_array, low_array, volume_array, new_time

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
        # print('Making rest call ... ')
        c = 0
        for i in range(9):
            c += 1
            try:
                klines = bin.get_klines(symbol=symbol, interval=interval)
            except Exception as fuck:
                cp.red(f'[error]: {fuck}')
                if c == 10:
                    return False
                sleep(c)
            else:
                # print(len(klines))
                break
        open_time = [int(entry[0]) for entry in klines]
        open = [float(entry[1]) for entry in klines]
        high = [float(entry[2]) for entry in klines]
        low = [float(entry[3]) for entry in klines]
        close = [float(entry[4]) for entry in klines]
        base_volume = [float(entry[5]) for entry in klines]
        close_time = [float(entry[6]) for entry in klines]
        quote_volume = [float(entry[7]) for entry in klines]
        # self.debug_print(close)
        open_array = np.asarray(open)
        volume_array = np.asarray(base_volume)
        close_array = np.asarray(close)
        high_array = np.asarray(high)
        low_array = np.asarray(low)
        open_array_ = pd.Series(low)
        high_array_ = pd.Series(high)
        low_array_ = pd.Series(low)
        close_array_ = pd.Series(close)
        new_time = [datetime.datetime.fromtimestamp(time / 1000) for time in open_time]
        # return open_array_, close_array_, high_array_, low_array_, open_array, close_array, high_array, low_array, volume_array, new_time
        return open_array_, close_array_, high_array_, low_array_, open_array, close_array, high_array, low_array, volume_array, new_time

    def get_change(self, current, previous):
        if current == previous:
            return 0
        try:
            return (abs(current - previous) / previous) * 100.0
        except ZeroDivisionError:
            return float('inf')

    def percent(self, part, whole):
        try:
            return 100 * float(part) / float(whole)
        except ZeroDivisionError:
            return 0

    def run_indicators(self, _period, market):
        vol_spike = False
        """
        Turn FTX intervals into human friendly time periods
        """
        self._print(f'[⚂] Analysis on {market}, {_period}')

        # long_score = 0
        # short_score = 0

        # open , close, high, low, open, close, high, low, volume, new_time
        open_array_, close_array_, high_array_, low_array_, open_array, close_array, high_array, low_array, \
        volume_array, new_time = self.query_ohlc(symbol=market, period=_period)

        # bop = self.balance_of_power()
        macdret, rogo = self.generate_macd(close_array, new_time)
        self._print(f'[⚂] MACD: {macdret}, {rogo}')

        bop_ret = talib.BOP(open_array_, high_array_, low_array_, close_array_)
        bop_ret = pd.DataFrame.from_dict(bop_ret[:1]).values.tolist()[0][0]
        self.cp.white(f'[⚂] BOP:{bop_ret}')
        adx_val = self.adx(high_array_, low_array_, close_array_)
        # print(pd.DataFrame.from_dict(adx_val[:1500]).values.tolist())
        adx_val = pd.DataFrame.from_dict(adx_val[:500]).values.tolist()[499][0]
        # if s.roc_periods.__contains__(_period):
        roc = self.roc(close_array_)
        vroc = self.roc(volume_array)
        print(f'ROC: {roc}, VROC: {vroc}')
        """roc_val = self.roc(close_array_)
        if roc_val >= .5:
            pl = {'market': market, 'roc': roc_val}
            mq.mqPublish(payload=pl, topic='/roc')"""

        self.cp.purple(f'[⚂] ADX: {adx_val}')

        self._print('[⚂]Calculating sar ... ')
        sar = self.generate_sar(high_array, low_array, market=market)
        self._print(f'[⚂] SAR {sar}')
        if not sar:
            return 0, 0, 0, 0
        rsi = self.generate_rsi(close_array=close_array_)
        self._print(f'[⚂] RSI: {rsi}')
        ema_long = self.exponential_moving_average(close_array, n=26)
        ema_short = self.exponential_moving_average(close_array, n=9)
        self.cp.white(f'[⚂] EMA: {ema_long}, {ema_short}')
        if s.volume_spike_periods.__contains__(_period):
            print(f'Checking {market}_{_period} for volume spike')
            vol_spike = self.volume_spike_detect(volume_array, symbol=market, interval=_period, window=200)

        return rogo, bop_ret, adx_val, sar, rsi, ema_long, ema_short, vol_spike

    def score(self, _period, market):

        rogo, bop_ret, adx_val, sar, rsi, ema_long, ema_short, vol_spike = self.run_indicators(_period, market)
        score = 0
        if rogo == 1:
            score += 1
        elif rogo == -1:
            score -= 1

        if sar[0] and sar[0] == 1:
            score += 1
        elif sar[0] and sar[0] == -1:
            score -= 1
        else:
            return 0, 0, (0, 0, 0), 0
        if rsi > 70:
            score -= 1
        elif rsi > 0 < 30:
            score += 1
        if ema_short > ema_long:
            score += 1
        elif ema_short < ema_long:
            score -= 1

        self.cp.red(f'[⋄] Analysis on {market} for {_period}:, {score}, {adx_val}')
        return score, adx_val, sar, vol_spike

    def calculate_score(self, market):
        sars = {}
        analysis = ''
        total_score = 0

        weighted_score = 0
        highest_score = 0
        p = 0

        adx_mean = 0
        adx_max = 0
        # print(f'We have {len(self.indicators)} indicators ')
        # sars = []
        events = []
        for x, period in enumerate(self.periods):
            score_ = 0

            psecs = period_mapper(p_string=period)
            # print(x, psecs)

            p += 1

            score, adx, sar, vol_spike = self.score(period, market)
            sars[x] = sar

            # if p <= len(self.periods) - 1:
            # print('Period', p)
            # if adx <= 20:
            #    return 'NEUTRAL', 0, 0

            # score_ = score * psecs
            total_score += score

            adx_max += psecs * 100
            adx_mean += adx * psecs
            # adx_mean += adx
            self._print(f'[⋄] Non weighed score: {total_score}, ADX: {adx}')

            highest_score += (psecs * len(self.indicators))
            weighted_score += (score * psecs)
            self.cp.yellow(f'[⋄] Weighted Score: {period}:{psecs}: {weighted_score}')
            if vol_spike:
                events.append(f'Volume spike: {vol_spike}x {period}_{market}')
        # weighted_score = weighted_score / len(self.periods)
        score_pct = self.percent(weighted_score, highest_score)
        adx_avg = self.percent(adx_mean, adx_max)
        self.cp.red(f'[⋄] Weighted Score: {weighted_score} / Highest Possible: {highest_score} ')

        self.cp.white(data='Calculating weighted score!')
        if weighted_score > 0:
            analysis = 'LONG'
            """if x <= 1:
                print('SAR', sars.get(x))
                if sars.get(x) == -1:
                    analysis = 'NEUTRAL'"""

        elif weighted_score < 0:
            analysis = 'SHORT'
            """if x <= 1:
                print('SAR', sars.get(x))
                if sars.get(x) == 1:
                    analysis = 'NEUTRAL'"""

        elif weighted_score == 0:
            analysis = 'NEUTRAL'
        if analysis == 'LONG':
            self.cp.green(f'[🔺] Finished: score: {analysis}, Percent: {score_pct}, ADX: {adx_mean}')
        elif analysis == 'SHORT':
            self.cp.red(f'[🔻] Finished: score: {analysis}, Percent: {score_pct}, ADX: {adx_mean}')
        else:
            self.cp.yellow(f'[🔸] Finished: score: {analysis}')
            score_pct = 0
        self.forward_tester(signal=analysis, market=market, score=score_pct, adx_avg=adx_avg, events=events)
        score_pct = round(score_pct, 2)
        adx_avg = round(adx_avg, 2)
        return analysis, score_pct, adx_avg

    def simulate_score(self, market):
        score = random.randrange(1, 100, 2)
        adx = random.randrange(1, 100, 2)
        analysis = random.choice(['LONG', 'SHORT', 'NEUTRAL'])
        self.forward_tester(signal=analysis, market=market, score=score, adx_avg=adx)
        return analysis, score


def interactive():
    cp = ColorPrint()
    cp.white('[⋄] This s*** is ridiculous!')
    markets = market_enumeration.InteractiveArgs()
    markets = markets.interactive_calL()
    strategy = Strategy(markets, periods=['1m', '5m', '30m', '1h', '4h', '6h', '12h'], quiet=args.quiet)
    while 1:
        for m in strategy.markets:
            strategy.calculate_score(m)


def main(args):
    def enum_markets():
        state.reset()
        _markets = market_enumeration.InteractiveArgs()
        if args.single:
            state.markets = [args.single]
        else:
            state.markets = _markets.interactive_calL(n=max_markets, reverse=args.reverse, shard=args.shard)
        state.running = True
        t = threading.Thread(target=run, args=(state.markets, periods))
        t.start()

    # state = State()
    cp = ColorPrint()
    cp.white('[💰] Lets find some unicorns!')
    cp.yellow('[~] Will search for Volume Spike Events on : ')
    print(s.volume_spike_periods)
    cp.yellow(f'[~] Min score to broadcast: {s.min_score}, Min ADX: {s.min_adx} ')
    # market = sys.argv[1]
    # print('Market', market)

    # if args.debug:
    #    periods = ['1m', '3m']

    if not args.custom_tfs:
        if args.mode == 'microscalp':
            periods = ['1m', '3m', '5m']
        if args.mode == 'scalp':
            periods = ['1m', '3m', '5m', '15m', '30m']
        elif args.mode == 'standard':
            periods = ['1m', '5m', '15m', '30m', '1h', '2h']
        elif args.mode == 'precise':
            periods = ['1m', '5m', '15m', '1h', '4h', '12h']
        elif args.mode == 'longtrends':
            periods = ['3d', '1d', '6h', '2h', '30m']
    else:
        periods = args.custom_tfs

    """supported binance periods:
    1m,3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M"""
    # periods = ['1m', '3m', '15m', '30m', '1h', '2h', '4h', '6h', '12h']

    max_markets = 200 / (len(periods) + 1)
    max_markets = math.floor(max_markets)
    cp.white(f'We can evaluate {max_markets} markets with {len(periods)} periods ... '
             f'requesting top {max_markets} markets by volume ...')
    enum_markets()

    # markets = ['BTCUSDT']

    xx = x = 30
    for x in range(x):
        xx -= 1
        print(f'[~] Booting ... {xx}')
        sleep(1)
    strategy = Strategy(state.markets, periods=periods, quiet=args.quiet)
    while state.running:
        if time.time() - state.start_time > 900:
            print('[⋄] Enumerating markets again ... ')
            enum_markets()

        for m in state.markets:
            if args.debug:
                signal_, score, _adx = strategy.calculate_score(m)
                cp.red(f'{signal_}, {score}, {_adx}')
                sleep(1)
            else:
                try:
                    signal_, score, adx_avg = strategy.calculate_score(m)
                except Exception as err:
                    print(err)
                else:
                    if not args.quiet:
                        cp.red(f'[~] {signal_}, {score}, {m}')
                    sleep(0.125)


def parse_uri(uri):
    try:
        host = uri.split(':')[0]
        port = args.uri.split(':')[1]
    except IndexError:
        cp.red('[!] Error parsing uri: "{uri}, format is incorrect!"')
    else:
        return host, port


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('-d', '--debug', dest='debug', action='store_true', help='Super verbose output')
    args.add_argument('-q', '--quiet', dest='quiet', action='store_true')
    args.add_argument('-m', '--mode', dest='mode', choices=['scalp', 'standard', 'precise'], default='standard',
                      help='Period settings')
    args.add_argument('-ms', '-minscore', dest='min_score', type=float, default=30, help='If set, only broadcast if '
                                                                                         'score is at least this.')
    args.add_argument('-ma', '--minadx', dest='min_adx', type=float, default=20, help='Filter out adx lt value')
    args.add_argument('-r', '--reverse', dest='reverse', action='store_true', help='Get lowest volume markets.')
    args.add_argument('-s', '--single', dest='single', default=0)
    args.add_argument('-S', '--shard_from', dest='shard', type=int, default=0)
    args.add_argument('-u', '--uri', default='localhost:1883', type=str, help='MqTT server in host:port format.')
    args.add_argument('-t', '--timeframes', dest='custom_tfs', type=str, nargs='+', help='Custom timeframes')
    args.add_argument('-sd', '--spikedetect', dest='spike_detect', type=str, nargs='+', help='Detect volume spikes'
                                                                                             'on these timeframes.')
    #args.add_argument('-p', '--patterns', dest='pattern_detection', type=str, nargs='+', help='Detect candlestick '
    #                                                                                          'patterns on these '
    #                                                                                          'timeframes.')
    args.add_argument('-mp', '--min_pattern_rating', dest='min_pattern_rating', type=int, default=50,
                      help='Minimum pattern rating to report.')
    args.add_argument('-v', '--verbosity', action='count', default=0)
    # args.add_argument('-rv', '--roc_periods', dest='roc_periods', default=None, type=str, nargs='+',
    #                  help='Monitor for roc/vroc increases')
    args = args.parse_args()

    cp.white_black('[x] BitSeive -- a High Bandwidth Market Analyser')
    restarts = 1
    s = Settings()
    host, port = parse_uri(args.uri)
    mq = MqSkel(host=host, port=int(port))
    if args.min_score != 0:
        s.min_score = args.min_score
    if args.min_adx != 0:
        s.min_adx = args.min_adx
    # if args.roc_periods:
    #    s.roc_periods = args.roc_periods
    if args.spike_detect:
        s.volume_spike_periods = args.spike_detect

    #if args.pattern_detection:
    #    s.pattern_detect = args.pattern_detection
    #    s.min_pattern_rating = args.min_pattern_rating

    while True:
        try:
            # mq.mqStart()
            main(args)
        finally:
            restarts += 1
            print(f'[⋄] Starting .. {restarts}')
