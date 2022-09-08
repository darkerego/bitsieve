#!/usr/bin/env python3
import argparse
import json
import sys
import threading
import time
from datetime import datetime

import numpy as np
import talib
from talib import stream
from binance.client import Client

from indicator_framework import mode_map
from indicator_framework import binance_ws_framework
from indicator_framework.binance_ws_framework import TickerQue
from utils import colorprint, program_state
from utils.current_mode import CurrentMode
from indicator_framework.binance_ws_framework import klines

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
        self._process = TheSARsAreAllAligning(debug=self.debug, verbose=self.verbose)

    def debug_print(self, text):
        if self.debug:
            cp.debug(text)


# def human_time():
#    return [str(datetime.now().today()), time.time()]

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
        t = TickerQue.ticker(instrument)
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
                    # time.sleep(2)
                    # print(self.inst_list)
                    # time.sleep(2)
                    # status_insts = _process.get_instrumnents()
                    # print(status_insts)

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
                        ret, json_ret = i._process.sar_scalper(i, current_mode_list)
                    except TypeError:
                        ret = False
                    else:
                        print(current_mode_list)
                        if ret and not i.sent and i.open:
                            que.append_signal(ret)
                            que.__append_mq__(json_ret)
                            print('New Signal')
                            i.sent = True
                        elif ret and i.sent:
                            print(f'{i.symbol} Signal open')
                        elif not ret and i.closing:
                            side = i.side
                            side = side.lower()
                            print(f'{i.symbol} - Signals closed!')
                            msg = f'[🚫] {i.symbol} Closed: Exit {side} @ {i.last_price}'
                            # json_msg = json.dumps({"signal": "long", "instrument": f"{instrument_string}", "entry":
                            #                     f"{instrument.signal_at}", f"time": instrument.signal_open_time})
                            json_msg = json.dumps({"Exit": i.side, "symbol": i.symbol, "signal_close_time": time.time(),
                                                   "last_price": i.last_price})
                            que.append_signal(msg)
                            que.__append_mq__(json_msg)
                            # que.__ws__signal__(msg)
                            i.closing = False
                            i.open = False
                            i.sent = False
                            i.open_at = None
                            i.closed_at = None
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
    tws.run_bot()


def main():
    args = cli_args()
    print(args)
    if args.symbol or args.symbol_list_file:
        tws = BlitzKreig()
        insts = tws.configure(args)

        t = threading.Thread(target=tws.start_websocket, args=(insts,))
        t.start()
        tws.run_bot()
        print('Data aggregator is demonized.')


if __name__ == '__main__':
    main()
else:
    print('Use the interactive_call, and InteractiveParser to generate signals for consumption in another program.')
