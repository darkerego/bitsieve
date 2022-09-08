import json


class KlineQue:
    def __init__(self, symbol_tf):
        print(f'Init {symbol_tf}')
        self.symbol_tf = symbol_tf
        self.candles = {}
        self.candles_list = []
        self.closed_candles = {}
        self.open_candles = {}
        self.closed_candles_list = []
        self.open_candles_list = []
        self.open_list_copy = []
        self.closed_list_copy = []

    def __update__(self, symbol_tf, kline):
        """"
        kline = f"{{'kline':}} 'symbol': {symbol} 'time': {timeframe}, 'open': {open_prices},'close':
        {close_prices}, 'high': {high_prices}, 'low': {low_prices} 'baseVol': {base_vol}, 'quoteVol'
        {quote_vol}, 'closed': {is_candle_closed}}}"
        """
        # kline = json.loads(kline)

        """if kline.get('is_candle_closed'):
            self.closed_candles[symbol_tf] = kline
            self.closed_candles_list.append(self.closed_candles[symbol_tf])
            self.closed_list_copy.append(self.closed_candles_list.copy)
        else:
            self.open_candles[symbol_tf] = kline
            self.open_candles_list.append(self.open_candles[symbol_tf])
            self.open_list_copy.append(self.open_candles_list.copy())"""
        self.candles[symbol_tf] = kline
        self.candles_list.append(kline)

    def __next__(self, closed=True):
        if closed:
            try:
                return self.closed_candles
            except KeyError:
                return None
        else:
            try:
                return self.open_candles
            except KeyError:
                return None

    def next(self, closed=False):
        if closed:
            try:
                return self.open_list_copy.pop()[0]
            except IndexError:
                return None
        else:
            try:
                return self.closed_list_copy.pop()[0]
            except IndexError:
                return None

