class KlineQue:
    def __init__(self, symbol_tf):
        print(f'Init {symbol_tf}')
        self.symbol_tf = symbol_tf
        self.closed_candles = {}
        self.open_candles = {}
        self.closed_candles_list = []
        self.open_candles_list = []
        self.open_list_copy = []
        self.closed_list_copy = []
        self.is_initialized = False

    def initialize(self, kline_500):
        for kline in kline_500:
            self.closed_candles_list.append(kline)
            self.closed_list_copy.append(kline)
        self.is_initialized = True

    def __update__(self, symbol_tf, kline):
        """"
        kline = f"{{'kline':}} 'symbol': {symbol} 'time': {timeframe}, 'open': {open_prices},'close':
        {close_prices}, 'high': {high_prices}, 'low': {low_prices} 'baseVol': {base_vol}, 'quoteVol'
        {quote_vol}, 'closed': {is_candle_closed}, "close_time": {close_time}}"
        """
        if len(self.closed_candles_list) > 2000:
            ccl = self.closed_candles_list.copy()
            self.closed_candles_list = ccl[-2000:]
        if len(self.open_candles_list) > 2000:
            ocl = self.open_candles_list.copy()
            self.open_candles_list = ocl[-2000:]
        last_ct = float(self.closed_candles_list[-1].get('kline').get('close_time'))
        this_ct = float(kline.get('kline').get('close_time'))
        if this_ct > last_ct:
            if kline.get('kline').get('closed'):
                print(f'New Closed Candle: for {symbol_tf}\n{kline}')
                self.closed_candles_list.append(kline)
            else:
                self.open_candles_list.append(kline)

    def next(self, closed=True):
        if closed:
            try:
                return self.closed_list_copy.pop()[0]
            except IndexError:
                return None
        else:
            try:
                return self.open_list_copy.pop()[0]
            except IndexError:
                return None

    def get(self, closed=True, history=500):
        # print(len(self.closed_candles_list))
        if closed:
            if len(self.closed_candles_list):
                return self.closed_candles_list[-history:]

