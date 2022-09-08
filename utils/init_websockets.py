# from kline_que import KlineQue
import argparse
import json
import logging
import threading

from unicorn_binance_websocket_api.manager import BinanceWebSocketApiManager

from indicator_framework import kline_que, ticker_que
from trade_engine import market_enumeration

TickerQue = ticker_que.TickerQue()
from binance import Client

rest_client = Client(None, None)
import time
logger = logging.getLogger()
# kline_que = KlineQue()
klines = {}

channel_map = [('1m', 'kline_1m'), ('3m', 'kline_3m'), ('5m', 'kline_5m'), ('15m', 'kline_15m'), ('30m', 'kline_30m'),
               ('1h', 'kline_1h', ), ('2h', 'kline_2h'), ('4h', 'kline_4h'), ('6h', 'kline_6h'), ('12h', 'kline_12h')]


class State:
    def __init__(self):
        self.running = False
        self.start_time = 0
        self.markets = []
        self.reset()

    def reset(self):
        self.running = False
        self.markets = []
        self.start_time = time.time()

state = State()

def aggravate(symbol, interval):
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
    klines_ = []
    print(f'Init {symbol}_{interval} Making rest call ... ')
    c = 0

    try:
        kls_ = rest_client.get_klines(symbol=symbol, interval=interval)
    except Exception as fuck:
        logger.error(f'[error]: {fuck}')
        if c == 10:
            return False
        time.sleep(c)
    else:

        for entry in kls_:
            open_time = int(entry[0])
            open = entry[1]
            high = float(entry[2])
            low = float(entry[3])
            close = float(entry[4])
            base_volume = float(entry[5])
            close_time = float(entry[6])
            quote_volume = float(entry[7])
            kline = ({'kline': {'symbol': symbol, 'event_time': 0, 'open_time': open_time, 'time': interval,
                                'open': open, 'close': close, 'high': high, 'low': low,
                                'baseVol': base_volume, 'quoteVol': quote_volume, 'closed': True,
                                'close_time': close_time}})
            klines_.append(kline)

        return klines_


def agg_wrap(s, i):
    for x in range(10):
        k500 = aggravate(s, i)
        if k500:
            return k500
        else:
            print(f'[!] Error initializing {s}:{i}')
            time.sleep(2.5)


class BinanceWs:

    def __init__(self, channels=[], markets=[]):
        # self.running = None
        self.tickers = TickerQue
        market = 'btc_usdt'
        #self.running = False
        tf = 'bookTicker'

        #ubwa = BinanceWebSocketApiManager(exchange="binance.com")
        #ubwa.create_stream(['trade', 'kline_1m'], ['btcusdt', 'bnbbtc', 'ethbtc'])

        self.binance_websocket_api_manager = BinanceWebSocketApiManager(exchange="binance.com-futures")
        # self.binance_websocket_api_manager2 = BinanceWebSocketApiManager(exchange="binance.com-futures")
        self.stream = self.binance_websocket_api_manager.create_stream(channels=channels, markets=markets)
        self.binance_websocket_api_manager.subscribe_to_stream(self.stream, channels, markets)
        #self.mq = MqSkel()

    #def add_to_stream(self, markets=None, channels=None):
    #    if markets:
    #        print(f'Subscribing to {markets} ...')
    #    if channels:
    #        print(f'Subscribing to {channels} ...')

        self.binance_websocket_api_manager.subscribe_to_stream(self.stream, channels, markets)

    def run(self, debug=False):
        while state.running:

            received_stream_data_json = self.binance_websocket_api_manager.pop_stream_data_from_stream_buffer()
            if received_stream_data_json:
                json_data = json.loads(received_stream_data_json)
                if json_data is not None:
                    # print(json_data)
                    data = json_data.get('data', {})
                    data = json.loads(json.dumps(data))
                    if data.get('e') == 'bookTicker':
                        bid = data.get('b')
                        ask = data.get('a')
                        sym = data.get('s')
                        self.tickers.__update__(symbol=sym, bid=bid, ask=ask)

                        ticker_data = self.tickers.ticker(sym)
                        if ticker_data is not None and debug:
                            #print(ticker_data)
                            pass
                    elif data.get('e') == 'kline':
                        """
                        {
                              "e": "kline",     // Event type
                              "E": 123456789,   // Event time
                              "s": "BNBBTC",    // Symbol
                              "k": {
                                "t": 123400000, // Kline start time
                                "T": 123460000, // Kline close time
                                "s": "BNBBTC",  // Symbol
                                "i": "1m",      // Interval
                                "f": 100,       // First trade ID
                                "L": 200,       // Last trade ID
                                "o": "0.0010",  // Open price
                                "c": "0.0020",  // Close price
                                "h": "0.0025",  // High price
                                "l": "0.0015",  // Low price
                                "v": "1000",    // Base asset volume
                                "n": 100,       // Number of trades
                                "x": false,     // Is this kline closed?
                                "q": "1.0000",  // Quote asset volume
                                "V": "500",     // Taker buy base asset volume
                                "Q": "0.500",   // Taker buy quote asset volume
                                "B": "123456"   // Ignore
                              }}"""
                        # print(json_data.get('data'))
                        candle_data = json_data.get('data', {})
                        # if candle_data is not None:
                        #    print(candle_data)

                        candle = candle_data.get('k', {})
                        event_time = candle.get('E')
                        symbol = candle.get('s', {})
                        open_time = candle.get('t', {})
                        close_time = candle.get('T', {})
                        timeframe = candle.get('i', {})
                        close_prices = candle.get('c', {})
                        open_prices = candle.get('o', {})
                        high_prices = candle.get('h', {})
                        low_prices = candle.get('l', {})
                        is_candle_closed = candle.get('x', {})
                        base_vol = candle.get('v')
                        quote_vol = candle.get('q')
                        kline = ({'kline': {'symbol': symbol, 'event_time': event_time, 'open_time': open_time,
                                            'time': timeframe, 'open': open_prices, 'close': close_prices,
                                            'high': high_prices, 'low': low_prices, 'baseVol': base_vol,
                                            'quoteVol': quote_vol, 'closed': is_candle_closed,
                                            'close_time': close_time}})
                        # print(kline)
                        kline_que_name = f'{symbol}_{timeframe}'
                        if not klines.get(kline_que_name):
                            klines[kline_que_name] = kline_que.KlineQue(symbol_tf=kline_que_name)
                            kls = aggravate(symbol, timeframe)
                            #print(kls)
                            klines[kline_que_name].initialize(kline_500=kls)
                        else:
                            klines[kline_que_name].__update__(symbol_tf=kline_que_name, kline=kline)
                            #print(klines[kline_que_name].get())

                            # klines.get(symbol).update(kline)
                            # print('[DEBUG]', klines[kline_que_name].next(closed=False))
                        #if kline:
                        #    self.mq.mqPublish(payload=str(kline), topic=f'/klines/{kline_que_name}')


def interactive(markets=None, time_periods=None, debug=False):
    restarts = 0
    if markets is not None:
        logger.info(f'Requested ws for {len(markets)} markets: ')
        [print(f'{x}') for x in markets]
    else:
        print('Enum markets .. ')
        m_list = market_enumeration.InteractiveArgs()
        markets = m_list.interactive_calL()

    if time_periods is None:
        channels = ['kline_1m', 'kline_5m', 'kline_15m', 'kline_30m', 'kline_1h', 'kline_2h', 'kline_4h', 'kline_6h',
                    'kline_12h', 'kline_1d', 'bookTicker']
    else:
        print(f'Received periods: {time_periods}')
        print(
            f'We have {len(markets)} markets, and {len(time_periods)} periods, requiring {len(markets) * len(time_periods)} '
            f'subscriptions ')

        channels = []

        for tp in time_periods:
            # print(tp)
            for xx in channel_map:
                if xx[0] == tp:
                    # print(xx[0], tp)
                    channels.append(xx[1])

        channels.append('bookTicker')
        # print(channels)
        # print(f'Channels are {channels}')
    print('Magical unicorns fly...')
    kl_socket = BinanceWs(channels=channels, markets=markets)
    t = threading.Thread(target=kl_socket.run)
    t.start()
    print('Running')
    t.join()


def run(markets, time_periods, debug=False):
    interactive(markets, time_periods, debug)


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--debug', action='store_true', default=False, help='Print all ws data')
    args.add_argument('-l', '--list', dest='market_list', default=None, help='List of markets run.')
    args.add_argument('-a', '--auto', dest='auto_markets', action='store_true', default=False,
                      help='Stream top markets '
                           'by volume.')

    args = args.parse_args()
    if args.auto_markets:
        m_list = market_enumeration.InteractiveArgs()
        m_list = m_list.interactive_calL()
        interactive(markets=m_list, debug=args.debug)


if __name__ == "__main__":
    main()
