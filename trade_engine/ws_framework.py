import asyncio
import parser
import threading
from unicorn_binance_websocket_api.manager import BinanceWebSocketApiManager
import json, numpy, talib


from trade_engine import kline_que, ticker_que
#from kline_que import KlineQue
import argparse

TickerQue = ticker_que.TickerQue()
# kline_que = KlineQue()
klines = {}



class BinanceWs:

    def __init__(self, channels=[], markets=[]):
        self.tickers = TickerQue
        market = 'theta_usdt'
        tf = 'bookTicker'
        self.binance_websocket_api_manager = BinanceWebSocketApiManager(exchange="binance.com-futures")
        self.stream = self.binance_websocket_api_manager.create_stream(tf, market)
        self.binance_websocket_api_manager.subscribe_to_stream(self.stream, channels, markets)

    def add_to_stream(self, markets=None, channels=None):
        if markets:
            print(f'Subscribing to {markets} ...')
        if channels:
            print(f'Subscribing to {channels} ...')


    def run(self, debug=False):
        while True:

            received_stream_data_json = self.binance_websocket_api_manager.pop_stream_data_from_stream_buffer()
            if received_stream_data_json:
                json_data = json.loads(received_stream_data_json)
                if json_data is not None:
                    #print(json_data)
                    data = json_data.get('data', {})
                    data = json.loads(json.dumps(data))
                    if data.get('e') == 'bookTicker':
                        bid = data.get('b')
                        ask = data.get('a')
                        sym = data.get('s')
                        self.tickers.__update__(symbol=sym, bid=bid, ask=ask)

                        ticker_data  = self.tickers.ticker(sym)
                        if ticker_data  is not None and debug:
                            print(ticker_data)
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
                              }
}
"""
                        #print(json_data.get('data'))
                        candle_data = json_data.get('data', {})
                        #if candle_data is not None:
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
                        kline = ({'kline':{ 'symbol': symbol, 'event_time': event_time, 'open_time': open_time, 'time': timeframe, 'open': open_prices,'close': close_prices, 'high': high_prices, 'low': low_prices, 'baseVol': base_vol, 'quoteVol': quote_vol, 'closed': is_candle_closed}})
                        # print(json.loads(kline))
                        kline_que_name = f'{symbol}_{timeframe}'
                        if not klines.get(kline_que_name):
                            klines[kline_que_name] = kline_que.KlineQue(symbol_tf=kline_que_name)

                        else:
                            klines[kline_que_name].__update__(symbol_tf=kline_que_name, kline=kline)
                            #klines.get(symbol).update(kline)
                        # print('[DEBUG]', klines[kline_que_name].__next__(closed=False))




def interactive(markets=None, debug=False):
    restarts = 0
    channels = ['kline_1m', 'kline_5m', 'kline_15m', 'kline_30m', 'kline_1h', 'kline_2h', 'kline_4h', 'kline_6h', 'kline_12h', 'bookTicker']
    # channels = ['bookTicker']
    if markets is None:
        markets = {}
    print('Magical unicorns fly...')
    kl_socket = BinanceWs(channels=channels, markets=markets)
    t = threading.Thread(target=kl_socket.run)
    t.start()
    print('Running')
    t.join()




if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--debug', action='store_true', default=False, help='Print all ws data')
    args = args.parse_args()
    interactive(debug=args.debug)
