class TickerQue:
    tickers = {}
    open_klines  = {}
    closed_klines = {}

    def __update__(self, symbol, bid, ask):
        symbol = symbol
        bid = float(bid)
        ask = float(ask)
        price = (bid + ask) / 2
        self.tickers.update({symbol: {'symbol': symbol, 'bid': bid,  'ask': ask, 'price': price}})

    def ticker(self, symbol=None):
        if symbol is None:
            return self.tickers
        return
