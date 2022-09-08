import argparse

import requests
from binance.client import Client as BinanceClient

bincli = BinanceClient(None, None)


def get_symbol(symbol_names):
    """
    Set the information about a symbol.
    Args:
        symbol_name (str): name of the symbol to retrieve
    Return:
        Symbol
    """
    have = []
    symbol_info = bincli.get_ticker()
    for y in symbol_names:
        for x in symbol_info:
           if x.get('symbol') == y:
               have.append(y)
    return have


class FtxPublicApi:

    def list_futures(self):
        return requests.get(f'https://ftx.com/api/futures').json()['result']

    def list_perpetual_futures(self):
        ret = []
        for x in requests.get(f'https://ftx.com/api/futures').json()['result']:
            if x['perpetual']:
                ret.append(x)
        return ret


    def list_markets(self):
        return requests.get(f'https://ftx.com/api/markets').json()['result']


class MarketChecker:
    def __init__(self, api):
        self.api = api


    def check_spot_markets(self, base_list, quote):
        markets = self.api.list_markets()
        ret = []
        for base in base_list:
            for m in markets:
                if m['type'] == 'spot':
                    if m['baseCurrency'] == base:
                        if m['quoteCurrency'] == quote:
                            ret.append(m)
        return ret

    def check_future_markets(self):
        base = []
        futures = self.api.list_perpetual_futures()
        for f in futures:
            if f['type'] == 'perpetual':
                base.append(f['underlying'])
        return base


    def get_quote_arbitrage_perpetuals(self, quote='USD', futures=False):
        """
        Check all perpetual futures to see if an existing underlying spot
        market exists. Returns a list of spot markets which have a corresponding
        perpetual future which can be arbitraged.
        """
        perps = []
        for f in self.api.list_perpetual_futures():
            # print(f)
            underlying = f['underlying']
            vol = f['volumeUsd24h']
            perps.append(underlying)
        if not futures:
            return self.check_spot_markets(perps, quote=quote)


    def parse_ftx_markets(self, quote='USD', futures=False):
        """
        Return a list of futures on ftx which
        have a spot market denoted with the supplied
        quote currency.
        """
        base_symbols = []
        volume_tuples = []
        if not futures:
            print('Enumerating spot markets')
            markets = self.get_quote_arbitrage_perpetuals(quote=quote, futures=futures)
            for m in markets:
                # print(m.get('name'))
                name = m.get('name').split('/')[0]
                # quoteVolume24h
                vol = m.get('quoteVolume24h')
                volume_tuples.append([vol, name])
        else:
            print('Enumerating future markets')
            futures = self.api.list_perpetual_futures()
            for f in futures:
                #print(f)
                name = f.get('name').split('-')[0]
                vol = f.get('volumeUsd24h')
                #print(f'Future: {name}. {vol}')
                volume_tuples.append([vol, name])


        print(volume_tuples)
        return volume_tuples

    def get_ftx_by_volume(self, futures=False, reverse=False):
        volumes = []
        markets = []
        sorted_list = []
        ret = self.parse_ftx_markets(futures=futures)
        for i in ret:
            #print(i[0])
            market = i[1]
            vol = i[0]
            volumes.append(vol)
            markets.append(market)
        if not reverse:
            sorted_volumes = sorted(volumes, key = lambda x:float(x), reverse=True)
        else:
            sorted_volumes = sorted(volumes, key=lambda x: float(x), reverse=False)
        for v in sorted_volumes:
            for mm in ret:
                if mm[0] == v:
                    ftx_base = mm[1]
                    #print(v, ftx_base)
                    sorted_list.append([v, ftx_base])
        return sorted_list


    def check_market_binance(self, futures, reverse):
        ret = self.get_ftx_by_volume(futures, reverse)
        symbols = []
        run_markets = []

        for x in ret:
            v = x[0]
            m = x[1]
            bin_mark = m + 'USDT'
            symbols.append(bin_mark)
        ret = get_symbol(symbols)
        print('We have ', len(ret), 'markets!')
        for x in ret:
            run_markets.append(x)
        return run_markets


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spot', dest='spot_markets', default=False, action='store_true', help='Get a list of FTX futures that also '
                                                                                                'have spot markets on both FTX and '
                                                                                                'binance. ')
    parser.add_argument('-c', '--count', dest='count', default=25, type=int, help='Return the top <count> markets by volume.'
                                                                                  'Default: 25')
    return parser.parse_args()

class InteractiveArgs:
    """
    Interactive Loader
    """
    spot_markets = False
    count = 50

    def interactive_calL(self, n=15, reverse=False, shard=0):
        api = FtxPublicApi()
        m = MarketChecker(api)
        if self.spot_markets:
            ret = m.check_market_binance(futures=False)
        else:
            ret = m.check_market_binance(futures=True, reverse=reverse)
        if shard:
            return ret[shard:][:n]
        return ret[:n]



if __name__ == '__main__':
    args = get_args()
    api = FtxPublicApi()
    m = MarketChecker(api)
    if args.spot_markets:
        ret = m.check_market_binance(futures=False)
    else:
        ret = m.check_market_binance(futures=True)

    print(ret[:args.count])

else:
    print('Imported market enumeration libary. Use `InteractiveArgs.interactive_call`.')