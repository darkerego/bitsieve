import datetime
import time

import binance
import trendet

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style='darkgrid')
import investpy
#df = investpy.crypto.get_crypto_historical_data(crypto='Ethereum', from_date='01/01/2022',
#                                                to_date='09/09/2022', interval='Daily')


class BinanceApi():
    def __init__(self, lookback_periods=26, tolerance=0.1, high_low_datapoints=5):
        self.lookback_periods = lookback_periods
        self.tolerance = tolerance
        self.high_low_datapoints = high_low_datapoints

        self.binance = binance.client.Client(None, None)
        self.rcparams = rc_params = {
            "lines.color": "white",
            "patch.edgecolor": "white",
            "text.color": "white",
            "axes.facecolor": "black",
            "axes.edgecolor": "lightgray",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "grid.color": "grey",
            "figure.facecolor": "black",
            "figure.edgecolor": "white",
            "figure.figsize": "25, 12"}

    def parse_candles(self, candles):
        #candles = self.aggravate(symbol, interval)
        open_time = [int(entry[0]) for entry in candles]
        _open = [float(entry[1]) for entry in candles]
        high = [float(entry[2]) for entry in candles]
        low = [float(entry[3]) for entry in candles]
        _close = [float(entry[4]) for entry in candles]
        base_volume = [float(entry[5]) for entry in candles]
        close_time = [float(entry[6]) for entry in candles]
        quote_volume = [float(entry[7]) for entry in candles]
        currency = ['USD' for x in range(len(open_time))]
        #new_time = [datetime.datetime.fromtimestamp(t / 1000) for t in close_time]
        date_array = [datetime.datetime.fromtimestamp(t/1000) for t in close_time]

        df = pd.DataFrame(data={'Open':_open, 'High': high, 'Low': low, 'Close': _close, 'Volume': base_volume,
                                   'Currency': currency}, index=date_array,
                            columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Currency'])
        df.index.name = 'Date'
        return df

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

    def calculate(self, symbol, interval, window=5):
        df = self.aggravate(symbol, interval)
        print(df)
        res = trendet.identify_df_trends(df=df, column='Close', window_size=window)

        res.reset_index(inplace=True)

        with plt.style.context(style='ggplot'):
            plt.rcParams.update(self.rcparams)

            plt.figure(figsize=(20, 10))

            ax = sns.lineplot(x=res['Date'], y=res['Close'])

            labels = res['Up Trend'].dropna().unique().tolist()

            for label in labels:
                sns.lineplot(x=res[res['Up Trend'] == label]['Date'],
                             y=res[res['Up Trend'] == label]['Close'],
                             color='green')

                ax.axvspan(res[res['Up Trend'] == label]['Date'].iloc[0],
                           res[res['Up Trend'] == label]['Date'].iloc[-1],
                           alpha=0.2,
                           color='green')

            labels = res['Down Trend'].dropna().unique().tolist()

            for label in labels:
                sns.lineplot(x=res[res['Down Trend'] == label]['Date'],
                             y=res[res['Down Trend'] == label]['Close'],
                             color='red')

                ax.axvspan(res[res['Down Trend'] == label]['Date'].iloc[0],
                           res[res['Down Trend'] == label]['Date'].iloc[-1],
                           alpha=0.2,
                           color='red')

            plt.title = f'Trends: {symbol} {interval}'

            plt.show()


api = BinanceApi()

#print(f'Calc for {}')
plot = api.calculate(symbol='ETHUSDT', interval='1m')


