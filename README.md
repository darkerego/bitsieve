
# BitSieve 

### About

<p>
BitSieve is a high-performance market analysis toolkit. The idea is to look at the top 30 or 40 crypto markets by volume 
and find the ones that have clear, strong trends. This tool identifies what the best possible markets to trade at the 
moment are. This tool is not intended to do all the TA work for you. It is intended to sort through a tremendous amount 
of information and figure out which markets you ought to look closer at to identify possible profitable trades.

Originally I wrote this specifically to be used with FTX, so it will only evaluate markets that are both present on 
Binance and have perpetual future markets on FTX. Because binance has much better support for technical analysis 
(klines over websockets, for example), the tool is doing all the TA using binance's websocket API. 
Special thanks to the developers of the unicorn-binance-websocket API tool, as this project would not have been possible
without it.
</p>

![alt text](img/demo.png "Bitsieve")


### Methodology
<p>
The logic for scoring a market in this program is based on my theory of identifying both trend and strength of trend. 
Trend is identified by creating an average based on what a collection of indicators is reporting. The formulas are: 
</p>
<b>TA Score</b>

<p>
For each indicator that reports a long signal, add +1 to the score, and subtract -1 for each short signal. Then multiply 
that score by the number of seconds in this period, giving us a weighted score. Likewise, we need to know what the highest 
possible score is for each period, which is the amount of indicators multiplied by the number of seconds in the period. 
Finally, we add all the weighted and highest scores together, then divide those two numbers and convert it into a 
percentage. This is the final score. This gives us a decent idea of what the trend is.
</p>
<b>ADX Average</b>
<p>
For each period calculate the ADX, multiply it by the number of seconds in the period. Highest ADX is 100, so multiply 
seconds in each period by 100 to get max adx. Divide average adx by max adx, this gives us the average ADX of all 
periods, which gives us a good idea of how strong the trend of this market is.
</p>

 ### The Data

<p>
Upon identifying a signal, a json object is broadcasting over mqtt to the topics `/signal` and `/stream`. 

<pre>
{
  "Signal": "LONG",
  "Status": "open",
  "Instrument": "EOSUSDT",
  "Score": 70.37037037037037,
  "Live_score": 56.48148148148148,
  "Mean_Adx": 25.24832970999006,
  "Periods": [
    "1m",
    "3m",
    "5m",
    "15m",
    "30m"
  ],
  "Open_time": "2022-09-25 00:42:28.034579",
  "Entry": 1.2115,
  "Exit": 0,
  "Closed_at": 0,
  "PNL": 0.08254230293024266,
  "Highest_PNL": 0.08254230293024266,
  "Lowest_PNL": -0.33016921172100727,
  "Events": [
    "Volume spike: 5.0x 1m_EOSUSDT",
    "Volume spike: 8.0x 3m_EOSUSDT",
    "Volume spike: 4.0x 5m_EOSUSDT"
  ]
}




</pre>

<p>
As a general rule of thumb, a mean adx greater than 30 is a decently strong strend, and anything over 40 is a very 
strong trend. The TA score is a bit less obvious to interpret. For automated trading, you probably want to use 
something over 50, but you also want to keep in mind that a high score may also indicate that this trend has been 
going for a while and may exhaust soon. It really depends on the situation.
</p>


### Patter Recognition

<p>
I am still trying to figure out the best way to handle this. Currently, I consider patters as just another indicator. 
If patterns are not present, then this indicator score is 0. Long patterns +1 and shorts -1. Of course this is hardly 
precise or complete. I think that patterns need to expire. For instance, a hammer on the two hour is not very useful 
six hours later. Currently, if you are using patterns then you may want to periodically restart the program to avoid 
that effect. 
</p>

### Volume Spike Detection
<p>
A new feature I am testing is volume spike detection. This is calculated by taking the moving average of the 
volume for each candle and comparing it with the last closed candles value (in the future I want to also 
check open candles. but nobody seems to be able to tell me how to do this *). If the last value > moving average *2, 
consider this a spike and report by how many magnitudes more than the 200 period moving average is volume is. Volume 
spikes will be reported in the `Events` field of the mqtt messages.
</p>
<p>
Now enabled by default. There is also now a seperate stream `/event` where you can see volume events. The rate of change 
is also reported:
</p>
<pre>
{
  "event": "spike",
  "instrument": "NEARUSDT",
  "period": "1m",
  "details": {
    "VolumeX": 49,
    "ROC": 0.0026
  }
}

</pre>
<p>
This can be really awesome for scalping. Run a few instances of the engine using the shard feature. Grab a cup of coffee 
and just sit and watch until you see a giant volume spike, like anything over 30x the moving average that also has a 
high rate of change (i suppose you can define "high" as over 0.02) and then go scalping. Right after I implement this I 
caught a sick near pump with a 49x spike!

</p>
<p>
* Footnote on open candles:

[Can I Analyse the Open Candles?](https://stackoverflow.com/questions/71811026/using-binance-websockets-with-ta-lib-can-i-analyse-the-open-candles)
</p>

### Usage 

<pre>
usage: engine.py [-h] [-d] [-q] [-m {scalp,standard,precise}] [-ms MIN_SCORE] [-ma MIN_ADX] [-r] [-s SINGLE] [-S SHARD] [-H HOST] [-t CUSTOM_TFS [CUSTOM_TFS ...]] [-sd SPIKE_DETECT [SPIKE_DETECT ...]]

optional arguments:
  -h, --help            show this help message and exit
  -d, --debug           Super verbose output
  -q, --quiet
  -m {scalp,standard,precise}, --mode {scalp,standard,precise}
                        Period settings
  -ms MIN_SCORE, -minscore MIN_SCORE
                        If set, only broadcast if score is at least this.
  -ma MIN_ADX, --minadx MIN_ADX
                        Filter out adx lt value
  -r, --reverse         Get lowest volume markets.
  -s SINGLE, --single SINGLE
  -S SHARD, --shard_from SHARD
  -H HOST, --host HOST
  -t CUSTOM_TFS [CUSTOM_TFS ...], --timeframes CUSTOM_TFS [CUSTOM_TFS ...]
                        Custom timeframes
  -sd SPIKE_DETECT [SPIKE_DETECT ...], --spikedetect SPIKE_DETECT [SPIKE_DETECT ...]
                        Detect volume spikeson these timeframes.


</pre>

<p>
<b>
How to Use
</b>
Make sure that you are running an mqtt server locally (mosquitto is fine). To start with default options, you can run 
it with no arguments.

The time periods are customizable. There are three default modes:


- Scalp:  ['1m', '3m', '5m', '15m', '30m']
- Standard: ['1m', '5m', '15m', '30m', '1h', '2h']
- Precise: ['1m', '5m', '15m', '1h', '4h', '12h']

Any period that binance's API is supported. Supported periods and their seconds:
</p>


<pre>
period_map = [(60, '1m'), (180, '3m'), (300, '5m'), (900, '15m'), (1800, '30m'), (3600, '1h'), (7200, '2h'),
              (14400, '4h'), (21600, '6h'), (43200, '12h'), (86400, '1d'), (259200, '3d'), (604800, '1w'),
              (2592000, '1M')]
</pre>
<p>
You can use the `--timeframes` argument to supply your own custom periods or you can use `--mode` to choose one of the 
ones that I programed. If you want to do pattern recognition, use `--patterns`, for example `--patterns 2h 1h`. You can 
also set the min score/adx thresholds with `-ms` and `-ma`. If you want to check the lowest volume markets instead of 
the highest, use `--reverse`. To supply an mqtt daemon other than localhost, use `--host`.

This is an incomplete list of the possible settings, I will write documentation for the rest when I get a chance. 
</p>


### Custom Indicators

<p>
You can easily add your own indicators if you know how to code a little. Talib supports tons of them and is easy to 
use. I will be refactoring this to make this more straightforward, but moderately experienced devs can figure it out.
Make sure that you also add the name of that indicator to the `self.indicators` variable of the `Strategy` class.
</p>

### FTX
<p>
Do you not have an FTX account yet? Please support my work by using my 
<a href="https://ftx.com/referrals#a=darkerego">referal code</a>: https://ftx.com/referrals#a=darkerego, you 
will get a discount of fees. FTX is definitely the best crypto futures exchange for these reasons:

- Support for many, many crypto markets
- Up to 20x leverage (anything higher is nuts and a scam)
- Forgiving liquidation engine - you will not ever lose more than 50% of your margin with FTX. Every other futures 
exchange will simply take it all. 
- Intuitive interface
- Awesome flexibility. You can use pretty much anything as collateral.
- Hedge any way that you see fit. 
</p>

<p>
<b><br>
Find this useful? Consider supporting development: </b><br>
BTC:3KD3sKiGSyipijVVJ8jVyLYeRDkkTKzKct<br>
ETH:0x0f7274f04d47c5A7cd08AF848e809396ef6B08A5<br>

Want to hire me? Contact info available on [linkedin](https://www.linkedin.com/in/chev-young-7a22ba152?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3Bdsq11bKxQ0uVwSRiLIH5Zg%3D%3D).


</p>