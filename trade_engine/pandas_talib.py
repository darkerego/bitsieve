import numpy as np
import pandas_ta
from pandas import DataFrame
from pandas_ta import Imports
from pandas_ta.overlap import ma
from pandas_ta.utils import get_drift, zero
from pandas_ta.utils import get_offset, non_zero_range, verify_series
from pandas_ta.volatility import atr


class PandasIndicators:



    def bop(self, open_, high, low, close, scalar=None, talib=None, offset=None, **kwargs):
        """Indicator: Balance of Power (BOP)"""
        # Validate Arguments
        open_ = verify_series(open_)
        high = verify_series(high)
        low = verify_series(low)
        close = verify_series(close)
        scalar = float(scalar) if scalar else 1
        offset = get_offset(offset)
        mode_tal = bool(talib) if isinstance(talib, bool) else True

        # Calculate Result
        if Imports["talib"] and mode_tal:
            from talib import BOP
            bop = BOP(open_, high, low, close)
        else:
            high_low_range = non_zero_range(high, low)
            close_open_range = non_zero_range(close, open_)
            bop = scalar * close_open_range / high_low_range

        # Offset
        if offset != 0:
            bop = bop.shift(offset)

        # Handle fills
        if "fillna" in kwargs:
            bop.fillna(kwargs["fillna"], inplace=True)
        if "fill_method" in kwargs:
            bop.fillna(method=kwargs["fill_method"], inplace=True)

        # Name and Categorize it
        bop.name = f"BOP"
        bop.category = "momentum"

        return bop

    def adx(self, high, low, close, length=None, lensig=None, scalar=None, mamode=None, drift=None, offset=None, **kwargs):
        """Indicator: ADX"""
        # Validate Arguments
        length = length if length and length > 0 else 14
        lensig = lensig if lensig and lensig > 0 else length
        mamode = mamode if isinstance(mamode, str) else "rma"
        scalar = float(scalar) if scalar else 100
        high = verify_series(high, length)
        low = verify_series(low, length)
        close = verify_series(close, length)
        drift = get_drift(drift)
        offset = get_offset(offset)

        if high is None or low is None or close is None: return

        # Calculate Result
        atr_ = atr(high=high, low=low, close=close, length=length)

        up = high - high.shift(drift)  # high.diff(drift)
        dn = low.shift(drift) - low    # low.diff(-drift).shift(drift)

        pos = ((up > dn) & (up > 0)) * up
        neg = ((dn > up) & (dn > 0)) * dn

        pos = pos.apply(zero)
        neg = neg.apply(zero)

        k = scalar / atr_
        dmp = k * pandas_ta.overlap.ma(mamode, pos, length=length)
        dmn = k * pandas_ta.overlap.ma(mamode, neg, length=length)

        dx = scalar * (dmp - dmn).abs() / (dmp + dmn)
        adx = pandas_ta.overlap.ma(mamode, dx, length=lensig)

        # Offset
        if offset != 0:
            dmp = dmp.shift(offset)
            dmn = dmn.shift(offset)
            adx = adx.shift(offset)

        # Handle fills
        if "fillna" in kwargs:
            adx.fillna(kwargs["fillna"], inplace=True)
            dmp.fillna(kwargs["fillna"], inplace=True)
            dmn.fillna(kwargs["fillna"], inplace=True)
        if "fill_method" in kwargs:
            adx.fillna(method=kwargs["fill_method"], inplace=True)
            dmp.fillna(method=kwargs["fill_method"], inplace=True)
            dmn.fillna(method=kwargs["fill_method"], inplace=True)

        # Name and Categorize it
        adx.name = f"ADX_{lensig}"
        dmp.name = f"DMP_{length}"
        dmn.name = f"DMN_{length}"

        adx.category = dmp.category = dmn.category = "trend"

        # Prepare DataFrame to return
        data = {adx.name: adx, dmp.name: dmp, dmn.name: dmn}
        adxdf = DataFrame(data)
        adxdf.name = f"ADX_{lensig}"
        adxdf.category = "trend"

        return adxdf


