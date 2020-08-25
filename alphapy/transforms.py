################################################################################
#
# Package   : AlphaPy
# Module    : transforms
# Created   : March 14, 2020
#
# Copyright 2020 ScottFree Analytics LLC
# Mark Conway & Robert D. Scott II
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################


#
# Imports
#

from alphapy.calendrical import biz_day_month
from alphapy.calendrical import biz_day_week
from alphapy.globals import NULLTEXT
from alphapy.globals import BSEP, PSEP, USEP
from alphapy.variables import vexec

import itertools
import logging
import math
import numpy as np
import pandas as pd


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function abovema
#

def abovema(f, c, p = 50):
    r"""Determine those values of the dataframe that are above the
    moving average.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    p : int
        The period of the moving average.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = f[c] > ma(f, c, p)
    return new_column


#
# Function adx
#

def adx(f, p = 14):
    r"""Calculate the Average Directional Index (ADX).

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with all columns required for calculation. If you
        are applying ADX through ``vapply``, then these columns are
        calculated automatically.
    p : int
        The period over which to calculate the ADX.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    The Average Directional Movement Index (ADX) was invented by J. Welles
    Wilder in 1978 [WIKI_ADX]_.  Its value reflects the strength of trend in any
    given instrument.

    .. [WIKI_ADX] https://en.wikipedia.org/wiki/Average_directional_movement_index

    """
    c1 = 'diplus'
    vexec(f, c1)
    c2 = 'diminus'
    vexec(f, c2)
    # calculations
    dip = f[c1]
    dim = f[c2]
    didiff = abs(dip - dim)
    disum = dip + dim
    new_column = 100 * didiff.ewm(span=p).mean() / disum
    return new_column


#
# Function belowma
#

def belowma(f, c, p = 50):
    r"""Determine those values of the dataframe that are below the
    moving average.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    p : int
        The period of the moving average.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = f[c] < ma(f, c, p)
    return new_column


#
# Function c2max
#
    
def c2max(f, c1, c2):
    r"""Take the maximum value between two columns in a dataframe.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``f``.
    c2 : str
        Name of the second column in the dataframe ``f``.

    Returns
    -------
    max_val : float
        The maximum value of the two columns.

    """
    max_val = max(f[c1], f[c2])
    return max_val


#
# Function c2min
#
    
def c2min(f, c1, c2):
    r"""Take the minimum value between two columns in a dataframe.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``f``.
    c2 : str
        Name of the second column in the dataframe ``f``.

    Returns
    -------
    min_val : float
        The minimum value of the two columns.

    """
    min_val = min(f[c1], f[c2])
    return min_val


#
# Function diff
#

def diff(f, c, n = 1):
    r"""Calculate the n-th order difference for the given variable.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    n : int
        The number of times that the values are differenced.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = np.diff(f[c], n)
    return new_column


#
# Function diminus
#

def diminus(f, p = 14):
    r"""Calculate the Minus Directional Indicator (-DI).

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.
    p : int
        The period over which to calculate the -DI.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *A component of the average directional index (ADX) that is used to
    measure the presence of a downtrend. When the -DI is sloping downward,
    it is a signal that the downtrend is getting stronger* [IP_NDI]_.

    .. [IP_NDI] http://www.investopedia.com/terms/n/negativedirectionalindicator.asp

    """
    tr = 'truerange'
    vexec(f, tr)
    atr = USEP.join(['atr', str(p)])
    vexec(f, atr)
    dmm = 'dmminus'
    f[dmm] = dminus(f)
    new_column = 100 * dminus(f).ewm(span=p).mean() / f[atr]
    return new_column


#
# Function diplus
#

def diplus(f, p = 14):
    r"""Calculate the Plus Directional Indicator (+DI).

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.
    p : int
        The period over which to calculate the +DI.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *A component of the average directional index (ADX) that is used to
    measure the presence of an uptrend. When the +DI is sloping upward,
    it is a signal that the uptrend is getting stronger* [IP_PDI]_.

    .. [IP_PDI] http://www.investopedia.com/terms/p/positivedirectionalindicator.asp

    """
    tr = 'truerange'
    vexec(f, tr)
    atr = USEP.join(['atr', str(p)])
    vexec(f, atr)
    dmp = 'dmplus'
    vexec(f, dmp)
    new_column = 100 * f[dmp].ewm(span=p).mean() / f[atr]
    return new_column


#
# Function dminus
#

def dminus(f):
    r"""Calculate the Minus Directional Movement (-DM).

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *Directional movement is negative (minus) when the prior low minus
    the current low is greater than the current high minus the prior high.
    This so-called Minus Directional Movement (-DM) equals the prior low
    minus the current low, provided it is positive. A negative value
    would simply be entered as zero* [SC_ADX]_.

    """
    c1 = 'downmove'
    f[c1] = -net(f, 'low')
    c2 = 'upmove'
    f[c2] = net(f, 'high')
    new_column = f.apply(gtval0, axis=1, args=[c1, c2])
    return new_column


#
# Function dmplus
#

def dmplus(f):
    r"""Calculate the Plus Directional Movement (+DM).

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *Directional movement is positive (plus) when the current high minus
    the prior high is greater than the prior low minus the current low.
    This so-called Plus Directional Movement (+DM) then equals the current
    high minus the prior high, provided it is positive. A negative value
    would simply be entered as zero* [SC_ADX]_.

    .. [SC_ADX] http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx

    """
    c1 = 'upmove'
    f[c1] = net(f, 'high')
    c2 = 'downmove'
    f[c2] = -net(f, 'low')
    new_column = f.apply(gtval0, axis=1, args=[c1, c2])
    return new_column


#
# Function down
#

def down(f, c):
    r"""Find the negative values in the series.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = f[c] < 0
    return new_column


#
# Function dpc
#

def dpc(f, c):
    r"""Get the negative values, with positive values zeroed.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with column ``c``.
    c : str
        Name of the column.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = f.apply(mval, axis=1, args=[c])
    return new_column


#
# Function ema
#

def ema(f, c, p = 20):
    r"""Calculate the mean on a rolling basis.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    p : int
        The period over which to calculate the rolling mean.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *An exponential moving average (EMA) is a type of moving average
    that is similar to a simple moving average, except that more weight
    is given to the latest data* [IP_EMA]_.

    .. [IP_EMA] http://www.investopedia.com/terms/e/ema.asp

    """
    new_column = pd.ewma(f[c], span=p)
    return new_column


#
# Function extract_bizday
#

def extract_bizday(f, c):
    r"""Extract business day of month and week.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the date column ``c``.
    c : str
        Name of the date column in the dataframe ``f``.

    Returns
    -------
    date_features : pandas.DataFrame
        The dataframe containing the date features.
    """

    date_features = pd.DataFrame()
    try:
        date_features = extract_date(f, c)
        rdate = date_features.apply(get_rdate, axis=1)
        bdm = pd.Series(rdate.apply(biz_day_month), name='bizday_month')
        bdw = pd.Series(rdate.apply(biz_day_week), name='bizday_week')
        frames = [date_features, bdm, bdw]
        date_features = pd.concat(frames, axis=1)
    except:
        logger.info("Could not extract business date information from %s column", c)
    return date_features


#
# Function extract_date
#

def extract_date(f, c):
    r"""Extract date into its components: year, month, day, dayofweek.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the date column ``c``.
    c : str
        Name of the date column in the dataframe ``f``.

    Returns
    -------
    date_features : pandas.DataFrame
        The dataframe containing the date features.
    """

    fc = pd.to_datetime(f[c])
    date_features = pd.DataFrame()
    try:
        fyear = pd.Series(fc.dt.year, name='year')
        fmonth = pd.Series(fc.dt.month, name='month')
        fday = pd.Series(fc.dt.day, name='day')
        fdow = pd.Series(fc.dt.dayofweek, name='day_of_week')
        frames = [fyear, fmonth, fday, fdow]
        date_features = pd.concat(frames, axis=1)
    except:
        logger.info("Could not extract date information from %s column", c)
    return date_features


#
# Function extract_time
#

def extract_time(f, c):
    r"""Extract time into its components: hour, minute, second.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the time column ``c``.
    c : str
        Name of the time column in the dataframe ``f``.

    Returns
    -------
    time_features : pandas.DataFrame
        The dataframe containing the time features.
    """

    fc = pd.to_datetime(f[c])
    time_features = pd.DataFrame()
    try:
        fhour = pd.Series(fc.dt.hour, name='year')
        fminute = pd.Series(fc.dt.minute, name='month')
        fsecond = pd.Series(fc.dt.second, name='day')
        frames = [fhour, fminute, fsecond]
        time_features = pd.concat(frames, axis=1)
    except:
        logger.info("Could not extract time information from %s column", c)
    return time_features


#
# Function gap
#

def gap(f):
    r"""Calculate the gap percentage between the current open and
    the previous close.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``open`` and ``close``.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *A gap is a break between prices on a chart that occurs when the
    price of a stock makes a sharp move up or down with no trading
    occurring in between* [IP_GAP]_.

    .. [IP_GAP] http://www.investopedia.com/terms/g/gap.asp

    """
    c1 = 'open'
    c2 = 'close[1]'
    vexec(f, c2)
    new_column = 100 * pchange2(f, c1, c2)
    return new_column


#
# Function gapbadown
#

def gapbadown(f):
    r"""Determine whether or not there has been a breakaway gap down.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``open`` and ``low``.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    References
    ----------
    *A breakaway gap represents a gap in the movement of a stock price
    supported by levels of high volume* [IP_BAGAP]_.

    .. [IP_BAGAP] http://www.investopedia.com/terms/b/breakawaygap.asp

    """
    new_column = f['open'] < f['low'].shift(1)
    return new_column


#
# Function gapbaup
#

def gapbaup(f):
    r"""Determine whether or not there has been a breakaway gap up.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``open`` and ``high``.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    References
    ----------
    *A breakaway gap represents a gap in the movement of a stock price
    supported by levels of high volume* [IP_BAGAP]_.

    """
    new_column = f['open'] > f['high'].shift(1)
    return new_column


#
# Function gapdown
#

def gapdown(f):
    r"""Determine whether or not there has been a gap down.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``open`` and ``close``.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    References
    ----------
    *A gap is a break between prices on a chart that occurs when the
    price of a stock makes a sharp move up or down with no trading
    occurring in between* [IP_GAP]_.

    """
    new_column = f['open'] < f['close'].shift(1)
    return new_column


#
# Function gapup
#

def gapup(f):
    r"""Determine whether or not there has been a gap up.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``open`` and ``close``.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    References
    ----------
    *A gap is a break between prices on a chart that occurs when the
    price of a stock makes a sharp move up or down with no trading
    occurring in between* [IP_GAP]_.

    """
    new_column = f['open'] > f['close'].shift(1)
    return new_column


#
# Function gtval
#

def gtval(f, c1, c2):
    r"""Determine whether or not the first column of a dataframe
    is greater than the second.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``f``.
    c2 : str
        Name of the second column in the dataframe ``f``.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = f[c1] > f[c2]
    return new_column


#
# Function gtval0
#

def gtval0(f, c1, c2):
    r"""For positive values in the first column of the dataframe
    that are greater than the second column, get the value in
    the first column, otherwise return zero. 

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``f``.
    c2 : str
        Name of the second column in the dataframe ``f``.

    Returns
    -------
    new_val : float
        A positive value or zero.

    """
    if f[c1] > f[c2] and f[c1] > 0:
        new_val = f[c1]
    else:
        new_val = 0
    return new_val


#
# Function higher
#

def higher(f, c, o = 1):
    r"""Determine whether or not a series value is higher than
    the value ``o`` periods back.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    o : int, optional
        Offset value for shifting the series.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = f[c] > f[c].shift(o)
    return new_column


#
# Function highest
#

def highest(f, c, p = 20):
    r"""Calculate the highest value on a rolling basis.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    p : int
        The period over which to calculate the rolling maximum.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = f[c].rolling(p).max()
    return new_column


#
# Function hlrange
#

def hlrange(f, p = 1):
    r"""Calculate the Range, the difference between High and Low.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.
    p : int
        The period over which the range is calculated.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = highest(f, 'high', p) - lowest(f, 'low', p)
    return new_column


#
# Function lower
#

def lower(f, c, o = 1):
    r"""Determine whether or not a series value is lower than
    the value ``o`` periods back.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    o : int, optional
        Offset value for shifting the series.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = f[c] < f[c].shift(o)
    return new_column


#
# Function lowest
#

def lowest(f, c, p = 20):
    r"""Calculate the lowest value on a rolling basis.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    p : int
        The period over which to calculate the rolling minimum.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    return f[c].rolling(p).min()


#
# Function ma
#

def ma(f, c, p = 20):
    r"""Calculate the mean on a rolling basis.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    p : int
        The period over which to calculate the rolling mean.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *In statistics, a moving average (rolling average or running average)
    is a calculation to analyze data points by creating series of averages
    of different subsets of the full data set* [WIKI_MA]_.

    .. [WIKI_MA] https://en.wikipedia.org/wiki/Moving_average

    """
    new_column = f[c].rolling(p).mean()
    return new_column


#
# Function maratio
#

def maratio(f, c, p1 = 1, p2 = 10):
    r"""Calculate the ratio of two moving averages.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    p1 : int
        The period of the first moving average.
    p2 : int
        The period of the second moving average.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = ma(f, c, p1) / ma(f, c, p2)
    return new_column


#
# Function mval
#
   
def mval(f, c):
    r"""Get the negative value, otherwise zero.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.

    Returns
    -------
    new_val : float
        Negative value or zero.

    """
    new_val = -f[c] if f[c] < 0 else 0
    return new_val


#
# Function net
#

def net(f, c='close', o = 1):
    r"""Calculate the net change of a given column.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    o : int, optional
        Offset value for shifting the series.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *Net change is the difference between the closing price of a security
    on the day's trading and the previous day's closing price. Net change
    can be positive or negative and is quoted in terms of dollars* [IP_NET]_.

    .. [IP_NET] http://www.investopedia.com/terms/n/netchange.asp

    """
    new_column = f[c] - f[c].shift(o)
    return new_column


#
# Function netreturn
#

def netreturn(f, c, o = 1):
    r"""Calculate the net return, or Return On Invesment (ROI)

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    o : int, optional
        Offset value for shifting the series.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *ROI measures the amount of return on an investment relative to the
    original cost. To calculate ROI, the benefit (or return) of an
    investment is divided by the cost of the investment, and the result
    is expressed as a percentage or a ratio* [IP_ROI]_.

    .. [IP_ROI] http://www.investopedia.com/terms/r/returnoninvestment.asp

    """
    new_column = 100 * pchange1(f, c, o)
    return new_column


#
# Function pchange1
#
    
def pchange1(f, c, o = 1):
    r"""Calculate the percentage change within the same variable.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    o : int
        Offset to the previous value.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = f[c] / f[c].shift(o) - 1.0
    return new_column


#
# Function pchange2
#

def pchange2(f, c1, c2):
    r"""Calculate the percentage change between two variables.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``f``.
    c2 : str
        Name of the second column in the dataframe ``f``.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = f[c1] / f[c2] - 1.0
    return new_column


#
# Function pval
#
  
def pval(f, c):
    r"""Get the positive value, otherwise zero.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.

    Returns
    -------
    new_val : float
        Positive value or zero.

    """
    new_val = f[c] if f[c] > 0 else 0
    return new_val


#
# Function rindex
#

def rindex(f, ci, ch, cl, p = 1):
    r"""Calculate the *range index* spanning a given period ``p``.

    The **range index** is a number between 0 and 100 that
    relates the value of the index column ``ci`` to the
    high column ``ch`` and the low column ``cl``. For example,
    if the low value of the range is 10 and the high value
    is 20, then the range index for a value of 15 would be 50%.
    The range index for 18 would be 80%.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the columns ``ci``, ``ch``, and ``cl``.
    ci : str
        Name of the index column in the dataframe ``f``.
    ch : str
        Name of the high column in the dataframe ``f``.
    cl : str
        Name of the low column in the dataframe ``f``.
    p : int
        The period over which the range index of column ``ci``
        is calculated.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    o = p-1 if f[ci].name == 'open' else 0
    hh = highest(f, ch, p)
    ll = lowest(f, cl, p)
    fn = f[ci].shift(o) - ll
    fd = hh - ll
    new_column = 100 * fn / fd
    return new_column


#
# Function rsi
#

def rsi(f, c, p = 14):
    r"""Calculate the Relative Strength Index (RSI).

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``net``.
    c : str
        Name of the column in the dataframe ``f``.
    p : int
        The period over which to calculate the RSI.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *Developed by J. Welles Wilder, the Relative Strength Index (RSI) is a momentum
    oscillator that measures the speed and change of price movements* [SC_RSI]_.

    .. [SC_RSI] http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:relative_strength_index_rsi

    """
    cdiff = 'net'
    vexec(f, cdiff)
    f['pval'] = upc(f, cdiff)
    f['mval'] = dpc(f, cdiff)
    upcs = ma(f, 'pval', p)
    dpcs = ma(f, 'mval', p)
    new_column = 100 - (100 / (1 + (upcs / dpcs)))
    return new_column


#
# Function rtotal
#

def rtotal(vec):
    r"""Calculate the running total.

    Parameters
    ----------
    vec : pandas.Series
        The input array for calculating the running total.

    Returns
    -------
    running_total : int
        The final running total.

    Example
    -------

    >>> vec.rolling(window=20).apply(rtotal)

    """
    tcount = np.count_nonzero(vec)
    fcount = len(vec) - tcount
    running_total = tcount - fcount
    return running_total


#
# Function runs
#

def runs(vec):
    r"""Calculate the total number of runs.

    Parameters
    ----------
    vec : pandas.Series
        The input array for calculating the number of runs.

    Returns
    -------
    runs_value : int
        The total number of runs.

    Example
    -------

    >>> vec.rolling(window=20).apply(runs)

    """
    runs_value = len(list(itertools.groupby(vec)))
    return runs_value


#
# Function runs_test
#

def runs_test(f, c, wfuncs, window):
    r"""Perform a runs test on binary series.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    wfuncs : list
        The set of runs test functions to apply to the column:

        ``'all'``:
            Run all of the functions below.
        ``'rtotal'``:
            The running total over the ``window`` period.
        ``'runs'``:
            Total number of runs in ``window``.
        ``'streak'``:
            The length of the latest streak.
        ``'zscore'``:
            The Z-Score over the ``window`` period.
    window : int
        The rolling period.

    Returns
    -------
    new_features : pandas.DataFrame
        The dataframe containing the runs test features.

    References
    ----------
    For more information about runs tests for detecting non-randomness,
    refer to [RUNS]_.

    .. [RUNS] http://www.itl.nist.gov/div898/handbook/eda/section3/eda35d.htm

    """

    fc = f[c]
    all_funcs = {'runs'   : runs,
                 'streak' : streak,
                 'rtotal' : rtotal,
                 'zscore' : zscore}
    # use all functions
    if 'all' in wfuncs:
        wfuncs = list(all_funcs.keys())
    # apply each of the runs functions
    new_features = pd.DataFrame()
    for w in wfuncs:
        if w in all_funcs:
            new_feature = fc.rolling(window=window).apply(all_funcs[w])
            new_feature.fillna(0, inplace=True)
            new_column_name = PSEP.join([c, w])
            new_feature = new_feature.rename(new_column_name)
            frames = [new_features, new_feature]
            new_features = pd.concat(frames, axis=1)
        else:
            logger.info("Runs Function %s not found", w)
    return new_features


#
# Function split_to_letters
#

def split_to_letters(f, c):
    r"""Separate text into distinct characters.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the text column in the dataframe ``f``.

    Returns
    -------
    new_feature : pandas.Series
        The array containing the new feature.

    Example
    -------
    The value 'abc' becomes 'a b c'.

    """
    fc = f[c]
    new_feature = None
    dtype = fc.dtypes
    if dtype == 'object':
        fc.fillna(NULLTEXT, inplace=True)
        maxlen = fc.astype(str).str.len().max()
        if maxlen > 1:
            new_feature = fc.apply(lambda x: BSEP.join(list(x)))
    return new_feature


#
# Function streak
#

def streak(vec):
    r"""Determine the length of the latest streak.

    Parameters
    ----------
    vec : pandas.Series
        The input array for calculating the latest streak.

    Returns
    -------
    latest_streak : int
        The length of the latest streak.

    Example
    -------

    >>> vec.rolling(window=20).apply(streak)

    """
    latest_streak = [len(list(g)) for k, g in itertools.groupby(vec)][-1]
    return latest_streak


#
# Function texplode
#

def texplode(f, c):
    r"""Get dummy values for a text column.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the text column in the dataframe ``f``.

    Returns
    -------
    dummies : pandas.DataFrame
        The dataframe containing the dummy variables.

    Example
    -------

    This function is useful for columns that appear to
    have separate character codes but are consolidated
    into a single column. Here, the column ``c`` is
    transformed into five dummy variables.

    === === === === === ===
     c  0_a 1_x 1_b 2_x 2_z
    === === === === === ===
    abz   1   0   1   0   1
    abz   1   0   1   0   1
    axx   1   1   0   1   0
    abz   1   0   1   0   1
    axz   1   1   0   0   1
    === === === === === ===

    """
    fc = f[c]
    maxlen = fc.astype(str).str.len().max()
    fc.fillna(maxlen * BSEP, inplace=True)
    fpad = str().join(['{:', BSEP, '>', str(maxlen), '}'])
    fcpad = fc.apply(fpad.format)
    fcex = fcpad.apply(lambda x: pd.Series(list(x)))
    dummies = pd.get_dummies(fcex)
    return dummies


#
# Function truehigh
#

def truehigh(f):
    r"""Calculate the *True High* value.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *Today's high, or the previous close, whichever is higher* [TS_TR]_.

    .. [TS_TR] http://help.tradestation.com/09_01/tradestationhelp/charting_definitions/true_range.htm

    """
    c1 = 'low[1]'
    vexec(f, c1)
    c2 = 'high'
    new_column = f.apply(c2max, axis=1, args=[c1, c2])
    return new_column


#
# Function truelow
#

def truelow(f):
    r"""Calculate the *True Low* value.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *Today's low, or the previous close, whichever is lower* [TS_TR]_.

    """
    c1 = 'high[1]'
    vexec(f, c1)
    c2 = 'low'
    new_column = f.apply(c2min, axis=1, args=[c1, c2])
    return new_column


#
# Function truerange
#

def truerange(f):
    r"""Calculate the *True Range* value.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *True High - True Low* [TS_TR]_.

    """
    new_column = truehigh(f) - truelow(f)
    return new_column


#
# Function up
#

def up(f, c):
    r"""Find the positive values in the series.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = f[c] > 0
    return new_column


#
# Function upc
#

def upc(f, c):
    r"""Get the positive values, with negative values zeroed.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with column ``c``.
    c : str
        Name of the column.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = f.apply(pval, axis=1, args=[c])
    return new_column


#
# Function xmadown
#

def xmadown(f, c='close', pfast = 20, pslow = 50):
    r"""Determine those values of the dataframe that are below the
    moving average.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str, optional
        Name of the column in the dataframe ``f``.
    pfast : int, optional
        The period of the fast moving average.
    pslow : int, optional
        The period of the slow moving average.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    References
    ----------
    *In the statistics of time series, and in particular the analysis
    of financial time series for stock trading purposes, a moving-average
    crossover occurs when, on plotting two moving averages each based
    on different degrees of smoothing, the traces of these moving averages
    cross* [WIKI_XMA]_.

    .. [WIKI_XMA] https://en.wikipedia.org/wiki/Moving_average_crossover

    """
    sma = ma(f, c, pfast)
    sma_prev = sma.shift(1)
    lma = ma(f, c, pslow)
    lma_prev = lma.shift(1)
    new_column = (sma < lma) & (sma_prev > lma_prev)
    return new_column


#
# Function xmaup
#

def xmaup(f, c='close', pfast = 20, pslow = 50):
    r"""Determine those values of the dataframe that are below the
    moving average.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str, optional
        Name of the column in the dataframe ``f``.
    pfast : int, optional
        The period of the fast moving average.
    pslow : int, optional
        The period of the slow moving average.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    References
    ----------
    *In the statistics of time series, and in particular the analysis
    of financial time series for stock trading purposes, a moving-average
    crossover occurs when, on plotting two moving averages each based
    on different degrees of smoothing, the traces of these moving averages
    cross* [WIKI_XMA]_.

    """
    sma = ma(f, c, pfast)
    sma_prev = sma.shift(1)
    lma = ma(f, c, pslow)
    lma_prev = lma.shift(1)
    new_column = (sma > lma) & (sma_prev < lma_prev)
    return new_column


#
# Function zscore
#

def zscore(vec):
    r"""Calculate the Z-Score.

    Parameters
    ----------
    vec : pandas.Series
        The input array for calculating the Z-Score.

    Returns
    -------
    zscore : float
        The value of the Z-Score.

    References
    ----------
    To calculate the Z-Score, you can find more information here [ZSCORE]_.

    .. [ZSCORE] https://en.wikipedia.org/wiki/Standard_score

    Example
    -------

    >>> vec.rolling(window=20).apply(zscore)

    """
    n1 = np.count_nonzero(vec)
    n2 = len(vec) - n1
    fac1 = float(2 * n1 * n2)
    fac2 = float(n1 + n2)
    rbar = fac1 / fac2 + 1
    sr2num = fac1 * (fac1 - n1 - n2)
    sr2den = math.pow(fac2, 2) * (fac2 - 1)
    sr = math.sqrt(sr2num / sr2den)
    if sr2den and sr:
        zscore = (runs(vec) - rbar) / sr
    else:
        zscore = 0
    return zscore
