MarketFlow
==========

**MarketFlow** transforms financial market data into machine learning
models for making market predictions. The platform gets stock price
data from Yahoo Finance (end-of-day) and Google Finance (intraday),
transforming the data into canonical form for training and testing.
MarketFlow is powerful because you can easily apply new features
to groups of stocks simultaneously using our *Variable Definition
Language* (VDL). All of the dataframes are aggregated and split
into training and testing files for input into *AlphaPy*.

.. image:: market_pipeline.png
   :alt: Market Pipeline
   :width: 100%
   :align: center

Data Sources
------------

MarketFlow gets daily stock prices from Yahoo Finance and intraday
stock prices from Google Finance. Both data sources have the standard
primitives: ``Open``, ``High``, ``Low``, ``Close``, and ``Volume``.
For daily data, there is a ``Date`` timestamp and for intraday data,
there is a ``Datetime`` timestamp. We augment the intraday data with
a ``bar_number`` field to mark the end of the trading day. All trading
days do not end at 4:00 pm EST, as there are holiday trading days
that are shortened.

.. csv-table:: Amazon Daily Stock Prices (Source: Yahoo)
   :file: amzn_daily.csv

.. note:: Normal market hours are 9:30 am to 4:00 pm EST. Here, we
   retrieved the data from the CST time zone, one hour ahead.

.. csv-table:: Amazon Intraday Stock Prices (Source: Google)
   :file: amzn_intraday.csv

.. note:: You can get Google intraday data going back a maximum of
   50 days. If you want to build your own historical record, then
   we recommend that you save the data on an ongoing basis for a
   a larger backtesting window.

Configuration
-------------

The market configuration file (``market.yml``) is written in YAML
and is divided into logical sections reflecting different parts
of **MarketFlow**. This file is stored in the ``config`` directory
of your project, along with the ``model.yml`` and ``algos.yml`` files.
The ``market`` section has the following parameters:

``data_history``:  
    Number of periods of historical data to retrieve.

``forecast_period``:
    Number of periods to forecast for the target variable.

``fractal``: 
    The time quantum for the data feed, represented by an integer
    followed by a character code. For example "1d" is one day, and
    "5m" is five minutes.

``leaders``: 
    A list of features that are coincident with the target variable.
    For example, with daily stock market data, the ``Open`` is
    considered to be a leader because it is recorded at the market
    open. In contrast, the daily ``High`` or ``Low`` cannot be
    known until the the market close.

``predict_history``: 
    This is the minimum number of periods required to derive all
    of the features in prediction mode on a given date. For example,
    if you use a rolling mean of 50 days, then the ``predict_history``
    must be set to at least 50 so it can be calculated on the

``schema``: 
    This string uniquely identifies the subject matter of the data.
    For example, a schema could be ``prices`` for identifying market
    data.

``target_group``:  
    The name of the group selected from the ``groups`` section,
    e.g., a set of stock symbols.

.. literalinclude:: market.yml
   :language: yaml
   :caption: **market.yml**

Group Analysis
--------------

The cornerstone of MarketFlow is the *Analysis*. You can create
models and forecasts for different groups of stocks. The purpose
of the analysis object is to gather data for all of the group
members and then consolidate the data into train and test files.
Further, some features and the target variable have to be adjusted
(lagged) to avoid data leakage.

A group is simply a collection of symbols for analysis. In this
example, we create different groups for technology stocks, ETFs,
and a smaller group for testing. To create a model for a given
group, simply set the ``target_group`` in the ``market`` section
of the market.yml file and run ``mflow``.

.. literalinclude:: market.yml
   :language: yaml
   :caption: **market.yml**
   :lines: 10-51

Variables and Aliases
---------------------

Because market analysis encompasses a wide array of technical indicators,
you can define features using the *Variable Definition Language* (VDL).







For example, suppose I want a feature that indicates whether
or not a stock is above its 50-day moving average.


.. code-block:: yaml
   :caption: **market.yml**

    treatments:
        doji : ['alphapy.features', 'runs_test', ['all'], 18]
        hc   : ['alphapy.features', 'runs_test', ['all'], 18]

Alias Examples:

.. literalinclude:: market.yml
   :language: yaml
   :caption: **market.yml**
   :lines: 71-104

Variable Examples:

.. literalinclude:: market.yml
   :language: yaml
   :caption: **market.yml**
   :lines: 106-134

Once the aliases and variables are defined, you have a foundation
for defining the features: 

.. literalinclude:: market.yml
   :language: yaml
   :caption: **market.yml**
   :lines: 53-69

Trading Systems
---------------

x

.. image:: system_pipeline.png
   :alt: Market Pipeline
   :width: 100%
   :align: center

Model Configuration
-------------------

MarketFlow runs on top of AlphaPy, so the ``model.yml`` file has
the same format. In the following example, note the use of treatments
to calculate runs for a set of features.

.. literalinclude:: market_model.yml
   :language: text
   :caption: **model.yml**
