MarketFlow
===========

**MarketFlow** transforms financial market data into machine learning
models for making market predictions. The platform gets stock price
data from Yahoo Finance (end-of-day) and Google Finance (intraday),
transforming the data into canonical form for training and testing.
MarketFlow is powerful because you can easily apply new features
to groups of stocks simultaneously using our *Feature Definition
Language* (FDL). All of the dataframes are aggregated and split
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

``forecast_period``:
    This directory contains all of the YAML files. At a minimum, it must
    contain ``model.yml`` and ``algos.yml``.

``fractal``: 
    If required, any data for the domain pipeline is stored here. Data
    from this directory will be transformed into ``train.csv`` and
    ``test.csv`` in the ``input`` directory.

``leaders``: 
    The training file ``train.csv`` and the testing file ``test.csv``
    are stored here. Note that these file names can be named anything
    as configured in the ``model.yml`` file.

``lookback_period``:  
    The final model is dumped here as a pickle file in the format
    ``model_[yyyymmdd].pkl``.

``predict_date``: 
    This directory contains predictions, probabilities, rankings,
    and any submission files:

    * ``predictions_[yyyymmdd].csv``
    * ``probabilities_[yyyymmdd].csv``
    * ``rankings_[yyyymmdd].csv``
    * ``submission_[yyyymmdd].csv``

``schema``: 
    All generated plots are stored here. The file name has the
    following elements:

    * plot name
    * 'train' or 'test'
    * algorithm abbreviation
    * format suffix

    For example, a calibration plot for the testing data for all
    algorithms will be named ``calibration_test.png``. The file
    name for a confusion matrix for XGBoost training data will be
    ``confusion_train_XGB.png``.

``train_date``:  
    The final model is dumped here as a pickle file in the format
    ``model_[yyyymmdd].pkl``.

``target_group``:  
    The final model is dumped here as a pickle file in the format
    ``model_[yyyymmdd].pkl``.

.. literalinclude:: market.yml
   :language: yaml
   :caption: **market.yml**

Analyses
--------

x

Groups
------

.. literalinclude:: market.yml
   :language: yaml
   :caption: **market.yml**
   :lines: 11-52

Features
--------

.. literalinclude:: market.yml
   :language: yaml
   :caption: **market.yml**
   :lines: 54-70

Aliases
-------

.. literalinclude:: market.yml
   :language: yaml
   :caption: **market.yml**
   :lines: 72-105

Variables
---------

Variable Definition Language (VDL) that would make it easy for data
scientists to define formulas and functions. Features are applied to groups,
so feature sets are uniformly applied across multiple frames.
The features are represented by variables, and
these variables map to functions with parameters.

# Numeric substitution is allowed for any number in the expression.
# Offsets are allowed in event expressions but cannot be substituted.
#
# Examples
# --------
#
# Variable('rrunder', 'rr_3_20 <= 0.9')
#
# 'rrunder_2_10_0.7'
# 'rrunder_2_10_0.9'
# 'xmaup_20_50_20_200'
# 'xmaup_10_50_20_50'

.. literalinclude:: market.yml
   :language: yaml
   :caption: **market.yml**
   :lines: 107-137

AlphaPy Configuration
---------------------

.. literalinclude:: market_model.yml
   :language: text
   :caption: **model.yml**

SystemStream
------------

**SystemStream** transforms financial market data into machine learning
models for making market predictions. The platform gets stock price
data from Yahoo Finance (end-of-day) and Google Finance (intraday),
transforming the data into canonical form for training and testing.
StockStream is powerful because you can easily apply new features
to groups of stocks simultaneously using our *Feature Definition
Language* (FDL). All of the dataframes are aggregated and split
into training and testing files for input into *AlphaPy*.

.. image:: system_pipeline.png
   :alt: Market Pipeline
   :width: 100%
   :align: center

Configuration
-------------

Here is an example of a model configuration file. It is written in YAML
and is divided into logical sections reflecting different parts of the
pipeline.

.. literalinclude:: systemstream.yml
   :language: yaml
   :caption: **market.yml**

Data Sources
------------

.. csv-table:: Amazon Daily Stock Prices (Source: Yahoo)
   :file: amzn_daily.csv

.. csv-table:: Amazon Intraday Stock Prices (Source: Google)
   :file: amzn_intraday.csv
