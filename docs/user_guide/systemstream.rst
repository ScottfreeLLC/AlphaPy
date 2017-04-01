SystemStream
============

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
