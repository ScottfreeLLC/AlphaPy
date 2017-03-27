MarketStream
============

Market data consist of standard primitives such as open, high, low, and close; the latter three are postdictive and cause data leakage. Leaders and laggards must be identified and possibly column-shifted, which is handled by the Model Pipeline.

Suppose we have five years of history for a group of stocks, each stock represented by rows of time series data on a daily basis.
2. We want to create a model that predicts whether or not a stock will generate a given return over the next n days, where n = the forecast period.
3. The goal is to generate canonical training and test data for the model pipeline, so we need to apply a series of transformations to the raw stock data.

.. image:: market_pipeline.png
   :height:  500 px
   :width:  1000 px
   :alt: Market Pipeline
   :align: center

Data Sources
------------

.. csv-table:: Frozen Delights!
   :header: "Treat", "Quantity", "Description"
   :widths: 15, 10, 30

   "Albatross", 2.99, "On a stick!"
   "Crunchy Frog", 1.49, "If we took the bones out, it wouldn't be
   crunchy, now would it?"
   "Gannet Ripple", 1.99, "On a stick!"

Configuration
-------------

Here is an example of a model configuration file. It is written in YAML
and is divided into logical sections reflecting different parts of the
pipeline.

.. literalinclude:: marketstream.yml
   :language: yaml
   :caption: **market.yml**

Groups
------

.. image:: ms_groups.png
   :height:  500 px
   :width:  1000 px
   :alt: Market Groups
   :align: center

Features
--------

Aliases
-------

Variables
---------

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

Along with treatments, we defined a Feature Definition Language (FDL) that would make it easy for data scientists to define formulas and functions.
•
Features are applied to groups, so feature sets are
•
uniformly applied across multiple frames.
The features are represented by variables, and
•
these variables map to functions with parameters.
  29
 •
Suppose we want to use the 50-day moving average (MA) in our model, as we believe that it has predictive power for a stock’s direction.
FDL Example
The moving average function ma has two parameters: a
•
feature (column) name and a time period.
To apply the 50-day MA, we can simply join the function ma_close_50.
•
name with its parameters, separated by “_”, or
If we want to use an alias, then we can define cma to be 30
•
the equivalent of ma_close and get cma_50.


.. image:: ms_variables.png
   :height:  500 px
   :width:  1000 px
   :alt: Market Variables
   :align: center

Market Pipeline
---------------

Pipeline Start
~~~~~~~~~~~~~~

.. image:: ms_pipeline.png
   :height:  500 px
   :width:  1000 px
   :alt: Pipeline Start
   :align: center

Data Feed
~~~~~~~~~

.. image:: ms_data.png
   :height:  500 px
   :width:  1000 px
   :alt: Data Feed
   :align: center

Variable Creation
~~~~~~~~~~~~~~~~~

.. image:: ms_apply.png
   :height:  500 px
   :width:  1000 px
   :alt: Variable Creation
   :align: center

