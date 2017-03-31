StockStream
===========

**StockStream** transforms financial market data into machine learning
models for making market predictions. 

.. image:: market_pipeline.png
   :height:  500 px
   :width:  1000 px
   :alt: Market Pipeline
   :align: center

Data Sources
------------

Market data consist of standard primitives such as open, high, low, and
close; the latter three are postdictive and cause data leakage. Leaders
and laggards must be identified and possibly column-shifted, which is
handled by the Model Pipeline.

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

Analysis
--------

x

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

Market Pipeline
---------------

x