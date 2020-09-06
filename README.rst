AlphaPy
=======

|badge_pypi| |badge_build| |badge_docs| |badge_downloads|

**AlphaPy** is a machine learning framework for both speculators and
data scientists. It is written in Python mainly with the ``scikit-learn``
and ``pandas`` libraries, as well as many other helpful
packages for feature engineering and visualization. Here are just
some of the things you can do with AlphaPy:

* Run machine learning models using ``scikit-learn``, ``Keras``, ``xgboost``, ``LightGBM``, and ``CatBoost``.
* Generate blended or stacked ensembles.
* Create models for analyzing the markets with *MarketFlow*.
* Predict sporting events with *SportFlow*.
* Develop trading systems and analyze portfolios using *MarketFlow*
  and Quantopian's ``pyfolio``.

.. image:: https://github.com/Alpha314/AlphaPy/blob/master/images/model_pipeline.png
    :width: 100%
    :alt: AlphaPy Model Pipeline
    :align: center

Documentation
-------------

http://alphapy.readthedocs.io/en/latest/

Installation
------------

You should already have pip, Python, and optionally XGBoost, LightGBM, and
CatBoost installed on your system (see below). Run the following command to install
AlphaPy::

    pip install -U alphapy

Pyfolio
~~~~~~~

Pyfolio is automatically installed by AlphaPy, but if you encounter
the following error when trying to create a tear sheet:

    *AttributeError: 'numpy.int64' object has no attribute 'to_pydatetime'*

Install pyfolio with this command:

    pip install git+https://github.com/quantopian/pyfolio

XGBoost
~~~~~~~

For Mac and Windows users, XGBoost will *not* install automatically
with ``pip``. For instructions to install XGBoost on your specific
platform, go to http://xgboost.readthedocs.io/en/latest/build.html.

LightGBM
~~~~~~~~

For instructions to install LightGBM on your specific
platform, go to https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html.

CatBoost
~~~~~~~~

For instructions to install CatBoost on your specific
platform, go to https://catboost.ai/docs/concepts/python-installation.html.

MarketFlow
----------

.. image:: https://github.com/Alpha314/AlphaPy/blob/master/images/market_pipeline.png
    :width: 100%
    :alt: MarketFlow Model
    :align: center

.. image:: https://github.com/Alpha314/AlphaPy/blob/master/images/system_pipeline.png
    :width: 100%
    :alt: MarketFlow System
    :align: center

SportFlow
---------

.. image:: https://github.com/Alpha314/AlphaPy/blob/master/images/sports_pipeline.png
    :width: 100%
    :alt: SportFlow
    :align: center

Support
-------

The official channel for support is to open an issue on Github.

http://github.com/ScottfreeLLC/AlphaPy/issues

Follow us on Twitter:

https://twitter.com/_AlphaPy_?lang=en

Donations
---------

If you like the software, please donate:

http://alphapy.readthedocs.io/en/latest/introduction/support.html#donations


.. |badge_pypi| image:: https://badge.fury.io/py/alphapy.svg
.. |badge_build| image:: https://travis-ci.org/ScottfreeLLC/AlphaPy.svg?branch=master
.. |badge_docs| image:: https://readthedocs.org/projects/alphapy/badge/?version=latest
.. |badge_downloads| image:: https://pepy.tech/badge/alphapy
