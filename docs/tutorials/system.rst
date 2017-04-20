Trading System Tutorial
=======================

*Running Time: Approximately 1 minute*

A trading system is a set of automated rules for buying and selling
stocks, options, futures, and other instruments. Trading is considered
to be both an art and a science; the scientific 

Trading systems generally operate in two contexts: trend and
counter-trend. You can
run a system such as Toby Crabel’s Open Range Breakout (ORB) for Widest-Range
(WR) days where you think the market or instrument is going to trend, or
alternatively you can fade support and resistance in a mean-reverting strategy.
[Note that Mr. Crabel runs a successful hedge fund and wrote a rare, groundbreaking
book on short-term trading: Day Trading with Short Term Price Patterns and
Opening Range Breakout]

Clearly, you will lose money by blindly executing any short-term system on
any given timeframe. About twenty years ago, when artificial intelligence
was first hot, some funds tried their hand but failed. The problem was that
using neural networks to predict positive return was misguided. Instead, you
need a bidirectional strategy that goes short as easily as it goes long,
but in the proper context.

Before running AlphaPy, let's briefly review the ``model.yml``
file. We will submit the actual predictions instead of the
probabilities, so ``submit_probas`` is set to ``False``. All
features will be included except for the ``PassengerId``. The
target variable is ``Survived``, the label we are trying to
accurately predict.

We'll compare random forests and XGBoost, run recursive
feature elimination and a grid search, and select the best
model. Note that a blended model of all the algorithms is
a candidate for best model. The details of each algorithm
are located in the ``algos.yml`` file.

.. literalinclude:: titanic.yml
   :language: yaml
   :caption: **model.yml**

From the ``examples`` directory, run the following commands::

    cd "Trading System"
    mflow
    jupyter notebook

From the ``examples`` directory, run the following commands::

    jupyter notebook
