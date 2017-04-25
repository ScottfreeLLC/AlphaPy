Trading System Tutorial
=======================

*MarketFlow Running Time: Approximately 1 minute*

.. image:: returns.png
   :alt: Cumulative Returns
   :width: 75%
   :align: center

A trading system is a set of automated rules for buying and selling
stocks, options, futures, and other instruments. Trading is considered
to be both an art and a science; the scientific 

Many technicians spend their lives chasing the Holy Grail: a system that will make them rich simply by detecting common patterns and executing trades just by following a special recipe. Technicians in history such as Edwards, Elliott, Fibonacci, Gann, and Gartley showed us visually appealing charts but no evidence that these techniques actually worked.
Once you come to the conclusion that there is no master algorithm, then you can move to the next level.

 Further, we have discovered that all of the systems you need have already been invented. The real magic is then using machine learning to decide which system to deploy at any given moment.

Trading systems generally operate in two contexts: trend and
counter-trend. You can
run a system such as Toby Crabel’s Open Range Breakout (ORB) for Widest-Range
(WR) days where you think the market or instrument is going to trend, or
alternatively you can fade support and resistance in a mean-reverting strategy.
[Note that Mr. Crabel runs a successful hedge fund and wrote a rare, groundbreaking
book on short-term trading: Day Trading with Short Term Price Patterns and
Opening Range Breakout]

Once you come to the conclusion that there is no master algorithm, then
you can move to the next level.  Further, we
have discovered that all of the systems you need have already been invented.
The real magic is then using machine learning to decide which system to
deploy at any given moment.

Clearly, you will lose money by blindly executing any short-term system on
any given timeframe. About twenty years ago, when artificial intelligence
was first hot, some funds tried their hand but failed. The problem was that
using neural networks to predict positive return was misguided. Instead, you
need a bidirectional strategy that goes short as easily as it goes long,
but in the proper context.

**Step 1**: From the ``examples`` directory, change your directory::

    cd "Trading System"

Before running MarketFlow, let's briefly review the ``model.yml``
file in the ``config`` directory. We will submit the actual predictions instead of the
probabilities, so ``submit_probas`` is set to ``False``. All
features will be included except for the ``PassengerId``. The
target variable is ``Survived``, the label we are trying to
accurately predict.

We'll compare random forests and XGBoost, run recursive
feature elimination and a grid search, and select the best
model. Note that a blended model of all the algorithms is
a candidate for best model. The details of each algorithm
are located in the ``algos.yml`` file.

.. literalinclude:: closer.yml
   :language: yaml
   :caption: **model.yml**

**Step 2**: Now, we are ready to run MarketFlow::

    mflow

Now, run the following command::

    jupyter notebook

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

Develop a model to predict days with ranges that
•
are greater than average.
We will use both random forests and gradient
•
boosting.
Get daily data from Yahoo over the past few years
•
to train our model.
•
Define technical analysis features with FDL.
