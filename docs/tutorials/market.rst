Market Prediction Tutorial
==========================

*MarketFlow Running Time: Approximately 6 minutes*

.. image:: amzn.png
   :alt: Amazon Candlestick Chart
   :width: 80%
   :align: center

Machine learning subsumes *technical analysis* because collectively,
technical analysis is an infinite set of computations and patterns
for supposedly predicting markets. We can use machine learning as
a feature blender for technical analysis with its moving averages,
indicators, and representations of visual chart formations.

We are not directly predicting net return in our models, although
that is the ultimate goal. By characterizing the market with models,
we can improve the Return On Investment (ROI). We have a wide range
of dependent or target variables from which to choose, not just net
return. There is more power in building a classifier rather than a
more traditional regression model, so we want to define binary
conditions such as whether or not today is going to be a trend day,
rather than a numerical prediction of today’s return.

In this tutorial, we will train a model that predicts whether or
not the next day will have a larger-than-average range. This is
important for deciding which system to deploy on the prediction
day. If our model gives us predictive power, then we can filter
out those days where trading a given system is a losing strategy.


Before running MarketFlow, let's briefly review the ``model.yml``
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

    cd "Trading Model"
    mflow


Training the data using Random Forests and XGBoost, we obtained the following results on our test sets:

For either algorithm, we can predict WR4 days with almost 70% accuracy. Consequently, we can then define our own trading regimes to select which system is most appropriate at the beginning of every trading day. We can also experiment with these models at different fractals, such as weekly or even intraday on an HFT (High Frequency Trading) level.

Let’s now examine the feature importances to see if both algorithms are picking up commonly significant features:

Of the top 10 features, we have 5 shared important features, although not in the same order or magnitude: 17, 42, 109, 110, and 130. Still, these bar charts of the relative importance of each feature bolster our confidence in the model.

Before deploying any model in real-time, we want to assess the degradation in the model between the training and testing set. For classifiers, I prefer both the ROC (Receiver Operating Characteristic) and the calibration curves. First, here are the ROC curves:

You can see that the mean of the AUC (Area Under Curve) decreased from 0.87 in the training set to 0.71 in the testing set, as expected. Now let’s compare the confidence levels when we calibrate each classifier. We want to make sure that all of our calibration curves slope upward from left to right.


From the ``examples`` directory, run the following commands::

    jupyter notebook


The perfectly calibrated classifier would yield a straight line, but we still see that even in the test case, the curves slope upward. So, which algorithm is best? The lower-right graph shows the distribution of the mean predicted value [deciles of 0.1] for RF and XGB. Clearly, the distribution of XGB counts (green line) is more uniform, so that gives us more confidence. In contrast, the RF decile counts are quite small at either tail. I prefer to use XGBoost in most cases, as its reputation is stellar, and it has won many Kaggle competitions.

For practitioners of technical analysis, all is not lost. The point of machine learning is to build useful models from data, and technical analysis is just that: data. But you no longer have to experiment with just a few variants of RSI or MACD on your charts. You can just dump thousands of these technical indicators into the feature blender and see what comes out.
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

We have identified some features that predict large- range days. This is important for determining whether or not to deploy an automated system on any given day.
Results are consistent across training and test data.
•
Results
The learning curves show that results may improve
•
with more data.