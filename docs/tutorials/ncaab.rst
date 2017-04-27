NCAA Basketball Tutorial
========================

*SportFlow Running Time: Approximately 15 minutes*

.. image:: ncaab_roc_curve.png
   :alt: NCAAB ROC Curve
   :width: 80%
   :align: center

In this tutorial, we use machine learning to predict whether
or not an NCAA Men's Basketball team will cover the *spread*.
The spread is set by Las Vegas bookmakers to balance the betting;
it is a way of giving points to the underdog to encourage bets
on both sides.

SportFlow starts with the basic data and derives time series
features based on streaks and runs (not the baseball runs).
In the table below, the game data includes both *line* and
*over_under* information consolidated from various sports Web sites.
For example, a line of -9 means the home team is favored by 9 points.
A line of +3 means the away team is favored by 3 points; the line
is always relative to the home team. An over_under is the predicted
total score for the game, with a bet being placed on whether not
the final total will be under or over that amount.

.. csv-table:: NCAA Basketball Data
   :file: ncaa.csv

**Step 1**: First, from the ``examples`` directory, change your
directory::

    cd NCAAB

Before running SportFlow, let's briefly review the configuration
files in the ``config`` directory:

``sport.yml``:
    The SportFlow configuration file

``model.yml``:
    The AlphaPy configuration file

In ``sport.yml``, the first three items are used for ``random_scoring``,
which we will not be doing here. By default, we will create a model
based on all ``seasons`` and calculate short-term streaks of 3 with
the ``rolling_window``.

.. literalinclude:: ncaab_sport.yml
   :language: yaml
   :caption: **sport.yml**

In each of the tutorials, we experiment with different options in
``model.yml`` to run AlphaPy. Here, we will run a random forest
classifier with Recursive Feature Elimination and Cross-Validation
(RFECV), and then an XGBoost classifier. We will also perform a
random grid search, which increases the total running time to
approximately 15 minutes. You can get in some two-ball dribbling
while waiting for SportFlow to finish.

In the ``features`` section, we identify the ``factors`` generated
by SportFlow. For example, we want to treat the various streaks
as factors. Other options are ``interactions``, standard ``scaling``,
and a ``threshold`` for removing low-variance features.

Our target variable is ``won_on_spread``, a Boolean indicator of
whether or not the home team covered the spread. This is what we
are trying to predict.

.. literalinclude:: ncaab_model.yml
   :language: yaml
   :caption: **model.yml**

**Step 2**: Now, let's run SportFlow::

    sflow --pdate 2016-03-01

As ``sflow`` runs, you will see the progress of the workflow,
and the logging output is saved in ``sport_flow.log``. When the
workflow completes, your project structure will look like this::

    NCAAB
    ├── sport_flow.log
    ├── config
        ├── algos.yml
        ├── market.yml
        ├── model.yml
    └── data
    └── input
        ├── test.csv
        ├── train.csv
    └── model
        ├── feature_map_20170420.pkl
        ├── model_20170420.pkl
    └── output
        ├── predictions_20170420.csv
        ├── probabilities_20170420.csv
        ├── rankings_20170420.csv
    └── plots
        ├── calibration_test.png
        ├── calibration_train.png
        ├── confusion_test_RF.png
        ├── confusion_train_RF.png
        ├── feature_importance_train_RF.png
        ├── learning_curve_train_RF.png
        ├── roc_curve_test.png
        ├── roc_curve_train.png

Let's look at the results in the ``plots`` directory. Since our
scoring function was ``roc_auc``, we examine the ROC Curve first.
The AUC is approximately 0.61, which is not very high but in the
context of the stock market, we may still be able to derive
some predictive power. Further, we are running the model on a
relatively small sample of stocks, as denoted by the jittery
line of the ROC Curve.

.. image:: rrover_roc_curve.png
   :alt: ROC Curve
   :width: 100%
   :align: center

We can benefit from more samples, as the learning curve shows
that the training and cross-validation lines have yet to converge.

.. image:: rrover_learning_curve.png
   :alt: ROC Curve
   :width: 100%
   :align: center

The good news is that even with a relatively small number of
testing points, the Reliability Curve slopes upward from left
to right, with the dotted line denoting a perfect classifier.

.. image:: rrover_calibration.png
   :alt: ROC Curve
   :width: 100%
   :align: center

To get better accuracy, we can raise our threshold to find the
best candidates, since they are ranked by probability, but this
also means limiting our pool of stocks. Let's take a closer
look at the rankings file.

**Step 3**: Now, let's run SportFlow in predict mode::

    sflow --predict --pdate 2016-04-01

**Step 4**: Check the predictions.

x

``Conclusion`` We can predict large-range days with some confidence,
but only at a higher probability threshold. This is important for
choosing the correct system on any given day. We can achieve
better results with more data, so we recommend expanding the
stock universe, e.g., a group with at least 100 members going
five years back.
