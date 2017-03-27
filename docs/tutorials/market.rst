Market Prediction Tutorial
==========================

From the ``examples`` directory, run the following commands::

    cd "Trading Model"
    marketstream -d './config'
    jupyter notebook

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