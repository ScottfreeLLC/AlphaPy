Command Line
============

The AlphaPy Command Line Interface (CLI) was designed to be
as simple as possible. First, change the directory to your
project location, where you have already followed the
:doc:`../user_guide/project` specifications::

    cd path/to/project

Run this command to train a model::

    alphapy

Usage::

    alphapy [--train | --predict]

The AlphaPy CLI has the following options:

--train     Train a new model and make predictions [Default]
--predict   Make predictions from a saved model

The domain pipelines have additional options for time series::

    mflow [--train | --predict] [--tdate yyyy-mm-dd] [--pdate yyyy-mm-dd]
    sflow [--train | --predict] [--tdate yyyy-mm-dd] [--pdate yyyy-mm-dd]

--train     Train a new model and make predictions (Default)
--predict   Make predictions from a saved model
--tdate     The training date in format YYYY-MM-DD (Default: Earliest Date in the Data)
--pdate     The prediction date in format YYYY-MM-DD (Default: Today's Date)
