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

    alphapy [--predict | --train]

The AlphaPy CLI has the following options:

--predict   Make predictions from a saved model
--train     Train a new model and make predictions [Default]

The domain pipelines have the same syntax::

    mflow [--predict | --train]
    sflow [--predict | --train]
