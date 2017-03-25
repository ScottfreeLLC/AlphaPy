Command Line
============

The AlphaPy Command Line Interface (CLI) was designed to be as
simple as possible. The main program ``alpha.py`` is located in
the *alphapy* directory.

First, change the directory to your project location, which has been
set up according to the :doc:`../user_guide/project` specifications::
  
    cd path/to/project
    alphapy -d './config'

Usage::

    alphapy -d config_dir [--score | --train]

The AlphaPy CLI has the following options:

-d          Directory location of the model.yml configuration file
--score     Make predictions from a saved model
--train     Train a new model and make predictions [Default]
