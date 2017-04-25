Installation
============

You should already have pip, Python, and XGBoost (see instructions
below) installed on your system. Run the following command to
install AlphaPy::

    pip install -U alphapy

XGBoost
-------

For most users, XGBoost will not install automatically with
``pip``. Please follow the instructions below for your platform
to install XGBoost before installing AlphaPy.

Unix
~~~~

Macintosh
~~~~~~~~~

In a Terminal window:

.. code-block:: shell

    git clone --recursive https://github.com/dmlc/xgboost.git
    cd xgboost/
    ./build.sh
    pip install -e python-package

Windows
~~~~~~~

x

Anaconda Python
---------------

.. note:: If you already have the Anaconda Python distribution,
   then you can create a virtual environment with *conda* using
   the following instructions.

    .. line-block::

        conda create -n alphapy python=3.5
        source activate alphapy
        conda install -c conda-forge pyyaml
        conda install -c conda-forge bokeh
        conda install -c conda-forge matplotlib
        conda install -c conda-forge seaborn
        pip install pandas_datareader
        pip install imbalanced-learn
        pip install category_encoders
        pip install pyfolio
