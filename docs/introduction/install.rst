Installation
============

You should already have pip, Python, and XGBoost (see below)
installed on your system. Run the following command to install
AlphaPy::

    pip install -U alphapy

XGBoost
-------

For Macintosh and Window users, XGBoost will *not* install automatically
with ``pip``. For instructions to install XGBoost on your specific
platform, go to http://xgboost.readthedocs.io/en/latest/build.html.

Anaconda Python
---------------

.. note:: If you already have the Anaconda Python distribution,
   then you can create a virtual environment for AlphaPy with
   *conda* with the following recipe.

    .. line-block::

        conda create -n alphapy python=3.5
        source activate alphapy
        conda install -c conda-forge bokeh
        conda install -c conda-forge ipython
        conda install -c conda-forge matplotlib
        conda install -c conda-forge numpy
        conda install -c conda-forge pandas
        conda install -c conda-forge pyyaml
        conda install -c conda-forge scikit-learn
        conda install -c conda-forge scipy
        conda install -c conda-forge seaborn
        conda install -c conda-forge xgboost
        pip install pandas_datareader
        pip install imbalanced-learn
        pip install category_encoders
        pip install pyfolio
