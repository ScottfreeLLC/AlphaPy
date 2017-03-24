Installation
============

You should already have Python and pip installed on
your system. Run the following command::

    pip install alphapy

.. important:: For MacOS users, the ``xgboost`` package may not
   install automatically through ``pip``. If you have errors,
   then install ``xgboost`` before installing ``alphapy``. You
   will first have to build the package using the following
   instructions. 

    .. line-block::

        git clone --recursive https://github.com/dmlc/xgboost.git
        cd xgboost/
        ./build.sh
        pip install -e python-package

.. note:: If you already have the Anaconda Python distribution,
   then you can create a virtual environment with *conda* using
   the following instructions.

    .. line-block::

        conda create -n alphapy python=3.5
        source activate alphapy
        conda install -c conda-forge tensorflow
        pip install pandas_datareader
        pip install imblearn
        pip install category_encoders
        pip install gplearn
        pip install pyfolio
        conda install -c conda-forge pyyaml
        conda install -c conda-forge bokeh
        conda install -c conda-forge matplotlib
        conda install -c conda-forge seaborn
