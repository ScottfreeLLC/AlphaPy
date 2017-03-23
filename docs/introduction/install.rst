Installation
============

You should already have Python and pip installed on
your system. Run the following command::

    pip install alphapy

.. important:: This is a note admonition.
   This is the second line of the first paragraph.

    .. line-block::

        git clone --recursive https://github.com/dmlc/xgboost.git
        cd xgboost/
        ./build.sh
        pip install -e python-package

.. note:: Advanced users can install this in a virtualenv if they wish.

    .. line-block::

        conda create -n alphapy python=3.5
        source activate alphapy
        conda install -c conda-forge tensorflow
        pip install pandas_datareader
        pip install imblearn
        pip install category_encoders
        pip install gplearn
        conda install -c conda-forge pyyaml
        conda install -c conda-forge bokeh
        conda install -c conda-forge matplotlib
        conda install -c conda-forge seaborn
        pip install pyfolio
