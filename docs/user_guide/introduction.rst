Introduction
============

AlphaPy is a machine learning framework for both speculators and
data scientists. It is written in Python with the scikit-learn and
pandas libraries, as well as many other helpful libraries for
feature engineering and visualization. As you can see from the
picture below, we separate the domain pipeline from the model
pipeline. The main job of a domain pipeline is to transform
the raw application data into canonical form, i.e., a training
set and a testing set. The model pipeline is flexible enough to
handle any project and evolved over many Kaggle competitions.

.. image:: alphapy_pipeline.png
   :height: 400 px
   :width:  800 px
   :alt: AlphaPy Model Pipeline
   :align: center

Let's review all of the components in the diagram:

``Domain Pipeline``:
    This is the Python code that creates the training and testing
    data. For example, you may be combining different data frames
    or collecting data from an external feed.

``Domain YAML``: 
    AlphaPy uses configuration files written in YAML to give the
    data scientist maximum flexibility. Typically, you will have
    a standard YAML template for each domain or application.

``Training Data``: 
    The training data is an external file that is read as a
    pandas dataframe. For classification, one of the columns will
    represent the target or dependent variable.

``Testing Data``:  
    The testing data is an external file that is read as a pandas
    dataframe. For classification, the labels may or may not be
    included.

``Model Pipeline``: 
    This Python code is generic for running all classification or
    regression models. The pipeline begins with data and ends with
    a model object for new predictions.

``Model YAML``: 
    The configuration file has specific sections for running the
    model pipeline. Every aspect of creating a model is controlled
    through this file.

``Model Object``: 
    All models are saved to disk. You can load and run your trained
    model on new data in scoring mode.

