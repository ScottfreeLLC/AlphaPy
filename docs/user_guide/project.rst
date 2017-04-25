Project Structure
=================

Setup
-----

Your initial configuration must have the following directories and files.
The directories ``config``, ``data``, and ``input`` store input, and
the directories ``model``, ``output``, and ``plots`` store output::

    project
    ├── config
        ├── model.yml
        ├── algos.yml
    └── data
    └── input
        ├── train.csv
        ├── test.csv
    └── model
    └── output
    └── plots

The top-level directory is the main project directory with a
unique name. There are six required subdirectories:

``config``:
    This directory contains all of the YAML files. At a minimum, it must
    contain ``model.yml`` and ``algos.yml``.

``data``: 
    If required, any data for the domain pipeline is stored here. Data
    from this directory will be transformed into ``train.csv`` and
    ``test.csv`` in the ``input`` directory.

``input``: 
    The training file ``train.csv`` and the testing file ``test.csv``
    are stored here. Note that these file names can be named anything
    as configured in the ``model.yml`` file.

``model``:  
    The final model is dumped here as a pickle file in the format
    ``model_[yyyymmdd].pkl``.

``output``: 
    This directory contains predictions, probabilities, rankings,
    and any submission files:

    * ``predictions_[yyyymmdd].csv``
    * ``probabilities_[yyyymmdd].csv``
    * ``rankings_[yyyymmdd].csv``
    * ``submission_[yyyymmdd].csv``

``plots``: 
    All generated plots are stored here. The file name has the
    following elements:

    * plot name
    * 'train' or 'test'
    * algorithm abbreviation
    * format suffix

    For example, a calibration plot for the testing data for all
    algorithms will be named ``calibration_test.png``. The file
    name for a confusion matrix for XGBoost training data will be
    ``confusion_train_XGB.png``.

Model Configuration
-------------------

Here is an example of a model configuration file. It is written in
YAML and is divided into logical sections reflecting the stages of
the pipeline. Within each section, you can control different aspects
for experimenting with model results. Please refer to the following
sections for more detail.

.. literalinclude:: titanic.yml
   :language: yaml
   :caption: **model.yml**

Project Section
~~~~~~~~~~~~~~~

The ``project`` section has the following keys:

``directory``:
    The full specification of the project location
``file_extension``:
    The extension is usually ``csv`` but could also be ``tsv`` or other
    types using different delimiters between values
``submission_file``:
    The file name of the submission template, which is usually provided
    in Kaggle competitions
``submit_probas``:
    Set the value to ``True`` if submitting probabilities, or set to
    ``False`` if the predictions are the actual labels or real values.

.. literalinclude:: titanic.yml
   :language: yaml
   :caption: **model.yml**
   :lines: 1-5

.. warning:: If you do not supply a value on the right-hand side of
   the colon [:], then Python will interpret that key as having
   a ``None`` value, which is correct. Do not spell out *None*;
   otherwise, the value will be interpreted as the string 'None'.

Data Section
~~~~~~~~~~~~

The ``data`` section has the following keys:

``drop``:
    A list of features to be dropped from the data frame
``features``:
    A list of features for training. ``'*'`` means all features
    will be used in training.
``sampling``:
    Resample imbalanced classes with one of the sampling methods
    in :py:data:`alphapy.data.SamplingMethod`
``sentinel``:
    The designated value to replace any missing values
``separator``:
    The delimiter separating values in the training and test files
``shuffle``:
    If ``True``, randomly shuffle the data.
``split``:
    The proportion of data to include in training, which is a fraction
    between 0 and 1
``target``:
    The name of the feature that designates the label to predict
``target_value``:
    The value of the target label to predict

.. literalinclude:: titanic.yml
   :language: yaml
   :caption: **model.yml**
   :lines: 7-19

Model Section
~~~~~~~~~~~~~

The ``model`` section has the following keys:

``algorithms``:
    The list of algorithms to test for model selection. Refer to
    :ref:`algo-config` for the abbreviation codes.
``balance_classes``:
    If ``True``, calculate sample weights to offset the majority
    class when training a model.
``calibration``:
    Calibrate final probabilities for a classification. Refer to
    the scikit-learn documentation for Calibration_.
``cv_folds``:
    The number of folds for cross-validation
``estimators``:
    The number of estimators to be used in the machine learning algorithm,
    e.g., the number of trees in a random forest
``feature_selection``:
    Perform univariate feature selection based on percentile. Refer to
    the scikit-learn documentation for FeatureSelection_.
``grid_search``:
    The grid search is either random with a fixed number of iterations, or
    it is a full grid search. Refer to the scikit-learn documentation
    for GridSearch_.
``pvalue_level``:
    The p-value threshold to determine whether or not a numerical feature is
    normally distributed.
``rfe``:
    Perform Recursive Feature Elimination (RFE). Refer to the scikit-learn
    documentation for RecursiveFeatureElimination_.
``scoring_function``:
    The scoring function is an objective function for model evaluation. Use one
    of the values in ScoringFunction_.
``type``:
    The model type is either ``classification`` or ``regression``.

.. _Calibration: http://scikit-learn.org/stable/modules/calibration.html#calibration

.. _FeatureSelection: http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection

.. _GridSearch: http://scikit-learn.org/stable/modules/grid_search.html#grid-search

.. _RecursiveFeatureElimination: http://scikit-learn.org/stable/modules/feature_selection.html#rfe

.. _ScoringFunction: http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values

.. literalinclude:: titanic.yml
   :language: yaml
   :caption: **model.yml**
   :lines: 21-45

Features Section
~~~~~~~~~~~~~~~~

The ``features`` section has the following keys:

``clustering``:
    For clustering, specify the minimum and maximum number of clusters
    and the increment from min-to-max.
``counts``:
    Create features that record counts of the NA values, zero values,
    and the digits 1-9 in each row.
``encoding``:
    Encode factors from features, selecting an encoding type and any
    rounding if necessary. Refer to :py:data:`alphapy.features.Encoders`
    for the encoding type.
``factors``:
    The list of features that are factors.
``interactions``:
    Calculate polynomical interactions of a given degree, and select
    the percentage of interactions included in the feature set.
``isomap``:
    Use isomap embedding. Refer to isomap_.
``logtransform``:
    For numerical features that do not fit a normal distribution, perform
    a log transformation.
``numpy``:
    Calculate the total, mean, standard deviation, and variance of
    each row.
``pca``:
    For Principal Component Analysis, specify the minimum and maximum
    number of components, the increment from min-to-max, and whether or
    not whitening is applied.
``scaling``:
    To scale features, specify ``standard`` or ``minmax``.
``scipy``:
    Calculate skew and kurtosis for row distributions.
``text``:
    If there are text features, then apply vectorization and TF-IDF. If
    vectorization does not work, then apply factorization.
``tsne``:
    Perform t-distributed Stochastic Neighbor Embedding (TSNE), which
    can be very memory-intensive. Refer to TSNE_.
``variance``:
    Remove low-variance features using a specified threshold. Refer to VAR_.

.. _isomap: http://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html#examples-using-sklearn-manifold-isomap

.. _TSNE: http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

.. _VAR: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html

.. literalinclude:: titanic.yml
   :language: yaml
   :caption: **model.yml**
   :lines: 47-92

Treatments Section
~~~~~~~~~~~~~~~~~~

Treatments are special functions for feature extraction. In the
``treatments`` section below, we are applying treatments to two
features *doji* and *hc*. Within the Python list, we are calling
the ``runs_test`` function of the module *alphapy.features*. The
module name is always the first element of the list, and the
the function name is always the second element of the list. The
remaining elements of the list are the actual parameters to the
function.

.. code-block:: yaml
   :caption: **model.yml**

    treatments:
        doji : ['alphapy.features', 'runs_test', ['all'], 18]
        hc   : ['alphapy.features', 'runs_test', ['all'], 18]

Here is the code for the ``runs_test`` function, which calculates
runs for Boolean features. For a treatment function, the first and
second arguments are always the same. The first argument ``f`` is
the data frame, and the second argument ``c`` is the column (or feature)
to which we are going to apply the treatment. The remaining function
arguments correspond to the actual parameters that were specified
in the configuration file, in this case ``wfuncs`` and ``window``.

.. code-block:: python
   :caption: **features.py**

    def runs_test(f, c, wfuncs, window):
        fc = f[c]
        all_funcs = {'runs'   : runs,
                     'streak' : streak,
                     'rtotal' : rtotal,
                     'zscore' : zscore}
        # use all functions
        if 'all' in wfuncs:
            wfuncs = all_funcs.keys()
        # apply each of the runs functions
        new_features = pd.DataFrame()
        for w in wfuncs:
            if w in all_funcs:
                new_feature = fc.rolling(window=window).apply(all_funcs[w])
                new_feature.fillna(0, inplace=True)
                frames = [new_features, new_feature]
                new_features = pd.concat(frames, axis=1)
            else:
                logger.info("Runs Function %s not found", w)
        return new_features

When the ``runs_test`` function is invoked, a new data frame is
created, as multiple feature columns may be generated from a
single treatment function. These new features are returned and
appended to the original data frame.

Pipeline Section
~~~~~~~~~~~~~~~~

The ``pipeline`` section has the following keys:

``number_jobs``:
    Number of jobs to run in parallel [-1 for all cores]
``seed``:
    A random seed integer to ensure reproducible results
``verbosity``:
    The logging level from 0 (no logging) to 10 (highest)

.. literalinclude:: titanic.yml
   :language: yaml
   :caption: **model.yml**
   :lines: 94-97

Plots Section
~~~~~~~~~~~~~

To turn on the automatic generation of any plot in the ``plots``
section, simply set the corresponding value to ``True``.

.. literalinclude:: titanic.yml
   :language: yaml
   :caption: **model.yml**
   :lines: 99-104

XGBoost Section
~~~~~~~~~~~~~~~

The ``xgboost`` section has the following keys:

``stopping_rounds``:
    early stopping rounds for XGBoost

.. literalinclude:: titanic.yml
   :language: yaml
   :caption: **model.yml**
   :lines: 106-107

.. _algo-config:

Algorithms Configuration
------------------------

Each algorithm has its own section in the ``algos.yml`` file, e.g.,
**AB** or **RF**. The following elements are required for every
algorithm entry in the YAML file:

``model_type``:
    Specify ``classification`` or ``regression``
``params``
    The initial parameters for the first fitting
``grid``:
    The grid search dictionary for hyperparameter tuning of an
    estimator. If you are using randomized grid search, then make
    sure that the total number of grid combinations exceeds the
    number of random iterations.
``scoring``:
    Set to ``True`` if a specific scoring function will be applied.

.. note:: The parameters ``n_estimators``, ``n_jobs``, ``seed``, and
   ``verbosity`` are informed by the ``model.yml`` file. When the
   estimators are created, the proper values for these parameters are
   automatically substituted in the ``algos.yml`` file on a global
   basis.

.. literalinclude:: algos.yml
   :language: yaml
   :caption: **algos.yml**

Final Output
------------

This is an example of your file structure after running the pipeline::

    project
    ├── alphapy.log
    ├── config
        ├── algos.yml
        ├── model.yml
    └── data
    └── input
        ├── test.csv
        ├── train.csv
    └── model
        ├── feature_map_20170325.pkl
        ├── model_20170325.pkl
    └── output
        ├── predictions_20170325.csv
        ├── probabilities_20170325.csv
        ├── rankings_20170325.csv
        ├── submission_20170325.csv
    └── plots
        ├── calibration_train.png
        ├── confusion_train_RF.png
        ├── confusion_train_XGB.png
        ├── feature_importance_train_RF.png
        ├── feature_importance_train_XGB.png
        ├── learning_curve_train_RF.png
        ├── learning_curve_train_XGB.png
        ├── roc_curve_train.png
