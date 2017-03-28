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

.. literalinclude:: model.yml
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

.. literalinclude:: model.yml
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
``dummy_limit``:
    Features with unique value counts less than or equal to this
    limit are converted to factors and encoded
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
``test``:
    Name of the testing file
``test_labels``:
    Set to ``True`` if labels are included in the testing file.
``train``:
    Name of the training file

.. literalinclude:: model.yml
   :language: yaml
   :caption: **model.yml**
   :lines: 7-23

Model Section
~~~~~~~~~~~~~

The ``model`` section has the following keys:

``algorithms``:
    Full specification of project location
``balance_classes``:
    usually ``csv`` but could also be ``tsv``
    or other types
``calibration``:
    file name of submission template, which
    is usually provided in Kaggle competitions
``cv_folds``:
    full specification of project location
``estimators``:
    usually ``csv`` but could also be ``tsv``
    or other types
``feature_selection``:
    file name of submission template, which
    is usually provided in Kaggle competitions
``grid_search``:
    full specification of project location
``pvalue_level``:
    usually ``csv`` but could also be ``tsv``
    or other types
``rfe``:
    file name of submission template, which
    is usually provided in Kaggle competitions
``scoring_function``:
    full specification of project location
``type``:
    usually ``csv`` but could also be ``tsv`` or other types

.. literalinclude:: model.yml
   :language: yaml
   :caption: **model.yml**
   :lines: 25-49

Features Section
~~~~~~~~~~~~~~~~

The ``features`` section has the following keys:

``clustering``:
    full specification of project location
``counts``:
    usually ``csv`` but could also be ``tsv`` or other types
``encoding``:
    file name of submission template, which
    is usually provided in Kaggle competitions
``genetic``:
    full specification of project location
``interactions``:
    usually ``csv`` but could also be ``tsv`` or other types
``isomap``:
    file name of submission template, which
    is usually provided in Kaggle competitions
``logtransform``:
    full specification of project location
``numpy``:
    usually ``csv`` but could also be ``tsv`` or other types
``pca``:
    file name of submission template, which
    is usually provided in Kaggle competitions
``scaling``:
    full specification of project location
``scipy``:
    usually ``csv`` but could also be ``tsv`` or other types
``text``:
    file name of submission template, which
    is usually provided in Kaggle competitions
``tsne``:
    file name of submission template, which
    is usually provided in Kaggle competitions

.. literalinclude:: model.yml
   :language: yaml
   :caption: **model.yml**
   :lines: 51-95

Treatments Section
~~~~~~~~~~~~~~~~~~

Although there is no 

.. code-block:: yaml
   :caption: **model.yml**

    treatments:
        doji : ['runs_test', ['all'], 18]
        hc   : ['runs_test', ['all'], 18]

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


Pipeline Section
~~~~~~~~~~~~~~~~

The ``pipeline`` section has the following keys:

``number_jobs``:
    full specification of project location
``seed``:
    usually ``csv`` but could also be ``tsv`` or other types
``verbosity``:
    file name of submission template, which is usually provided in
    Kaggle competitions

.. literalinclude:: model.yml
   :language: yaml
   :caption: **model.yml**
   :lines: 97-100

Plots Section
~~~~~~~~~~~~~

To turn on the automatic generation of any plot in the ``plots``
section, simply set the corresponding value to ``True``.

.. literalinclude:: model.yml
   :language: yaml
   :caption: **model.yml**
   :lines: 102-107

XGBoost Section
~~~~~~~~~~~~~~~

The ``xgboost`` section has the following keys:

``stopping_rounds``:
    early stopping rounds for XGBoost

.. literalinclude:: model.yml
   :language: yaml
   :caption: **model.yml**
   :lines: 109-110

Algorithms Configuration
------------------------

Each algorithm section:

``directory``:
    full specification of project location
``file_extension``
    usually ``csv`` but could also be ``tsv`` or other types
``submission_file``:
    file name of submission template, which is usually provided in
    Kaggle competitions
``submit_probas``:
    ``True`` if submitting probabilities, else ``False`` if
    predictions are just the labels

.. literalinclude:: algos.yml
   :language: yaml
   :caption: **algos.yml**

Final Output
------------

This is an example of your file structure after running the pipeline::

    project
    ├── config
        ├── model.yml
        ├── algos.yml
    └── data
    └── input
        ├── train.csv
        ├── test.csv
    └── model
        ├── model_20170325.pkl
    └── output
        ├── predictions_20170325.csv
        ├── probabilities_20170325.csv
        ├── rankings_20170325.csv
    └── plots
        ├── calibration_train.png
        ├── confusion_train_RF.png
        ├── confusion_train_XGB.png
        ├── feature_importance_train_RF.png
        ├── feature_importance_train_XGB.png
        ├── learning_curve_train_RF.png
        ├── learning_curve_train_XGB.png
        ├── roc_curve_train.png
