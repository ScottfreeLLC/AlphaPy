################################################################################
#
# Package   : AlphaPy
# Module    : estimators
# Created   : July 11, 2013
#
# Copyright 2017 ScottFree Analytics LLC
# Mark Conway & Robert D. Scott II
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################


#
# Imports
#

from alphapy.estimator import Estimator
from alphapy.globs import SSEP

from enum import Enum, unique
import logging
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLasso
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVC
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
import tensorflow as tf
import tensorflow.contrib.learn as skflow
import xgboost as xgb
import yaml


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Model Types
#

@unique
class ModelType(Enum):
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Parameters
    ----------
    msg : str
        Human readable string describing the exception.
    code : :obj:`int`, optional
        Numeric error code.

    Attributes
    ----------
    msg : str
        Human readable string describing the exception.
    code : int
        Numeric error code.

    """
    classification = 1
    clustering = 2
    multiclass = 3
    oneclass = 4
    regression = 5


#
# Objective Functions
#

@unique
class Objective(Enum):
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Parameters
    ----------
    msg : str
        Human readable string describing the exception.
    code : :obj:`int`, optional
        Numeric error code.

    Attributes
    ----------
    msg : str
        Human readable string describing the exception.
    code : int
        Numeric error code.

    """
    maximize = 1
    minimize = 2


#
# Define scorers
#

scorers = {'accuracy'               : (ModelType.classification, Objective.maximize),
           'average_precision'      : (ModelType.classification, Objective.maximize),
           'f1'                     : (ModelType.classification, Objective.maximize),
           'f1_macro'               : (ModelType.classification, Objective.maximize),
           'f1_micro'               : (ModelType.classification, Objective.maximize),
           'f1_samples'             : (ModelType.classification, Objective.maximize),
           'f1_weighted'            : (ModelType.classification, Objective.maximize),
           'neg_log_loss'           : (ModelType.classification, Objective.minimize),
           'precision'              : (ModelType.classification, Objective.maximize),
           'recall'                 : (ModelType.classification, Objective.maximize),
           'roc_auc'                : (ModelType.classification, Objective.maximize),
           'adjusted_rand_score'    : (ModelType.clustering,     Objective.maximize),
           'mean_absolute_error'    : (ModelType.regression,     Objective.minimize),
           'neg_mean_squared_error' : (ModelType.regression,     Objective.minimize),
           'median_absolute_error'  : (ModelType.regression,     Objective.minimize),
           'r2'                     : (ModelType.regression,     Objective.maximize)}


#
# Define XGB scoring map
#

xgb_score_map = {'neg_log_loss'           : 'logloss',
                 'mean_absolute_error'    : 'mae',
                 'neg_mean_squared_error' : 'rmse',
                 'precision'              : 'map',
                 'roc_auc'                : 'auc'}


#
# Classes
#

class AdaBoostClassifierCoef(AdaBoostClassifier):
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Parameters
    ----------
    msg : str
        Human readable string describing the exception.
    code : :obj:`int`, optional
        Numeric error code.

    Attributes
    ----------
    msg : str
        Human readable string describing the exception.
    code : int
        Numeric error code.

    """

    def fit(self, *args, **kwargs):
        super(AdaBoostClassifierCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


class ExtraTreesClassifierCoef(ExtraTreesClassifier):
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Parameters
    ----------
    msg : str
        Human readable string describing the exception.
    code : :obj:`int`, optional
        Numeric error code.

    Attributes
    ----------
    msg : str
        Human readable string describing the exception.
    code : int
        Numeric error code.

    """

    def fit(self, *args, **kwargs):
        super(ExtraTreesClassifierCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


class RandomForestClassifierCoef(RandomForestClassifier):
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Parameters
    ----------
    msg : str
        Human readable string describing the exception.
    code : :obj:`int`, optional
        Numeric error code.

    Attributes
    ----------
    msg : str
        Human readable string describing the exception.
    code : int
        Numeric error code.

    """

    def fit(self, *args, **kwargs):
        super(RandomForestClassifierCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

class GradientBoostingClassifierCoef(GradientBoostingClassifier):
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Parameters
    ----------
    msg : str
        Human readable string describing the exception.
    code : :obj:`int`, optional
        Numeric error code.

    Attributes
    ----------
    msg : str
        Human readable string describing the exception.
    code : int
        Numeric error code.

    """

    def fit(self, *args, **kwargs):
        super(GradientBoostingClassifierCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


#
# Define estimator map
#

estimator_map = {'AB'     : AdaBoostClassifierCoef,
                 'GB'     : GradientBoostingClassifierCoef,
                 'GBR'    : GradientBoostingRegressor,
                 'KNN'    : KNeighborsClassifier,
                 'KNR'    : KNeighborsRegressor,
                 'LOGR'   : LogisticRegression,
                 'LR'     : LinearRegression,
                 'LSVC'   : LinearSVC,
                 'LSVM'   : SVC,
                 'NB'     : MultinomialNB,
                 'RBF'    : SVC,
                 'RF'     : RandomForestClassifierCoef,
                 'RFR'    : RandomForestRegressor,
                 'SVM'    : SVC,
                 'TF_DNN' : skflow.DNNClassifier,
                 'XGB'    : xgb.XGBClassifier,
                 'XGBM'   : xgb.XGBClassifier,
                 'XGBR'   : xgb.XGBRegressor,
                 'XT'     : ExtraTreesClassifierCoef,
                 'XTR'    : ExtraTreesRegressor
                }


#
# Function get_algos_config
#

def get_algos_config(cfg_dir):
    r"""Read in data from the given directory in a given format.

    Several sentences providing an extended description. Refer to
    variables using back-ticks, e.g. `var`.

    Parameters
    ----------
    var1 : array_like
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.  We can also refer to
        variables like `var1`.
    var2 : int
        The type above can either refer to an actual Python type
        (e.g. ``int``), or describe the type of the variable in more
        detail, e.g. ``(N,) ndarray`` or ``array_like``.
    long_var_name : {'hi', 'ho'}, optional
        Choices in brackets, default first when optional.

    Returns
    -------
    describe : type
        Explanation of return value named `describe`.

    Other Parameters
    ----------------
    only_seldom_used_keywords : type
        Explanation
    common_parameters_listed_above : type
        Explanation

    Raises
    ------
    BadException
        Because you shouldn't have done that.

    See Also
    --------
    otherfunc : relationship (optional)
    newfunc : Relationship (optional), which could be fairly long, in which
              case the line wraps here.
    thirdfunc, fourthfunc, fifthfunc

    Notes
    -----
    Notes about the implementation algorithm (if needed).

    This can have multiple paragraphs.

    You may include some math:

    .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}

    And even use a greek symbol like :math:`omega` inline.

    References
    ----------
    Cite the relevant literature, e.g. [1]_.  You may also cite these
    references in the notes section above.

    .. [1] O. McNoleg, "The integration of GIS, remote sensing,
       expert systems and adaptive co-kriging for environmental habitat
       modelling of the Highland Haggis using object-oriented, fuzzy-logic
       and neural-network techniques," Computers & Geosciences, vol. 22,
       pp. 585-588, 1996.

    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> a = [1, 2, 3]
    >>> print [x + 3 for x in a]
    [4, 5, 6]
    >>> print "a\n\nb"
    a
    b

    """

    logger.info("Algorithm Configuration")

    # Read the configuration file

    full_path = SSEP.join([cfg_dir, 'algos.yml'])
    with open(full_path, 'r') as ymlfile:
        specs = yaml.load(ymlfile)

    # Ensure each algorithm has required keys

    required_keys = ['model_type', 'params', 'grid', 'scoring']
    for algo in specs:
        algo_keys = specs[algo].keys()
        if set(algo_keys) != set(required_keys):
            logger.warning("Algorithm %s is missing the required keys %s",
                           algo, required_keys)
            logger.warning("Keys found instead: %s", algo_keys)
        else:
            # determine whether or not model type is valid
            model_types = {x.name: x.value for x in ModelType}
            model_type = specs[algo]['model_type']
            if model_type in model_types:
                specs[algo]['model_type'] = ModelType(model_types[model_type])
            else:
                raise ValueError("algos.yml model:type %s unrecognized", model_type)

    # Algorithm Specifications
    return specs


#
# Function get_estimators
#

# AdaBoost (feature_importances_)
# Gradient Boosting (feature_importances_)
# K-Nearest Neighbors (NA)
# Linear Regression (coef_)
# Linear Support Vector Machine (coef_)
# Logistic Regression (coef_)
# Naive Bayes (coef_)
# Radial Basis Function (NA)
# Random Forest (feature_importances_)
# Support Vector Machine (NA)
# XGBoost Binary (NA)
# XGBoost Multi (NA)
# Extra Trees (feature_importances_)
# Random Forest (feature_importances_)
# Randomized Lasso

def get_estimators(model):
    r"""Read in data from the given directory in a given format.

    Several sentences providing an extended description. Refer to
    variables using back-ticks, e.g. `var`.

    Parameters
    ----------
    var1 : array_like
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.  We can also refer to
        variables like `var1`.

    Returns
    -------
    type
        Explanation of anonymous return value of type ``type``.
    describe : type
        Explanation of return value named `describe`.
    out : type
        Explanation of `out`.

    Other Parameters
    ----------------
    only_seldom_used_keywords : type
        Explanation
    common_parameters_listed_above : type
        Explanation

    See Also
    --------
    otherfunc : relationship (optional)
    newfunc : Relationship (optional), which could be fairly long, in which
              case the line wraps here.
    thirdfunc, fourthfunc, fifthfunc

    Notes
    -----
    Notes about the implementation algorithm (if needed).

    This can have multiple paragraphs.

    You may include some math:

    .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}

    And even use a greek symbol like :math:`omega` inline.

    References
    ----------
    Cite the relevant literature, e.g. [1]_.  You may also cite these
    references in the notes section above.

    .. [1] O. McNoleg, "The integration of GIS, remote sensing,
       expert systems and adaptive co-kriging for environmental habitat
       modelling of the Highland Haggis using object-oriented, fuzzy-logic
       and neural-network techniques," Computers & Geosciences, vol. 22,
       pp. 585-588, 1996.

    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> a = [1, 2, 3]
    >>> print [x + 3 for x in a]
    [4, 5, 6]
    >>> print "a\n\nb"
    a
    b

    """

    # Extract model data

    directory = model.specs['directory']
    n_estimators = model.specs['n_estimators']
    n_jobs = model.specs['n_jobs']
    seed = model.specs['seed']
    verbosity = model.specs['verbosity']

    # Initialize estimator dictionary
    estimators = {}

    # Global parameter substitution fields
    ps_fields = {'n_estimators' : 'n_estimators',
                 'n_jobs'       : 'n_jobs',
                 'nthread'      : 'n_jobs',
                 'random_state' : 'seed',
                 'seed'         : 'seed',
                 'verbose'      : 'verbosity'}

    # Get algorithm specifications

    config_dir = SSEP.join([directory, 'config'])
    algo_specs = get_algos_config(config_dir)

    # Create estimators for all of the algorithms

    for algo in algo_specs:
        model_type = algo_specs[algo]['model_type']
        params = algo_specs[algo]['params']
        for param in params:
            if param in ps_fields and isinstance(param, str):
                algo_specs[algo]['params'][param] = eval(ps_fields[param])
        func = estimator_map[algo]
        est = func(**params)
        grid = algo_specs[algo]['grid']
        scoring = algo_specs[algo]['scoring']
        estimators[algo] = Estimator(algo, model_type, est, grid, scoring)

    # return the entire classifier list
    return estimators
