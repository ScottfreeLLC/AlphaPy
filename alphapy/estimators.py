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

from alphapy.globals import ModelType
from alphapy.globals import Objective
from alphapy.globals import SSEP

from keras.layers import *
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
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
import xgboost as xgb
import yaml


#
# Initialize logger
#

logger = logging.getLogger(__name__)


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
# Class Estimator
#

class Estimator:
    """Store information about each estimator.

    Parameters
    ----------
    algorithm : str
        Abbreviation representing the given algorithm.
    model_type : enum ModelType
        The machine learning task for this algorithm.
    estimator : function
        A scikit-learn, TensorFlow, or XGBoost function.
    grid : dict
        The dictionary of hyperparameters for grid search.

    """

    # __new__
    
    def __new__(cls,
                algorithm,
                model_type,
                estimator,
                grid):
        return super(Estimator, cls).__new__(cls)
    
    # __init__
    
    def __init__(self,
                 algorithm,
                 model_type,
                 estimator,
                 grid):
        self.algorithm = algorithm.upper()
        self.model_type = model_type
        self.estimator = estimator
        self.grid = grid
        
    # __str__

    def __str__(self):
        return self.name


#
# Define estimator map
#

estimator_map = {'AB'     : AdaBoostClassifier,
                 'GB'     : GradientBoostingClassifier,
                 'GBR'    : GradientBoostingRegressor,
                 'KERASC' : KerasClassifier,
                 'KERASR' : KerasRegressor,
                 'KNN'    : KNeighborsClassifier,
                 'KNR'    : KNeighborsRegressor,
                 'LOGR'   : LogisticRegression,
                 'LR'     : LinearRegression,
                 'LSVC'   : LinearSVC,
                 'LSVM'   : SVC,
                 'NB'     : MultinomialNB,
                 'RBF'    : SVC,
                 'RF'     : RandomForestClassifier,
                 'RFR'    : RandomForestRegressor,
                 'SVM'    : SVC,
                 'XGB'    : xgb.XGBClassifier,
                 'XGBM'   : xgb.XGBClassifier,
                 'XGBR'   : xgb.XGBRegressor,
                 'XT'     : ExtraTreesClassifier,
                 'XTR'    : ExtraTreesRegressor
                }


#
# Function get_algos_config
#

def get_algos_config(cfg_dir):
    r"""Read the algorithms configuration file.

    Parameters
    ----------
    cfg_dir : str
        The directory where the configuration file ``algos.yml``
        is stored.

    Returns
    -------
    specs : dict
        The specifications for determining which algorithms to run.

    """

    logger.info("Algorithm Configuration")

    # Read the configuration file

    full_path = SSEP.join([cfg_dir, 'algos.yml'])
    with open(full_path, 'r') as ymlfile:
        specs = yaml.load(ymlfile)

    # Ensure each algorithm has required keys

    minimum_keys = ['model_type', 'params', 'grid']
    required_keys_keras = minimum_keys + ['layers', 'compiler']
    for algo in specs:
        if 'KERAS' in algo:
            required_keys = required_keys_keras
        else:
            required_keys = minimum_keys
        algo_keys = list(specs[algo].keys())
        if set(algo_keys) != set(required_keys):
            logger.warning("Algorithm %s has the wrong keys %s",
                           algo, required_keys)
            logger.warning("Keys found instead: %s", algo_keys)
        else:
            # determine whether or not model type is valid
            model_types = {x.name: x.value for x in ModelType}
            model_type = specs[algo]['model_type']
            if model_type in model_types:
                specs[algo]['model_type'] = ModelType(model_types[model_type])
            else:
                raise ValueError("algos.yml model:type %s unrecognized" % model_type)

    # Algorithm Specifications
    return specs


#
# Function create_keras_model
#

def create_keras_model(nlayers,
                       layer1=None,
                       layer2=None,
                       layer3=None,
                       layer4=None,
                       layer5=None,
                       layer6=None,
                       layer7=None,
                       layer8=None,
                       layer9=None,
                       layer10=None,
                       optimizer=None,
                       loss=None,
                       metrics=None):
    r"""Create a Keras Sequential model.

    Parameters
    ----------
    nlayers : int
        Number of layers of the Sequential model.
    layer1...layer10 : str
        Ordered layers of the Sequential model.
    optimizer : str
        Compiler optimizer for the Sequential model.
    loss : str
        Compiler loss function for the Sequential model.
    metrics : str
        Compiler evaluation metric for the Sequential model.

    Returns
    -------
    model : keras.models.Sequential
        Compiled Keras Sequential Model.

    """

    model = Sequential()
    for i in range(nlayers):
        lvar = 'layer' + str(i+1)
        layer = eval(lvar)
        model.add(eval(layer))
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    return model


#
# Function get_estimators
#

def get_estimators(model):
    r"""Define all the AlphaPy estimators based on the contents
    of the ``algos.yml`` file.

    Parameters
    ----------
    model : alphapy.Model
        The model object containing global AlphaPy parameters.

    Returns
    -------
    estimators : dict
        All of the estimators required for running the pipeline.

    """

    # Extract model data

    directory = model.specs['directory']
    n_estimators = model.specs['n_estimators']
    n_jobs = model.specs['n_jobs']
    seed = model.specs['seed']
    verbosity = model.specs['verbosity']

    # Reference training data for Keras input_dim
    X_train = model.X_train

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
        if 'KERAS' in algo:
            params['build_fn'] = create_keras_model
            layers = algo_specs[algo]['layers']
            params['nlayers'] = len(layers)
            input_dim_string = ', input_dim={})'.format(X_train.shape[1])
            layers[0] = layers[0].replace(')', input_dim_string)
            for i, layer in enumerate(layers):
                params['layer'+str(i+1)] = layer
            compiler = algo_specs[algo]['compiler']
            params['optimizer'] = compiler['optimizer']
            params['loss'] = compiler['loss']
            try:
                params['metrics'] = compiler['metrics']
            except:
                pass
        est = func(**params)
        grid = algo_specs[algo]['grid']
        estimators[algo] = Estimator(algo, model_type, est, grid)

    # return the entire classifier list
    return estimators
