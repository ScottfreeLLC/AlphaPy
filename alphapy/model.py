################################################################################
#
# Package   : AlphaPy
# Module    : model
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

from alphapy.estimators import scorers
from alphapy.estimators import xgb_score_map
from alphapy.features import feature_scorers
from alphapy.frame import read_frame
from alphapy.frame import write_frame
from alphapy.globals import Encoders
from alphapy.globals import ModelType
from alphapy.globals import Objective
from alphapy.globals import Partition, datasets
from alphapy.globals import PSEP, SSEP, USEP
from alphapy.globals import SamplingMethod
from alphapy.globals import Scalers
from alphapy.utilities import get_datestamp
from alphapy.utilities import most_recent_file

from copy import copy
from datetime import datetime
from keras.models import load_model
import logging
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import explained_variance_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import train_test_split
import sys
import yaml


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Class Model
#
# model unifies algorithms and we use hasattr to list the available attrs for each
# algorithm so users can query an algorithm and get the list of attributes
#

class Model:
    """Create a new model.

    Parameters
    ----------
    specs : dict
        The model specifications obtained by reading the ``model.yml``
        file.

    Attributes
    ----------
    specs : dict
        The model specifications.
    X_train : pandas.DataFrame
        Training features in matrix format.
    X_test  : pandas.Series
        Testing features in matrix format.
    y_train : pandas.DataFrame
        Training labels in vector format.
    y_test  : pandas.Series
        Testing labels in vector format.
    algolist : list
        Algorithms to use in training.
    estimators : dict
        Dictionary of estimators (key: algorithm)
    importances : dict
        Feature Importances (key: algorithm)
    coefs : dict
        Coefficients, if applicable (key: algorithm)
    support : dict
        Support Vectors, if applicable (key: algorithm)
    preds : dict
        Predictions or labels (keys: algorithm, partition)
    probas : dict
        Probabilities from classification (keys: algorithm, partition)
    metrics : dict
        Model evaluation metrics (keys: algorith, partition, metric)

    Raises
    ------
    KeyError
        Model specs must include the key *algorithms*, which is
        stored in ``algolist``.

    """
            
    # __init__
            
    def __init__(self,
                 specs):
        # specifications
        self.specs = specs
        # data in memory
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        # test labels
        self.test_labels = False
        # datasets
        self.train_file = datasets[Partition.train]
        self.test_file = datasets[Partition.test]
        self.predict_file = datasets[Partition.predict]
        # algorithms
        try:
            self.algolist = self.specs['algorithms']
        except:
            raise KeyError("Model specs must include the key: algorithms")
        self.best_algo = None
        # feature map
        self.feature_map = {}
        # Key: (algorithm)
        self.estimators = {}
        self.importances = {}
        self.coefs = {}
        self.support = {}
        # Keys: (algorithm, partition)
        self.preds = {}
        self.probas = {}
        # Keys: (algorithm, partition, metric)
        self.metrics = {}
                
    # __str__

    def __str__(self):
        return self.name

    # __getnewargs__

    def __getnewargs__(self):
        return (self.specs,)


#
# Function get_model_config
#

def get_model_config():
    r"""Read in the configuration file for AlphaPy.

    Parameters
    ----------
    None : None

    Returns
    -------
    specs : dict
        The parameters for controlling AlphaPy.

    Raises
    ------
    ValueError
        Unrecognized value of a ``model.yml`` field.

    """

    logger.info("Model Configuration")

    # Read the configuration file

    full_path = SSEP.join([PSEP, 'config', 'model.yml'])
    with open(full_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Store configuration parameters in dictionary

    specs = {}

    # Section: project [this section must be first]

    specs['directory'] = cfg['project']['directory']
    specs['extension'] = cfg['project']['file_extension']
    specs['submission_file'] = cfg['project']['submission_file']
    specs['submit_probas'] = cfg['project']['submit_probas']

    # Section: data

    specs['drop'] = cfg['data']['drop']
    specs['features'] = cfg['data']['features']
    specs['sentinel'] = cfg['data']['sentinel']
    specs['separator'] = cfg['data']['separator']
    specs['shuffle'] = cfg['data']['shuffle']
    specs['split'] = cfg['data']['split']
    specs['target'] = cfg['data']['target']
    specs['target_value'] = cfg['data']['target_value']
    # sampling
    specs['sampling'] = cfg['data']['sampling']['option']
    # determine whether or not sampling method is valid
    samplers = {x.name: x.value for x in SamplingMethod}
    sampling_method = cfg['data']['sampling']['method']
    if sampling_method in samplers:
        specs['sampling_method'] = SamplingMethod(samplers[sampling_method])
    else:
        raise ValueError("model.yml data:sampling:method %s unrecognized" %
                         sampling_method)
    # end of sampling method
    specs['sampling_ratio'] = cfg['data']['sampling']['ratio']

    # Section: features

    # clustering
    specs['clustering'] = cfg['features']['clustering']['option']
    specs['cluster_min'] = cfg['features']['clustering']['minimum']
    specs['cluster_max'] = cfg['features']['clustering']['maximum']
    specs['cluster_inc'] = cfg['features']['clustering']['increment']
    # counts
    specs['counts'] = cfg['features']['counts']['option']
    # encoding
    specs['rounding'] = cfg['features']['encoding']['rounding']
    # determine whether or not encoder is valid
    encoders = {x.name: x.value for x in Encoders}
    encoder = cfg['features']['encoding']['type']
    if encoder in encoders:
        specs['encoder'] = Encoders(encoders[encoder])
    else:
        raise ValueError("model.yml features:encoding:type %s unrecognized" % encoder)
    # factors
    specs['factors'] = cfg['features']['factors']
    # interactions
    specs['interactions'] = cfg['features']['interactions']['option']
    specs['isample_pct'] = cfg['features']['interactions']['sampling_pct']
    specs['poly_degree'] = cfg['features']['interactions']['poly_degree']
    # isomap
    specs['isomap'] = cfg['features']['isomap']['option']
    specs['iso_components'] = cfg['features']['isomap']['components']
    specs['iso_neighbors'] = cfg['features']['isomap']['neighbors']
    # log transformation
    specs['logtransform'] = cfg['features']['logtransform']['option']
    # low-variance features
    specs['lv_remove'] = cfg['features']['variance']['option']
    specs['lv_threshold'] = cfg['features']['variance']['threshold']
    # NumPy
    specs['numpy'] = cfg['features']['numpy']['option']
    # pca
    specs['pca'] = cfg['features']['pca']['option']
    specs['pca_min'] = cfg['features']['pca']['minimum']
    specs['pca_max'] = cfg['features']['pca']['maximum']
    specs['pca_inc'] = cfg['features']['pca']['increment']
    specs['pca_whiten'] = cfg['features']['pca']['whiten']
    # Scaling
    specs['scaler_option'] = cfg['features']['scaling']['option']
    # determine whether or not scaling type is valid
    scaler_types = {x.name: x.value for x in Scalers}
    scaler_type = cfg['features']['scaling']['type']
    if scaler_type in scaler_types:
        specs['scaler_type'] = Scalers(scaler_types[scaler_type])
    else:
        raise ValueError("model.yml features:scaling:type %s unrecognized" % scaler_type)
    # SciPy
    specs['scipy'] = cfg['features']['scipy']['option']
    # text
    specs['ngrams_max'] = cfg['features']['text']['ngrams']
    specs['vectorize'] = cfg['features']['text']['vectorize']
    # t-sne
    specs['tsne'] = cfg['features']['tsne']['option']
    specs['tsne_components'] = cfg['features']['tsne']['components']
    specs['tsne_learn_rate'] = cfg['features']['tsne']['learning_rate']
    specs['tsne_perplexity'] = cfg['features']['tsne']['perplexity']

    # Section: model

    specs['algorithms'] = cfg['model']['algorithms']
    specs['cv_folds'] = cfg['model']['cv_folds']
    # determine whether or not model type is valid
    model_types = {x.name: x.value for x in ModelType}
    model_type = cfg['model']['type']
    if model_type in model_types:
        specs['model_type'] = ModelType(model_types[model_type])
    else:
        raise ValueError("model.yml model:type %s unrecognized" % model_type)
    # end of model type
    specs['n_estimators'] = cfg['model']['estimators']
    specs['pvalue_level'] = cfg['model']['pvalue_level']
    specs['scorer'] = cfg['model']['scoring_function']
    # calibration
    specs['calibration'] = cfg['model']['calibration']['option']
    specs['cal_type'] = cfg['model']['calibration']['type']
    # feature selection
    specs['feature_selection'] = cfg['model']['feature_selection']['option']
    specs['fs_percentage'] = cfg['model']['feature_selection']['percentage']
    specs['fs_uni_grid'] = cfg['model']['feature_selection']['uni_grid']
    score_func = cfg['model']['feature_selection']['score_func']
    if score_func in feature_scorers:
        specs['fs_score_func'] = feature_scorers[score_func]
    else:
        raise ValueError("model.yml model:feature_selection:score_func %s unrecognized" %
                         score_func)
    # grid search
    specs['grid_search'] = cfg['model']['grid_search']['option']
    specs['gs_iters'] = cfg['model']['grid_search']['iterations']
    specs['gs_random'] = cfg['model']['grid_search']['random']
    specs['gs_sample'] = cfg['model']['grid_search']['subsample']
    specs['gs_sample_pct'] = cfg['model']['grid_search']['sampling_pct']
    # rfe
    specs['rfe'] = cfg['model']['rfe']['option']
    specs['rfe_step'] = cfg['model']['rfe']['step']

    # Section: pipeline

    specs['n_jobs'] = cfg['pipeline']['number_jobs']
    specs['seed'] = cfg['pipeline']['seed']
    specs['verbosity'] = cfg['pipeline']['verbosity']

    # Section: plots

    specs['calibration_plot'] = cfg['plots']['calibration']
    specs['confusion_matrix'] = cfg['plots']['confusion_matrix']
    specs['importances'] = cfg['plots']['importances']
    specs['learning_curve'] = cfg['plots']['learning_curve']
    specs['roc_curve'] = cfg['plots']['roc_curve']

    # Section: treatments

    try:
        specs['treatments'] = cfg['treatments']
    except:
        specs['treatments'] = None
        logger.info("No Treatments Found")

    # Section: xgboost

    specs['esr'] = cfg['xgboost']['stopping_rounds']

    # Log the configuration parameters

    logger.info('MODEL PARAMETERS:')
    logger.info('algorithms        = %s', specs['algorithms'])
    logger.info('calibration       = %r', specs['calibration'])
    logger.info('cal_type          = %s', specs['cal_type'])
    logger.info('calibration_plot  = %r', specs['calibration'])
    logger.info('clustering        = %r', specs['clustering'])
    logger.info('cluster_inc       = %d', specs['cluster_inc'])
    logger.info('cluster_max       = %d', specs['cluster_max'])
    logger.info('cluster_min       = %d', specs['cluster_min'])
    logger.info('confusion_matrix  = %r', specs['confusion_matrix'])
    logger.info('counts            = %r', specs['counts'])
    logger.info('cv_folds          = %d', specs['cv_folds'])
    logger.info('directory         = %s', specs['directory'])
    logger.info('extension         = %s', specs['extension'])
    logger.info('drop              = %s', specs['drop'])
    logger.info('encoder           = %r', specs['encoder'])
    logger.info('esr               = %d', specs['esr'])
    logger.info('factors           = %s', specs['factors'])
    logger.info('features [X]      = %s', specs['features'])
    logger.info('feature_selection = %r', specs['feature_selection'])
    logger.info('fs_percentage     = %d', specs['fs_percentage'])
    logger.info('fs_score_func     = %s', specs['fs_score_func'])
    logger.info('fs_uni_grid       = %s', specs['fs_uni_grid'])
    logger.info('grid_search       = %r', specs['grid_search'])
    logger.info('gs_iters          = %d', specs['gs_iters'])
    logger.info('gs_random         = %r', specs['gs_random'])
    logger.info('gs_sample         = %r', specs['gs_sample'])
    logger.info('gs_sample_pct     = %f', specs['gs_sample_pct'])
    logger.info('importances       = %r', specs['importances'])
    logger.info('interactions      = %r', specs['interactions'])
    logger.info('isomap            = %r', specs['isomap'])
    logger.info('iso_components    = %d', specs['iso_components'])
    logger.info('iso_neighbors     = %d', specs['iso_neighbors'])
    logger.info('isample_pct       = %d', specs['isample_pct'])
    logger.info('learning_curve    = %r', specs['learning_curve'])
    logger.info('logtransform      = %r', specs['logtransform'])
    logger.info('lv_remove         = %r', specs['lv_remove'])
    logger.info('lv_threshold      = %f', specs['lv_threshold'])
    logger.info('model_type        = %r', specs['model_type'])
    logger.info('n_estimators      = %d', specs['n_estimators'])
    logger.info('n_jobs            = %d', specs['n_jobs'])
    logger.info('ngrams_max        = %d', specs['ngrams_max'])
    logger.info('numpy             = %r', specs['numpy'])
    logger.info('pca               = %r', specs['pca'])
    logger.info('pca_inc           = %d', specs['pca_inc'])
    logger.info('pca_max           = %d', specs['pca_max'])
    logger.info('pca_min           = %d', specs['pca_min'])
    logger.info('pca_whiten        = %r', specs['pca_whiten'])
    logger.info('poly_degree       = %d', specs['poly_degree'])
    logger.info('pvalue_level      = %f', specs['pvalue_level'])
    logger.info('rfe               = %r', specs['rfe'])
    logger.info('rfe_step          = %d', specs['rfe_step'])
    logger.info('roc_curve         = %r', specs['roc_curve'])
    logger.info('rounding          = %d', specs['rounding'])
    logger.info('sampling          = %r', specs['sampling'])
    logger.info('sampling_method   = %r', specs['sampling_method'])
    logger.info('sampling_ratio    = %f', specs['sampling_ratio'])
    logger.info('scaler_option     = %r', specs['scaler_option'])
    logger.info('scaler_type       = %r', specs['scaler_type'])
    logger.info('scipy             = %r', specs['scipy'])
    logger.info('scorer            = %s', specs['scorer'])
    logger.info('seed              = %d', specs['seed'])
    logger.info('sentinel          = %d', specs['sentinel'])
    logger.info('separator         = %s', specs['separator'])
    logger.info('shuffle           = %r', specs['shuffle'])
    logger.info('split             = %f', specs['split'])
    logger.info('submission_file   = %s', specs['submission_file'])
    logger.info('submit_probas     = %r', specs['submit_probas'])
    logger.info('target [y]        = %s', specs['target'])
    logger.info('target_value      = %d', specs['target_value'])
    logger.info('treatments        = %s', specs['treatments'])
    logger.info('tsne              = %r', specs['tsne'])
    logger.info('tsne_components   = %d', specs['tsne_components'])
    logger.info('tsne_learn_rate   = %f', specs['tsne_learn_rate'])
    logger.info('tsne_perplexity   = %f', specs['tsne_perplexity'])
    logger.info('vectorize         = %r', specs['vectorize'])
    logger.info('verbosity         = %d', specs['verbosity'])

    # Specifications to create the model
    return specs


#
# Function load_predictor
#

def load_predictor(directory):
    r"""Load the model predictor from storage. By default, the
    most recent model is loaded into memory.

    Parameters
    ----------
    directory : str
        Full directory specification of the predictor's location.

    Returns
    -------
    predictor : function
        The scoring function.

    """

    # Locate the model Pickle or HD5 file

    search_dir = SSEP.join([directory, 'model'])
    file_name = most_recent_file(search_dir, 'model_*.*')

    # Load the model from the file

    file_ext = file_name.split(PSEP)[-1]
    if file_ext == 'pkl' or file_ext == 'h5':
        logger.info("Loading model predictor from %s", file_name)
        # load the model predictor
        if file_ext == 'pkl':
            predictor = joblib.load(file_name)
        elif file_ext == 'h5':
            predictor = load_model(file_name)
    else:
        logging.error("Could not find model predictor in %s", search_path)

    # Return the model predictor
    return predictor


#
# Function save_predictor
#

def save_predictor(model, timestamp):
    r"""Save the time-stamped model predictor to disk.

    Parameters
    ----------
    model : alphapy.Model
        The model object that contains the best estimator.
    timestamp : str
        Date in yyyy-mm-dd format.

    Returns
    -------
    None : None

    """

    logger.info("Saving Model Predictor")

    # Extract model parameters.
    directory = model.specs['directory']

    # Get the best predictor
    predictor = model.estimators['BEST']

    # Save model object

    if 'KERAS' in model.best_algo:
        filename = 'model_' + timestamp + '.h5'
        full_path = SSEP.join([directory, 'model', filename])
        logger.info("Writing model predictor to %s", full_path)
        predictor.model.save(full_path)
    else:
        filename = 'model_' + timestamp + '.pkl'
        full_path = SSEP.join([directory, 'model', filename])
        logger.info("Writing model predictor to %s", full_path)
        joblib.dump(predictor, full_path)


#
# Function load_feature_map
#

def load_feature_map(model, directory):
    r"""Load the feature map from storage. By default, the
    most recent feature map is loaded into memory.

    Parameters
    ----------
    model : alphapy.Model
        The model object to contain the feature map.
    directory : str
        Full directory specification of the feature map's location.

    Returns
    -------
    model : alphapy.Model
        The model object containing the feature map.

    """

    # Locate the feature map and load it

    try:
        search_dir = SSEP.join([directory, 'model'])
        file_name = most_recent_file(search_dir, 'feature_map_*.pkl')
        logger.info("Loading feature map from %s", file_name)
        # load the feature map
        feature_map = joblib.load(file_name)
        model.feature_map = feature_map
    except:
        logging.error("Could not find feature map in %s", search_path)

    # Return the model with the feature map
    return model


#
# Function save_feature_map
#

def save_feature_map(model, timestamp):
    r"""Save the feature map to disk.

    Parameters
    ----------
    model : alphapy.Model
        The model object containing the feature map.
    timestamp : str
        Date in yyyy-mm-dd format.

    Returns
    -------
    None : None

    """

    logger.info("Saving Feature Map")

    # Extract model parameters.
    directory = model.specs['directory']

    # Create full path name.

    filename = 'feature_map_' + timestamp + '.pkl'
    full_path = SSEP.join([directory, 'model', filename])

    # Save model object

    logger.info("Writing feature map to %s", full_path)
    joblib.dump(model.feature_map, full_path)


#
# Function first_fit
#

def first_fit(model, algo, est):
    r"""Fit the model before optimization.

    Parameters
    ----------
    model : alphapy.Model
        The model object with specifications.
    algo : str
        Abbreviation of the algorithm to run.
    est : alphapy.Estimator
        The estimator to fit.

    Returns
    -------
    model : alphapy.Model
        The model object with the initial estimator.

    Notes
    -----
    AlphaPy fits an initial model because the user may choose to get
    a first score without any additional feature selection or grid
    search. XGBoost is a special case because it has the advantage
    of an ``eval_set`` and ``early_stopping_rounds``, which can
    speed up the estimation phase.

    """

    logger.info("Fitting Initial Model")

    # Extract model parameters.

    esr = model.specs['esr']
    model_type = model.specs['model_type']
    scorer = model.specs['scorer']
    seed = model.specs['seed']
    split = model.specs['split']

    # Extract model data.

    X_train = model.X_train
    y_train = model.y_train

    # Fit the initial model.

    algo_keras = 'KERAS' in algo
    algo_xgb = 'XGB' in algo

    if algo_xgb and scorer in xgb_score_map:
        X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=split,
                                          random_state=seed)
        eval_set = [(X1, y1), (X2, y2)]
        eval_metric = xgb_score_map[scorer]
        est.fit(X1, y1, eval_set=eval_set, eval_metric=eval_metric,
                early_stopping_rounds=esr)
    else:
        est.fit(X_train, y_train)

    # Store the estimator

    model.estimators[algo] = est

    # Record importances and coefficients if necessary.

    if hasattr(est, "feature_importances_"):
        model.importances[algo] = est.feature_importances_

    if hasattr(est, "coef_"):
        model.coefs[algo] = est.coef_

    # Save the estimator in the model and return the model
    return model


#
# Function make_predictions
#

def make_predictions(model, algo, calibrate):
    r"""Make predictions for the training and testing data.

    Parameters
    ----------
    model : alphapy.Model
        The model object with specifications.
    algo : str
        Abbreviation of the algorithm to make predictions.
    calibrate : bool
        If ``True``, calibrate the probabilities for a classifier.

    Returns
    -------
    model : alphapy.Model
        The model object with the predictions.

    Notes
    -----
    For classification, calibration is a precursor to making the
    actual predictions. In this case, AlphaPy predicts both labels
    and probabilities. For regression, real values are predicted.

    """

    logger.info("Final Model Predictions for %s", algo)

    # Extract model parameters.

    cal_type = model.specs['cal_type']
    cv_folds = model.specs['cv_folds']
    model_type = model.specs['model_type']

    # Get the estimator

    est = model.estimators[algo]

    # Extract model data.

    try:
        support = model.support[algo]
        X_train = model.X_train[:, support]
        X_test = model.X_test[:, support]
    except:
        X_train = model.X_train
        X_test = model.X_test
    y_train = model.y_train

    # Calibration

    if model_type == ModelType.classification:
        if calibrate:
            logger.info("Calibrating Classifier")
            est = CalibratedClassifierCV(est, cv=cv_folds, method=cal_type)
            est.fit(X_train, y_train)
            model.estimators[algo] = est
            logger.info("Calibration Complete")
        else:
            logger.info("Skipping Calibration")

    # Make predictions on original training and test data.

    logger.info("Making Predictions")
    model.preds[(algo, Partition.train)] = est.predict(X_train)
    model.preds[(algo, Partition.test)] = est.predict(X_test)
    if model_type == ModelType.classification:
        model.probas[(algo, Partition.train)] = est.predict_proba(X_train)[:, 1]
        model.probas[(algo, Partition.test)] = est.predict_proba(X_test)[:, 1]
    logger.info("Predictions Complete")

    # Return the model
    return model


#
# Function predict_best
#

def predict_best(model):
    r"""Select the best model based on score.

    Parameters
    ----------
    model : alphapy.Model
        The model object with all of the estimators.

    Returns
    -------
    model : alphapy.Model
        The model object with the best estimator.

    Notes
    -----
    Best model selection is based on a scoring function. If the
    objective is to minimize (e.g., negative log loss), then we
    select the model with the algorithm that has the lowest score.
    If the objective is to maximize, then we select the algorithm
    with the highest score (e.g., AUC).

    For multiple algorithms, AlphaPy always creates a blended model.
    Therefore, the best algorithm that is selected could actually
    be the blended model itself.

    """

    logger.info('='*80)
    logger.info("Selecting Best Model")

    # Define model tags

    best_tag = 'BEST'
    blend_tag = 'BLEND'

    # Extract model parameters.

    model_type = model.specs['model_type']
    rfe = model.specs['rfe']
    scorer = model.specs['scorer']
    test_labels = model.test_labels

    # Determine the correct partition to select the best model

    partition = Partition.test if test_labels else Partition.train
    logger.info("Scoring for: %s", partition)

    # Initialize best parameters.

    maximize = True if scorers[scorer][1] == Objective.maximize else False
    if maximize:
        best_score = -sys.float_info.max
    else:
        best_score = sys.float_info.max

    # Initialize the model selection process.

    start_time = datetime.now()
    logger.info("Best Model Selection Start: %s", start_time)

    # Add blended model to the list of algorithms.

    if len(model.algolist) > 1:
        algolist = copy(model.algolist)
        algolist.append(blend_tag)
    else:
        algolist = model.algolist

    # Iterate through the models, getting the best score for each one.

    for algorithm in algolist:
        logger.info("Scoring %s Model", algorithm)
        top_score = model.metrics[(algorithm, partition, scorer)]
        # objective is to either maximize or minimize score
        if maximize:
            if top_score > best_score:
                best_score = top_score
                best_algo = algorithm
        else:
            if top_score < best_score:
                best_score = top_score
                best_algo = algorithm

    # Record predictions of best estimator

    logger.info("Best Model is %s with a %s score of %.4f", best_algo, scorer, best_score)
    model.best_algo = best_algo
    model.estimators[best_tag] = model.estimators[best_algo]
    model.preds[(best_tag, Partition.train)] = model.preds[(best_algo, Partition.train)]
    model.preds[(best_tag, Partition.test)] = model.preds[(best_algo, Partition.test)]
    if model_type == ModelType.classification:
        model.probas[(best_tag, Partition.train)] = model.probas[(best_algo, Partition.train)]
        model.probas[(best_tag, Partition.test)] = model.probas[(best_algo, Partition.test)]

    # Record support vector for any recursive feature elimination

    if rfe and 'XGB' not in best_algo:
        try:
            model.feature_map['rfe_support'] = model.support[best_algo]
        except:
            # no RFE support for best algorithm
            pass

    # Return the model with best estimator and predictions.

    end_time = datetime.now()
    time_taken = end_time - start_time
    logger.info("Best Model Selection Complete: %s", time_taken)

    return model


#
# Function predict_blend
#

def predict_blend(model):
    r"""Make predictions from a blended model.

    Parameters
    ----------
    model : alphapy.Model
        The model object with all of the estimators.

    Returns
    -------
    model : alphapy.Model
        The model object with the blended estimator.

    Notes
    -----
    For classification, AlphaPy uses logistic regression for creating
    a blended model. For regression, ridge regression is applied.

    """

    logger.info("Blending Models")

    # Extract model paramters.

    model_type = model.specs['model_type']
    cv_folds = model.specs['cv_folds']

    # Extract model data.

    X_train = model.X_train
    X_test = model.X_test
    y_train = model.y_train

    # Add blended algorithm.

    blend_tag = 'BLEND'

    # Create blended training and test sets.

    n_models = len(model.algolist)
    X_blend_train = np.zeros((X_train.shape[0], n_models))
    X_blend_test = np.zeros((X_test.shape[0], n_models))

    # Iterate through the models, cross-validating for each one.

    start_time = datetime.now()
    logger.info("Blending Start: %s", start_time)

    for i, algorithm in enumerate(model.algolist):
        # get the best estimator
        estimator = model.estimators[algorithm]
        # update coefficients and feature importances
        if hasattr(estimator, "coef_"):
            model.coefs[algorithm] = estimator.coef_
        if hasattr(estimator, "feature_importances_"):
            model.importances[algorithm] = estimator.feature_importances_
        # store predictions in the blended training set
        if model_type == ModelType.classification:
            X_blend_train[:, i] = model.probas[(algorithm, Partition.train)]
            X_blend_test[:, i] = model.probas[(algorithm, Partition.test)]
        else:
            X_blend_train[:, i] = model.preds[(algorithm, Partition.train)]
            X_blend_test[:, i] = model.preds[(algorithm, Partition.test)]

    # Use the blended estimator to make predictions

    if model_type == ModelType.classification:
        clf = LogisticRegression()
        clf.fit(X_blend_train, y_train)
        model.estimators[blend_tag] = clf
        model.preds[(blend_tag, Partition.train)] = clf.predict(X_blend_train)
        model.preds[(blend_tag, Partition.test)] = clf.predict(X_blend_test)
        model.probas[(blend_tag, Partition.train)] = clf.predict_proba(X_blend_train)[:, 1]
        model.probas[(blend_tag, Partition.test)] = clf.predict_proba(X_blend_test)[:, 1]
    else:
        alphas = [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5,
                  1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]    
        rcvr = RidgeCV(alphas=alphas, normalize=True, cv=cv_folds)
        rcvr.fit(X_blend_train, y_train)
        model.estimators[blend_tag] = rcvr
        model.preds[(blend_tag, Partition.train)] = rcvr.predict(X_blend_train)
        model.preds[(blend_tag, Partition.test)] = rcvr.predict(X_blend_test)

    # Return the model with blended estimator and predictions.

    end_time = datetime.now()
    time_taken = end_time - start_time
    logger.info("Blending Complete: %s", time_taken)

    return model


#
# Function generate_metrics
#

def generate_metrics(model, partition):
    r"""Generate model evaluation metrics for all estimators.

    Parameters
    ----------
    model : alphapy.Model
        The model object with stored predictions.
    partition : alphapy.Partition
        Reference to the dataset.

    Returns
    -------
    model : alphapy.Model
        The model object with the completed metrics.

    Notes
    -----
    AlphaPy takes a brute-force approach to calculating each metric.
    It calls every scikit-learn function without exception. If the
    calculation fails for any reason, then the evaluation will still
    continue without error.

    References
    ----------
    For more information about model evaluation and the associated metrics,
    refer to [EVAL]_.

    .. [EVAL] http://scikit-learn.org/stable/modules/model_evaluation.html

    """

    logger.info('='*80)
    logger.info("Metrics for: %s", partition)

    # Extract model paramters.

    model_type = model.specs['model_type']

    # Extract model data.

    if partition == Partition.train:
        expected = model.y_train
    else:
        expected = model.y_test

    # Generate Metrics

    if expected.any():
        # Add blended model to the list of algorithms.
        if len(model.algolist) > 1:
            algolist = copy(model.algolist)
            algolist.append('BLEND')
        else:
            algolist = model.algolist

        # get the metrics for each algorithm
        for algo in algolist:
            # get predictions for the given algorithm
            predicted = model.preds[(algo, partition)]
            # classification metrics
            if model_type == ModelType.classification:
                probas = model.probas[(algo, partition)]
                try:
                    model.metrics[(algo, partition, 'accuracy')] = accuracy_score(expected, predicted)
                except:
                    logger.info("Accuracy Score not calculated")
                try:
                    model.metrics[(algo, partition, 'average_precision')] = average_precision_score(expected, probas)
                except:
                    logger.info("Average Precision Score not calculated")
                try:
                    model.metrics[(algo, partition, 'brier_score')] = brier_score_loss(expected, probas)
                except:
                    logger.info("Brier Score not calculated")
                try:
                    model.metrics[(algo, partition, 'cohen_kappa')] = cohen_kappa_score(expected, predicted)
                except:
                    logger.info("Cohen's Kappa Score not calculated")
                try:
                    model.metrics[(algo, partition, 'confusion_matrix')] = confusion_matrix(expected, predicted)
                except:
                    logger.info("Confusion Matrix not calculated")
                try:
                    model.metrics[(algo, partition, 'f1')] = f1_score(expected, predicted)
                except:
                    logger.info("F1 Score not calculated")
                try:
                    model.metrics[(algo, partition, 'neg_log_loss')] = log_loss(expected, probas)
                except:
                    logger.info("Log Loss not calculated")
                try:
                    model.metrics[(algo, partition, 'precision')] = precision_score(expected, predicted)
                except:
                    logger.info("Precision Score not calculated")
                try:
                    model.metrics[(algo, partition, 'recall')] = recall_score(expected, predicted)
                except:
                    logger.info("Recall Score not calculated")
                try:
                    fpr, tpr, _ = roc_curve(expected, probas)
                    model.metrics[(algo, partition, 'roc_auc')] = auc(fpr, tpr)
                except:
                    logger.info("ROC AUC Score not calculated")
            # regression metrics
            elif model_type == ModelType.regression:
                try:
                    model.metrics[(algo, partition, 'explained_variance')] = explained_variance_score(expected, predicted)
                except:
                    logger.info("Explained Variance Score not calculated")
                try:
                    model.metrics[(algo, partition, 'mean_absolute_error')] = mean_absolute_error(expected, predicted)
                except:
                    logger.info("Mean Absolute Error not calculated")
                try:
                    model.metrics[(algo, partition, 'median_absolute_error')] = median_absolute_error(expected, predicted)
                except:
                    logger.info("Median Absolute Error not calculated")
                try:
                    model.metrics[(algo, partition, 'neg_mean_squared_error')] = mean_squared_error(expected, predicted)
                except:
                    logger.info("Mean Squared Error not calculated")
                try:
                    model.metrics[(algo, partition, 'r2')] = r2_score(expected, predicted)
                except:
                    logger.info("R-Squared Score not calculated")
        # log the metrics for each algorithm
        for algo in model.algolist:
            logger.info('-'*80)
            logger.info("Algorithm: %s", algo)
            metrics = [(k[2], v) for k, v in list(model.metrics.items()) if k[0] == algo and k[1] == partition]
            for key, value in sorted(metrics):
                svalue = str(value)
                svalue.replace('\n', ' ')
                logger.info("%s: %s", key, svalue)
    else:
        logger.info("No labels for generating %s metrics", partition)

    return model


#
# Function save_predictions
#

def save_predictions(model, tag, partition):
    r"""Save the predictions to disk.

    Parameters
    ----------
    model : alphapy.Model
        The model object to save.
    tag : str
        A unique identifier for the output files, e.g., a date stamp.
    partition : alphapy.Partition
        Reference to the dataset.

    Returns
    -------
    preds : numpy array
        The prediction vector.
    probas : numpy array
        The probability vector.

    """

    # Extract model parameters.

    directory = model.specs['directory']
    extension = model.specs['extension']
    model_type = model.specs['model_type']
    separator = model.specs['separator']

    # Get date stamp to record file creation
    timestamp = get_datestamp()

    # Specify input and output directories

    input_dir = SSEP.join([directory, 'input'])
    output_dir = SSEP.join([directory, 'output'])

    # Read the prediction frame
    pf = read_frame(input_dir, datasets[partition], extension, separator)

    # Cull records before the prediction date

    try:
        predict_date = model.specs['predict_date']
        found_pdate = True
    except:
        found_pdate = False

    if found_pdate:
        pd_indices = pf[pf.date >= predict_date].index.tolist()
        pf = pf.ix[pd_indices]
    else:
        pd_indices = pf.index.tolist()

    # Save predictions for all projects

    logger.info("Saving Predictions")
    output_file = USEP.join(['predictions', timestamp])
    preds = model.preds[(tag, partition)].squeeze()
    if found_pdate:
        preds = np.take(preds, pd_indices)
    pred_series = pd.Series(preds, index=pd_indices)
    df_pred = pd.DataFrame(pred_series, columns=['prediction'])
    write_frame(df_pred, output_dir, output_file, extension, separator)

    # Save probabilities for classification projects

    probas = None
    if model_type == ModelType.classification:
        logger.info("Saving Probabilities")
        output_file = USEP.join(['probabilities', timestamp])
        probas = model.probas[(tag, partition)].squeeze()
        if found_pdate:
            probas = np.take(probas, pd_indices)
        prob_series = pd.Series(probas, index=pd_indices)
        df_prob = pd.DataFrame(prob_series, columns=['probability'])
        write_frame(df_prob, output_dir, output_file, extension, separator)

    # Save ranked predictions

    logger.info("Saving Ranked Predictions")
    pf['prediction'] = pred_series
    if model_type == ModelType.classification:
        pf['probability'] = prob_series
        pf.sort_values('probability', ascending=False, inplace=True)
    else:
        pf.sort_values('prediction', ascending=False, inplace=True)
    output_file = USEP.join(['rankings', timestamp])
    write_frame(pf, output_dir, output_file, extension, separator)

    # Return predictions and any probabilities
    return preds, probas


#
# Function save_model
#

def save_model(model, tag, partition):
    r"""Save the results in the model file.

    Parameters
    ----------
    model : alphapy.Model
        The model object to save.
    tag : str
        A unique identifier for the output files, e.g., a date stamp.
    partition : alphapy.Partition
        Reference to the dataset.

    Returns
    -------
    None : None

    Notes
    -----

    The following components are extracted from the model object
    and saved to disk:

    * Model predictor (via joblib/pickle)
    * Predictions
    * Probabilities (classification only)
    * Rankings
    * Submission File (optional)

    """

    logger.info('='*80)

    # Extract model parameters.

    directory = model.specs['directory']
    extension = model.specs['extension']
    model_type = model.specs['model_type']
    submission_file = model.specs['submission_file']
    submit_probas = model.specs['submit_probas']

    # Get date stamp to record file creation

    d = datetime.now()
    f = "%Y%m%d"
    timestamp = d.strftime(f)

    # Save the model predictor
    save_predictor(model, timestamp)

    # Save the feature map
    save_feature_map(model, timestamp)

    # Specify input and output directories

    input_dir = SSEP.join([directory, 'input'])
    output_dir = SSEP.join([directory, 'output'])

    # Save predictions
    preds, probas = save_predictions(model, tag, partition)

    # Generate submission file

    if submission_file:
        sample_spec = PSEP.join([submission_file, extension])
        sample_input = SSEP.join([input_dir, sample_spec])
        ss = pd.read_csv(sample_input)
        if submit_probas and model_type == ModelType.classification:
            ss[ss.columns[1]] = probas
        else:
            ss[ss.columns[1]] = preds
        submission_base = USEP.join(['submission', timestamp])
        submission_spec = PSEP.join([submission_base, extension])
        submission_output = SSEP.join([output_dir, submission_spec])
        logger.info("Saving Submission to %s", submission_output)
        ss.to_csv(submission_output, index=False)
