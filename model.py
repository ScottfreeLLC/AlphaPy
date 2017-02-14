################################################################################
#
# Package   : AlphaPy
# Module    : model
# Version   : 1.0
# Date      : July 11, 2013
#
# Copyright 2017 @ Alpha314
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

from data import SamplingMethod
from datetime import datetime
from estimators import Objective
from estimators import ModelType
from estimators import scorers
from estimators import xgb_score_map
from features import Encoders
from features import feature_scorers
from frame import write_frame
import glob
from globs import PSEP
from globs import SSEP
from globs import USEP
import logging
import numpy as np
import os
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
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
            
    # __init__
            
    def __init__(self,
                 specs):
        self.specs = specs
        # initialize model
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        try:
            self.algolist = self.specs['algorithms']
        except:
            raise KeyError("Model specs must include the key: algorithms")
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

def get_model_config(cfg_dir):

    logger.info("Model Configuration")

    # Read the configuration file

    full_path = SSEP.join([cfg_dir, 'model.yml'])
    with open(full_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Store configuration parameters in dictionary

    specs = {}

    # Section: project [this section must be first]

    specs['directory'] = cfg['project']['directory']
    specs['extension'] = cfg['project']['file_extension']
    specs['sample_submission'] = cfg['project']['sample_submission']
    specs['scoring_mode'] = cfg['project']['scoring_mode']
    specs['submission_file'] = cfg['project']['submission_file']

    # Section: data

    specs['drop'] = cfg['data']['drop']
    specs['dummy_limit'] = cfg['data']['dummy_limit']
    specs['features'] = cfg['data']['features']
    specs['sentinel'] = cfg['data']['sentinel']
    specs['separator'] = cfg['data']['separator']
    specs['shuffle'] = cfg['data']['shuffle']
    specs['split'] = cfg['data']['split']
    specs['target'] = cfg['data']['target']
    specs['target_value'] = cfg['data']['target_value']
    specs['test_file'] = cfg['data']['test']
    specs['test_labels'] = cfg['data']['test_labels']
    specs['train_file'] = cfg['data']['train']
    # sampling
    specs['sampling'] = cfg['data']['sampling']['option']
    # determine whether or not sampling method is valid
    samplers = {x.name: x.value for x in SamplingMethod}
    sampling_method = cfg['data']['sampling']['method']
    if sampling_method in samplers:
        specs['sampling_method'] = SamplingMethod(samplers[sampling_method])
    else:
        raise ValueError(".yml data:sampling:method %s unrecognized",
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
        raise ValueError(".yml features:encoding:type %s unrecognized", encoder)
    # genetic
    specs['genetic'] = cfg['features']['genetic']['option']
    specs['gfeatures'] = cfg['features']['genetic']['features']
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
    # NumPy
    specs['numpy'] = cfg['features']['numpy']['option']
    # pca
    specs['pca'] = cfg['features']['pca']['option']
    specs['pca_min'] = cfg['features']['pca']['minimum']
    specs['pca_max'] = cfg['features']['pca']['maximum']
    specs['pca_inc'] = cfg['features']['pca']['increment']
    specs['pca_whiten'] = cfg['features']['pca']['whiten']
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
    specs['balance_classes'] = cfg['model']['balance_classes']
    specs['cv_folds'] = cfg['model']['cv_folds']
    # determine whether or not model type is valid
    model_types = {x.name: x.value for x in ModelType}
    model_type = cfg['model']['type']
    if model_type in model_types:
        specs['model_type'] = ModelType(model_types[model_type])
    else:
        raise ValueError(".yml model:type %s unrecognized", model_type)
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
        raise ValueError('.yml model:feature_selection:score_func %s unrecognized',
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
    logger.info('balance_classes   = %s', specs['balance_classes'])
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
    logger.info('dummy_limit       = %d', specs['dummy_limit'])
    logger.info('encoder           = %r', specs['encoder'])
    logger.info('esr               = %d', specs['esr'])
    logger.info('features [X]      = %s', specs['features'])
    logger.info('feature_selection = %r', specs['feature_selection'])
    logger.info('fs_percentage     = %d', specs['fs_percentage'])
    logger.info('fs_score_func     = %s', specs['fs_score_func'])
    logger.info('fs_uni_grid       = %s', specs['fs_uni_grid'])
    logger.info('genetic           = %r', specs['genetic'])
    logger.info('gfeatures         = %d', specs['gfeatures'])
    logger.info('grid_search       = %r', specs['grid_search'])
    logger.info('gs_iters          = %d', specs['gs_iters'])
    logger.info('gs_random         = %r', specs['gs_random'])
    logger.info('gs_sample         = %r', specs['gs_sample'])
    logger.info('gs_sample_pct     = %f', specs['gs_sample_pct'])
    logger.info('importances       = %r', specs['importances'])
    logger.info('interactions      = %r', specs['interactions'])
    logger.info('isomap            = %r', specs['interactions'])
    logger.info('iso_components    = %d', specs['iso_components'])
    logger.info('iso_neighbors     = %d', specs['iso_neighbors'])
    logger.info('isample_pct       = %d', specs['isample_pct'])
    logger.info('learning_curve    = %r', specs['learning_curve'])
    logger.info('logtransform      = %r', specs['logtransform'])
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
    logger.info('sample_submission = %r', specs['sample_submission'])
    logger.info('sampling          = %r', specs['sampling'])
    logger.info('sampling_method   = %r', specs['sampling_method'])
    logger.info('sampling_ratio    = %f', specs['sampling_ratio'])
    logger.info('scipy             = %r', specs['scipy'])
    logger.info('scorer            = %s', specs['scorer'])
    logger.info('scoring_mode      = %r', specs['scoring_mode'])
    logger.info('seed              = %d', specs['seed'])
    logger.info('sentinel          = %d', specs['sentinel'])
    logger.info('separator         = %s', specs['separator'])
    logger.info('shuffle           = %r', specs['shuffle'])
    logger.info('split             = %f', specs['split'])
    logger.info('submission_file   = %s', specs['submission_file'])
    logger.info('target [y]        = %s', specs['target'])
    logger.info('target_value      = %d', specs['target_value'])
    logger.info('test_file         = %s', specs['test_file'])
    logger.info('test_labels       = %r', specs['test_labels'])
    logger.info('train_file        = %s', specs['train_file'])
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
# Function load_model_object
#

def load_model_object(directory):
    """
    Load the model from storage.
    """

    logger.info("Loading Model")

    # Create search path

    search_path = SSEP.join([directory, 'model', 'model*.pkl'])

    # Locate the Pickle model file

    try:
        filename = max(glob.iglob(search_path), key=os.path.getctime)
    except:
        logging.error("Could not find model %s", search_path)

    # Load the model predictor

    predictor = joblib.load(filename)
    return predictor


#
# Function save_model_object
#

def save_model_object(model, timestamp):
    """
    Save the model to storage.
    """

    logger.info("Saving Model Object")

    # Extract model parameters.

    directory = model.specs['directory']

    # Get the best predictor

    predictor = model.estimators['BEST']

    # Create full path name.

    filename = 'model_' + timestamp + '.pkl'
    full_path = SSEP.join([directory, 'model', filename])

    # Save model object

    joblib.dump(predictor, full_path)


#
# Function get_sample_weights
#

def get_sample_weights(model):
    """
    Set sample weights for fitting the model
    """

    # Extract model parameters.

    balance_classes = model.specs['balance_classes']
    target = model.specs['target']
    target_value = model.specs['target_value']

    # Extract model data.

    y_train = model.y_train

    # Calculate sample weights

    sw = None
    if balance_classes:
        logger.info("Getting Sample Weights")
        uv, uc = np.unique(y_train, return_counts=True)
        weight = uc[not target_value] / uc[target_value]
        logger.info("Sample Weight for target %s [%r]: %f",
                    target, target_value, weight)
        sw = [weight if x==target_value else 1.0 for x in y_train]
    else:
        logger.info("Skipping Sample Weights")  

    # Set weights

    model.specs['sample_weights'] = sw
    return model


#
# Function first_fit
#

def first_fit(model, algo, est):
    """
    Fit the model before optimization.
    """

    logger.info("Fitting Initial Model")

    # Extract model parameters.

    esr = model.specs['esr']
    sample_weights = model.specs['sample_weights']
    scorer = model.specs['scorer']
    seed = model.specs['seed']
    split = model.specs['split']

    # Extract model data.

    X_train = model.X_train
    y_train = model.y_train

    # Fit the initial model.

    if 'XGB' in algo and scorer in xgb_score_map:
        X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=split,
                                          random_state=seed)
        eval_set = [(X1, y1), (X2, y2)]
        eval_metric = xgb_score_map[scorer]
        est.fit(X1, y1, eval_set=eval_set, eval_metric=eval_metric,
                early_stopping_rounds=esr)
    elif sample_weights:
        est.fit(X_train, y_train, sample_weight=sample_weights)
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
    """
    Make predictions for training and test set.
    """

    logger.info("Final Model Predictions for %s", algo)

    # Extract model parameters.

    cal_type = model.specs['cal_type']
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

    if calibrate and model_type == ModelType.classification:
        logger.info("Calibrating Classifier")
        est = CalibratedClassifierCV(est, cv="prefit", method=cal_type)
        est.fit(X_train, y_train)
        model.estimators[algo] = est
        logger.info("Calibration Complete")
    else:
        logger.info("Skipping Calibration")

    # Make predictions on original training and test data.

    logger.info("Making Predictions")
    model.preds[(algo, 'train')] = est.predict(X_train)
    model.preds[(algo, 'test')] = est.predict(X_test)
    if model_type == ModelType.classification:
        model.probas[(algo, 'train')] = est.predict_proba(X_train)[:, 1]
        model.probas[(algo, 'test')] = est.predict_proba(X_test)[:, 1]
    logger.info("Predictions Complete")

    # Return the model
    return model


#
# Function predict_best
#

def predict_best(model):
    """
    Select the best model based on score.
    """

    logger.info("Selecting Best Model")

    # Extract model parameters.

    model_type = model.specs['model_type']
    scorer = model.specs['scorer']

    # Extract model data.

    X_train = model.X_train
    X_test = model.X_test
    y_test = model.y_test

    # Initialize best parameters.

    best_tag = 'BEST'
    partition = 'train' if y_test is None else 'test'
    maximize = True if scorers[scorer][1] == Objective.maximize else False
    if maximize:
        best_score = -sys.float_info.max
    else:
        best_score = sys.float_info.max

    # Iterate through the models, getting the best score for each one.

    start_time = datetime.now()
    logger.info("Best Model Selection Start: %s", start_time)

    for algorithm in model.algolist:
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

    # Store predictions of best estimator

    logger.info("Best Model is %s with a %s score of %.4f", best_algo, scorer, best_score)
    model.estimators[best_tag] = model.estimators[best_algo]
    model.preds[(best_tag, 'train')] = model.preds[(best_algo, 'train')]
    model.preds[(best_tag, 'test')] = model.preds[(best_algo, 'test')]
    if model_type == ModelType.classification:
        model.probas[(best_tag, 'train')] = model.probas[(best_algo, 'train')]
        model.probas[(best_tag, 'test')] = model.probas[(best_algo, 'test')]

    # Return the model with best estimator and predictions.

    end_time = datetime.now()
    time_taken = end_time - start_time
    logger.info("Best Model Selection Complete: %s", time_taken)

    return model


#
# Function predict_blend
#

def predict_blend(model):
    """
    Make predictions from a blended model.
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
        if hasattr(estimator, "coef_"):
            model.coefs[algorithm] = estimator.coef_
        if hasattr(estimator, "feature_importances_"):
            model.importances[algorithm] = estimator.feature_importances_
        # store predictions in the blended training set
        if model_type == ModelType.classification:
            X_blend_train[:, i] = model.probas[(algorithm, 'train')]
            X_blend_test[:, i] = model.probas[(algorithm, 'test')]
        else:
            X_blend_train[:, i] = model.preds[(algorithm, 'train')]
            X_blend_test[:, i] = model.preds[(algorithm, 'test')]

    # Use the blended estimator to make predictions

    if model_type == ModelType.classification:
        clf = LogisticRegression()
        clf.fit(X_blend_train, y_train)
        model.estimators[blend_tag] = clf
        model.preds[(blend_tag, 'train')] = clf.predict(X_blend_train)
        model.preds[(blend_tag, 'test')] = clf.predict(X_blend_test)
        model.probas[(blend_tag, 'train')] = clf.predict_proba(X_blend_train)[:, 1]
        model.probas[(blend_tag, 'test')] = clf.predict_proba(X_blend_test)[:, 1]
    else:
        alphas = [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5,
                  1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]    
        rcvr = RidgeCV(alphas=alphas, normalize=True, cv=cv_folds)
        rcvr.fit(X_blend_train, y_train)
        model.estimators[blend_tag] = rcvr
        model.preds[(blend_tag, 'train')] = rcvr.predict(X_blend_train)
        model.preds[(blend_tag, 'test')] = rcvr.predict(X_blend_test)

    # Return the model with blended estimator and predictions.

    end_time = datetime.now()
    time_taken = end_time - start_time
    logger.info("Blending Complete: %s", time_taken)

    return model


#
# Function generate_metrics
#

def generate_metrics(model, partition):

    logger.info('='*80)
    logger.info("Metrics for Partition: %s", partition)

    # Extract model paramters.

    model_type = model.specs['model_type']

    # Extract model data.

    if partition == 'train':
        expected = model.y_train
    else:
        expected = model.y_test

    # Generate Metrics

    if expected is not None:
        # get the metrics for each algorithm
        for algo in model.algolist:
            # get predictions for the given algorithm
            predicted = model.preds[(algo, partition)]
            try:
                model.metrics[(algo, partition, 'accuracy')] = accuracy_score(expected, predicted)
            except:
                logger.info("Accuracy Score not calculated")
            try:
                model.metrics[(algo, partition, 'adjusted_rand_score')] = adjusted_rand_score(expected, predicted)
            except:
                logger.info("Adjusted Rand Index not calculated")
            try:
                model.metrics[(algo, partition, 'confusion_matrix')] = confusion_matrix(expected, predicted)
            except:
                logger.info("Confusion Matrix not calculated")
            try:
                model.metrics[(algo, partition, 'explained_variance')] = explained_variance_score(expected, predicted)
            except:
                logger.info("Explained Variance Score not calculated")
            try:
                model.metrics[(algo, partition, 'f1')] = f1_score(expected, predicted)
            except:
                logger.info("F1 Score not calculated")
            try:
                model.metrics[(algo, partition, 'mean_absolute_error')] = mean_absolute_error(expected, predicted)
            except:
                logger.info("Mean Absolute Error not calculated")
            try:
                model.metrics[(algo, partition, 'median_absolute_error')] = median_absolute_error(expected, predicted)
            except:
                logger.info("Median Absolute Error not calculated")
            try:
                model.metrics[(algo, partition, 'mean_squared_error')] = mean_squared_error(expected, predicted)
            except:
                logger.info("Mean Squared Error not calculated")
            try:
                model.metrics[(algo, partition, 'precision')] = precision_score(expected, predicted)
            except:
                logger.info("Precision Score not calculated")
            try:
                model.metrics[(algo, partition, 'r2')] = r2_score(expected, predicted)
            except:
                logger.info("R-Squared Score not calculated")
            try:
                model.metrics[(algo, partition, 'recall')] = recall_score(expected, predicted)
            except:
                logger.info("Recall Score not calculated")
            # Probability-Based Metrics
            if model_type == ModelType.classification:
                predicted = model.probas[(algo, partition)]
                try:
                    model.metrics[(algo, partition, 'average_precision')] = average_precision_score(expected, predicted)
                except:
                    logger.info("Average Precision Score not calculated")
                try:
                    model.metrics[(algo, partition, 'neg_log_loss')] = log_loss(expected, predicted)
                except:
                    logger.info("Log Loss not calculated")
                try:
                    fpr, tpr, _ = roc_curve(expected, predicted)
                    model.metrics[(algo, partition, 'roc_auc')] = auc(fpr, tpr)
                except:
                    logger.info("ROC AUC Score not calculated")
        # log the metrics for each algorithm
        for algo in model.algolist:
            logger.info('-'*80)
            logger.info("Algorithm: %s", algo)
            metrics = [(k[2], v) for k, v in model.metrics.items() if k[0] == algo and k[1] == partition]
            for key, value in sorted(metrics):
                svalue = str(value)
                svalue.replace('\n', ' ')
                logger.info("%s: %s", key, svalue)
    else:
        logger.info("No labels for generating %s metrics", partition)

    logger.info('='*80)

    return model


#
# Function np_store_data
#

def np_store_data(data, dir_name, file_name, extension, separator):
    """
    Store NumPy data in a file.
    """

    output_file = PSEP.join([file_name, extension])
    output = SSEP.join([dir_name, output_file])
    np.savetxt(output, data, delimiter=separator)


#
# Function save_model
#

def save_model(model, tag, partition):
    """
    Save the results in the model file.
    """

    # Extract model parameters.

    directory = model.specs['directory']
    extension = model.specs['extension']
    model_type = model.specs['model_type']
    sample_submission = model.specs['sample_submission']
    separator = model.specs['separator']
    submission_file = model.specs['submission_file']
    test_file = model.specs['test_file']

    # Extract model data.

    X_train = model.X_train
    X_test = model.X_test

    # Get date stamp to record file creation

    d = datetime.now()
    f = "%Y%m%d"
    timestamp = d.strftime(f)

    # Dump the model object itself

    save_model_object(model, timestamp)

    # Specify input and output directories

    input_dir = SSEP.join([directory, 'input'])
    output_dir = SSEP.join([directory, 'output'])

    # Save predictions for all projects

    logger.info("Saving Predictions")
    output_file = USEP.join(['predictions', timestamp])
    preds = model.preds[(tag, partition)]
    np_store_data(preds, output_dir, output_file, extension, separator)

    # Save probabilities for classification projects

    if model_type == ModelType.classification:
        logger.info("Saving Probabilities")
        output_file = USEP.join(['probabilities', timestamp])
        probas = model.probas[(tag, partition)]
        np_store_data(probas, output_dir, output_file, extension, separator)

    # Save ranked predictions

    logger.info("Saving Ranked Predictions")
    tf = read_frame(input_dir, test_file, extension, separator)
    tf['prediction'] = pd.Series(preds, index=tf.index)
    tf['probability'] = pd.Series(probas, index=tf.index)
    tf.sort_values('probability', ascending=False, inplace=True)
    output_file = USEP.join(['rankings', timestamp])
    write_frame(tf, output_dir, output_file, extension, separator)

    # Generate submission file

    if sample_submission:
        logger.info("Saving Submission")
        sample_spec = PSEP.join([submission_file, extension])
        sample_input = SSEP.join([input_dir, sample_spec])
        ss = pd.read_csv(sample_input)
        if model_type == ModelType.classification:
            ss[ss.columns[1]] = probas
        else:
            ss[ss.columns[1]] = preds
        submission_base = USEP.join(['submission', timestamp])
        submission_spec = PSEP.join([submission_base, extension])
        submission_output = SSEP.join([output_dir, submission_spec])
        ss.to_csv(submission_output, index=False)
