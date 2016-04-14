##############################################################
#
# Package   : AlphaPy
# Module    : model
# Version   : 1.0
# Copyright : Mark Conway
# Date      : June 29, 2013
#
##############################################################


#
# Imports
#

import _pickle as pickle
from data import SamplingMethod
from datetime import datetime
from estimators import Objective
from estimators import ModelType
from estimators import scorers
from estimators import xgb_score_map
from globs import PSEP
from globs import SSEP
from globs import USEP
from group import Group
import logging
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score
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
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
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

    # class variable to track all models

    models = {}

    # __new__
    
    def __new__(cls,
                specs):
        # create model name
        try:
            mn = specs['project']
        except:
            raise KeyError("Model specs must include the key: project")
        if not mn in Model.models:
            return super(Model, cls).__new__(cls)
        else:
            print ("Model ", mn, " already exists")
            
    # __init__
            
    def __init__(self,
                 specs):
        self.specs = specs
        self.name = specs['project']
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
        self.scores = {}
        self.preds = {}
        self.probas = {}
        # Keys: (algorithm, partition, metric)
        self.metrics = {}
        # add model to models list
        try:
            Model.models[specs['project']] = self
        except:
            raise KeyError("Model specs must include the key: project")
                
    # __str__

    def __str__(self):
        return self.name


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

    specs['base_dir'] = cfg['project']['base_directory']
    specs['extension'] = cfg['project']['file_extension']
    specs['kaggle'] = cfg['project']['kaggle_submission']
    specs['project'] = cfg['project']['project_name']

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
    # interactions
    specs['interactions'] = cfg['data']['interactions']['option']
    specs['isample_pct'] = cfg['data']['interactions']['sampling_pct']
    specs['poly_degree'] = cfg['data']['interactions']['poly_degree']
    # sampling
    specs['sampling'] = cfg['data']['sampling']['option']
    specs['sampling_method'] = SamplingMethod(cfg['data']['sampling']['method'])
    specs['sampling_ratio'] = cfg['data']['sampling']['ratio']

    # Section: features

    # clustering
    specs['clustering'] = cfg['features']['clustering']['option']
    specs['cluster_min'] = cfg['features']['clustering']['minimum']
    specs['cluster_max'] = cfg['features']['clustering']['maximum']
    specs['cluster_inc'] = cfg['features']['clustering']['increment']
    # genetic
    specs['genetic'] = cfg['features']['genetic']['option']
    specs['gfeatures'] = cfg['features']['genetic']['features']
    # pca
    specs['pca'] = cfg['features']['pca']['option']
    specs['pca_min'] = cfg['features']['pca']['minimum']
    specs['pca_max'] = cfg['features']['pca']['maximum']
    specs['pca_inc'] = cfg['features']['pca']['increment']
    specs['pca_whiten'] = cfg['features']['pca']['whiten']
    # text
    specs['ngrams_max'] = cfg['features']['text']['ngrams']
    specs['vectorize'] = cfg['features']['text']['vectorize']

    # Section: model

    specs['algorithms'] = cfg['model']['algorithms']
    specs['balance_classes'] = cfg['model']['balance_classes']
    specs['cv_folds'] = cfg['model']['cv_folds']
    specs['model_type'] = ModelType(cfg['model']['type'])
    specs['n_estimators'] = cfg['model']['estimators']
    specs['pvalue_level'] = cfg['model']['pvalue_level']
    specs['scorer'] = cfg['model']['scoring_function']
    # calibration
    specs['calibration'] = cfg['model']['calibration']['option']
    specs['cal_type'] = cfg['model']['calibration']['type']
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
    logger.info('algorithms       = %s', specs['algorithms'])
    logger.info('balance_classes  = %s', specs['balance_classes'])
    logger.info('base_dir         = %s', specs['base_dir'])
    logger.info('calibration      = %r', specs['calibration'])
    logger.info('cal_type         = %s', specs['cal_type'])
    logger.info('calibration_plot = %r', specs['calibration'])
    logger.info('clustering       = %r', specs['clustering'])
    logger.info('cluster_inc      = %d', specs['cluster_inc'])
    logger.info('cluster_max      = %d', specs['cluster_max'])
    logger.info('cluster_min      = %d', specs['cluster_min'])
    logger.info('confusion_matrix = %r', specs['confusion_matrix'])
    logger.info('cv_folds         = %d', specs['cv_folds'])
    logger.info('extension        = %s', specs['extension'])
    logger.info('drop             = %s', specs['drop'])
    logger.info('dummy_limit      = %d', specs['dummy_limit'])
    logger.info('esr              = %d', specs['esr'])
    logger.info('features [X]     = %s', specs['features'])
    logger.info('genetic          = %r', specs['genetic'])
    logger.info('gfeatures        = %d', specs['gfeatures'])
    logger.info('grid_search      = %r', specs['grid_search'])
    logger.info('gs_iters         = %d', specs['gs_iters'])
    logger.info('gs_random        = %r', specs['gs_random'])
    logger.info('gs_sample        = %r', specs['gs_sample'])
    logger.info('gs_sample_pct    = %f', specs['gs_sample_pct'])
    logger.info('importances      = %r', specs['importances'])
    logger.info('interactions     = %r', specs['interactions'])
    logger.info('isample_pct      = %d', specs['isample_pct'])
    logger.info('kaggle           = %r', specs['kaggle'])
    logger.info('learning_curve   = %r', specs['learning_curve'])
    logger.info('model_type       = %r', specs['model_type'])
    logger.info('n_estimators     = %d', specs['n_estimators'])
    logger.info('n_jobs           = %d', specs['n_jobs'])
    logger.info('ngrams_max       = %d', specs['ngrams_max'])
    logger.info('pca              = %r', specs['pca'])
    logger.info('pca_inc          = %d', specs['pca_inc'])
    logger.info('pca_max          = %d', specs['pca_max'])
    logger.info('pca_min          = %d', specs['pca_min'])
    logger.info('pca_whiten       = %r', specs['pca_whiten'])
    logger.info('poly_degree      = %d', specs['poly_degree'])
    logger.info('project          = %s', specs['project'])
    logger.info('pvalue_level     = %f', specs['pvalue_level'])
    logger.info('rfe              = %r', specs['rfe'])
    logger.info('rfe_step         = %d', specs['rfe_step'])
    logger.info('roc_curve        = %r', specs['roc_curve'])
    logger.info('sampling         = %r', specs['sampling'])
    logger.info('sampling_method  = %r', specs['sampling_method'])
    logger.info('sampling_ratio   = %f', specs['sampling_ratio'])
    logger.info('scorer           = %s', specs['scorer'])
    logger.info('seed             = %d', specs['seed'])
    logger.info('sentinel         = %d', specs['sentinel'])
    logger.info('separator        = %s', specs['separator'])
    logger.info('shuffle          = %r', specs['shuffle'])
    logger.info('split            = %f', specs['split'])
    logger.info('target [y]       = %s', specs['target'])
    logger.info('target_value     = %d', specs['target_value'])
    logger.info('test_file        = %s', specs['test_file'])
    logger.info('test_labels      = %r', specs['test_labels'])
    logger.info('train_file       = %s', specs['train_file'])
    logger.info('treatments       = %s', specs['treatments'])
    logger.info('vectorize        = %r', specs['vectorize'])
    logger.info('verbosity        = %d', specs['verbosity'])

    # Specifications to create the model

    return specs


#
# Function load_model
#

def load_model():
    """
    Load the model from storage.
    """

    logger.info("Loading Model")

    # Open model object

    with open ('model.save', 'rb') as f:
        model = pickle.load(f)
    f.close()

    return model


#
# Function save_model
#

def save_model(model):
    """
    Save the model to storage.
    """

    logger.info("Saving Model")

    # Extract model parameters.

    base_dir = model.specs['base_dir']
    project = model.specs['project']

    # Create full path name.

    directory = SSEP.join([base_dir, project])
    full_path = SSEP.join([directory, 'model.save'])

    # Save the model object to a YAML representation.

    # with open(full_path, 'wb') as f:
    #     yaml.dump(model, f)
    # f.close()

    # Save model object (previously saved in Pickle format)

    # with open(full_path, 'wb') as f:
    #     pickle.dump(model, f)
    # f.close()


#
# Function get_sample_weights
#

def get_sample_weights(model):
    """
    Set sample weights for fitting the model
    """

    logger.info("Getting Sample Weights")

    # Extract model parameters.

    balance_classes = model.specs['balance_classes']
    target = model.specs['target']
    target_value = model.specs['target_value']

    # Extract model data.

    y_train = model.y_train

    # Calculate sample weights

    sw = None
    if balance_classes:
        uv, uc = np.unique(y_train, return_counts=True)
        weight = uc[not target_value] / uc[target_value]
        logger.info("Sample Weight for target %s [%r]: %f",
                    target, target_value, weight)
        sw = [weight if x==target_value else 1.0 for x in y_train]    

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

    cv_folds = model.specs['cv_folds']
    esr = model.specs['esr']
    n_jobs = model.specs['n_jobs']
    scorer = model.specs['scorer']
    seed = model.specs['seed']
    split = model.specs['split']
    sample_weights = model.specs['sample_weights']
    verbosity = model.specs['verbosity']

    # Extract model data.

    X_train = model.X_train
    y_train = model.y_train

    # Get initial estimates of our score.

    sss = StratifiedShuffleSplit(y_train, n_iter=cv_folds, test_size=split,
                                 random_state=seed)

    scores = cross_val_score(est, X_train, y_train, cv=sss, scoring=scorer,
                             n_jobs=n_jobs, verbose=verbosity,
                             fit_params={'sample_weight': sample_weights})

    # Save the estimator in the model

    model.estimators[algo] = est

    # Record scores

    logger.info("CV Scores: %s", scores)

    # Record importances and coefficients, if available.

    if hasattr(est, "feature_importances_"):
        model.importances[algo] = est.feature_importances_

    if hasattr(est, "coef_"):
        model.coefs[algo] = est.coef_

    return model


#
# Function make_predictions
#

def make_predictions(model, algo, calibrate):
    """
    Make predictions for training and test set.
    """

    start_time = datetime.now()
    logger.info("Final Model Predictions for %s", algo)

    # Extract model parameters.

    cal_type = model.specs['cal_type']
    model_type = model.specs['model_type']
    sample_weights = model.specs['sample_weights']
    seed = model.specs['seed']
    split = model.specs['split']
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

    # Fit the final model.

    est.fit(X_train, y_train, sample_weight=sample_weights)
    model.estimators[algo] = est

    # Calibration

    if calibrate and model_type == ModelType.classification:
        logger.info("Calibration")
        est = CalibratedClassifierCV(est, cv="prefit", method=cal_type)
        est.fit(X_train, y_train)
    else:
        logger.info("Skipping Calibration")

    # Make predictions on original training and test data.

    model.preds[(algo, 'train')] = est.predict(X_train)
    model.preds[(algo, 'test')] = est.predict(X_test)
    if model_type == ModelType.classification:
        model.probas[(algo, 'train')] = est.predict_proba(X_train)[:, 1]
        model.probas[(algo, 'test')] = est.predict_proba(X_test)[:, 1]

    # Record the final score.

    score = est.score(X_train, y_train)
    logger.info("Final Score: %.6f", score)
    model.scores[algo] = score

    # Training Log Loss

    if model_type == ModelType.classification:
        loss = log_loss(y_train, model.probas[(algo, 'train')])
        logger.info("Log Loss for %s: %.6f", algo, loss)

    # Return the model.

    end_time = datetime.now()
    time_taken = end_time - start_time
    logger.info("Final Predictions Complete: %s", time_taken)

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

    logger.info("Best Model is %s with a %s score of %.6f", best_algo, scorer, best_score)
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
                    model.metrics[(algo, partition, 'log_loss')] = log_loss(expected, predicted)
                except:
                    logger.info("Log Loss not calculated")
                try:
                    model.metrics[(algo, partition, 'roc_auc')] = roc_auc_score(expected, predicted)
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
# Function save_results
#

def save_results(model, tag, partition):
    """
    Save results in the given output file.
    """

    # Extract model parameters.

    base_dir = model.specs['base_dir']
    extension = model.specs['extension']
    kaggle = model.specs['kaggle']
    model_type = model.specs['model_type']
    project = model.specs['project']
    separator = model.specs['separator']

    # Extract model data.

    X_train = model.X_train
    X_test = model.X_test

    # Get date stamp to record file creation

    d = datetime.now()
    f = "%Y%m%d"

    # Save the model

    save_model(model)

    # Save predictions and final features

    # training data
    # output_dir = SSEP.join([base_dir, project])
    # output_file = USEP.join(['train', d.strftime(f)])
    # output_file = PSEP.join([output_file, extension])
    # output = SSEP.join([output_dir, output_file])
    # np.savetxt(output, X_train, delimiter=separator)
    # test data
    # output_dir = SSEP.join([base_dir, project])
    # output_file = USEP.join(['test', d.strftime(f)])
    # output_file = PSEP.join([output_file, extension])
    # output = SSEP.join([output_dir, output_file])
    # np.savetxt(output, X_test, delimiter=separator)
    # predictions
    # output_dir = SSEP.join([base_dir, project])
    # output_file = USEP.join(['predictions', d.strftime(f)])
    # output_file = PSEP.join([output_file, extension])
    # output = SSEP.join([output_dir, output_file])
    # np.savetxt(output, preds, delimiter=separator)
    # probabilities

    output_dir = SSEP.join([base_dir, project])
    if model_type == ModelType.classification:
        output_file = USEP.join(['probas', d.strftime(f)])
        preds = model.probas[(tag, partition)]
    else:
        output_file = USEP.join(['preds', d.strftime(f)])
        preds = model.preds[(tag, partition)]
    output_file = PSEP.join([output_file, extension])
    output = SSEP.join([output_dir, output_file])
    np.savetxt(output, preds, delimiter=separator)

    # Generate Kaggle submission file

    if kaggle:
        sample_file = SSEP.join([output_dir, 'sample_submission.csv'])
        ss = pd.read_csv(sample_file)
        ss[ss.columns[1]] = preds
        kaggle_base = USEP.join(['kaggle', 'submission', d.strftime(f)])
        kaggle_file = PSEP.join([kaggle_base, 'csv'])
        kaggle_output = SSEP.join([output_dir, kaggle_file])
        ss.to_csv(kaggle_output, index=False)
