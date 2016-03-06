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

import cPickle as pickle
from datetime import datetime
from estimators import objective
from estimators import scorers
from estimators import xgb_score_map
from globs import PSEP, SSEP, USEP
import logging
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
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
            print "Model %s already exists" % mn
            
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
            separator = self.specs['separator']
        except:
            raise KeyError("Model specs must include the key: separator")
        try:
            self.algolist = self.specs['algorithms'].upper().split(separator)
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
# Function load_model
#

def load_model():
    """
    Load the model from storage.
    """

    logger.info("Loading Model")

    # Open model object

    f = file('model.save', 'rb')
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

    # Save model object

    f = file(full_path, 'wb')
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


#
# Function fit_model
#

def fit_model(model, algo, est, X1, y1, X2=None, y2=None):
    """
    Fit a scikit-learn model.
    """

    logger.info("Fitting Model for %s", algo)

    # Extract model parameters.

    esr = model.specs['esr']
    scorer = model.specs['scorer']

    # First Fit

    if 'XGB' in algo:
        try:
            if X2 is not None and y2 is not None:
                eval_set = [(X1, y1), (X2, y2)]
            else:
                eval_set = [(X1, y1)]
            eval_metric = xgb_score_map[scorer]
            est.fit(X1, y1, eval_set=eval_set, eval_metric=eval_metric,
                    early_stopping_rounds=esr)
        except:
            est.fit(X1, y1)
    else:
        est.fit(X1, y1)

    return est


#
# Function first_fit
#

def first_fit(model, algo, est):
    """
    Fit the model before optimization.
    """

    logger.info("Fitting Initial Model")

    # Extract model parameters.

    n_folds = model.specs['n_folds']
    regression = model.specs['regression']
    seed = model.specs['seed']
    split = model.specs['split']

    # Extract model data.

    X_train = model.X_train
    y_train = model.y_train

    # First Fit

    X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=split,
                                      random_state=seed)
    est = fit_model(model, algo, est, X1, y1, X2, y2)
    model.estimators[algo] = est

    # Record scores, importances, and coefficients, if available.

    score = est.score(X1, y1)
    logger.info("Training Score: %.6f", score)

    score = est.score(X2, y2)
    logger.info("Testing Score: %.6f", score)

    if hasattr(est, "feature_importances_"):
        model.importances[algo] = est.feature_importances_

    if hasattr(est, "coef_"):
        model.coefs[algo] = est.coef_

    return model


#
# Function make_predictions
#

def make_predictions(model, algo):
    """
    Make predictions for training and test set.
    """

    logger.info("Final Model Predictions for %s", algo)

    # Extract model parameters.

    regression = model.specs['regression']
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

    # Fit the final model

    est = fit_model(model, algo, est, X_train, y_train)

    # Make predictions on original training and test data

    model.preds[(algo, 'train')] = est.predict(X_train)
    model.preds[(algo, 'test')] = est.predict(X_test)
    if not regression:
        model.probas[(algo, 'train')] = est.predict_proba(X_train)[:, 1]
        model.probas[(algo, 'test')] = est.predict_proba(X_test)[:, 1]

    # Training Log Loss

    if not regression:
        lloss = log_loss(y_train, model.probas[(algo, 'train')])
        logger.info("Log Loss for %s: %.6f", algo, lloss)

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

    regression = model.specs['regression']
    scorer = model.specs['scorer']

    # Extract model data.

    X_train = model.X_train
    X_test = model.X_test
    y_test = model.y_test

    # Initialize best parameters.

    best_tag = 'BEST'
    partition = 'train' if y_test is None else 'test'
    maximize = True if scorers[scorer][1] == objective.maximize else False
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
    if not regression:
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

    n_folds = model.specs['n_folds']
    regression = model.specs['regression']

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
        if not regression:
            X_blend_train[:, i] = model.probas[(algorithm, 'train')]
            X_blend_test[:, i] = model.probas[(algorithm, 'test')]
        else:
            X_blend_train[:, i] = model.preds[(algorithm, 'train')]
            X_blend_test[:, i] = model.preds[(algorithm, 'test')]

    # Use the blended estimator to make predictions

    if not regression:
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
        rcvr = RidgeCV(alphas=alphas, normalize=True, cv=n_folds)
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

    regression = model.specs['regression']

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
            probas = model.probas[(algo, partition)]
            try:
                model.metrics[(algo, partition, 'accuracy')] = accuracy_score(expected, predicted)
            except:
                logger.info("Accuracy Score not calculated")
            try:
                model.metrics[(algo, partition, 'adjusted_rand_score')] = adjusted_rand_score(expected, predicted)
            except:
                logger.info("Adjusted Rand Index not calculated")
            try:
                model.metrics[(algo, partition, 'average_precision')] = average_precision_score(expected, predicted)
            except:
                logger.info("Average Precision Score not calculated")
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
                model.metrics[(algo, partition, 'log_loss')] = log_loss(expected, probas)
            except:
                logger.info("Log Loss not calculated")
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
            try:
                model.metrics[(algo, partition, 'roc_auc')] = roc_auc_score(expected, predicted)
            except:
                logger.info("ROC AUC Score not calculated")
        # log the metrics for each algorithm
        for algo in model.algolist:
            logger.info('-'*80)
            logger.info("Algorithm: %s", algo)
            metrics = [(k[2], v) for k, v in model.metrics.iteritems() if k[0] == algo and k[1] == partition]
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
    project = model.specs['project']
    extension = model.specs['extension']
    separator = model.specs['separator']
    regression = model.specs['regression']

    # Extract model data.

    X_train = model.X_train
    X_test = model.X_test

    # Get date stamp to record file creation

    d = datetime.now()
    f = "%m%d%y"

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
    if not regression:
        output_file = USEP.join(['probas', d.strftime(f)])
        preds = model.probas[(tag, partition)]
    else:
        output_file = USEP.join(['preds', d.strftime(f)])
        preds = model.preds[(tag, partition)]
    output_file = PSEP.join([output_file, extension])
    output = SSEP.join([output_dir, output_file])
    np.savetxt(output, preds, delimiter=separator)
