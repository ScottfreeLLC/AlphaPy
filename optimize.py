##############################################################
#
# Package  : AlphaPy
# Module   : optimize
# Version  : 1.0
# Copyright: Mark Conway
# Date     : August 13, 2015
#
##############################################################


#
# Imports
#

from __future__ import division
from datetime import datetime
import logging
import numpy as np
from scoring import report_scores
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import log_loss
from time import time


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function rfecv_search
#

def rfecv_search(model, estimator):
    """
    Return the best feature set using recursive feature elimination
    with cross-validation.
    """

    logger.info("Recursive Feature Elimination with CV")

    # Extract model data and parameters.

    X_train = model.X_train
    y_train = model.y_train

    n_step = model.specs['n_step']
    n_folds = model.specs['n_folds']
    scorer = model.specs['scorer']
    verbosity = model.specs['verbosity']

    # Extract estimator parameters.

    algorithm = estimator.algorithm
    estimator = estimator.estimator

    # Perform Recursive Feature Elimination

    rfecv = RFECV(estimator, step=n_step, cv=n_folds,
                  scoring=scorer, verbose=verbosity)
    start = time()
    selector = rfecv.fit(X_train, y_train)
    logger.info("RFECV took %.2f seconds for step %d and %d folds",
                (time() - start), n_step, n_folds)
    logger.info("Algorithm: %s, Selected Features: %d, Ranking: %s",
                algorithm, selector.n_features_, selector.ranking_)

    # Record the support vector

    model.support[algorithm] = selector.support_

    # Return the model with the support vector

    return model


#
# Function rfe_search
#

def rfe_search(model, estimator):
    """
    Return the best feature set using recursive feature elimination.
    """

    logger.info("Recursive Feature Elimination")

    # Extract model data and parameters.

    X_train = model.X_train
    y_train = model.y_train

    n_step = model.specs['n_step']
    verbosity = model.specs['verbosity']

    # Extract estimator parameters.

    algorithm = estimator.algorithm
    estimator = estimator.estimator

    # Perform Recursive Feature Elimination

    rfe = RFE(estimator, step=n_step, verbose=verbosity)
    start = time()
    selector = rfe.fit(X_train, y_train)
    logger.info("RFE took %.2f seconds for step %d",
                (time() - start), n_step)
    logger.info("Algorithm: %s, Selected Features: %d, Ranking: %s",
                algorithm, selector.n_features_, selector.ranking_)

    # Record the support vector

    model.support[algorithm] = selector.support_

    # Return the model with the support vector

    return model


#
# Function sfm_search
#

def sfm_search(model, algorithm, feature_select, estimator):
    """
    Select From Model [SFM] Feature Selection
    """

    logger.info("Select From Model: %s", feature_select)

    # Extract model data and parameters.

    X_train = model.X_train
    y_train = model.y_train
    clf = estimator.estimator

    # Select features based on extra trees.

    start = time()
    clf = clf.fit(X_train, y_train)
    sfm = SelectFromModel(clf, prefit=True)
    logger.info("Select From Model: %s took %.2f seconds", feature_select, (time() - start))

    # Record the support vector

    model.support[algorithm] = sfm.get_support()

    # Return the model with the support vector

    return model


#
# Function grid_search
#

def grid_search(model, estimator):
    """
    Return the best hyperparameters using a randomized grid search.
    """

    logger.info("Parameter Grid Search")

    # Extract estimator parameters.

    algorithm = estimator.algorithm
    grid = estimator.grid
    if not grid:
        raise AttributeError("A grid must be defined to use grid search")
    estimator = estimator.estimator

    # Extract model parameters.

    try:
        support = model.support[algorithm]
        X_train = model.X_train[:, support]
    except:
        X_train = model.X_train

    y_train = model.y_train
    n_iters = model.specs['n_iters']
    n_folds = model.specs['n_folds']
    n_jobs = model.specs['n_jobs']
    scorer = model.specs['scorer']
    subsample = model.specs['subsample']
    subsample_pct = model.specs['subsample_pct']
    verbosity = model.specs['verbosity']

    # Subsample if necessary to reduce grid search duration.

    if subsample:
        length = len(X_train)
        subset = int(length * subsample_pct)
        indices = np.random.choice(length, subset, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    # Create the randomized grid search iterator.

    if n_iters > 0:
        gscv = RandomizedSearchCV(estimator, param_distributions=grid,
                                  n_iter=n_iters, scoring=scorer, n_jobs=n_jobs,
                                  cv=n_folds, verbose=verbosity)
    else:
        gscv = GridSearchCV(estimator, param_grid=grid, scoring=scorer,
                            n_jobs=n_jobs, cv=n_folds, verbose=verbosity)

    # Fit the randomized search and time it.

    start = time()
    gscv.fit(X_train, y_train)
    if n_iters > 0:
        logger.info("Random Grid Search took %.2f seconds for %d iterations",
                    (time() - start), n_iters)
    else:
        logger.info("Full Grid Search took %.2f seconds", (time() - start))
    logger.info("Algorithm: %s, Best Score: %.4f, Best Parameters: %s",
                algorithm, gscv.best_score_, gscv.best_params_)

    # Assign the Grid Search estimator for this algorithm

    model.estimators[algorithm] = gscv.best_estimator_
    model.scores[algorithm] = gscv.best_score_

    # Return the model with Grid Search estimators

    return model


#
# Function calibrate_model
#

def calibrate_model(model, estimator):
    """
    Calibrate a classifier.
    """

    # Extract model data and parameters.

    try:
        support = model.support[algorithm]
        X_train = model.X_train[:, support]
    except:
        X_train = model.X_train

    X_test = model.X_test
    y_train = model.y_train

    calibration = model.specs['calibration']
    n_folds = model.specs['n_folds']
    regression = model.specs['regression']

    # Extract estimator parameters.

    algo = estimator.algorithm
    clf = estimator.estimator

    # Iterate through the models, getting the best score for each one.

    start_time = datetime.now()
    logger.info("Calibration Start: %s", start_time)

    # Calibration

    cal_clf = CalibratedClassifierCV(clf, method=calibration, cv=n_folds)
    cal_clf.fit(X_train, y_train)
    model.estimators[algo] = cal_clf

    # Make predictions on original training and test data

    model.preds[(algo, 'train')] = cal_clf.predict(X_train)
    model.preds[(algo, 'test')] = cal_clf.predict(X_test)
    if not regression:
        model.probas[(algo, 'train')] = cal_clf.predict_proba(X_train)[:, 1]
        model.probas[(algo, 'test')] = cal_clf.predict_proba(X_test)[:, 1]

    # Training Log Loss

    lloss = log_loss(y_train, model.probas[(algo, 'train')], eps=1e-15, normalize=True)
    logger.info("Log Loss for %s: %.6f", algo, lloss)

    # Return the model with the calibrated classifier.

    end_time = datetime.now()
    time_taken = end_time - start_time
    logger.info("Calibration Complete: %s", time_taken)

    return model
