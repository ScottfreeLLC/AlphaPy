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
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from time import time


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function rfecv_search
#

def rfecv_search(model, algo):
    """
    Return the best feature set using recursive feature elimination
    with cross-validation.
    """

    # Extract model data.

    X_train = model.X_train
    y_train = model.y_train

    # Extract model parameters.

    n_step = model.specs['n_step']
    n_folds = model.specs['n_folds']
    scorer = model.specs['scorer']
    verbosity = model.specs['verbosity']
    estimator = model.estimators[algo]

    # Perform Recursive Feature Elimination

    logger.info("Recursive Feature Elimination with CV")
    rfecv = RFECV(estimator, step=n_step, cv=n_folds,
                  scoring=scorer, verbose=verbosity)
    start = time()
    selector = rfecv.fit(X_train, y_train)
    logger.info("RFECV took %.2f seconds for step %d and %d folds",
                (time() - start), n_step, n_folds)
    logger.info("Algorithm: %s, Selected Features: %d, Ranking: %s",
                algo, selector.n_features_, selector.ranking_)

    # Record the support vector

    model.support[algo] = selector.support_

    # Return the model with the support vector

    return model


#
# Function rfe_search
#

def rfe_search(model, algo):
    """
    Return the best feature set using recursive feature elimination.
    """

    # Extract model data.

    X_train = model.X_train
    y_train = model.y_train

    # Extract model parameters.

    n_step = model.specs['n_step']
    verbosity = model.specs['verbosity']
    estimator = model.estimators[algo]

    # Perform Recursive Feature Elimination

    logger.info("Recursive Feature Elimination")
    rfe = RFE(estimator, step=n_step, verbose=verbosity)
    start = time()
    selector = rfe.fit(X_train, y_train)
    logger.info("RFE took %.2f seconds for step %d",
                (time() - start), n_step)
    logger.info("Algorithm: %s, Selected Features: %d, Ranking: %s",
                algo, selector.n_features_, selector.ranking_)

    # Record the support vector

    model.support[algo] = selector.support_

    # Return the model with the support vector

    return model


#
# Function hyper_grid_search
#

def hyper_grid_search(model, estimator):
    """
    Return the best hyperparameters using a randomized grid search.
    """

    # Extract estimator parameters.

    grid = estimator.grid
    if not grid:
        raise AttributeError("A grid must be defined to use grid search")
    algo = estimator.algorithm
    est = model.estimators[algo]

    # Extract model data.

    try:
        support = model.support[algo]
        X_train = model.X_train[:, support]
    except:
        X_train = model.X_train
    y_train = model.y_train

    # Extract model parameters.

    gs_iters = model.specs['gs_iters']
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

    if gs_iters > 0:
        logger.info("Randomized Grid Search")
        gscv = RandomizedSearchCV(est, param_distributions=grid, n_iter=gs_iters,
                                  scoring=scorer, n_jobs=n_jobs, cv=n_folds,
                                  verbose=verbosity)
    else:
        logger.info("Full Grid Search")
        gscv = GridSearchCV(est, param_grid=grid, scoring=scorer, n_jobs=n_jobs,
                            cv=n_folds, verbose=verbosity)

    # Fit the randomized search and time it.

    start = time()
    gscv.fit(X_train, y_train)
    if gs_iters > 0:
        logger.info("Randomized Grid Search took %.2f seconds for %d iterations",
                    (time() - start), gs_iters)
    else:
        logger.info("Full Grid Search took %.2f seconds", (time() - start))
    logger.info("Algorithm: %s, Best Score: %.4f, Best Parameters: %s",
                algo, gscv.best_score_, gscv.best_params_)

    # Assign the Grid Search estimator for this algorithm

    model.estimators[algo] = gscv.best_estimator_
    model.scores[algo] = gscv.best_score_

    # Return the model with Grid Search estimators

    return model


#
# Function calibrate_model
#

def calibrate_model(model, algo):
    """
    Calibrate a classifier.
    """

    # Extract model parameters

    calibration = model.specs['calibration']
    esr = model.specs['esr']
    grid_search = model.specs['grid_search']
    n_folds = model.specs['n_folds']
    regression = model.specs['regression']
    seed = model.specs['seed']
    split = model.specs['split']
    clf = model.estimators[algo]

    # Extract model data.

    try:
        support = model.support[algo]
        X_train = model.X_train[:, support]
        X_test = model.X_test[:, support]
    except:
        X_train = model.X_train
        X_test = model.X_test
    y_train = model.y_train

    # Iterate through the models, getting the best score for each one.

    start_time = datetime.now()
    logger.info("Calibration Start: %s", start_time)

    # Calibration

    if 'XGB' in algo:
        X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=split,
                                          random_state=seed)
        es = [(X1, y1), (X2, y2)]
        clf.fit(X1, y1, eval_set=es, early_stopping_rounds=esr)
    else:
        clf = CalibratedClassifierCV(clf, method=calibration, cv=n_folds)
        clf.fit(X_train, y_train)

    # Record the training score

    model.estimators[algo] = clf
    score = clf.score(X_train, y_train)
    model.scores[algo] = score
    logger.info("Calibrated Score: %.6f", score)

    # Return the model with the calibrated classifier.

    end_time = datetime.now()
    time_taken = end_time - start_time
    logger.info("Calibration Complete: %s", time_taken)

    return model
