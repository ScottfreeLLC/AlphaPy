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
from estimators import ModelType
import logging
import numpy as np
from scoring import report_scores
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

    cv_folds = model.specs['cv_folds']
    rfe_step = model.specs['rfe_step']
    scorer = model.specs['scorer']
    verbosity = model.specs['verbosity']
    estimator = model.estimators[algo]

    # Perform Recursive Feature Elimination

    logger.info("Recursive Feature Elimination with CV")
    rfecv = RFECV(estimator, step=rfe_step, cv=cv_folds,
                  scoring=scorer, verbose=verbosity)
    start = time()
    selector = rfecv.fit(X_train, y_train)
    logger.info("RFECV took %.2f seconds for step %d and %d folds",
                (time() - start), rfe_step, cv_folds)
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

    rfe_step = model.specs['rfe_step']
    verbosity = model.specs['verbosity']
    estimator = model.estimators[algo]

    # Perform Recursive Feature Elimination

    logger.info("Recursive Feature Elimination")
    rfe = RFE(estimator, step=rfe_step, verbose=verbosity)
    start = time()
    selector = rfe.fit(X_train, y_train)
    logger.info("RFE took %.2f seconds for step %d",
                (time() - start), rfe_step)
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
        logger.info("No grid is defined for grid search")
        return model

    # Get estimator.

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

    cv_folds = model.specs['cv_folds']
    gs_iters = model.specs['gs_iters']
    gs_random = model.specs['gs_random']
    gs_sample = model.specs['gs_sample']
    gs_sample_pct = model.specs['gs_sample_pct']
    n_jobs = model.specs['n_jobs']
    scorer = model.specs['scorer']
    verbosity = model.specs['verbosity']

    # Subsample if necessary to reduce grid search duration.

    if gs_sample:
        length = len(X_train)
        subset = int(length * gs_sample_pct)
        indices = np.random.choice(length, subset, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    # Create the randomized grid search iterator.

    if gs_random:
        logger.info("Randomized Grid Search")
        gscv = RandomizedSearchCV(est, param_distributions=grid, n_iter=gs_iters,
                                  scoring=scorer, n_jobs=n_jobs, cv=cv_folds,
                                  verbose=verbosity)
    else:
        logger.info("Full Grid Search")
        gscv = GridSearchCV(est, param_grid=grid, scoring=scorer, n_jobs=n_jobs,
                            cv=cv_folds, verbose=verbosity)

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
