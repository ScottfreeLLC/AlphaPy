################################################################################
#
# Package   : AlphaPy
# Module    : optimize
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

from datetime import datetime
import logging
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from time import time


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function rfecv_search
#

def rfecv_search(model, algo):
    r"""Return the best feature set using recursive feature elimination
    with cross-validation.

    Parameters
    ----------
    model : alphapy.Model
        The model object with RFE parameters.
    algo : str
        Abbreviation of the algorithm to run.

    Returns
    -------
    model : alphapy.Model
        The model object with the RFE support vector and the best
        estimator.

    See Also
    --------
    rfe_search

    Notes
    -----
    If a scoring function is available, then AlphaPy can perform RFE
    with Cross-Validation (CV), as in this function; otherwise, it just
    does RFE without CV.

    References
    ----------
    For more information about Recursive Feature Elimination,
    refer to [RFECV]_.

    .. [RFECV] http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html

    """

    # Extract model data.

    X_train = model.X_train
    y_train = model.y_train

    # Extract model parameters.

    cv_folds = model.specs['cv_folds']
    n_jobs = model.specs['n_jobs']
    rfe_step = model.specs['rfe_step']
    scorer = model.specs['scorer']
    verbosity = model.specs['verbosity']
    estimator = model.estimators[algo]

    # Perform Recursive Feature Elimination

    logger.info("Recursive Feature Elimination with CV")
    rfecv = RFECV(estimator, step=rfe_step, cv=cv_folds,
                  scoring=scorer, verbose=verbosity, n_jobs=n_jobs)
    start = time()
    selector = rfecv.fit(X_train, y_train)
    logger.info("RFECV took %.2f seconds for step %d and %d folds",
                (time() - start), rfe_step, cv_folds)
    logger.info("Algorithm: %s, Selected Features: %d, Ranking: %s",
                algo, selector.n_features_, selector.ranking_)

    # Record the new estimator and support vector

    model.estimators[algo] = selector.estimator_
    model.support[algo] = selector.support_

    # Return the model with the support vector

    return model


#
# Function grid_report
#

def grid_report(results, n_top=3):
    r"""Report the top grid search scores.

    Parameters
    ----------
    results : dict of numpy arrays
        Mean test scores for each grid search iteration.
    n_top : int, optional
        The number of grid search results to report.

    Returns
    -------
    None : None

    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            logger.info("Model with rank: {0}".format(i))
            logger.info("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                        results['mean_test_score'][candidate],
                        results['std_test_score'][candidate]))
            logger.info("Parameters: {0}".format(results['params'][candidate]))


#
# Function hyper_grid_search
#

def hyper_grid_search(model, estimator):
    r"""Return the best hyperparameters for a grid search.

    Parameters
    ----------
    model : alphapy.Model
        The model object with grid search parameters.
    estimator : alphapy.Estimator
        The estimator containing the hyperparameter grid.

    Returns
    -------
    model : alphapy.Model
        The model object with the grid search estimator.

    Notes
    -----
    To reduce the time required for grid search, use either
    randomized grid search with a fixed number of iterations
    or a full grid search with subsampling. AlphaPy uses
    the scikit-learn Pipeline with feature selection to
    reduce the feature space.

    References
    ----------
    For more information about grid search, refer to [GRID]_.

    .. [GRID] http://scikit-learn.org/stable/modules/grid_search.html#grid-search

    To learn about pipelines, refer to [PIPE]_.

    .. [PIPE] http://scikit-learn.org/stable/modules/pipeline.html#pipeline

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
    feature_selection = model.specs['feature_selection']
    fs_percentage = model.specs['fs_percentage']
    fs_score_func = model.specs['fs_score_func']
    fs_uni_grid = model.specs['fs_uni_grid']
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

    # Convert the grid to pipeline format

    grid_new = {}
    for k, v in list(grid.items()):
        new_key = '__'.join(['est', k])
        grid_new[new_key] = grid[k]

    # Create the pipeline for grid search

    if feature_selection:
        # Augment the grid for feature selection.
        fs = SelectPercentile(score_func=fs_score_func,
                              percentile=fs_percentage)
        # Combine the feature selection and estimator grids.
        fs_grid = dict(fs__percentile=fs_uni_grid)
        grid_new.update(fs_grid)
        # Create a pipeline with the selected features and estimator.
        pipeline = Pipeline([("fs", fs), ("est", est)])
    else:
        pipeline = Pipeline([("est", est)])

    # Create the randomized grid search iterator.

    if gs_random:
        logger.info("Randomized Grid Search")
        gscv = RandomizedSearchCV(pipeline, param_distributions=grid_new,
                                  n_iter=gs_iters, scoring=scorer,
                                  n_jobs=n_jobs, cv=cv_folds, verbose=verbosity)
    else:
        logger.info("Full Grid Search")
        gscv = GridSearchCV(pipeline, param_grid=grid_new, scoring=scorer,
                            n_jobs=n_jobs, cv=cv_folds, verbose=verbosity)

    # Fit the randomized search and time it.

    start = time()
    gscv.fit(X_train, y_train)
    if gs_iters > 0:
        logger.info("Grid Search took %.2f seconds for %d candidate"
                    " parameter settings." % ((time() - start), gs_iters))
    else:
        logger.info("Grid Search took %.2f seconds for %d candidate parameter"
                    " settings." % (time() - start, len(gscv.cv_results_['params'])))

    # Log the grid search scoring statistics.

    grid_report(gscv.cv_results_)
    logger.info("Algorithm: %s, Best Score: %.4f, Best Parameters: %s",
                algo, gscv.best_score_, gscv.best_params_)

    # Assign the Grid Search estimator for this algorithm

    model.estimators[algo] = gscv

    # Return the model with Grid Search estimators
    return model
