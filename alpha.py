##############################################################
#
# Package  : AlphaPy
# Module   : alpha
# Version  : 1.0
# Copyright: Mark Conway
# Date     : June 29, 2013
#
##############################################################


#
# Imports
#

from __future__ import division
import argparse
from data import load_data
from estimators import get_estimators
from estimators import scorers
from features import create_features
from features import create_interactions
from features import drop_features
from features import save_features
from globs import CSEP
from globs import PSEP
from globs import SSEP
from globs import WILDCARD
import logging
from model import first_fit
from model import generate_metrics
from model import make_predictions
from model import Model
from model import predict_best
from model import predict_blend
from model import save_results
import numpy as np
from optimize import calibrate_model
from optimize import hyper_grid_search
from optimize import rfe_search
from optimize import rfecv_search
import pandas as pd
from plots import generate_plots


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function pipeline
#

def pipeline(model):
    """
    AlphaPy Main Program
    :rtype : object
    """

    # Unpack the model specifications

    base_dir = model.specs['base_dir']
    drop = model.specs['drop']
    extension = model.specs['extension']
    features = model.specs['features']
    grid_search = model.specs['grid_search']
    n_estimators = model.specs['n_estimators']
    n_jobs = model.specs['n_jobs']
    plots = model.specs['plots']
    project = model.specs['project']
    regression = model.specs['regression']
    rfe = model.specs['rfe']
    scorer = model.specs['scorer']
    seed = model.specs['seed']
    separator = model.specs['separator']
    shuffle = model.specs['shuffle']
    split = model.specs['split']
    target = model.specs['target']
    test_file = model.specs['test_file']
    test_labels = model.specs['test_labels']
    train_file = model.specs['train_file']
    verbosity = model.specs['verbosity']

    # Initialize feature variables

    X_train = None
    X_test = None
    y_train = None
    y_test = None    

    # Load data based on whether there are 1 or 2 files

    logger.info("Loading Data")

    directory = SSEP.join([base_dir, project])
    # load training data
    X_train, y_train = load_data(directory, train_file, extension,
                                 separator, features, target)
    # load test data
    if test_labels:
        X_test, y_test = load_data(directory, test_file, extension,
                                   separator, features, target,
                                   return_labels=test_labels)
    else:
        X_test = load_data(directory, test_file, extension,
                           separator, features, target,
                           return_labels=test_labels)
    # merge training and test data
    if X_train.shape[1] == X_test.shape[1]:
        split_point = X_train.shape[0]
        X = pd.concat([X_train, X_test])
    else:
        raise IndexError("The number of training and test columns must match.")

    # Feature Statistics

    logger.info("Original Feature Statistics")
    logger.info("Number of Training Rows    : %d", X_train.shape[0])
    logger.info("Number of Training Columns : %d", X_train.shape[1])
    logger.info("Number of Testing Rows    : %d", X_test.shape[0])
    logger.info("Number of Testing Columns : %d", X_test.shape[1])

    # Drop features

    X = drop_features(X, drop)

    # Create initial features

    new_features = create_features(X, model)
    X_train, X_test = np.array_split(new_features, [split_point])
    model = save_features(model, X_train, X_test, y_train, y_test)

    # Generate interactions

    all_features = create_interactions(new_features, model)
    X_train, X_test = np.array_split(all_features, [split_point])
    model = save_features(model, X_train, X_test, y_train, y_test)

    # Shuffle the data if necessary

    if shuffle:
        logger.info("Shuffling Training Data")
        np.random.seed(seed)
        new_indices = np.random.permutation(y_train.size)
        X_train = X_train[new_indices]
        y_train = y_train[new_indices]
        model = save_features(model, X_train, X_test, y_train, y_test)

    # Get the available classifiers and regressors 

    logger.info("Getting All Estimators")
    estimators = get_estimators(n_estimators, seed, n_jobs, verbosity)

    # Get the available scorers

    if scorer not in scorers:
        raise KeyError("Scorer function %s not found", scorer)

    # Model Selection

    logger.info("Selecting Models")

    for algo in model.algolist:
        logger.info("Algorithm: %s", algo)
        # select estimator
        try:
            estimator = estimators[algo]
            scoring = estimator.scoring
            est = estimator.estimator
        except KeyError:
            logger.info("Algorithm %s not found", algo)
        # initial fit
        model = first_fit(model, algo, est)
        # feature selection
        if rfe:
            if scoring:
                model = rfecv_search(model, algo)
            elif hasattr(est, "coef_"):
                model = rfe_search(model, algo)
        # grid search
        if grid_search:
            model = hyper_grid_search(model, estimator)
        # calibration
        if not regression:
            model = calibrate_model(model, algo)
        # predictions
        model = make_predictions(model, algo)

    # Create a blended estimator

    model = predict_blend(model)

    # Generate metrics

    model = generate_metrics(model, 'train')
    model = generate_metrics(model, 'test')

    # Store the best estimator

    model = predict_best(model)

    # Generate plots

    if plots:
        generate_plots(model, 'train')
        if test_labels:
            generate_plots(model, 'test')

    # Save best features and predictions

    save_results(model, 'BEST', 'test')

    # Return the completed model

    return model


#
# MAIN PROGRAM
#

if __name__ == '__main__':

    # Logging

    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="alpha314.log", filemode='a', level=logging.DEBUG,
                        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    logger = logging.getLogger(__name__)

    # Start the pipeline

    logger.info('='*80)
    logger.info("START")
    logger.info('='*80)

    # Argument Parsing

    parser = argparse.ArgumentParser(description="Alpha314 Parser")
    parser.add_argument('-algos', dest="algorithms", action='store', default='XGB',
                        help='algorithms for either classification or regression')
    parser.add_argument('-base', dest="base_dir", default="/Users/markconway/Projects",
                        help="base directory location")
    parser.add_argument('-cali', dest="calibration", action='store', default='isotonic',
                        help='calibration: isotonic [default] or sigmoid')
    parser.add_argument('-drop', dest="drop", action='store', default=None,
                        help='features to drop')
    parser.add_argument('-dummy', dest="dummy_limit", type=int, default=100,
                        help="maximum limit for distinct categorical values")
    parser.add_argument('-esr', dest="esr", type=int, default=30,
                        help="early stopping rounds for XGBoost")
    parser.add_argument('-ext', dest="extension", action='store', default='csv',
                        help='file extension for features and predictions')
    parser.add_argument('-fsp', dest="fsample_pct", type=int, default=10,
                        help="feature sampling percentage for interactions")
    parser.add_argument('-grid', dest="grid_search", action="store_true",
                        help="perform a grid search [False]")
    parser.add_argument('-gsub', dest="subsample", action="store_true",
                        help="use subsampling to reduce grid search time [False]")
    parser.add_argument('-gsp', dest="subsample_pct", type=float, default=0.25,
                        help="subsampling percentage for grid search")
    parser.add_argument('-label', dest="test_labels", action="store_true",
                        help="test labels are available [False]")
    parser.add_argument('-na', dest="na_fill", type=int, default=0,
                        help="fill value for NA variables [use mean]")
    parser.add_argument("-name", dest="project", default="project",
                        help="unique project name")
    parser.add_argument('-nest', dest="n_estimators", type=int, default=201,
                        help="default number of estimators [201]")
    parser.add_argument('-nfold', dest="n_folds", type=int, default=5,
                        help="number of folds for cross-validation")
    parser.add_argument('-ngen', dest="gp_learn", type=int, default=0,
                        help="number of genetic learning features")
    parser.add_argument('-ngram', dest="ngrams_max", type=int, default=1,
                        help="number of maximum ngrams for text features")
    parser.add_argument('-ngs', dest="gs_iters", type=int, default=100,
                        help="number of grid search iterations")
    parser.add_argument('-njobs', dest="n_jobs", type=int, default=-1,
                        help="number of jobs to run in parallel (-1 use all cores)")
    parser.add_argument('-nstep', dest="n_step", type=int, default=5,
                        help="step increment for recursive feature elimination")
    parser.add_argument('-plots', dest="plots", action="store_true",
                        help="show plots [False]")
    parser.add_argument('-poly', dest="poly_degree", type=int, default=0,
                        help="polynomial degree for interactions")
    parser.add_argument('-reg', dest="regression", action="store_true",
                        help="classification [default] or regression")
    parser.add_argument('-rfe', dest="rfe", action="store_true",
                        help="recursive feature elimination [False]")
    parser.add_argument('-score', dest="scorer", action='store', default='roc_auc',
                        help='scorer function')
    parser.add_argument('-seed', dest="seed", type=int, default=42,
                        help="random seed for reproducibility")
    parser.add_argument('-sep', dest="separator", action='store', default=',',
                        help='separator for input file')
    parser.add_argument('-shuf', dest="shuffle", action="store_true",
                        help="shuffle the data [False]")
    parser.add_argument('-split', dest="split", type=float, default=0.3,
                        help="percentage of data withheld for testing")
    parser.add_argument("-test", dest="test_file", default="test",
                        help="test file containing features and/or labels")
    parser.add_argument("-train", dest="train_file", default="train",
                        help="training file containing features and labels")
    parser.add_argument('-v', dest="verbosity", type=int, default=2,
                        help="verbosity level")
    parser.add_argument('-X', dest="features", action='store', default=WILDCARD,
                        help='features [default is all features]')
    parser.add_argument("-y", dest="target", action='store', default='target',
                        help="target variable [y]")

    # Print the arguments

    args = parser.parse_args()

    logger.info('PARAMETERS:')
    logger.info('algorithms      = %s', args.algorithms)
    logger.info('base_dir        = %s', args.base_dir)
    logger.info('calibration     = %s', args.calibration)
    logger.info('dummy_limit     = %d', args.dummy_limit)
    logger.info('extension       = %s', args.extension)
    logger.info('drop            = %s', args.drop)
    logger.info('esr             = %d', args.esr)
    logger.info('features [X]    = %s', args.features)
    logger.info('fsample_pct     = %d', args.fsample_pct)
    logger.info('gp_learn        = %d', args.gp_learn)
    logger.info('grid_search     = %r', args.grid_search)
    logger.info('gs_iters        = %d', args.gs_iters)
    logger.info('n_estimators    = %d', args.n_estimators)
    logger.info('n_folds         = %d', args.n_folds)
    logger.info('n_jobs          = %d', args.n_jobs)
    logger.info('n_step          = %d', args.n_step)
    logger.info('na_fill         = %d', args.na_fill)
    logger.info('ngrams_max      = %d', args.ngrams_max)
    logger.info('plots           = %r', args.plots)
    logger.info('poly_degree     = %d', args.poly_degree)
    logger.info('project         = %s', args.project)
    logger.info('regression      = %r', args.regression)
    logger.info('rfe             = %r', args.rfe)
    logger.info('scorer          = %s', args.scorer)
    logger.info('seed            = %d', args.seed)
    logger.info('separator       = %s', args.separator)
    logger.info('shuffle         = %r', args.shuffle)
    logger.info('split           = %f', args.split)
    logger.info('subsample       = %r', args.subsample)
    logger.info('subsample_pct   = %f', args.subsample_pct)
    logger.info('test_file       = %s', args.test_file)
    logger.info('test_labels     = %r', args.test_labels)
    logger.info('train_file      = %s', args.train_file)
    logger.info('target [y]      = %s', args.target)
    logger.info('verbosity       = %d', args.verbosity)

    # Debug the program

    logger.debug('\n' + '='*50 + '\n')

    # Create a model from the arguments

    logger.info("Creating Model")

    model = Model(vars(args))

    # Start the pipeline

    logger.info("Calling Pipeline")

    model = pipeline(model)

    # Complete the pipeline

    logger.info('='*80)
    logger.info("END")
    logger.info('='*80)
