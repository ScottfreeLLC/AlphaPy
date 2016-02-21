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
from estimators import get_scorers
from features import create_features
from features import drop_features
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
    categoricals = model.specs['categoricals']
    drop = model.specs['drop']
    dummy_limit = model.specs['dummy_limit']
    extension = model.specs['extension']
    features = model.specs['features']
    grid_search = model.specs['grid_search']
    interactions = model.specs['interactions']
    n_estimators = model.specs['n_estimators']
    n_jobs = model.specs['n_jobs']
    ngrams_max = model.specs['ngrams_max']
    plots = model.specs['plots']
    poly_degree = model.specs['poly_degree']
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
    text_features = model.specs['text_features']
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

    # Drop features

    logger.info("Dropping Features")

    X = drop_features(X, drop)

    # Feature Generation

    logger.info("Generating Features")
    logger.info("Number of Training Rows    : %d", X_train.shape[0])
    logger.info("Number of Training Columns : %d", X_train.shape[1])
    logger.info("Number of Testing Rows    : %d", X_test.shape[0])
    logger.info("Number of Testing Columns : %d", X_test.shape[1])

    X = create_features(X, categoricals, dummy_limit, interactions, poly_degree,
                        text_features, ngrams_max, separator)

    # Split the data back into training and test

    logger.info("Splitting Data")

    X_train, X_test = np.array_split(X, [split_point])

    # Shuffle the data if necessary

    if shuffle:
        logger.info("Shuffling Data")
        np.random.seed(seed)
        new_indices = np.random.permutation(y.size)
        X_train = X_train[new_indices]
        y_train = y_train[new_indices]

    # Save the new features in the model object

    logger.info("Saving New Features")

    model.X_train = X_train
    model.X_test = X_test
    model.y_train = y_train
    model.y_test = y_test

    # Get the available classifiers and regressors 

    logger.info("Getting All Estimators")

    estimators = get_estimators(n_estimators, seed, n_jobs, verbosity)

    # Get the available scorers

    scorers = get_scorers()
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

    logger.info("Blending Multiple Models")
    model = predict_blend(model)

    # Generate metrics

    model = generate_metrics(model, 'train')
    model = generate_metrics(model, 'test')

    # Store the best estimator

    logger.info("Selecting Best Model")
    model = predict_best(model)

    # Generate plots

    if plots:
        generate_plots(model, 'train')
        generate_plots(model, 'test')

    # Save best features and predictions

    logger.info("Saving Model")
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

    # Argument Parsing

    parser = argparse.ArgumentParser(description="Alpha314 Parser")
    parser.add_argument('-algos', dest="algorithms", action='store', default='XGB',
                        help='algorithms for either classification or regression')
    parser.add_argument('-base', dest="base_dir", default="/Users/markconway/Projects",
                        help="base directory location")
    parser.add_argument('-cali', dest="calibration", action='store', default='isotonic',
                        help='calibration: isotonic [default] or sigmoid')
    parser.add_argument('-cats', dest="categoricals", action='store', default=None,
                        help='categorical features')
    parser.add_argument('-drop', dest="drop", action='store', default=None,
                        help='features to drop')
    parser.add_argument('-dummy', dest="dummy_limit", type=int, default=100,
                        help="maximum limit for distinct categorical values")
    parser.add_argument('-ext', dest="extension", action='store', default='csv',
                        help='file extension for features and predictions')
    parser.add_argument('-grid', dest="grid_search", action="store_true",
                        help="perform a grid search [False]")
    parser.add_argument('-gsub', dest="subsample", action="store_true",
                        help="use subsampling to reduce grid search time [False]")
    parser.add_argument('-gspct', dest="subsample_pct", type=float, default=0.25,
                        help="subsampling percentage for grid search")
    parser.add_argument('-inter', dest="interactions", action="store_true",
                        help="compute feature interactions [False]")
    parser.add_argument('-label', dest="test_labels", action="store_true",
                        help="test labels are available [False]")
    parser.add_argument("-name", dest="project", default="project",
                        help="unique project name")
    parser.add_argument('-nest', dest="n_estimators", type=int, default=201,
                        help="default number of estimators [201]")
    parser.add_argument('-nfold', dest="n_folds", type=int, default=5,
                        help="number of folds for cross-validation")
    parser.add_argument('-ngram', dest="ngrams_max", type=int, default=1,
                        help="number of maximum ngrams for text features")
    parser.add_argument('-ngs', dest="gs_iters", type=int, default=200,
                        help="number of grid search iterations")
    parser.add_argument('-njobs', dest="n_jobs", type=int, default=-1,
                        help="number of jobs to run in parallel (-1 use all cores)")
    parser.add_argument('-nstep', dest="n_step", type=int, default=5,
                        help="step increment for recursive feature elimination")
    parser.add_argument('-plots', dest="plots", action="store_true",
                        help="show plots [False]")
    parser.add_argument('-polyd', dest="poly_degree", type=int, default=0,
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
    parser.add_argument('-text', dest="text_features", action='store', default=None,
                        help='text features')
    parser.add_argument("-train", dest="train_file", default="train",
                        help="training file containing features and labels")
    parser.add_argument('-v', dest="verbosity", type=int, default=0,
                        help="verbosity level")
    parser.add_argument('-X', dest="features", action='store', default=WILDCARD,
                        help='features [default is all features]')
    parser.add_argument("-y", dest="target", action='store', default='target',
                        help="target variable [y]")

    # Print the arguments

    args = parser.parse_args()
    print '\nPARAMETERS:\n'
    print 'algorithms      =', args.algorithms
    print 'base_dir        =', args.base_dir
    print 'calibration     =', args.calibration
    print 'categoricals    =', args.categoricals
    print 'dummy_limit     =', args.dummy_limit
    print 'extension       =', args.extension
    print 'drop            =', args.drop
    print 'features [X]    =', args.features
    print 'grid_search     =', args.grid_search
    print 'gs_iters        =', args.gs_iters
    print 'interactions    =', args.interactions
    print 'n_estimators    =', args.n_estimators
    print 'n_folds         =', args.n_folds
    print 'n_jobs          =', args.n_jobs
    print 'n_step          =', args.n_step
    print 'ngrams_max      =', args.ngrams_max
    print 'plots           =', args.plots
    print 'poly_degree     =', args.poly_degree
    print 'project         =', args.project
    print 'regression      =', args.regression
    print 'rfe             =', args.rfe
    print 'scorer          =', args.scorer
    print 'seed            =', args.seed
    print 'separator       =', args.separator
    print 'shuffle         =', args.shuffle
    print 'split           =', args.split
    print 'subsample       =', args.subsample
    print 'subsample_pct   =', args.subsample_pct
    print 'test_file       =', args.test_file
    print 'test_labels     =', args.test_labels
    print 'text_features   =', args.text_features
    print 'train_file      =', args.train_file
    print 'target [y]      =', args.target
    print 'verbosity       =', args.verbosity

    # Debug the program

    logger.debug('\n' + '='*50 + '\n')

    # Create a model from the arguments

    logger.info("Creating Model")

    model = Model(vars(args))

    # Call the pipeline

    logger.info("Starting Model Pipeline")

    model = pipeline(model)

    logger.info("Completed Model Pipeline")
