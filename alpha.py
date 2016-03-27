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
from estimators import ModelType
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
from model import get_model_config
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
import yaml


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
    calibration = model.specs['calibration']
    drop = model.specs['drop']
    extension = model.specs['extension']
    features = model.specs['features']
    grid_search = model.specs['grid_search']
    n_estimators = model.specs['n_estimators']
    n_jobs = model.specs['n_jobs']
    project = model.specs['project']
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

    new_features = create_features(X, model, X_train, y_train)
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
            else:
                logger.info("No RFE Available for %s", algo)
        # grid search
        if grid_search:
            model = hyper_grid_search(model, estimator)
        # calibration
        if calibration:
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

    # Start the pipeline

    logger.info('*'*80)
    logger.info("START PIPELINE")
    logger.info('*'*80)

    # Argument Parsing

    parser = argparse.ArgumentParser(description="Alpha314 Parser")
    parser.add_argument("-d", dest="cfg_dir", default=".",
                        help="directory location of configuration file")
    args = parser.parse_args()

    # Read configuration file

    specs = get_model_config(args.cfg_dir)

    # Debug the program

    logger.debug('\n' + '='*50 + '\n')

    # Create a model from the arguments

    logger.info("Creating Model")

    model = Model(specs)

    # Start the pipeline

    logger.info("Calling Pipeline")

    model = pipeline(model)

    # Complete the pipeline

    logger.info('*'*80)
    logger.info("END PIPELINE")
    logger.info('*'*80)
