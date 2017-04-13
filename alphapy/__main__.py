################################################################################
#
# Package   : AlphaPy
# Module    : __main__
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

from alphapy.data import get_data
from alphapy.data import sample_data
from alphapy.data import shuffle_data
from alphapy.estimators import get_estimators
from alphapy.estimators import scorers
from alphapy.features import create_features
from alphapy.features import create_interactions
from alphapy.features import drop_features
from alphapy.features import remove_lv_features
from alphapy.features import save_features
from alphapy.features import select_features
from alphapy.globals import CSEP, PSEP, SSEP
from alphapy.globals import ModelType
from alphapy.globals import Partition, datasets
from alphapy.globals import WILDCARD
from alphapy.model import first_fit
from alphapy.model import generate_metrics
from alphapy.model import get_model_config
from alphapy.model import get_class_weights
from alphapy.model import load_predictor
from alphapy.model import make_predictions
from alphapy.model import Model
from alphapy.model import predict_best
from alphapy.model import predict_blend
from alphapy.model import save_model
from alphapy.optimize import hyper_grid_search
from alphapy.optimize import rfe_search
from alphapy.optimize import rfecv_search
from alphapy.plots import generate_plots

import argparse
import logging
import numpy as np
import pandas as pd


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function data_pipeline
#

def data_pipeline(model):
    r"""AlphaPy Data Pipeline

    Parameters
    ----------
    model : alphapy.Model
        The initial model specifications for locating the data.

    Returns
    -------
    model : alphapy.Model
        All of the new features will be contained in the model object.

    Raises
    ------
    IndexError
        If the number of columns of the train and test data do not match,
        then this exception is raised.

    Notes
    -----
    The data are loaded, the features are processed, and the model
    object is updated for either the training or prediction stage.

    """

    logger.info("Data Pipeline")

    # Unpack the model specifications

    drop = model.specs['drop']
    model_type = model.specs['model_type']
    target = model.specs['target']

    # Get train and test data

    X_train, y_train = get_data(model, Partition.train)
    X_test, y_test = get_data(model, Partition.test)
    if y_test:
        model.test_labels = True

    # Drop features

    logger.info("Dropping Features: %s", drop)
    X_train = drop_features(X_train, drop)
    X_test = drop_features(X_test, drop)

    # Log feature statistics

    logger.info("Original Feature Statistics")
    logger.info("Number of Training Rows    : %d", X_train.shape[0])
    logger.info("Number of Training Columns : %d", X_train.shape[1])
    if model_type == ModelType.classification:
        uv, uc = np.unique(y_train, return_counts=True)
        logger.info("Unique Training Values for %s : %s", target, uv)
        logger.info("Unique Training Counts for %s : %s", target, uc)
    logger.info("Number of Testing Rows     : %d", X_test.shape[0])
    logger.info("Number of Testing Columns  : %d", X_test.shape[1])
    if model_type == ModelType.classification and model.test_labels:
        uv, uc = np.unique(y_test, return_counts=True)
        logger.info("Unique Testing Values for %s : %s", target, uv)
        logger.info("Unique Testing Counts for %s : %s", target, uc)

    # Merge training and test data

    if X_train.shape[1] == X_test.shape[1]:
        split_point = X_train.shape[0]
        X = pd.concat([X_train, X_test])
    else:
        raise IndexError("The number of training and test columns [%s, %s] must match.",
                         X_train.shape[1], X_test.shape[1])

    # Create initial features

    new_features = create_features(X, model, split_point, y_train)
    X_train, X_test = np.array_split(new_features, [split_point])
    model = save_features(model, X_train, X_test, y_train, y_test)

    # Generate interactions

    all_features = create_interactions(new_features, model)
    X_train, X_test = np.array_split(all_features, [split_point])
    model = save_features(model, X_train, X_test)

    # Remove low-variance features

    sig_features = remove_lv_features(all_features)

    # Save the model's feature set

    X_train, X_test = np.array_split(sig_features, [split_point])
    model = save_features(model, X_train, X_test)

    # Return the model
    return model


#
# Function model_pipeline
#

def model_pipeline(model):
    r"""AlphaPy Model Pipeline

    Parameters
    ----------
    model : alphapy.Model
        The model object for controlling the pipeline.

    Returns
    -------
    model : alphapy.Model
        The final results are stored in the model object.

    Raises
    ------
    KeyError
        If the number of columns of the train and test data do not match,
        then this exception is raised.

    """

    logger.info("Model Pipeline")

    # Unpack the model specifications

    calibration = model.specs['calibration']
    feature_selection = model.specs['feature_selection']
    grid_search = model.specs['grid_search']
    model_type = model.specs['model_type']
    rfe = model.specs['rfe']
    sampling = model.specs['sampling']
    scorer = model.specs['scorer']
    test_labels = model.test_labels

    # Shuffle the data [if specified]

    model = shuffle_data(model)

    # Oversampling or Undersampling [if specified]

    if model_type == ModelType.classification:
        if sampling:
            model = sample_data(model)
        else:
            logger.info("Skipping Sampling")
        # Get sample weights (classification only)
        model = get_class_weights(model)

    # Get the available classifiers and regressors 

    logger.info("Getting All Estimators")
    estimators = get_estimators(model)

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
        # feature selection
        if feature_selection:
            model = select_features(model)
        # initial fit
        model = first_fit(model, algo, est)
        # recursive feature elimination
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
        # predictions
        model = make_predictions(model, algo, calibration)

    # Create a blended estimator

    if len(model.algolist) > 1:
        model = predict_blend(model)

    # Generate metrics

    model = generate_metrics(model, Partition.train)
    model = generate_metrics(model, Partition.test)

    # Store the best estimator

    model = predict_best(model)

    # Generate plots

    generate_plots(model, Partition.train)
    if test_labels:
        generate_plots(model, Partition.test)

    # Save best features and predictions

    if test_labels:
        save_model(model, 'BEST', Partition.test)
    else:
        save_model(model, 'BEST', Partition.train)

    # Return the model
    return model


#
# Function predict_with
#

def predict_with(model):
    r"""AlphaPy Model Prediction

    Parameters
    ----------
    model : alphapy.Model
        The initial model specifications for locating the data.

    Returns
    -------
    None : None

    Notes
    -----
    The saved model is loaded from disk, and predictions are made
    on the new testing data.

    """

    logger.info("Predict Mode")

    # Unpack the model data

    X_test = model.X_test

    # Unpack the model specifications

    directory = model.specs['directory']
    model_type = model.specs['model_type']

    # Load model object

    predictor = load_model_object(directory)

    # Make predictions
    
    preds = predictor.predict(X_test)
    logger.info("Predictions: %s", preds)
    if model_type == ModelType.classification:
        probas = predictor.predict_proba(X_test)[:, 1]
        logger.info("Probabilities: %s", probas)


#
# Function main_pipeline
#

def main_pipeline(model):
    r"""AlphaPy Main Pipeline

    Parameters
    ----------
    model : alphapy.Model
        The model specifications for training, testing, and prediction.

    Returns
    -------
    model : alphapy.Model
        The trained model.

    """

    # Extract any model specifications

    predict_mode = model.specs['predict_mode']

    # Call the data pipeline

    model = data_pipeline(model, predict_mode)

    # Scoring Only or Calibration

    if predict_mode:
        predict_with(model)
    else:
        model = model_pipeline(model)

    # Return the completed model
    return model


#
# Function main
#

def main(args=None):
    r"""AlphaPy Main Program

    Notes
    -----
    (1) Initialize logging.
    (2) Parse the command line arguments.
    (3) Get the model configuration.
    (4) Create the model object.
    (5) Call the main AlphaPy pipeline.

    """

    # Logging

    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="alphapy.log", filemode='a', level=logging.DEBUG,
                        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # Start the pipeline

    logger.info('*'*80)
    logger.info("AlphaPy Start")
    logger.info('*'*80)

    # Argument Parsing

    parser = argparse.ArgumentParser(description="AlphaPy Parser")
    parser.add_mutually_exclusive_group(required=False)
    parser.add_argument('--predict', dest='predict_mode', action='store_true')
    parser.add_argument('--train', dest='predict_mode', action='store_false')
    parser.set_defaults(predict_mode=False)
    args = parser.parse_args()

    # Read configuration file

    specs = get_model_config()
    specs['predict_mode'] = args.predict_mode

    # Create a model from the arguments

    logger.info("Creating Model")
    model = Model(specs)

    # Start the pipeline

    logger.info("Calling Pipeline")
    model = main_pipeline(model)

    # Complete the pipeline

    logger.info('*'*80)
    logger.info("AlphaPy End")
    logger.info('*'*80)


#
# MAIN PROGRAM
#

if __name__ == "__main__":
    main()
