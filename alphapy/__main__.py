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
from alphapy.features import apply_treatments
from alphapy.features import create_crosstabs
from alphapy.features import create_features
from alphapy.features import create_interactions
from alphapy.features import drop_features
from alphapy.features import remove_lv_features
from alphapy.features import save_features
from alphapy.features import select_features
from alphapy.frame import write_frame
from alphapy.globals import CSEP, PSEP, SSEP, USEP
from alphapy.globals import ModelType
from alphapy.globals import Partition, datasets
from alphapy.globals import WILDCARD
from alphapy.model import first_fit
from alphapy.model import generate_metrics
from alphapy.model import get_model_config
from alphapy.model import load_feature_map
from alphapy.model import load_predictor
from alphapy.model import make_predictions
from alphapy.model import Model
from alphapy.model import predict_best
from alphapy.model import predict_blend
from alphapy.model import save_model
from alphapy.model import save_predictions
from alphapy.optimize import hyper_grid_search
from alphapy.optimize import rfecv_search
from alphapy.plots import generate_plots
from alphapy.utilities import get_datestamp

import argparse
from datetime import datetime
import logging
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import warnings


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function training_pipeline
#

def training_pipeline(model):
    r"""AlphaPy Training Pipeline

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

    logger.info("Training Pipeline")

    # Unpack the model specifications

    calibration = model.specs['calibration']
    directory = model.specs['directory']
    drop = model.specs['drop']
    extension = model.specs['extension']
    feature_selection = model.specs['feature_selection']
    grid_search = model.specs['grid_search']
    model_type = model.specs['model_type']
    predict_mode = model.specs['predict_mode']
    rfe = model.specs['rfe']
    sampling = model.specs['sampling']
    scorer = model.specs['scorer']
    separator = model.specs['separator']
    target = model.specs['target']

    # Get train and test data

    X_train, y_train = get_data(model, Partition.train)
    X_test, y_test = get_data(model, Partition.test)

    # Determine if there are any test labels

    if y_test.any():
        logger.info("Test Labels Found")
        model.test_labels = True
    model = save_features(model, X_train, X_test, y_train, y_test)

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
        raise IndexError("The number of training and test columns [%d, %d] must match." %
                         (X_train.shape[1], X_test.shape[1]))

    # Apply treatments to the feature matrix
    all_features = apply_treatments(model, X)

    # Drop features
    all_features = drop_features(all_features, drop)

    # Save the train and test files with extracted and dropped features

    datestamp = get_datestamp()
    data_dir = SSEP.join([directory, 'input'])
    df_train = all_features.iloc[:split_point, :]
    df_train = pd.concat([df_train, pd.DataFrame(y_train, columns=[target])], axis=1)
    output_file = USEP.join([model.train_file, datestamp])
    write_frame(df_train, data_dir, output_file, extension, separator)
    df_test = all_features.iloc[split_point:, :]
    if y_test.any():
        df_test = pd.concat([df_test, pd.DataFrame(y_test, columns=[target])], axis=1)
    output_file = USEP.join([model.test_file, datestamp])
    write_frame(df_test, data_dir, output_file, extension, separator)

    # Create crosstabs for any categorical features

    if model_type == ModelType.classification:
        create_crosstabs(model)

    # Create initial features

    all_features = create_features(model, all_features)
    X_train, X_test = np.array_split(all_features, [split_point])
    model = save_features(model, X_train, X_test)

    # Generate interactions

    all_features = create_interactions(model, all_features)
    X_train, X_test = np.array_split(all_features, [split_point])
    model = save_features(model, X_train, X_test)

    # Remove low-variance features

    all_features = remove_lv_features(model, all_features)
    X_train, X_test = np.array_split(all_features, [split_point])
    model = save_features(model, X_train, X_test)

    # Shuffle the data [if specified]
    model = shuffle_data(model)

    # Oversampling or Undersampling [if specified]

    if model_type == ModelType.classification:
        if sampling:
            model = sample_data(model)
        else:
            logger.info("Skipping Sampling")

    # Perform feature selection, independent of algorithm

    if feature_selection:
        model = select_features(model)

    # Get the available classifiers and regressors 

    logger.info("Getting All Estimators")
    estimators = get_estimators(model)

    # Get the available scorers

    if scorer not in scorers:
        raise KeyError("Scorer function %s not found" % scorer)

    # Model Selection

    logger.info("Selecting Models")

    for algo in model.algolist:
        logger.info("Algorithm: %s", algo)
        # select estimator
        try:
            estimator = estimators[algo]
            est = estimator.estimator
        except KeyError:
            logger.info("Algorithm %s not found", algo)
        # initial fit
        model = first_fit(model, algo, est)
        # recursive feature elimination
        if rfe:
            has_coef = hasattr(est, "coef_")
            has_fimp = hasattr(est, "feature_importances_")
            if has_coef or has_fimp:
                model = rfecv_search(model, algo)
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
    if model.test_labels:
        generate_plots(model, Partition.test)

    # Save best features and predictions
    save_model(model, 'BEST', Partition.test)

    # Return the model
    return model


#
# Function prediction_pipeline
#

def prediction_pipeline(model):
    r"""AlphaPy Prediction Pipeline

    Parameters
    ----------
    model : alphapy.Model
        The model object for controlling the pipeline.

    Returns
    -------
    None : None

    Notes
    -----
    The saved model is loaded from disk, and predictions are made
    on the new testing data.

    """

    logger.info("Predict Mode")

    # Unpack the model specifications

    directory = model.specs['directory']
    drop = model.specs['drop']
    extension = model.specs['extension']
    feature_selection = model.specs['feature_selection']
    model_type = model.specs['model_type']
    rfe = model.specs['rfe']
    separator = model.specs['separator']

    # Get all data. We need original train and test for interactions.

    partition = Partition.predict
    X_predict, _ = get_data(model, partition)

    # Load feature_map
    model = load_feature_map(model, directory)

    # Log feature statistics

    logger.info("Feature Statistics")
    logger.info("Number of Prediction Rows    : %d", X_predict.shape[0])
    logger.info("Number of Prediction Columns : %d", X_predict.shape[1])

    # Apply treatments to the feature matrix
    all_features = apply_treatments(model, X_predict)

    # Drop features
    all_features = drop_features(all_features, drop)

    # Create initial features
    all_features = create_features(model, all_features)

    # Generate interactions
    all_features = create_interactions(model, all_features)

    # Remove low-variance features
    all_features = remove_lv_features(model, all_features)

    # Load the univariate support vector, if any

    if feature_selection:
        logger.info("Getting Univariate Support")
        try:
            support = model.feature_map['uni_support']
            all_features = all_features[:, support]
            logger.info("New Feature Count : %d", all_features.shape[1])
        except:
            logger.info("No Univariate Support")

    # Load the RFE support vector, if any

    if rfe:
        logger.info("Getting RFE Support")
        try:
            support = model.feature_map['rfe_support']
            all_features = all_features[:, support]
            logger.info("New Feature Count : %d", all_features.shape[1])
        except:
            logger.info("No RFE Support")

    # Load predictor
    predictor = load_predictor(directory)

    # Make predictions
    
    logger.info("Making Predictions")
    tag = 'BEST'
    model.preds[(tag, partition)] = predictor.predict(all_features)
    if model_type == ModelType.classification:
        model.probas[(tag, partition)]  = predictor.predict_proba(all_features)[:, 1]

    # Get date stamp to record file creation

    d = datetime.now()
    f = "%Y%m%d"
    timestamp = d.strftime(f)

    # Save predictions
    save_predictions(model, tag, partition)


#
# Function main_pipeline
#

def main_pipeline(model):
    r"""AlphaPy Main Pipeline

    Parameters
    ----------
    model : alphapy.Model
        The model specifications for the pipeline.

    Returns
    -------
    model : alphapy.Model
        The final model.

    """

    # Extract any model specifications
    predict_mode = model.specs['predict_mode']

    # Prediction Only or Calibration

    if predict_mode:
        model = prediction_pipeline(model)
    else:
        model = training_pipeline(model)

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

    # Create directories if necessary

    output_dirs = ['config', 'data', 'input', 'model', 'output', 'plots']
    for od in output_dirs:
        output_dir = SSEP.join([specs['directory'], od])
        if not os.path.exists(output_dir):
            logger.info("Creating directory %s", output_dir)
            os.makedirs(output_dir)

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
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    mp.set_start_method('forkserver')
    main()
