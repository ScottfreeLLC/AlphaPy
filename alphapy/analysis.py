################################################################################
#
# Package   : AlphaPy
# Module    : analysis
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

from alphapy.__main__ import main_pipeline
from alphapy.frame import load_frames
from alphapy.frame import sequence_frame
from alphapy.frame import write_frame
from alphapy.globals import SSEP, TAG_ID, USEP
from alphapy.utilities import subtract_days

from datetime import timedelta
import logging
import pandas as pd
from pandas.tseries.offsets import BDay


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function analysis_name
#

def analysis_name(gname, target):
    r"""Get the name of the analysis.

    Parameters
    ----------
    gname : str
        Group name.
    target : str
        Target of the analysis.

    Returns
    -------
    name : str
        Value for the corresponding key.

    """
    name = USEP.join([gname, target])
    return name


#
# Class Analysis
#

class Analysis(object):
    """Create a new analysis for a group. All analyses are stored
    in ``Analysis.analyses``. Duplicate keys are not allowed.

    Parameters
    ----------
    model : alphapy.Model
        Model object for the analysis.
    group : alphapy.Group
        The group of members in the analysis.

    Attributes
    ----------
    Analysis.analyses : dict
        Class variable for storing all known analyses

    """

    analyses = {}

    # __new__
    
    def __new__(cls,
                model,
                group):
        # set analysis name
        name = model.specs['directory'].split(SSEP)[-1]
        target = model.specs['target']
        an = analysis_name(name, target)
        if not an in Analysis.analyses:
            return super(Analysis, cls).__new__(cls)
        else:
            logger.info("Analysis %s already exists", an)

    # function __init__

    def __init__(self,
                 model,
                 group):
        # set analysis name
        name = model.specs['directory'].split(SSEP)[-1]
        target = model.specs['target']
        an = analysis_name(name, target)
        # initialize analysis
        self.name = an
        self.model = model
        self.group = group
        # add analysis to analyses list
        Analysis.analyses[an] = self
        
    # __str__

    def __str__(self):
        return self.name


#
# Function run_analysis
#

def run_analysis(analysis, lag_period, forecast_period, leaders,
                 predict_history, splits=True):
    r"""Run an analysis for a given model and group.

    First, the data are loaded for each member of the analysis group.
    Then, the target value is lagged for the ``forecast_period``, and
    any ``leaders`` are lagged as well. Each frame is split along
    the ``predict_date`` from the ``analysis``, and finally the
    train and test files are generated.

    Parameters
    ----------
    analysis : alphapy.Analysis
        The analysis to run.
    lag_period : int
        The number of lagged features for the analysis.
    forecast_period : int
        The period for forecasting the target of the analysis.
    leaders : list
        The features that are contemporaneous with the target.
    predict_history : int
        The number of periods required for lookback calculations.
    splits : bool, optional
        If ``True``, then the data for each member of the analysis
        group are in separate files.

    Returns
    -------
    analysis : alphapy.Analysis
        The completed analysis.

    """

    # Unpack analysis

    name = analysis.name
    model = analysis.model
    group = analysis.group

    # Unpack model data

    predict_file = model.predict_file
    test_file = model.test_file
    train_file = model.train_file

    # Unpack model specifications

    directory = model.specs['directory']
    extension = model.specs['extension']
    predict_date = model.specs['predict_date']
    predict_mode = model.specs['predict_mode']
    separator = model.specs['separator']
    target = model.specs['target']
    train_date = model.specs['train_date']

    # Calculate split date
    logger.info("Analysis Dates")
    split_date = subtract_days(predict_date, predict_history)
    logger.info("Train Date: %s", train_date)
    logger.info("Split Date: %s", split_date)
    logger.info("Test  Date: %s", predict_date)

    # Load the data frames
    data_frames = load_frames(group, directory, extension, separator, splits)

    # Create dataframes

    if predict_mode:
        # create predict frame
        predict_frame = pd.DataFrame()
    else:
        # create train and test frames
        train_frame = pd.DataFrame()
        test_frame = pd.DataFrame()

    # Subset each individual frame and add to the master frame

    for df in data_frames:
        try:
            tag = df[TAG_ID].unique()[0]
        except:
            tag = 'Unknown'
        first_date = df.index[0]
        last_date = df.index[-1]
        logger.info("Analyzing %s from %s to %s", tag, first_date, last_date)
        # sequence leaders, laggards, and target(s)
        df = sequence_frame(df, target, leaders, lag_period, forecast_period,
                            exclude_cols=[TAG_ID])
        # get frame subsets
        if predict_mode:
            new_predict = df.loc[(df.index >= split_date) & (df.index <= last_date)]
            if len(new_predict) > 0:
                predict_frame = predict_frame.append(new_predict)
            else:
                logger.info("Prediction frame %s has zero rows. Check prediction date.",
                            tag)
        else:
            # split data into train and test
            new_train = df.loc[(df.index >= train_date) & (df.index < split_date)]
            if len(new_train) > 0:
                new_train = new_train.dropna()
                train_frame = train_frame.append(new_train)
                new_test = df.loc[(df.index >= split_date) & (df.index <= last_date)]
                if len(new_test) > 0:
                    # check if target column has NaN values
                    nan_count = df[target].isnull().sum()
                    forecast_check = forecast_period - 1
                    if nan_count != forecast_check:
                        logger.info("%s has %d records with NaN targets", tag, nan_count)
                    # drop records with NaN values in target column
                    new_test = new_test.dropna(subset=[target])
                    # append selected records to the test frame
                    test_frame = test_frame.append(new_test)
                else:
                    logger.info("Testing frame %s has zero rows. Check prediction date.",
                                tag)
            else:
                logger.info("Training frame %s has zero rows. Check data source.", tag)

    # Write out the frames for input into the AlphaPy pipeline

    directory = SSEP.join([directory, 'input'])
    if predict_mode:
        # write out the predict frame
        write_frame(predict_frame, directory, predict_file, extension, separator,
                    index=True, index_label='date')
    else:
        # write out the train and test frames
        write_frame(train_frame, directory, train_file, extension, separator,
                    index=True, index_label='date')
        write_frame(test_frame, directory, test_file, extension, separator,
                    index=True, index_label='date')

    # Run the AlphaPy pipeline
    analysis.model = main_pipeline(model)

    # Return the analysis
    return analysis
