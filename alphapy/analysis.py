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
from alphapy.frame import write_frame
from alphapy.globs import SSEP, USEP

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
    train_date : pandas.datetime, optional
        The starting date for training the model.
    predict_date : pandas.datetime, optional
        The starting date for model predictions.

    Attributes
    ----------
    Analysis.analyses : dict
        Class variable for storing all known analyses

    Raises
    ------
    ValueError
        ``predict_date`` must be later than ``train_date``.

    """

    analyses = {}

    # __new__
    
    def __new__(cls,
                model,
                group,
                train_date = pd.datetime(1900, 1, 1),
                predict_date = pd.datetime.today() - BDay(2)):
        # verify that dates are in sequence
        if train_date >= predict_date:
            raise ValueError("Training date must be before prediction date")
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
                 group,
                 train_date = pd.datetime(1900, 1, 1),
                 predict_date = pd.datetime.today() - BDay(2)):
        # set analysis name
        name = model.specs['directory'].split(SSEP)[-1]
        target = model.specs['target']
        an = analysis_name(name, target)
        # initialize analysis
        self.name = an
        self.model = model
        self.group = group
        self.train_date = train_date.strftime('%Y-%m-%d')
        self.predict_date = predict_date.strftime('%Y-%m-%d')
        self.target = target
        # add analysis to analyses list
        Analysis.analyses[an] = self
        
    # __str__

    def __str__(self):
        return self.name


#
# Function run_analysis
#

def run_analysis(analysis, forecast_period, leaders, splits=True):
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
    forecast_period : int
        The period for forecasting the target of the analysis.
    leaders : list
        The features that are contemporaneous with the target.
    splits : bool, optional
        If ``True``, then the data for each member of the analysis
        group are in separate files.

    Returns
    -------
    analysis : alphapy.Analysis
        The completed analysis.

    """

    name = analysis.name
    model = analysis.model
    group = analysis.group
    target = analysis.target
    train_date = analysis.train_date
    predict_date = analysis.predict_date

    # Unpack model data

    directory = model.specs['directory']
    extension = model.specs['extension']
    separator = model.specs['separator']
    test_file = model.specs['test_file']
    test_labels = model.specs['test_labels']
    train_file = model.specs['train_file']

    # Load the data frames

    data_frames = load_frames(group, directory, extension, separator, splits)
    if data_frames:
        # create training and test frames
        train_frame = pd.DataFrame()
        test_frame = pd.DataFrame()
        # Subset each frame and add to the model frame
        for df in data_frames:
            # shift the target for the forecast period
            if forecast_period > 0:
                df[target] = df[target].shift(-forecast_period)
            # shift any leading features if necessary
            if leaders:
                df[leaders] = df[leaders].shift(-1)
            # split data into train and test
            new_train = df.loc[(df.index >= train_date) & (df.index < predict_date)]
            if len(new_train) > 0:
                # train frame
                new_train = new_train.dropna()
                train_frame = train_frame.append(new_train)
                # test frame
                new_test = df.loc[df.index >= predict_date]
                if len(new_test) > 0:
                    if test_labels:
                        new_test = new_test.dropna()
                    test_frame = test_frame.append(new_test)
                else:
                    logger.info("A test frame has zero rows. Check prediction date.")
            else:
                logger.warning("A training frame has zero rows. Check data source.")
        # write out the training and test files
        if len(train_frame) > 0 and len(test_frame) > 0:
            directory = SSEP.join([directory, 'input'])
            write_frame(train_frame, directory, train_file, extension, separator,
                        index=True, index_label='date')
            write_frame(test_frame, directory, test_file, extension, separator,
                        index=True, index_label='date')
        # run the AlphaPy pipeline
        analysis.model = main_pipeline(model)
    else:
        # no frames found
        logger.info("No frames were found for analysis %s", name)
    # return the analysis
    return analysis
