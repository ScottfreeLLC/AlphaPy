##############################################################
#
# Package   : AlphaPy
# Module    : analysis
# Version   : 1.0
# Copyright : Mark Conway
# Date      : June 29, 2013
#
##############################################################


#
# Imports
#

from alpha import pipeline
import datetime
from datetime import date
from frame import load_frames
from frame import write_frame
from globs import SSEP
from globs import USEP
import logging
import pandas as pd


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function analysis_name
#

def analysis_name(gname, target):
    return USEP.join([gname, target])


#
# Class Analysis
#

class Analysis(object):

    # class variable to track all analyses

    analyses = {}

    # __new__
    
    def __new__(cls,
                model,
                group,
                train_date = datetime.date(1900, 1, 1),
                predict_date = date.today()):
        # verify that dates are in sequence
        if train_date >= predict_date:
            raise ValueError("Training date must be before prediction date")
        # set analysis name
        name = model.specs['project']
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
                 train_date = datetime.date(1900, 1, 1),
                 predict_date = date.today()):
        # set analysis name
        name = model.specs['project']
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
    """
    Run an analysis for a given model and group
    """
    name = analysis.name
    model = analysis.model
    group = analysis.group
    target = analysis.target
    train_date = analysis.train_date
    predict_date = analysis.predict_date

    base_dir = model.specs['base_dir']
    extension = model.specs['extension']
    project = model.specs['project']
    separator = model.specs['separator']
    test_file = model.specs['test_file']
    train_file = model.specs['train_file']

    # Load the data frames

    directory = SSEP.join([base_dir, project])
    data_frames = load_frames(group, directory, extension,
                              separator, splits)
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
            # drop any rows with NA
            df.dropna(inplace=True)
            # split data into train and test
            new_train_frame = df.loc[(df.date >= train_date) & (df.date < predict_date)]
            if len(new_train_frame) > 0:
                train_frame = train_frame.append(new_train_frame)
            else:
                logger.info("Training frame has length 0")
            new_test_frame = df.loc[df.date >= predict_date]
            if len(new_test_frame) > 0:
                test_frame = test_frame.append(new_test_frame)
            else:
                logger.info("Test frame has length 0")
        # write out the training and test files
        write_frame(train_frame, directory, train_file, extension, separator)
        write_frame(test_frame, directory, test_file, extension, separator)
        # run the model pipeline
        analysis.model = pipeline(model)
    else:
        # no frames found
        logger.info("No frames were found for analysis %s", name)
    # return the analysis
    return analysis
