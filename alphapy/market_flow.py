################################################################################
#
# Package   : AlphaPy
# Module    : market_flow
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

from alphapy.alias import Alias
from alphapy.analysis import Analysis
from alphapy.analysis import run_analysis
from alphapy.data import get_market_data
from alphapy.globals import PD_INTRADAY_OFFSETS
from alphapy.globals import PSEP, SSEP
from alphapy.group import Group
from alphapy.market_variables import Variable
from alphapy.market_variables import vmapply
from alphapy.model import get_model_config
from alphapy.model import Model
from alphapy.portfolio import gen_portfolio
from alphapy.space import Space
from alphapy.system import run_system
from alphapy.system import System
from alphapy.utilities import valid_date

import argparse
import datetime
import logging
import multiprocessing as mp
import os
import pandas as pd
import warnings
import yaml


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_market_config
#

def get_market_config():
    r"""Read the configuration file for MarketFlow.

    Parameters
    ----------
    None : None

    Returns
    -------
    specs : dict
        The parameters for controlling MarketFlow.

    """

    logger.info("MarketFlow Configuration")

    # Read the configuration file

    full_path = SSEP.join([PSEP, 'config', 'market.yml'])
    with open(full_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Store configuration parameters in dictionary

    specs = {}

    # Section: market [this section must be first]

    specs['create_model'] = cfg['market']['create_model']
    fractal = cfg['market']['data_fractal']
    try:
        test_interval = pd.to_timedelta(fractal)
    except:
        logger.info("data_fractal [%s] is an invalid pandas offset",
                    fractal)
    specs['data_fractal'] = fractal
    specs['data_history'] = cfg['market']['data_history']
    specs['forecast_period'] = cfg['market']['forecast_period']
    fractal = cfg['market']['fractal']
    try:
        test_interval = pd.to_timedelta(fractal)
    except:
        logger.info("fractal [%s] is an invalid pandas offset",
                    fractal)
    specs['fractal'] = fractal
    specs['lag_period'] = cfg['market']['lag_period']
    specs['leaders'] = cfg['market']['leaders']
    specs['predict_history'] = cfg['market']['predict_history']
    specs['schema'] = cfg['market']['schema']
    specs['subject'] = cfg['market']['subject']
    specs['target_group'] = cfg['market']['target_group']

    # Create the subject/schema/fractal namespace

    sspecs = [specs['subject'], specs['schema'], specs['fractal']]    
    space = Space(*sspecs)

    # Section: features

    try:
        logger.info("Getting Features")
        specs['features'] = cfg['features']
    except:
        logger.info("No Features Found")
        specs['features'] = {}

    # Section: groups

    try:
        logger.info("Defining Groups")
        for g, m in list(cfg['groups'].items()):
            Group(g, space)
            Group.groups[g].add(m)
    except:
        logger.info("No Groups Found")

    # Section: aliases

    try:
        logger.info("Defining Aliases")
        for k, v in list(cfg['aliases'].items()):
            Alias(k, v)
    except:
        logger.info("No Aliases Found")

    # Section: system

    try:
        logger.info("Getting System Parameters")
        specs['system'] = cfg['system']
    except:
        logger.info("No System Parameters Found")
        specs['system'] = {}

    # Section: variables

    logger.info("Defining AlphaPy Variables [phigh, plow]")

    Variable('phigh', 'probability >= 0.7')
    Variable('plow', 'probability <= 0.3')

    try:
        logger.info("Defining User Variables")
        for k, v in list(cfg['variables'].items()):
            Variable(k, v)
    except:
        logger.info("No Variables Found")

    # Section: functions

    try:
        logger.info("Getting Variable Functions")
        specs['functions'] = cfg['functions']
    except:
        logger.info("No Variable Functions Found")
        specs['functions'] = {}

    # Log the stock parameters

    logger.info('MARKET PARAMETERS:')
    logger.info('create_model    = %r', specs['create_model'])
    logger.info('data_fractal    = %s', specs['data_fractal'])
    logger.info('data_history    = %d', specs['data_history'])
    logger.info('features        = %s', specs['features'])
    logger.info('forecast_period = %d', specs['forecast_period'])
    logger.info('fractal         = %s', specs['fractal'])
    logger.info('lag_period      = %d', specs['lag_period'])
    logger.info('leaders         = %s', specs['leaders'])
    logger.info('predict_history = %s', specs['predict_history'])
    logger.info('schema          = %s', specs['schema'])
    logger.info('subject         = %s', specs['subject'])
    logger.info('system          = %s', specs['system'])
    logger.info('target_group    = %s', specs['target_group'])

    # Market Specifications
    return specs


#
# Function market_pipeline
#

def market_pipeline(model, market_specs):
    r"""AlphaPy MarketFlow Pipeline

    Parameters
    ----------
    model : alphapy.Model
        The model object for AlphaPy.
    market_specs : dict
        The specifications for controlling the MarketFlow pipeline.

    Returns
    -------
    model : alphapy.Model
        The final results are stored in the model object.

    Notes
    -----
    (1) Define a group.
    (2) Get the market data.
    (3) Apply system features.
    (4) Create an analysis.
    (5) Run the analysis, which calls AlphaPy.

    """

    logger.info("Running MarketFlow Pipeline")

    # Get model specifications

    predict_mode = model.specs['predict_mode']
    target = model.specs['target']

    # Get market specifications

    create_model = market_specs['create_model']
    data_fractal = market_specs['data_fractal']
    data_history = market_specs['data_history']
    features = market_specs['features']
    forecast_period = market_specs['forecast_period']
    fractal = market_specs['fractal']
    functions = market_specs['functions']
    lag_period = market_specs['lag_period']
    leaders = market_specs['leaders']
    predict_history = market_specs['predict_history']
    target_group = market_specs['target_group']

    # Set the target group

    group = Group.groups[target_group]
    logger.info("All Symbols: %s", group.members)

    # Determine whether or not this is an intraday analysis.

    intraday = any(substring in fractal for substring in PD_INTRADAY_OFFSETS)

    # Get stock data. If we can't get all the data, then
    # predict_history resets to the actual history obtained.

    lookback = predict_history if predict_mode else data_history
    new_history = get_market_data(model, group, lookback,
                                  data_fractal, intraday)
    if new_history < data_history:
        logger.info("Maximum Data History is %d, not %d",
                    new_history, data_history)
        if new_history == 0:
            raise ValueError("Could not get market data from source")

    # Run an analysis to create the model

    if create_model:
        # apply features to all of the frames
        vmapply(group, features, functions)
        vmapply(group, [target], functions)
        # run the analysis, including the model pipeline
        a = Analysis(model, group)
        results = run_analysis(a, lag_period, forecast_period,
                               leaders, predict_history)

    # Run a system

    system_specs = market_specs['system']
    if system_specs:
        # get the system specs
        system_name = system_specs['name']
        longentry = system_specs['longentry']
        shortentry = system_specs['shortentry']
        longexit = system_specs['longexit']
        shortexit = system_specs['shortexit']
        holdperiod = system_specs['holdperiod']
        scale = system_specs['scale']
        logger.info("Running System %s", system_name)
        logger.info("Long Entry  : %s", longentry)
        logger.info("Short Entry : %s", shortentry)
        logger.info("Long Exit   : %s", longexit)
        logger.info("Short Exit  : %s", shortexit)
        logger.info("Hold Period : %d", holdperiod)
        logger.info("Scale       : %r", scale)
        # create and run the system
        system = System(system_name, longentry, shortentry,
                        longexit, shortexit, holdperiod, scale)
        tfs = run_system(model, system, group, intraday)
        # generate a portfolio
        gen_portfolio(model, system_name, group, tfs)

    # Return the completed model
    return model


#
# Function main
#

def main(args=None):
    r"""MarketFlow Main Program

    Notes
    -----
    (1) Initialize logging.
    (2) Parse the command line arguments.
    (3) Get the market configuration.
    (4) Get the model configuration.
    (5) Create the model object.
    (6) Call the main MarketFlow pipeline.

    Raises
    ------
    ValueError
        Training date must be before prediction date.

    """

    # Logging

    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="market_flow.log", filemode='a', level=logging.DEBUG,
                        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # Start the pipeline

    logger.info('*'*80)
    logger.info("MarketFlow Start")
    logger.info('*'*80)

    # Argument Parsing

    parser = argparse.ArgumentParser(description="MarketFlow Parser")
    parser.add_argument('--pdate', dest='predict_date',
                        help="prediction date is in the format: YYYY-MM-DD",
                        required=False, type=valid_date)
    parser.add_argument('--tdate', dest='train_date',
                        help="training date is in the format: YYYY-MM-DD",
                        required=False, type=valid_date)
    parser.add_mutually_exclusive_group(required=False)
    parser.add_argument('--predict', dest='predict_mode', action='store_true')
    parser.add_argument('--train', dest='predict_mode', action='store_false')
    parser.set_defaults(predict_mode=False)
    args = parser.parse_args()

    # Set train and predict dates

    if args.train_date:
        train_date = args.train_date
    else:
        train_date = pd.datetime(1900, 1, 1).strftime("%Y-%m-%d")

    if args.predict_date:
        predict_date = args.predict_date
    else:
        predict_date = datetime.date.today().strftime("%Y-%m-%d")

    # Verify that the dates are in sequence.

    if train_date >= predict_date:
        raise ValueError("Training date must be before prediction date")
    else:
        logger.info("Training Date: %s", train_date)
        logger.info("Prediction Date: %s", predict_date)

    # Read stock configuration file
    market_specs = get_market_config()

    # Read model configuration file

    model_specs = get_model_config()
    model_specs['predict_mode'] = args.predict_mode
    model_specs['predict_date'] = predict_date
    model_specs['train_date'] = train_date

    # Create directories if necessary

    output_dirs = ['config', 'data', 'input', 'model', 'output', 'plots', 'systems']
    for od in output_dirs:
        output_dir = SSEP.join([model_specs['directory'], od])
        if not os.path.exists(output_dir):
            logger.info("Creating directory %s", output_dir)
            os.makedirs(output_dir)

    # Create a model from the arguments

    logger.info("Creating Model")
    model = Model(model_specs)

    # Start the pipeline
    model = market_pipeline(model, market_specs)

    # Complete the pipeline

    logger.info('*'*80)
    logger.info("MarketFlow End")
    logger.info('*'*80)


#
# MAIN PROGRAM
#

if __name__ == "__main__":
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    mp.set_start_method('forkserver')
    main()
