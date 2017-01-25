##############################################################
#
# Package   : AlphaPy
# Module    : alpha_market
# Version   : 1.0
# Copyright : Mark Conway
# Date      : September 13, 2015
#
# python ../AlphaPy/alpha_market.py -d 'Stocks/config'
#
##############################################################


#
# Imports
#

from alias import Alias
from alpha import pipeline
from analysis import Analysis
from analysis import run_analysis
import argparse
from datetime import datetime
from datetime import timedelta
from data import get_remote_data
from frame import dump_frames
from globs import SSEP
from group import Group
import logging
from model import get_model_config
from model import Model
from portfolio import gen_portfolio
from space import Space
from system import System
from system import run_system
from var import Variable
from var import vmapply
import yaml


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_market_config
#

def get_market_config(cfg_dir):

    logger.info("Market Configuration")

    # Read the configuration file

    full_path = SSEP.join([cfg_dir, 'market.yml'])
    with open(full_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Store configuration parameters in dictionary

    specs = {}

    # Section: market [this section must be first]

    specs['forecast_period'] = cfg['market']['forecast_period']
    specs['fractal'] = cfg['market']['fractal']
    specs['leaders'] = cfg['market']['leaders']
    specs['lookback_period'] = cfg['market']['lookback_period']
    specs['predict_date'] = cfg['market']['predict_date']
    specs['schema'] = cfg['market']['schema']
    specs['target_group'] = cfg['market']['target_group']
    specs['train_date'] = cfg['market']['train_date']

    # Create the subject/schema/fractal namespace

    sspecs = ['market', specs['schema'], specs['fractal']]    
    space = Space(*sspecs)

    # Section: features

    try:
        logger.info("Getting Features")
        specs['features'] = cfg['features']
    except:
        logger.info("No Features Found")

    # Section: groups

    try:
        logger.info("Defining Groups")
        for g, m in cfg['groups'].items():
            command = 'Group(\'' + g + '\', space)'
            exec(command)
            Group.groups[g].add(m)
    except:
        logger.info("No Groups Found")

    # Section: aliases

    try:
        logger.info("Defining Aliases")
        for k, v in cfg['aliases'].items():
            Alias(k, v)
    except:
        logger.info("No Aliases Found")

    # Section: variables

    try:
        logger.info("Defining Variables")
        for k, v in cfg['variables'].items():
            Variable(k, v)
    except:
        logger.info("No Variables Found")

    # Log the stock parameters

    logger.info('MARKET PARAMETERS:')
    logger.info('features        = %s', specs['features'])
    logger.info('forecast_period = %d', specs['forecast_period'])
    logger.info('fractal         = %s', specs['fractal'])
    logger.info('leaders         = %s', specs['leaders'])
    logger.info('lookback_period = %d', specs['lookback_period'])
    logger.info('predict_date    = %s', specs['predict_date'])
    logger.info('schema          = %s', specs['schema'])
    logger.info('target_group    = %s', specs['target_group'])
    logger.info('train_date      = %s', specs['train_date'])

    # Stock Specifications

    return specs


#
# Function pipeline
#

def pipeline(model, market_specs):
    """
    AlphaPy Market Pipeline
    :rtype : object
    """

    # Get any model specifications

    target = model.specs['target']
    base_dir = model.specs['base_dir']
    extension = model.specs['extension']
    project = model.specs['project']
    separator = model.specs['separator']
    directory = SSEP.join([base_dir, project, 'data'])

    # Get any market specifications

    features = market_specs['features']
    forecast_period = market_specs['forecast_period']
    leaders = market_specs['leaders']
    lookback_period = market_specs['lookback_period']
    predict_date = market_specs['predict_date']
    target_group = market_specs['target_group']
    train_date = market_specs['train_date']

    # Set the target group

    gs = Group.groups[target_group]
    logger.info("All Members: %s", gs.members)

    # Get stock data

    time_frame = datetime.now() - timedelta(lookback_period)
    get_remote_data(gs, time_frame)

    # Apply the features to all of the frames

    vmapply(gs, features)
    vmapply(gs, [target])

    # Save the frames with all the new features

    # dump_frames(gs, directory, extension, separator)

    # Run the analysis, including the model pipeline

    a = Analysis(model, gs, train_date, predict_date)
    results = run_analysis(a, forecast_period, leaders)

    # Create and run systems

    # ts = System('trend', 'bigup', 'bigdown')
    # run_system(ts, gs)
    # gen_portfolio(ts, gs)

    # cs = System('closer', 'hc', 'lc')
    # tfs = run_system(model, cs, gs)
    # gen_portfolio(model, cs, gs, tfs)

    # Return the completed model

    return model


#
# MAIN PROGRAM
#

if __name__ == '__main__':

    # Logging

    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="alpha_market.log", filemode='a', level=logging.DEBUG,
                        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # Start the pipeline

    logger.info('*'*80)
    logger.info("START MARKET PIPELINE")
    logger.info('*'*80)

    # Argument Parsing

    parser = argparse.ArgumentParser(description="AlphaPy Market Parser")
    parser.add_argument("-d", dest="cfg_dir", default=".",
                        help="directory location of configuration file")
    args = parser.parse_args()

    # Read stock configuration file

    market_specs = get_market_config(args.cfg_dir)

    # Read model configuration file

    model_specs = get_model_config(args.cfg_dir)

    # Debug the program

    logger.debug('\n' + '='*50 + '\n')

    # Create a model from the arguments

    logger.info("Creating Model")

    model = Model(model_specs)

    # Start the pipeline

    logger.info("Calling Pipeline")

    model = pipeline(model, market_specs)

    # Complete the pipeline

    logger.info('*'*80)
    logger.info("END MARKET PIPELINE")
    logger.info('*'*80)
