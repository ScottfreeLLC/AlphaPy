##############################################################
#
# Package   : AlphaPy
# Module    : alpha_stock
# Version   : 1.0
# Copyright : Mark Conway
# Date      : September 13, 2015
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
from frame import Frame
from globs import SSEP
from globs import WILDCARD
from group import Group
import logging
from model import get_model_config
from model import Model
from space import Space
from system import System
from var import Variable
from var import vmapply
import yaml


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_stock_config
#

def get_stock_config(cfg_dir):

    logger.info("Stock Configuration")

    # Read the configuration file

    full_path = SSEP.join([cfg_dir, 'stock.yml'])
    with open(full_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Store configuration parameters in dictionary

    specs = {}

    # Section: stock [this section must be first]

    specs['forecast_period'] = cfg['stock']['forecast_period']
    specs['fractal'] = cfg['stock']['fractal']
    specs['leaders'] = cfg['stock']['leaders']
    specs['lookback_period'] = cfg['stock']['lookback_period']
    specs['predict_date'] = cfg['stock']['predict_date']
    specs['schema'] = cfg['stock']['schema']
    specs['target_group'] = cfg['stock']['target_group']
    specs['train_date'] = cfg['stock']['train_date']

    # Create the subject/schema/fractal namespace

    sspecs = ['stock', specs['schema'], specs['fractal']]    
    space = Space(*sspecs)

    # Section: groups

    try:
        for g, m in cfg['groups'].items():
            command = 'Group(\'' + g + '\', space)'
            exec(command)
            Group.groups[g].add(m)
    except:
        logger.info("No Groups Found")

    # Section: aliases

    try:
        for k, v in cfg['aliases'].items():
            Alias(k, v)
    except:
        logger.info("No Aliases Found")

    # Section: variables

    try:
        for k, v in cfg['variables'].items():
            Variable(k, v)
    except:
        logger.info("No Variables Found")

    # Log the stock parameters

    logger.info('STOCK PARAMETERS:')
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

def pipeline(model, stock_specs):
    """
    AlphaPy Stock Pipeline
    :rtype : object
    """

    # Get any model specifications

    target = model.specs['target']

    # Get any stock specifications

    forecast_period = stock_specs['forecast_period']
    leaders = stock_specs['leaders']
    lookback_period = stock_specs['lookback_period']
    predict_date = stock_specs['predict_date']
    target_group = stock_specs['target_group']
    train_date = stock_specs['train_date']

    # Set the target group

    gs = Group.groups[target_group]
    logger.info("All Members: %s", gs.members)

    # Get stock data

    get_remote_data(gs, datetime.now() - timedelta(lookback_period))

    # Define feature sets

    features_gap = ['gap', 'gapbadown', 'gapbaup', 'gapdown', 'gapup']
    features_ma = ['cma_10', 'cma_20', 'cma_50']
    features_range = ['net', 'netup', 'netdown', 'rr', 'rr_2', 'rr_3', 'rr_4', 'rr_5', \
                      'rr_6', 'rr_7', 'rrunder', 'rrover']
    features_roi = ['roi', 'roi_2', 'roi_3', 'roi_4', 'roi_5', 'roi_10', 'roi_20']
    features_sep = ['sepa', 'sepa_2', 'sepa_3', 'sepa_4', 'sepa_5', 'sepa_6', 'sepa_7', \
                    'sephigh', 'seplow', 'sepover', 'sepunder']
    features_simple = ['hc', 'hh', 'ho', 'hl', 'lc', 'lh', 'll', 'lo']
    features_trend = ['adx', 'diplus', 'diminus', 'trend', 'rsi_8', 'rsi_14', 'bigdown', 'bigup']
    features_volatility = ['atr', 'volatility', 'nr_4', 'nr_7', 'nr_10', 'wr_4', 'wr_7', 'wr_10']
    features_volume = ['vmover', 'vmunder', 'vma', 'vmratio']
    features_all = features_gap + features_ma + features_range + features_roi + \
                   features_sep + features_simple + features_trend + features_volatility + \
                   features_volume

    # Apply the features to all of the frames

    vmapply(gs, features_all)
    vmapply(gs, [target])

    # Run the analysis, including the model pipeline

    a = Analysis(model, gs, train_date, predict_date)
    results = run_analysis(a, forecast_period, leaders)

    # Create and run systems

    pass

    # ts = System('trend', 'bigup', 'bigdown')
    # run_system(ts, gs)
    # gen_portfolio(ts, gs)

    # cs = System('closer', 'hc', 'lc')
    # run_system(cs, gs)
    # gen_portfolio(cs, gs)

    # Return the completed model

    return model


#
# MAIN PROGRAM
#

if __name__ == '__main__':

    # Logging

    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="alpha314_stock.log", filemode='a', level=logging.DEBUG,
                        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # Start the pipeline

    logger.info('*'*80)
    logger.info("START STOCK PIPELINE")
    logger.info('*'*80)

    # Argument Parsing

    parser = argparse.ArgumentParser(description="Alpha314 Stock Parser")
    parser.add_argument("-d", dest="cfg_dir", default=".",
                        help="directory location of configuration file")
    args = parser.parse_args()

    # Read stock configuration file

    stock_specs = get_stock_config(args.cfg_dir)

    # Read configuration file

    specs = get_model_config(args.cfg_dir)

    # Debug the program

    logger.debug('\n' + '='*50 + '\n')

    # Create a model from the arguments

    logger.info("Creating Model")

    model = Model(specs)

    # Start the pipeline

    logger.info("Calling Pipeline")

    model = pipeline(model, stock_specs)

    # Complete the pipeline

    logger.info('*'*80)
    logger.info("END STOCK PIPELINE")
    logger.info('*'*80)
