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
from datetime import datetime
from datetime import timedelta
from data import get_remote_data
from frame import Frame
from globs import WILDCARD
from group import Group
import logging
from model import Model
from space import Space
from system import System
from var import Variable
from var import vmapply


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function pipeline
#

def pipeline(model):
    """
    AlphaPy Stock Pipeline
    :rtype : object
    """

    # Unpack the model specifications

    base_dir = model.specs['base_dir']

    # Create default space stock_prices_1d

    space = Space()

    # Create groups for each genre of stock

    gs = Group('my', space)
    ge = Group('psp', space)
    gt = Group('tech', space)

    # Populate groups with members and other groups

    gs.add([repr(ge), repr(gt)])
    ge.add(['qqq', 'spy', 'tna', 'tza', 'nugt', 'dust', 'fas', 'faz', 'eem', 'iwm', \
            'tvix', 'vxx', 'tlt', 'tbt', 'edc', 'edz'])
    gt.add(['aapl', 'amzn', 'fb', 'goog', 'lnkd', 'nflx', 'yhoo'])

    # Display all members

    print gs.all_members()

    # Get stock data

    get_remote_data(gs, datetime.now() - timedelta(1000))

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

    vmapply(gs, features_simple)
    vmapply(gs, features_gap)
    vmapply(gs, ['sephigh'])

    # Run the analysis, including the model pipeline

    a = Analysis(m, gs)

    forecast_period = 1
    leaders = ['open', 'gap', 'gapbadown', 'gapbaup', 'gapdown', 'gapup']
    a = run_analysis(a, forecast_period, leaders)

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
    logger.info("END STOCK PIPELINE")
    logger.info('*'*80)
