################################################################################
#
# Package   : AlphaPy
# Module    : alpha_market
# Created   : July 11, 2013
#
# Copyright 2017 @ Alpha314
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
# Example: python alpha_market.py -d 'Stocks/config'
#
################################################################################


#
# Imports
#

from alphapy.alias import Alias
from alphapy.analysis import Analysis
from alphapy.analysis import run_analysis
from alphapy.config import get_market_config
from alphapy.config import get_model_config
from alphapy.data import get_feed_data
from alphapy.globs import SSEP
from alphapy.group import Group
from alphapy.model import get_model_config
from alphapy.model import Model
from alphapy.space import Space
from alphapy.var import Variable
from alphapy.var import vmapply

import argparse
import logging


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function market_pipeline
#

def market_pipeline(model, market_specs):
    """
    AlphaPy Market Pipeline
    :rtype : object
    """

    # Get any model specifications

    target = model.specs['target']

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

    get_feed_data(gs, lookback_period)

    # Apply the features to all of the frames

    vmapply(gs, features)
    vmapply(gs, [target])

    # Run the analysis, including the model pipeline

    a = Analysis(model, gs, train_date, predict_date)
    results = run_analysis(a, forecast_period, leaders)

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
    model = market_pipeline(model, market_specs)

    # Complete the pipeline

    logger.info('*'*80)
    logger.info("END MARKET PIPELINE")
    logger.info('*'*80)
