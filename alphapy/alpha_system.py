################################################################################
#
# Package   : AlphaPy
# Module    : alpha_system
# Version   : 1.0
# Date      : July 11, 2013
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
# Example: python ../AlphaPy/alpha_system.py -d 'Stocks/config'
#
################################################################################


#
# Imports
#

from alpha_market import get_market_config
import argparse
from data import get_feed_data
from group import Group
import logging
from model import get_model_config
from model import Model
from portfolio import gen_portfolio
from space import Space
from system import System
from system import run_system
from var import vmapply


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function system_pipeline
#

def system_pipeline(model, market_specs):
    """
    AlphaPy System Pipeline
    :rtype : object
    """

    # Get any model specifications

    directory = model.specs['directory']
    extension = model.specs['extension']
    separator = model.specs['separator']
    target = model.specs['target']

    # Get any market specifications

    features = market_specs['features']
    lookback_period = market_specs['lookback_period']
    target_group = market_specs['target_group']

    # Set the target group

    gs = Group.groups[target_group]
    logger.info("All Members: %s", gs.members)

    # Get stock data

    get_feed_data(gs, lookback_period)

    # Apply the features to all of the frames

    # vmapply(gs, features)

    # Create and run systems

    system_name = 'open_range_breakout'
    tfs = run_system(model, system_name, gs)
    gen_portfolio(model, system_name, gs, tfs)

    # cs = System('closer', 'hc', 'lc')
    # tfs = run_system(model, cs, gs)
    # gen_portfolio(model, cs.name, gs, tfs)

    return


#
# MAIN PROGRAM
#

if __name__ == '__main__':

    # Logging

    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="alpha_system.log", filemode='a', level=logging.DEBUG,
                        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # Start the pipeline

    logger.info('*'*80)
    logger.info("START SYSTEM PIPELINE")
    logger.info('*'*80)

    # Argument Parsing

    parser = argparse.ArgumentParser(description="AlphaPy System Parser")
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
    model = system_pipeline(model, market_specs)

    # Complete the pipeline

    logger.info('*'*80)
    logger.info("END SYSTEM PIPELINE")
    logger.info('*'*80)
