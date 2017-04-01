################################################################################
#
# Package   : AlphaPy
# Module    : systemstream
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
# Example: systemstream -d './config'
#
################################################################################


#
# Imports
#

from alphapy.data import get_feed_data
from alphapy.group import Group
from alphapy.model import get_model_config
from alphapy.model import Model
from alphapy.portfolio import gen_portfolio
from alphapy.stockstream import get_market_config
from alphapy.system import System
from alphapy.system import run_system
from alphapy.var import vmapply

import argparse
import logging


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function system_pipeline
#

def system_pipeline(model, market_specs):
    r"""AlphaPy Model Pipeline

    Parameters
    ----------
    model : alphapy.Model
        The model object for controlling the pipeline.
    market_specs : dict
        The market specifications for running the system.

    Returns
    -------
    None : None

    Notes
    -----
    (1) Define a group.
    (2) Get the market data.
    (3) Apply system features.
    (4) Create the system.
    (5) Run the system.
    (6) Generate a portfolio.

    """

    # Get any model specifications

    directory = model.specs['directory']
    extension = model.specs['extension']
    separator = model.specs['separator']
    target = model.specs['target']

    # Get any market specifications

    features = market_specs['features']
    functions = market_specs['functions']
    lookback_period = market_specs['lookback_period']
    target_group = market_specs['target_group']

    # Set the target group

    gs = Group.groups[target_group]
    logger.info("All Members: %s", gs.members)

    # Get stock data
    get_feed_data(gs, lookback_period)

    # Apply the features to all of the frames
    vmapply(gs, features, functions)

    # Get the system specifications

    system_name = market_specs['system']['name']
    longentry = market_specs['system']['longentry']
    shortentry = market_specs['system']['shortentry']
    longexit = market_specs['system']['longexit']
    shortexit = market_specs['system']['shortexit']
    holdperiod = market_specs['system']['holdperiod']
    scale = market_specs['system']['scale']

    # Create and run the system

    cs = System(system_name, longentry, shortentry,
                longexit, shortexit, holdperiod, scale)
    tfs = run_system(model, cs, gs)
 
    # Generate a portfolio
    gen_portfolio(model, system_name, gs, tfs)
    return


#
# Function main
#

def main(args=None):
    r"""SystemStream Main Program

    Notes
    -----
    (1) Initialize logging.
    (2) Parse the command line arguments.
    (3) Get the market configuration.
    (4) Get the model configuration.
    (5) Create the model object.
    (6) Call the main SystemStream pipeline.

    """

    # Logging

    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="systemstream.log", filemode='a', level=logging.DEBUG,
                        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # Start the pipeline

    logger.info('*'*80)
    logger.info("START SystemStream PIPELINE")
    logger.info('*'*80)

    # Argument Parsing

    parser = argparse.ArgumentParser(description="SystemStream Parser")
    parser.add_argument("-d", dest="cfg_dir", default=".",
                        help="directory location of configuration file")
    parser.add_mutually_exclusive_group(required=False)
    parser.add_argument('--score', dest='scoring', action='store_true')
    parser.add_argument('--train', dest='scoring', action='store_false')
    parser.set_defaults(scoring=False)
    args = parser.parse_args()

    # Read stock configuration file

    market_specs = get_market_config(args.cfg_dir)

    # Read model configuration file

    model_specs = get_model_config(args.cfg_dir)
    model_specs['scoring'] = args.scoring

    # Create a model from the arguments

    logger.info("Creating Model")
    model = Model(model_specs)

    # Start the pipeline

    model = system_pipeline(model, market_specs)

    # Complete the pipeline

    logger.info('*'*80)
    logger.info("END SystemStream PIPELINE")
    logger.info('*'*80)


#
# MAIN PROGRAM
#

if __name__ == "__main__":
    main()
