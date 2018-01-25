################################################################################
#
# Package   : AlphaPy
# Module    : system
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

from alphapy.frame import Frame
from alphapy.frame import frame_name
from alphapy.frame import read_frame
from alphapy.frame import write_frame
from alphapy.globals import Orders
from alphapy.globals import BSEP, SSEP
from alphapy.market_variables import vexec
from alphapy.space import Space
from alphapy.portfolio import Trade
from alphapy.utilities import most_recent_file

import logging
import numbers
import pandas as pd
from pandas import DataFrame


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Class System
#

class System(object):
    """Create a new system. All systems are stored in
    ``System.systems``. Duplicate names are not allowed.

    Parameters
    ----------
    name : str
        The system name.
    longentry : str
        Name of the conditional feature for a long entry.
    shortentry : str, optional
        Name of the conditional feature for a short entry.
    longexit : str, optional
        Name of the conditional feature for a long exit.
    shortexit : str, optional
        Name of the conditional feature for a short exit.
    holdperiod : int, optional
        Holding period of a position.
    scale : bool, optional
        Add to a position for a signal in the same direction.

    Attributes
    ----------
    systems : dict
        Class variable for storing all known systems

    Examples
    --------
    
    >>> System('closer', hc, lc)

    """

    # class variable to track all systems

    systems = {}

    # __new__
    
    def __new__(cls,
                name,
                longentry,
                shortentry = None,
                longexit = None,
                shortexit = None,
                holdperiod = 0,
                scale = False):
        # create system name
        if name not in System.systems:
            return super(System, cls).__new__(cls)
        else:
            logger.info("System %s already exists", name)
    
    # __init__
    
    def __init__(self,
                 name,
                 longentry,
                 shortentry = None,
                 longexit = None,
                 shortexit = None,
                 holdperiod = 0,
                 scale = False):
        # initialization
        self.name = name
        self.longentry = longentry
        self.shortentry = shortentry
        self.longexit = longexit
        self.shortexit = shortexit
        self.holdperiod = holdperiod
        self.scale = scale
        # add system to systems list
        System.systems[name] = self
        
    # __str__

    def __str__(self):
        return self.name


#
# Function trade_system
#

def trade_system(model, system, space, intraday, name, quantity):
    r"""Trade the given system.

    Parameters
    ----------
    model : alphapy.Model
        The model object with specifications.
    system : alphapy.System
        The long/short system to run.
    space : alphapy.Space
        Namespace of instrument prices.
    intraday : bool
        If True, then run an intraday system.
    name : str
        The symbol to trade.
    quantity : float
        The amount of the ``name`` to trade, e.g., number of shares

    Returns
    -------
    tradelist : list
        List of trade entries and exits.

    Other Parameters
    ----------------
    Frame.frames : dict
        All of the data frames containing price data.

    """

    # Unpack the model data.

    directory = model.specs['directory']
    extension = model.specs['extension']
    separator = model.specs['separator']

    # Unpack the system parameters.

    longentry = system.longentry
    shortentry = system.shortentry
    longexit = system.longexit
    shortexit = system.shortexit
    holdperiod = system.holdperiod
    scale = system.scale

    # Determine whether or not this is a model-driven system.

    entries_and_exits = [longentry, shortentry, longexit, shortexit]
    active_signals = [x for x in entries_and_exits if x is not None]
    use_model = False
    for signal in active_signals:
        if any(x in signal for x in ['phigh', 'plow']):
            use_model = True

    # Read in the price frame
    pf = Frame.frames[frame_name(name, space)].df

    # Use model output probabilities as input to the system

    if use_model:
        # get latest probabilities file
        probs_dir = SSEP.join([directory, 'output'])
        file_path = most_recent_file(probs_dir, 'probabilities*')
        file_name = file_path.split(SSEP)[-1].split('.')[0]
        # read the probabilities frame and trim the price frame
        probs_frame = read_frame(probs_dir, file_name, extension, separator)
        pf = pf[-probs_frame.shape[0]:]
        probs_frame.index = pf.index
        probs_frame.columns = ['probability']
        # add probability column to price frame
        pf = pd.concat([pf, probs_frame], axis=1)

    # Evaluate the long and short events in the price frame

    for signal in active_signals:
        vexec(pf, signal)

    # Initialize trading state variables

    inlong = False
    inshort = False
    h = 0
    p = 0
    q = quantity
    tradelist = []

    # Loop through prices and generate trades

    for dt, row in pf.iterrows():
        # get closing price
        c = row['close']
        if intraday:
            bar_number = row['bar_number']
            end_of_day = row['end_of_day']            
        # evaluate entry and exit conditions
        lerow = row[longentry] if longentry else None
        serow = row[shortentry] if shortentry else None
        lxrow = row[longexit] if longexit else None
        sxrow = row[shortexit] if shortexit else None
        # process the long and short events
        if lerow:
            if p < 0:
                # short active, so exit short
                tradelist.append((dt, [name, Orders.sx, -p, c]))
                inshort = False
                h = 0
                p = 0
            if p == 0 or scale:
                # go long (again)
                tradelist.append((dt, [name, Orders.le, q, c]))
                inlong = True
                p = p + q
        elif serow:
            if p > 0:
                # long active, so exit long
                tradelist.append((dt, [name, Orders.lx, -p, c]))
                inlong = False
                h = 0
                p = 0
            if p == 0 or scale:
                # go short (again)
                tradelist.append((dt, [name, Orders.se, -q, c]))
                inshort = True
                p = p - q
        # check exit conditions
        if inlong and h > 0 and lxrow:
            # long active, so exit long
            tradelist.append((dt, [name, Orders.lx, -p, c]))
            inlong = False
            h = 0
            p = 0
        if inshort and h > 0 and sxrow:
            # short active, so exit short
            tradelist.append((dt, [name, Orders.sx, -p, c]))
            inshort = False
            h = 0
            p = 0
        # if a holding period was given, then check for exit
        if holdperiod and h >= holdperiod:
            if inlong:
                tradelist.append((dt, [name, Orders.lh, -p, c]))
                inlong = False
            if inshort:
                tradelist.append((dt, [name, Orders.sh, -p, c]))
                inshort = False
            h = 0
            p = 0
        # increment the hold counter
        if inlong or inshort:
            h += 1
            if intraday and end_of_day:
                if inlong:
                    # long active, so exit long
                    tradelist.append((dt, [name, Orders.lx, -p, c]))
                    inlong = False
                if inshort:
                    # short active, so exit short
                    tradelist.append((dt, [name, Orders.sx, -p, c]))
                    inshort = False
                h = 0
                p = 0
    return tradelist


#
# Function run_system
#

def run_system(model,
               system,
               group,
               intraday = False,
               quantity = 1):
    r"""Run a system for a given group, creating a trades frame.

    Parameters
    ----------
    model : alphapy.Model
        The model object with specifications.
    system : alphapy.System
        The system to run.
    group : alphapy.Group
        The group of symbols to trade.
    intraday : bool, optional
        If true, this is an intraday system.
    quantity : float, optional
        The amount to trade for each symbol, e.g., number of shares

    Returns
    -------
    tf : pandas.DataFrame
        All of the trades for this ``group``.

    """

    system_name = system.name
    logger.info("Generating Trades for System %s", system_name)

    # Unpack the model data.

    directory = model.specs['directory']
    extension = model.specs['extension']
    separator = model.specs['separator']

    # Extract the group information.

    gname = group.name
    gmembers = group.members
    gspace = group.space

    # Run the system for each member of the group

    gtlist = []
    for symbol in gmembers:
        # generate the trades for this member
        tlist = trade_system(model, system, gspace, intraday, symbol, quantity)
        if tlist:
            # add trades to global trade list
            for item in tlist:
                gtlist.append(item)
        else:
            logger.info("No trades for symbol %s", symbol)

    # Create group trades frame

    tf = None
    if gtlist:
        tspace = Space(system_name, "trades", group.space.fractal)
        gtlist = sorted(gtlist, key=lambda x: x[0])
        tf = DataFrame.from_items(gtlist, orient='index', columns=Trade.states)
        tfname = frame_name(gname, tspace)
        system_dir = SSEP.join([directory, 'systems'])
        labels = ['date']
        if intraday:
            labels.append('time')
        write_frame(tf, system_dir, tfname, extension, separator,
                    index=True, index_label=labels)
        del tspace
    else:
        logger.info("No trades were found")

    # Return trades frame
    return tf
