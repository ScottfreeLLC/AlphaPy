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
from alphapy.frame import write_frame
from alphapy.globals import Orders
from alphapy.globals import SSEP
from alphapy.market_variables import vexec
from alphapy.space import Space
from alphapy.portfolio import Trade

import logging
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
# Function long_short
#

def long_short(system, name, space, quantity):
    r"""Run a long/short system.

    A long/short system is always in the market. At any given
    time, either a long position is active, or a short position
    is active.

    Parameters
    ----------
    system : alphapy.System
        The long/short system to run.
    name : str
        The symbol to trade.
    space : alphapy.Space
        Namespace of instrument prices.
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
    # extract the system parameters
    longentry = system.longentry
    shortentry = system.shortentry
    longexit = system.longexit
    shortexit = system.shortexit
    holdperiod = system.holdperiod
    scale = system.scale
    # price frame
    pf = Frame.frames[frame_name(name, space)].df
    # initialize the trade list
    tradelist = []
    # evaluate the long and short events
    if longentry:
        vexec(pf, longentry)
    if shortentry:
        vexec(pf, shortentry)
    if longexit:
        vexec(pf, longexit)
    if shortexit:
        vexec(pf, shortexit)
    # generate trade file
    inlong = False
    inshort = False
    h = 0
    p = 0
    q = quantity
    for dt, row in pf.iterrows():
        # evaluate entry and exit conditions
        lerow = None
        if longentry:
            lerow = row[longentry]
        serow = None
        if shortentry:
            serow = row[shortentry]
        lxrow = None
        if longexit:
            lxrow = row[longexit]
        sxrow = None
        if shortexit:
            sxrow = row[shortexit]
        # get closing price
        c = row['close']
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
        if holdperiod > 0 and h >= holdperiod:
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
    return tradelist


#
# Function open_range_breakout
#

def open_range_breakout(name, space, quantity, t1=3, t2=12):
    r"""Run an Opening Range Breakout (ORB) system.

    An ORB system is an intraday strategy that waits for price to
    "break out" in a certain direction after establishing an
    initial High-Low range. The timing of the trade is either
    time-based (e.g., 30 minutes after the Open) or price-based
    (e.g., 20% of the average daily range). Either the position
    is held until the end of the trading day, or the position is
    closed with a stop loss (e.g., the other side of the opening
    range).

    Parameters
    ----------
    name : str
        The symbol to trade.
    space : alphapy.Space
        Namespace of instrument prices.
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
    # price frame
    pf = Frame.frames[frame_name(name, space)].df
    # initialize the trade list
    tradelist = []
    # generate trade file
    for dt, row in pf.iterrows():
        # extract data from row
        bar_number = row['bar_number']
        h = row['high']
        l = row['low']
        c = row['close']
        end_of_day = row['end_of_day']
        # open range breakout
        if bar_number == 0:
            # new day
            traded = False
            inlong = False
            inshort = False
            hh = h
            ll = l
        elif bar_number < t1:
            # set opening range
            if h > hh:
                hh = h
            if l < ll:
                ll = l
        else:
            if not traded and bar_number < t2:
                # trigger trade
                if h > hh:
                    # long breakout triggers
                    tradelist.append((dt, [name, Orders.le, quantity, hh]))
                    inlong = True
                    traded = True
                if l < ll and not traded:
                    # short breakout triggers
                    tradelist.append((dt, [name, Orders.se, -quantity, ll]))
                    inshort = True
                    traded = True
            # test stop loss
            if inlong and l < ll:
                tradelist.append((dt, [name, Orders.lx, -quantity, ll]))
                inlong = False
            if inshort and h > hh:
                tradelist.append((dt, [name, Orders.sx, quantity, hh]))
                inshort = False
        # exit any positions at the end of the day
        if inlong and end_of_day:
            # long active, so exit long
            tradelist.append((dt, [name, Orders.lx, -quantity, c]))
        if inshort and end_of_day:
            # short active, so exit short
            tradelist.append((dt, [name, Orders.sx, quantity, c]))
    return tradelist


#
# Function run_system
#

def run_system(model,
               system,
               group,
               system_params=None,
               quantity = 1):
    r"""Run a system for a given group, creating a trades frame.

    Parameters
    ----------
    model : alphapy.Model
        The model object with specifications.
    system : alphapy.System or str
        The system to run, either a long/short system or a local one
        identified by function name, e.g., 'open_range_breakout'.
    group : alphapy.Group
        The group of symbols to test.
    system_params : list, optional
        The parameters for the given system.
    quantity : float, optional
        The amount to trade for each symbol, e.g., number of shares

    Returns
    -------
    tf : pandas.DataFrame
        All of the trades for this ``group``.

    """

    if system.__class__ == str:
        system_name = system
    else:
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
        if system.__class__ == str:
            try:
                tlist = globals()[system_name](symbol, gspace, quantity,
                                               *system_params)
            except:
                logger.info("Could not execute system for %s", symbol)
        else:
            # call default long/short system
            tlist = long_short(system, symbol, gspace, quantity)
        if tlist:
            # create the local trades frame
            df = DataFrame.from_items(tlist, orient='index', columns=Trade.states)
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
        write_frame(tf, system_dir, tfname, extension, separator,
                    index=True, index_label='date')
        del tspace
    else:
        logger.info("No trades were found")

    # Return trades frame
    return tf
