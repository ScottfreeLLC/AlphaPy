################################################################################
#
# Package   : AlphaPy
# Module    : system
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
################################################################################


#
# Imports
#

from frame import Frame
from frame import frame_name
from frame import write_frame
from globs import SSEP
import logging
from pandas import DataFrame
from space import Space
from portfolio import Trade
from var import vexec


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Class System
#

class System(object):

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
# Class Orders
#

class Orders:
    le = 'le'
    se = 'se'
    lx = 'lx'
    sx = 'sx'
    lh = 'lh'
    sh = 'sh'


#
# Function long_short
#

def long_short(system, name, group, quantity):
    """
    Generate the list of trades based on the long and short events
    """
    # extract the system parameters
    longentry = system.longentry
    shortentry = system.shortentry
    longexit = system.longexit
    shortexit = system.shortexit
    holdperiod = system.holdperiod
    scale = system.scale
    # price frame
    pf = Frame.frames[frame_name(name, group.space)].df
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

def open_range_breakout(name, space, quantity):
    """
    Open Range Breakout
    """
    # system parameters
    trigger_bar = 3
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
        elif bar_number < trigger_bar:
            # set opening range
            if h > hh:
                hh = h
            if l < ll:
                ll = l
        else:
            if not traded:
                # trigger trade
                if h > hh and not inlong:
                    # long breakout triggers
                    tradelist.append((dt, [name, Orders.le, quantity, hh]))
                    inlong = True
                if l < ll and not inshort:
                    # short breakout triggers
                    tradelist.append((dt, [name, Orders.se, -quantity, ll]))
                    inshort = True
                # set traded flag
                if inlong or inshort:
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
               quantity = 1):
    """
    Run a system for a given group, creating a trades frame
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
                tlist = locals()[system_name](symbol, gspace, quantity)
            except:
                logger.error('Could not find system %s', system_name)
        else:
            # call default long/short system
            tlist = long_short(system, symbol, gspace, quantity)
        # create the local trades frame
        df = DataFrame.from_items(tlist, orient='index', columns=Trade.states)
        # add trades to global trade list
        for item in tlist:
            gtlist.append(item)

    # Create group trades frame

    tspace = Space(system_name, "trades", group.space.fractal)
    gtlist = sorted(gtlist, key=lambda x: x[0])
    tf = DataFrame.from_items(gtlist, orient='index', columns=Trade.states)
    tfname = frame_name(gname, tspace)
    system_dir = SSEP.join([directory, 'systems'])
    write_frame(tf, system_dir, tfname, extension, separator, index=True)
    del tspace

    # Return trades frame
    return tf
