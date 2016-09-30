##############################################################
#
# Package   : AlphaPy
# Module    : system
# Version   : 1.0
# Copyright : Mark Conway
# Date      : June 29, 2013
#
##############################################################


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
                longevent = '',
                shortevent = '',
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
                 longevent = '',
                 shortevent = '',
                 holdperiod = 0,
                 scale = False):
        # initialization
        self.name = name
        self.longevent = longevent
        self.shortevent = shortevent
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
    lt = 'lt'
    st = 'st'
    lb = 'lb'
    sb = 'sb'


#
# Function gen_trades
#

def gen_trades(system, name, group, quantity):
    """
    Generate the list of trades based on the long and short events
    """
    # extract the system parameters
    longevent = system.longevent
    shortevent = system.shortevent
    holdperiod = system.holdperiod
    scale = system.scale
    # price frame
    pf = Frame.frames[frame_name(name, group.space)].df
    # initialize the trade list
    tradelist = []
    # evaluate the long and short events
    if longevent:
        vexec(pf, longevent)
    if shortevent:
        vexec(pf, shortevent)
    # generate trade file
    inlong = False
    inshort = False
    h = 0
    p = 0
    q = quantity
    for tdate, row in pf.iterrows():
        lrow = None
        if longevent:
            lrow = row[longevent]
        srow = None
        if shortevent:
            srow = row[shortevent]
        c = row['close']
        # process the long and short events
        if lrow:
            if p < 0:
                # short active
                tradelist.append((tdate, [name, Orders.sx, -p, c]))
                inshort = False
                h = 0
                p = 0
            if p == 0 or scale:
                tradelist.append((tdate, [name, Orders.le, q, c]))
                inlong = True
                p = p + q
        elif srow:
            if p > 0:
                # long active
                tradelist.append((tdate, [name, Orders.lx, -p, c]))
                inlong = False
                h = 0
                p = 0
            if p == 0 or scale:
                tradelist.append((tdate, [name, Orders.se, -q, c]))
                inshort = True
                p = p - q
        # if a holding period was given, then check for exit
        if holdperiod > 0 and h >= holdperiod:
            if inlong:
                tradelist.append((tdate, [name, Orders.lt, -p, c]))
                inlong = False
            if inshort:
                tradelist.append((tdate, [name, Orders.st, -p, c]))
                inshort = False
            h = 0
            p = 0
        # increment the hold counter
        if inlong or inshort:
            h += 1
    return tradelist


#
# Function run_system
#

def run_system(model,
               system,
               group,
               startcap = 100000,
               quantity = 1000,
               posby = 'close',
               bperiod = 0):
    """
    Run a system for a given group, creating a trades frame
    """
    logger.info("Generating Trades for System %s", system.name)

    # Unpack model data

    base_dir = model.specs['base_dir']
    extension = model.specs['extension']
    project = model.specs['project']
    separator = model.specs['separator']
    directory = SSEP.join([base_dir, project])

    # Extract the group information
    gname = group.name
    gmembers = group.members

    # Run the system for each member of the group

    gtlist = []
    for m in gmembers:
        # generate the trades for this member
        tlist = gen_trades(system, m, group, quantity)
        # create the local trades frame
        df = DataFrame.from_items(tlist, orient='index', columns=Trade.states)
        # add trades to global trade list
        for item in tlist:
            gtlist.append(item)

    # Create group trades frame

    tspace = Space(system.name, "trades", group.space.fractal)
    gtlist = sorted(gtlist, key=lambda x: x[0])
    tf = DataFrame.from_items(gtlist, orient='index', columns=Trade.states)
    tfname = frame_name(gname, tspace)
    write_frame(tf, directory, tfname, extension, separator, index=True)
    del tspace

    # Return trades frame
    return tf
