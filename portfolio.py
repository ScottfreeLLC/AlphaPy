##############################################################
#
# Package  : AlphaPy
# Module   : portfolio
# Version  : 1.0
# Copyright: Mark Conway
# Date     : June 29, 2013
#
##############################################################


#
# Imports
#

from frame import Frame
from frame import frame_name
from frame import write_frame
from globs import MULTIPLIERS
from globs import SSEP
import logging
import math
from pandas import DataFrame
from pandas import date_range
from pandas import Series
from space import Space


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function portfolio_name
#

def portfolio_name(name, tag):
    """
    Return the name of the portfolio
    """
    return '.'.join([name, tag, "portfolio"])


#
# Class Portfolio
#

class Portfolio():

    # class variable to track all portfolios

    portfolios = {}
    
    # portfolio states

    states = ['value', 'profit', 'netreturn']

    # __new__
    
    def __new__(cls,
                name,
                tag,
                space = Space(),
                maxpos = 10,
                posby = 'close',
                kopos = 0,
                koby = '-profit',
                restricted = False,
                weightby = 'quantity',
                startcap = 100000,
                margin = 0.5,
                mincash = 0.2,
                fixedfrac = 0.1,
                maxloss = 0.1,
                balance = 'M'):
        # create portfolio name
        pn = portfolio_name(name, tag)
        if not pn in Portfolio.portfolios:
            return super(Portfolio, cls).__new__(cls)
        else:
            logger.info("Portfolio %s already exists", pn)
    
    # __init__
    
    def __init__(self,
                 name,
                 tag,
                 space = Space(),
                 maxpos = 10,
                 posby = 'close',
                 kopos = 0,
                 koby = '-profit',
                 restricted = False,
                 weightby = 'quantity',
                 startcap = 100000,
                 margin = 0.5,
                 mincash = 0.2,
                 fixedfrac = 0.1,
                 maxloss = 0.1,
                 balance = 'M'):
        # initialization
        self.name = name
        self.tag = tag
        self.space = space
        self.positions = {}
        self.startdate = None
        self.enddate = None
        self.npos = 0
        self.maxpos = maxpos
        self.posby = posby
        self.kopos = kopos
        self.koby = koby
        self.restricted = restricted
        self.weightby = weightby
        self.weights = []
        self.startcap = startcap
        self.cash = startcap
        self.margin = margin
        self.mincash = mincash
        self.fixedfrac = fixedfrac
        self.maxloss = maxloss
        self.balance = 'M'
        self.value = startcap
        self.profit = 0.0
        self.netreturn = 0.0
        self.runup = 0.0
        self.drawdown = 0.0
        # add portfolio to portfolios list
        pn = portfolio_name(name, tag)
        Portfolio.portfolios[pn] = self

    # __str__

    def __str__(self):
        return portfolio_name(self.name, self.tag)


#
# Class Position
#

class Position:
    
    # __init__
    
    def __init__(self,
                 portfolio,
                 name,
                 opendate):
        space = portfolio.space
        self.date = opendate
        self.name = name
        self.status = 'opened'
        self.mpos = 'flat'
        self.quantity = 0
        self.price = 0.0
        self.value = 0.0
        self.profit = 0.0
        self.netreturn = 0.0
        self.opened = opendate
        self.held = 0
        self.costbasis = 0.0
        self.trades = []
        self.ntrades = 0
        self.pdata = Frame.frames[frame_name(name, space)].df
        self.multiplier = MULTIPLIERS[space.subject]

    # __str__
    
    def __str__(self):
        return self.name


#
# Class Trade
#

class Trade:
    
    states = ['name', 'order', 'quantity', 'price']

    # __init__

    def __init__(self,
                 name,
                 order,
                 quantity,
                 price,
                 tdate):
        self.name = name
        self.order = order
        self.quantity = quantity
        self.price = price
        self.tdate = tdate


#
# Function add_position
#

def add_position(p, name, pos):
    """
    Add a position to a portfolio by name
    """
    if name not in p.positions:
        p.positions[name] = pos


#
# Function remove_position
#

def remove_position(p, name):
    """
    Remove a position from a portfolio by name
    """
    del p.positions[name]


#
# Function valuate_position
#

#
# Example of Cost Basis:
#
# | +100 | * 10 =  1,000
# | +200 | * 15 =  3,000
# | -500 | * 20 = 10,000
# --------        ------
#    800          14,000  =>  14,000 / 800 = 17.5
#
#    Position is -200 (net short) @ 17.5
#

def valuate_position(position, tdate):
    """
    Valuate the position based on the trade list
    """
    # get current price
    pdata = position.pdata
    if tdate in pdata.index:
        cp = pdata.ix[tdate]['close']
        # start valuation
        multiplier = position.multiplier
        netpos = 0
        tts = 0     # total traded shares
        ttv = 0     # total traded value
        totalprofit = 0.0
        for trade in position.trades:
            tq = trade.quantity
            netpos = netpos + tq
            tts = tts + abs(tq)
            tp = trade.price
            pfactor = tq * multiplier
            cv = pfactor * cp
            cvabs = abs(cv)
            ttv = ttv + cvabs
            ev = pfactor * tp
            totalprofit = totalprofit + cv - ev
        position.quantity = netpos
        position.price = cp
        position.value = abs(netpos) * multiplier * cp
        position.profit = totalprofit
        position.costbasis = ttv / tts
        position.netreturn = totalprofit / cvabs - 1.0


#
# Function update_position
#

def update_position(position, trade):
    """
    Update the position status, trade list, and valuation
    """
    position.trades.append(trade)
    position.ntrades = position.ntrades + 1
    position.date = trade.tdate
    position.held = trade.tdate - position.opened
    valuate_position(position, trade.tdate)
    if position.quantity > 0:
        position.mpos = 'long'
    if position.quantity < 0:
        position.mpos = 'short'


#
# Function close_position
#

def close_position(p, position, tdate):
    """
    Close the position and remove it from the portfolio
    """
    pq = position.quantity
    # if necessary, put on an offsetting trade
    if pq != 0:
        tradesize = -pq
        position.date = tdate
        pdata = position.pdata
        cp = pdata.ix[tdate]['close']
        newtrade = Trade(position.name, tradesize, cp, tdate)
        update_portfolio(p, position, newtrade, tradesize)
        position.quantity = 0
    position.status = 'closed'
    remove_position(p, position.name)

    
#
# Function deposit_portfolio
#

def deposit_portfolio(p, cash, tdate):
    """
    Deposit cash into a given portfolio
    """
    p.cash = p.cash + cash
    valuate_portfolio(p, tdate)
    return p.value


#
# Function withdraw_portfolio
#

def withdraw_portfolio(p, cash, tdate):
    """
    Withdraw cash from a given portfolio
    """
    currentcash = p.cash
    availcash = currentcash - (p.mincash * p.value)
    if cash > availcash:
        logger.info("Withdrawal of %s would exceed reserve amount", cash)
    else:
        p.cash = currentcash - cash
        valuate_portfolio(p, tdate)
    return p.value


#
# Function update_portfolio
#

def update_portfolio(p, pos, trade, allocation):
    """
    Update the portfolio positions and valuation
    """
    # update position
    ppq = abs(pos.quantity)
    update_position(pos, trade)
    cpq = abs(pos.quantity)
    npq = cpq - ppq
    # update portfolio
    p.date = trade.tdate
    multiplier = pos.multiplier
    cv = trade.price * multiplier * npq
    p.cash -= cv
    valuate_portfolio(p, trade.tdate)


#
# Function delete_portfolio
#

def delete_portfolio(p):
    """ Delete the portfolio, closing all positions """
    positions = p.positions
    for key in positions:
        close_position(p, positions[key])
    del p


#
# Function balance
#

def balance(p, tdate, cashlevel):
    """
    Balance the portfolio using a weighting variable
    """
    currentcash = p.cash
    mincash = p.mincash
    weightby = p.weightby
    if not weightby:
        weightby = 'close'
    valuate_portfolio(p, tdate)
    pvalue = p.value - cashlevel * p.value
    positions = p.positions
    bdata = np.ones(len(positions))
    # get weighting variable values
    if weightby[0] == "-":
        invert = True
        weightby = weightby[1:]
    else:
        invert = False
    attrs = filter(lambda aname: not aname.startswith('_'), dir(positions[0]))
    for i, pos in enumerate(positions):
        if weightby in attrs:
            estr = '.'.join('pos', weightby)
            bdata[i] = eval(estr)
        else:
            bdata[i] = pos.pdata.ix[tdate][weightby]
    if invert:
        bweights = (2 * bdata.mean() - bdata) / sum(bdata)
    else:
        bweights = bdata / sum(bdata)
    # rebalance
    for i, pos in enumerate(positions):
        multiplier = pos.multiplier
        bdelta = bweights[i] * pvalue - pos.value
        cp = pos.pdata.ix[tdate]['close']
        tradesize = math.trunc(bdelta / cp)
        ntv = abs(tradesize) * cp * multiplier
        if tradesize > 0:
            order = Orders.lb
        if tradesize < 0:
            order = Orders.sb
        exec_trade(p, pos.name, order, tradesize, cp, tdate)
        p.cash = currentcash + bdelta - ntv


#
# Function kick_out
#

def kick_out(p, tdate, freepos):
    """
    Trim the portfolio based on filter criteria
    """
    positions = p.positions
    kovalue = np.zeros(len(positions))
    koby = p.koby
    if not koby:
        koby = 'profit'
    if koby[0] == "-":
        descending = True
        koby = koby[1:]
    else:
        descending = False
    attrs = filter(lambda aname: not aname.startswith('_'), dir(positions[0]))
    for i, pos in enumerate(positions):
        if koby in attrs:
            estr = '.'.join('pos', koby)
            kovalue[i] = eval(estr)
        else:
            kovalue[i] = pos.pdata.ix[tdate][koby]
    koorder = np.argsort(np.argsort(kovalues))
    if descending:
        koorder = [i for i in reversed(koorder)]
    if freepos == 0:
        kopos = p.kopos
        maxpos = p.maxpos
        opos = maxpos - npos
        if opos == 0:
            freepos = kopos - opos
    # close the top freepos positions
    for i in range(freepos):
        close_position(p, positions[koorder[i]], tdate)


#
# Function stop_loss
#

def stop_loss(p, tdate):
    """
    Trim the portfolio based on stop-loss criteria
    """
    positions = p.positions
    maxloss = p.maxloss
    for key in positions:
        pos = positions[key]
        nr = pos.netreturn
        if nr <= -maxloss:
            close_position(p, pos, tdate)


#
# Function valuate_portfolio
#

def valuate_portfolio(p, tdate):
    """
    Value the portfolio based on the current positions
    """
    positions = p.positions
    poslen = len(positions)
    vpos = [0] * poslen
    p.weights = [0] * poslen
    posenum = enumerate(positions)
    # compute the total portfolio value
    value = p.cash
    for i, key in posenum:
        pos = positions[key]
        valuate_position(pos, tdate)
        vpos[i] = pos.value
        value = value + vpos[i]
    # now compute the weights
    for i, key in posenum:
        p.weights[i] = vpos[i] / value
    p.value = value
    p.profit = p.value - p.startcap
    p.netreturn = p.value / p.startcap - 1.0


#
# Function pstats1
#

def pstats1():
    """
    Portfolio Statistics, Phase I
    """
    pstat = {}
    pstat['totalperiod'] = 0
    pstat['totaltrades'] = 0
    pstat['longtrades'] = 0
    pstat['shortrades'] = 0
    pstat['lsratio'] = 0.0
    pstat['winners'] = 0
    pstat['losers'] = 0
    pstat['winpct'] = 0.0
    pstat['losepct'] = 0.0
    pstat['maxwin'] = 0.0
    pstat['maxloss'] = 0.0
    pstat['grossprofit'] = 0.0
    pstat['grossloss'] = 0.0
    pstat['profitfactor'] = 0.0
    pstat['totalreturn'] = 0.0
    pstat['car'] = 0.0
    pstat['avgtrade'] = 0.0
    pstat['avgwin'] = 0.0
    pstat['avgloss'] = 0.0
    pstat['avghold'] = 0
    pstat['optimalf'] = 0.0
    pstat['totalruns'] = 0
    pstat['longruns'] = 0
    pstat['shortruns'] = 0
    pstat['avgrun'] = 0
    pstat['avglongrun'] = 0
    pstat['avgshortrun'] = 0
    pstat['maxrunup'] = 0.0
    pstat['maxdrawdown'] = 0.0
    pstat['avgrunup'] = 0.0
    pstat['avgdrawdown'] = 0.0
    pstat['coverage'] = 0.0
    pstat['longcoverage'] = 0.0
    pstat['shortcoverage'] = 0.0
    pstat['frequency'] = 0
    pstat['longfrequency'] = 0
    pstat['shortfrequency'] = 0 
    return pstat


#
# Function pstats2
#

def pstats2(p, pstat, pos):
    """
    Portfolio Statistics, Phase II
    """
    pstat['totaltrades'] += 1
    if pos.mpos == 'long':
        pstat['longtrades'] += 1
    if pos.mpos == 'short':
        pstat['shortrades'] += 1
    if pos.profit > 0:
        pstat['winners'] += 1
        pstat['grossprofit'] += pos.profit
        if pos.profit > pstat['maxwin']:
            pstat['maxwin'] = pos.profit
    if pos.profit < 0:
        pstat['losers'] += 1
        pstat['grossloss'] += pos.profit
        if -pos.profit > pstat['maxloss']:
            pstat['maxloss'] = pos.profit
    pstat['totalreturn'] *= 1.0 + pos.netreturn


#
# Function pstats3
#

def pstats3(p, pstat):
    """
    Portfolio Statistics, Phase III
    """
    pstat['profitfactor'] = 0
    if pstat['grossloss'] > 0:
        pstat['profitfactor'] = pstat['grossprofit'] / pstat['grossloss']
    pstat['lsratio'] = 0
    if pstat['shortrades'] > 0:
        pstat['lsratio'] = pstat['longtrades'] / pstat['shortrades']
    pstat['winpct'] = 0.0
    if pstat['totaltrades'] > 0:
        pstat['winpct'] = pstat['winners'] / pstat['totaltrades']
    pstat['losepct'] = 0.0
    if pstat['totaltrades'] > 0:
        pstat['losepct'] = pstat['losers'] / pstat['totaltrades']


#
# Function allocate_trade
#

def allocate_trade(p, pos, trade):
    """
    Determine the trade allocation for a given portfolio
    """
    cash = p.cash
    margin = p.margin
    mincash = p.mincash
    restricted = p.restricted
    if restricted:
        kick_out(p, trade.tdate)
        stop_loss(p, trade.tdate)
    multiplier = pos.multiplier
    qpold = pos.quantity
    qtrade = trade.quantity
    qpnew = qpold + qtrade
    allocation = abs(qpnew) - abs(qpold)
    addedvalue = trade.price * multiplier * abs(allocation)
    if restricted:
        cashreserve = mincash * cash
        freemargin = (cash - cashreserve) / margin
        if addedvalue > freemargin:
            logger.info("Required free margin: %d < added value: %d",
                        freemargin, addedvalue)
            allocation = 0
        else:
            freecash = cash - addedvalue
            if freecash < 0:
                p.cash = cash + freecash
    return allocation


#
# Function exec_trade
#

def exec_trade(p, name, order, quantity, price, tdate):
    """
    Execute a trade within a portfolio
    """
    # see if the position already exists
    if name in p.positions:
        pos = p.positions[name]
        newpos = False
    else:
        pos = Position(p, name, tdate)
        newpos = True
    # check the dynamic position sizing variable
    if not p.posby:
        psize = quantity
    else:
        if order == 'le' or order == 'se':
            pf = Frame.frames[frame_name(name, p.space)].df
            cv = pf.ix[tdate][p.posby]
            psize = math.trunc((p.value * p.fixedfrac) / cv)
            if quantity < 0:
                psize = -psize
        else:
            psize = -pos.quantity
    # instantiate and allocate the trade
    newtrade = Trade(name, order, psize, price, tdate)
    allocation = allocate_trade(p, pos, newtrade)
    if allocation != 0:
        # create a new position if necessary
        if newpos:
            add_position(p, name, pos)
            p.npos += 1
        # update the portfolio
        update_portfolio(p, pos, newtrade, allocation)
        # if net position is zero, then close the position
        pflat = pos.quantity == 0
        if pflat:
            close_position(p, pos, tdate)
            p.npos -= 1
        return pos
    else:
        return None


#
# Function gen_portfolio
#

def gen_portfolio(model, system, group, tframe, startcap=100000, posby='close'):
    """
    Create a portfolio from a trades frame
    """
    logger.info("Creating Portfolio for System %s", system.name)

    # Unpack model data

    base_dir = model.specs['base_dir']
    extension = model.specs['extension']
    project = model.specs['project']
    separator = model.specs['separator']
    directory = SSEP.join([base_dir, project])

    # Create portfolio

    gname = group.name
    gspace = group.space
    p = Portfolio(gname,
                  system.name,
                  startcap = startcap,
                  posby = posby,
                  restricted = False)
    if not p:
        sys.exit("Error creating Portfolio")

    # Build a portfolio from the trades frame

    start = tframe.index[0]
    end = tframe.index[-1]
    drange = date_range(start, end, freq='B')
    # initialize portfolio states and stats
    ps = []
    pstat = pstats1()
    # iterate through the date range, updating the portfolio
    for i, d in enumerate(drange):
        logger.info("Updating Portfolio for %s", d)
        # process today's trades
        if drange[i] in tframe.index:
            trades = tframe.ix[drange[i]]
            if isinstance(trades, Series):
                trades = DataFrame(trades).transpose()
            for t in trades.iterrows():
                tdate = t[0]
                row = t[1]
                pos = exec_trade(p, row['name'], row['order'], row['quantity'], row['price'], tdate)
                if pos:
                    if pos.status == 'closed':
                        pstats2(p, pstat, pos)
                else:
                    logger.info("Trade could not be allocated")
        # update the portfolio valuation
        valuate_portfolio(p, d)
        ps.append((d, [p.value, p.profit, p.netreturn]))

    # Create the portfolio states frame for this system

    logger.info("Writing Portfolio Frame")
    pspace = Space(system.name, 'portfolio', gspace.fractal)
    pf = DataFrame.from_items(ps, orient='index', columns=Portfolio.states)
    pfname = frame_name(gname, pspace)
    write_frame(pf, directory, pfname, extension, separator, index=True)
    del pspace

    # Compute the final portfolio statistics
    pstats3(p, pstat)

    # return the portfolio
    return p
