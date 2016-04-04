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
import logging
import math
from pandas import DataFrame
from pandas import date_range
from pandas import Series
import position
from space import Space
import trade


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
    position.update_position(pos, trade)
    cpq = abs(pos.quantity)
    npq = cpq - ppq
    # update portfolio
    p.date = trade.tdate
    multiplier = pos.multiplier
    cv = trade.price * multiplier * npq
    p.cash -= cv
    valuate_portfolio(p, trade.tdate)


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
        trade.exec_trade(p, pos.name, order, tradesize, cp, tdate)
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
        position.close_position(p, positions[koorder[i]], tdate)


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
            position.close_position(p, pos, tdate)


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
        position.valuate_position(pos, tdate)
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
# Function gen_portfolio
#

def gen_portfolio(system, group, startcap=100000, posby='close'):
    """
    Create a portfolio from a trades file
    """
    gname = group.name
    gspace = group.space
    p = Portfolio(gname,
                  system.name,
                  startcap = startcap,
                  posby = posby,
                  restricted = False)
    if not p:
        sys.exit("Error creating Portfolio")
    # build a portfolio from the trades in the file
    tspace = Space(system.name, 'trades', gspace.fractal)
    tframe = Frame.frames[frame_name(gname, tspace)].df
    start = tframe.index[0]
    end = tframe.index[-1]
    drange = date_range(start, end, freq='B')
    del tspace
    # initialize portfolio states and stats
    ps = []
    pstat = pstats1()
    # iterate through the date range, updating the portfolio
    for i, d in enumerate(drange):
        # process today's trades
        if drange[i] in tframe.index:
            trades = tframe.ix[drange[i]]
            if isinstance(trades, Series):
                trades = DataFrame(trades).transpose()
            for t in trades.iterrows():
                tdate = t[0]
                row = t[1]
                pos = trade.exec_trade(p, row['name'], row['order'], row['quantity'], row['price'], tdate)
                if pos:
                    if pos.status == 'closed':
                        pstats2(p, pstat, pos)
                else:
                    logger.info("Trade could not be allocated")
        # update the portfolio valuation
        valuate_portfolio(p, d)
        ps.append((d, [p.value, p.profit, p.netreturn]))
    # create the portfolio states frame for this system
    pspace = Space(system.name, 'portfolio', gspace.fractal)
    psf = DataFrame.from_items(ps, orient='index', columns=Portfolio.states)
    Frame(gname, pspace, psf)
    del pspace
    # compute the final portfolio statistics
    pstats3(p, pstat)
    # return the portfolio
    return p


#
# Function delete_portfolio
#

def delete_portfolio(p):
    """ Delete the portfolio, closing all positions """
    positions = p.positions
    for key in positions:
        close_position(p, positions[key])
    del p
