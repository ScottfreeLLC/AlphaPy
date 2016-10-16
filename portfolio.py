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
from frame import read_frame
from frame import write_frame
from globs import MULTIPLIERS
from globs import SSEP
import logging
import math
from pandas import DataFrame
from pandas import date_range
from pandas import Series
from pyfolio import plot_annual_returns
from pyfolio import plot_drawdown_periods
from pyfolio import plot_drawdown_underwater
from pyfolio import plot_monthly_returns_dist
from pyfolio import plot_monthly_returns_heatmap
from pyfolio import plot_return_quantiles
from pyfolio import plot_returns
from pyfolio import plot_rolling_returns
from pyfolio import plot_rolling_sharpe
from pyfolio import show_perf_stats
from pyfolio import show_worst_drawdown_periods
from pyfolio.utils import get_symbol_rets
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
        self.netprofit = 0.0
        self.netreturn = 0.0
        self.totalprofit = 0.0
        self.totalreturn = 0.0
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
    Update the position status and valuate it.
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
    Close the position and remove it from the portfolio.
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
    Update the portfolio positions
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
    # save the current portfolio value
    prev_value = p.value
    # compute the total portfolio value
    value = p.cash
    for i, key in posenum:
        pos = positions[key]
        valuate_position(pos, tdate)
        vpos[i] = pos.value
        value = value + vpos[i]
    p.value = value
    # now compute the weights
    for i, key in posenum:
        p.weights[i] = vpos[i] / p.value
    # update portfolio stats
    p.netprofit = p.value - prev_value
    p.netreturn = p.value / prev_value - 1.0
    p.totalprofit = p.value - p.startcap
    p.totalreturn = p.value / p.startcap - 1.0


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
        tsize = quantity
    else:
        if order == 'le' or order == 'se':
            pf = Frame.frames[frame_name(name, p.space)].df
            cv = pf.ix[tdate][p.posby]
            tsize = math.trunc((p.value * p.fixedfrac) / cv)
            if quantity < 0:
                tsize = -tsize
        else:
            tsize = -pos.quantity
    # instantiate and allocate the trade
    newtrade = Trade(name, order, tsize, price, tdate)
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
    # return trade size
    return tsize


#
# Function gen_portfolio
#

def gen_portfolio(model, system, group, tframe,
                  startcap=100000, posby='close'):
    """
    Create a portfolio from a trades frame
    """

    logger.info("Creating Portfolio for System %s", system.name)

    # Unpack the model data.

    base_dir = model.specs['base_dir']
    extension = model.specs['extension']
    project = model.specs['project']
    separator = model.specs['separator']
    directory = SSEP.join([base_dir, project])

    # Create the portfolio.

    gname = group.name
    gspace = group.space
    gmembers = group.members
    ff = 1.0 / len(gmembers)

    p = Portfolio(gname,
                  system.name,
                  startcap = startcap,
                  posby = posby,
                  restricted = False,
                  fixedfrac = ff)
    if not p:
        logger.error("Error creating Portfolio")

    # Build pyfolio data from the trades frame.

    start = tframe.index[0]
    end = tframe.index[-1]
    drange = date_range(start, end, freq=gspace.fractal)

    # Initialize return, position, and transaction data.

    rs = []
    pcols = list(gmembers)
    pcols.extend(['cash'])
    pf = DataFrame(index=drange, columns=pcols).fillna(0.0)
    ts = []

    # Iterate through the date range, updating the portfolio.

    for d in drange:
        logger.info("Updating Portfolio for %s", d)
        # process today's trades
        if d in tframe.index:
            trades = tframe.ix[d]
            if isinstance(trades, Series):
                trades = DataFrame(trades).transpose()
            for t in trades.iterrows():
                tdate = t[0]
                row = t[1]
                tsize = exec_trade(p, row['name'], row['order'], row['quantity'], row['price'], tdate)
                if tsize != 0:
                    ts.append((d, [tsize, row['price'], row['name']]))
                else:
                    logger.info("Trade could not be executed for %s", row['name'])
        # iterate through current positions
        positions = p.positions
        pfrow = pf.ix[d]
        for key in positions:
            pos = positions[key]
            if pos.quantity > 0:
                value = pos.value
            else:
                value = -pos.value
            pfrow[pos.name] = value
        pfrow['cash'] = p.cash
        # update the portfolio returns
        valuate_portfolio(p, d)
        rs.append((d, [p.netreturn]))

    # Create and record the returns frame for this system.

    logger.info("Recording Returns Frame")
    rspace = Space(system.name, 'returns', gspace.fractal)
    rf = DataFrame.from_items(rs, orient='index', columns=['return'])
    rfname = frame_name(gname, rspace)
    write_frame(rf, directory, rfname, extension, separator,
                index=True, index_label='date')
    del rspace

    # Record the positions frame for this system.

    logger.info("Recording Positions Frame")
    pspace = Space(system.name, 'positions', gspace.fractal)
    pfname = frame_name(gname, pspace)
    write_frame(pf, directory, pfname, extension, separator,
                index=True, index_label='date')
    del pspace

    # Create and record the transactions frame for this system.

    logger.info("Recording Transactions Frame")
    tspace = Space(system.name, 'transactions', gspace.fractal)
    tf = DataFrame.from_items(ts, orient='index', columns=['amount', 'price', 'symbol'])
    tfname = frame_name(gname, tspace)
    write_frame(tf, directory, tfname, extension, separator,
                index=True, index_label='date')
    del tspace

    # Return the portfolio.
    return p


#
# Function plot_returns
#

def plot_returns(model, system, group,
                 benchmark='SPY',
                 drawdown_periods=5):
    """
    Plot portfolio return information.
    """

    # Unpack the model data.

    base_dir = model.specs['base_dir']
    extension = model.specs['extension']
    project = model.specs['project']
    separator = model.specs['separator']
    directory = SSEP.join([base_dir, project])

    # Form file name

    gname = group.name
    gspace = group.space
    rspace = Space(system.name, 'returns', gspace.fractal)
    rfname = frame_name(gname, rspace)
    del rspace

    # Read in the returns file

    rf = read_frame(directory, rfname, extension, separator,
                    index_col='date', squeeze=True)
    rf.index = pd.to_datetime(rf.index, utc=True)

    # Show and plot returns

    benchmark_rets = get_symbol_rets(benchmark)
    show_perf_stats(rf, benchmark_rets)
    plot_returns(rf)
    plot_rolling_returns(rf, benchmark_rets)
    plot_rolling_returns(rf, benchmark_rets, volatility_match=True)
    plot_monthly_returns_heatmap(rf)
    plot_annual_returns(rf)
    plot_return_quantiles(rf)
    plot_rolling_sharpe(rf)

    # Show and plot drawdowns

    show_worst_drawdown_periods(rf)
    plot_drawdown_periods(rf, top=drawdown_periods)
    plot_drawdown_underwater(rf)


#
# Function plot_positions
#

def plot_positions(model, system, group):
    """
    Plot portfolio position information.
    """

    # Unpack the model data.

    base_dir = model.specs['base_dir']
    extension = model.specs['extension']
    project = model.specs['project']
    separator = model.specs['separator']
    directory = SSEP.join([base_dir, project])

    # Form file name

    gname = group.name
    gspace = group.space
    pspace = Space(system.name, 'positions', gspace.fractal)
    pfname = frame_name(gname, pspace)
    del pspace

    # Read in the positions file

    pf = read_frame(directory, pfname, extension, separator,
                    index_col='date', squeeze=True)
    pf.index = pd.to_datetime(pf.index, utc=True)

    # Position Plots TBD


#
# Function plot_transactions
#

def plot_transactions(model, system, group):
    """
    Plot portfolio transaction information.
    """

    # Unpack the model data.

    base_dir = model.specs['base_dir']
    extension = model.specs['extension']
    project = model.specs['project']
    separator = model.specs['separator']
    directory = SSEP.join([base_dir, project])

    # Form file name

    gname = group.name
    gspace = group.space
    tspace = Space(system.name, 'transactions', gspace.fractal)
    tfname = frame_name(gname, tspace)
    del tspace

    # Read in the transactions file

    tf = read_frame(directory, tfname, extension, separator,
                    index_col='date', squeeze=True)
    tf.index = pd.to_datetime(tf.index, utc=True)

    # Transaction Plots TBD
