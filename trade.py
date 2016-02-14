##############################################################
#
# Package   : AlphaPy
# Module    : trade
# Version   : 1.0
# Copyright : Mark Conway
# Date      : June 29, 2013
#
##############################################################


#
# Imports
#

from frame import frame_name
from frame import Frame
from math import trunc
import portfolio
import position


#
# class Trade
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
        portfolio.kick_out(p, trade.tdate)
        portfolio.stop_loss(p, trade.tdate)
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
            print "Required free margin: %d < added value: %d" % (freemargin, addedvalue)
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
        pos = position.Position(p, name, tdate)
        newpos = True
    # check the dynamic position sizing variable
    if not p.posby:
        psize = quantity
    else:
        if order == 'le' or order == 'se':
            pf = Frame.frames[frame_name(name, p.subject, p.space)]
            cv = pf.ix[tdate][p.posby]
            psize = trunc((p.value * p.fixedfrac) / cv)
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
            portfolio.add_position(p, name, pos)
            p.npos += 1
        # update the portfolio
        portfolio.update_portfolio(p, pos, newtrade, allocation)
        # if net position is zero, then close the position
        pflat = pos.quantity == 0
        if pflat:
            position.close_position(p, pos, tdate)
            p.npos -= 1
        return pos
    else:
        return None
