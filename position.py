##############################################################
#
# Package   : AlphaPy
# Module    : position
# Version   : 1.0
# Copyright : Mark Conway
# Date      : June 29, 2013
#
##############################################################


#
# Imports
#

from globs import MULTIPLIERS
from frame import Frame
from frame import frame_name
import portfolio
import trade


#
# class Position
#

class Position:
	
	# __init__
	
	def __init__(self,
				 portfolio,
				 name,
				 opendate):
		subject = portfolio.subject
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
		self.pdata = Frame.frames[frame_name(name, space)]
		self.multiplier = multipliers[subject]

	# __str__
	
	def __str__(self):
		return self.name


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
		tts = 0		# total traded shares
		ttv = 0		# total traded value
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
		newtrade = trade.Trade(position.name, tradesize, cp, tdate)
		portfolio.update_portfolio(p, position, newtrade, tradesize)
		position.quantity = 0
	position.status = 'closed'
	portfolio.remove_position(p, position.name)
