################################################################################
#
# Package   : AlphaPy
# Module    : var
# Created   : July 11, 2013
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
# Variables
# ---------
#
# Numeric substitution is allowed for any number in the expression.
# Offsets are allowed in event expressions but cannot be substituted.
#
# Examples
# --------
#
# Variable('rrunder', 'rr_3_20 <= 0.9')
#
# 'rrunder_2_10_0.7'
# 'rrunder_2_10_0.9'
# 'xmaup_20_50_20_200'
# 'xmaup_10_50_20_50'
#


#
# Imports
#

from alphapy.alias import get_alias
from alphapy.frame import Frame
from alphapy.frame import frame_name
from alphapy.globs import BSEP, LOFF, ROFF, USEP
from alphapy.util import valid_name

from collections import OrderedDict
import logging
import numpy as np
import pandas as pd
import parser
import re
import sys


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Class Variable
#

class Variable(object):

    # class variable to track all variables

    variables = {}

    # function __new__

    def __new__(cls,
                name,
                expr,
                replace = False):
        # code
        efound = expr in [Variable.variables[key].expr for key in Variable.variables]
        if efound:
            key = [key for key in Variable.variables if expr in Variable.variables[key].expr]
            logger.info("Expression '%s' already exists for key %s", expr, key)
            return
        else:
            if replace or not name in Variable.variables:
                if not valid_name(name):
                    logger.info("Invalid variable key: %s", name)
                    return
                try:
                    result = parser.expr(expr)
                except:
                    logger.info("Invalid expression: %s", expr)
                    return
                return super(Variable, cls).__new__(cls)
            else:
                logger.info("Key %s already exists", name)

    # function __init__

    def __init__(self,
                 name,
                 expr,
                 replace = False):
        # code
        self.name = name;
        self.expr = expr;
        # add key with expression
        Variable.variables[name] = self
            
    # function __str__

    def __str__(self):
        return self.expr


#
# Function vparse
#

def vparse(vname):
    # split along lag first
    lsplit = vname.split(LOFF)
    vxlag = lsplit[0]
    # if necessary, substitute any alias
    root = vxlag.split(USEP)[0]
    alias = get_alias(root)
    if alias:
        vxlag = vxlag.replace(root, alias)
    vsplit = vxlag.split(USEP)
    root = vsplit[0]
    plist = vsplit[1:]
    # extract lag
    lag = 0
    if len(lsplit) > 1:
        # lag is present
        slag = lsplit[1].replace(ROFF, '')
        if len(slag) > 0:
            lpat = r'(^-?[0-9]+$)'
            lre = re.compile(lpat)
            if lre.match(slag):
                lag = int(slag)
    # return all components
    return vxlag, root, plist, lag


#
# Function allvars
#

def allvars(expr):
    regex = re.compile('\w+')
    items = regex.findall(expr)
    vlist = []
    for item in items:
        if valid_name(item):
            vlist.append(item)
    return vlist


#
# Function vtree
#

def vtree(vname):
    allv = []
    def vwalk(allv, vname):
        vxlag, root, plist, lag = vparse(vname)
        if root in Variable.variables:
            root_expr = Variable.variables[root].expr
            expr = vsub(vname, root_expr)
            av = allvars(expr)
            for v in av:
                vwalk(allv, v)
        else:
            for p in plist:
                if valid_name(p):
                    vwalk(allv, p)
        allv.append(vname)
        return allv
    allv = vwalk(allv, vname)
    return list(OrderedDict.fromkeys(allv))


#
# Function vsub
#

def vsub(v, expr):
    # numbers pattern
    npat = '[-+]?[0-9]*\.?[0-9]+'
    nreg = re.compile(npat)
    # find all number locations in variable name
    vnums = nreg.findall(v)
    viter = nreg.finditer(v)
    vlocs = []
    for match in viter:
        vlocs.append(match.span())
    # find all number locations in expression
    # find all non-number locations as well
    elen = len(expr)
    enums = nreg.findall(expr)
    eiter = nreg.finditer(expr)
    elocs = []
    enlocs = []
    index = 0
    for match in eiter:
        eloc = match.span()
        elocs.append(eloc)
        enlocs.append((index, eloc[0]))
        index = eloc[1]
    # build new expression
    newexpr = str()
    for i, enloc in enumerate(enlocs):
        if i < len(vlocs):
            newexpr += expr[enloc[0]:enloc[1]] + v[vlocs[i][0]:vlocs[i][1]]
        else:
            newexpr += expr[enloc[0]:enloc[1]] + expr[elocs[i][0]:elocs[i][1]]
    if elocs:
        estart = elocs[len(elocs)-1][1]
    else:
        estart = 0
    newexpr += expr[estart:elen]
    return newexpr


#
# Function vquote
#

def vquote(param):
    npat = '[-+]?[0-9]*\.?[0-9]+'
    nreg = re.compile(npat)
    if nreg.findall(param):
        return "%s" % param
    else:
        return '\'%s\'' % param

    
#
# Function vexec
#

def vexec(f, v):
    vxlag, root, plist, lag = vparse(v)
    logger.debug("vexec : %s", v)
    logger.debug("vxlag : %s", vxlag)
    logger.debug("root  : %s", root)
    logger.debug("plist : %s", plist)
    logger.debug("lag   : %s", lag)
    if vxlag not in f.columns:
        if root in Variable.variables:
            logger.debug("Found variable %s: ", root)
            vroot = Variable.variables[root]
            expr = vroot.expr
            expr_new = vsub(vxlag, expr)
            estr = "%s" % expr_new
            estr = BSEP.join([vxlag, '=', estr])
            logger.debug("Expression: %s", estr)
            # pandas eval
            f.eval(estr, inplace=True)
        else:
            logger.debug("Did not find variable: %s", root)
            # must be a function call
            fname = root
            modname = globals()['__name__']
            module = sys.modules[modname]
            # create a call if the function was found
            if fname in dir(module):
                if plist:
                    params = [vquote(p) for p in plist]
                    fcall = fname + '(f, ' + ', '.join(params) + ')'
                    logger.debug("Function Call: %s", fcall)
                else:
                    fcall = fname + '(f)'    
                estr = "%s" % fcall
                vstr = "f['%s'] = " % v
                estr = vstr + estr
                exec(estr)
            else:
                logger.debug("Could not find function %s", fname)
    # if necessary, add the lagged variable
    if lag > 0 and vxlag in f.columns:
        f[v] = f[vxlag].shift(lag)
    # output frame
    return f


#
# Function vapply
#

def vapply(group, vname):
    # get all frame names to apply variables
    gnames = [item.lower() for item in group.members]
    # get all the precedent variables
    allv = vtree(vname)
    # apply the variables to each frame
    for g in gnames:
        fname = frame_name(g, group.space)
        if fname in Frame.frames:
            f = Frame.frames[fname].df
            for v in allv:
                logger.debug("Applying variable %s to %s", v, g)
                f = vexec(f, v)
        else:
            logger.info("Frame not found: %s", fname)
                

#
# Function vmapply
#

def vmapply(group, vs):
    for v in vs:
        logger.info("Applying variable: %s", v)
        vapply(group, v)

        
#
# Function vunapply
#

def vunapply(group, vname):
    # get all frame names to apply variables
    gnames = [item.lower() for item in group.all_members()]
    # apply the variables to each frame
    for g in gnames:
        fname = frame_name(g, group.space)
        if fname in Frame.frames:
            f = Frame.frames[fname].df
            logger.info("Unapplying variable %s from %s", vname, g)
            if vname not in f.columns:
                logger.info("Variable %s not in %s frame", vname, g)
            else:
                estr = "Frame.frames['%s'].df = f.df.drop('%s', axis=1)" \
                        % (fname, vname)
                exec(estr)
        else:
            logger.info("Frame not found: %s", fname)
            

#
# Function vmunapply
#

def vmunapply(group, vs):
    for v in vs:
        vunapply(group, v)


#
# This is the reference for all internal and external variable functions.
#
#
# 1. datetime functions
#
#    date, datetime, time, timedelta
#
# 2. numpy unary ufuncs (PDA p. 96)
#
#    abs, ceil, cos, exp, floor, log, log10, log2, modf, rint, sign,
#    sin, square, sqrt, tan
#
# 3. moving window and exponential functions (PDA p. 323)
#
#    rolling, ewm
#
# 5. pandas descriptive and summary statistical functions (PDA p. 139)
#
#    argmin, argmax, count, cummax, cummin, cumprod, cumsum, describe,
#    diff, idxmin, idxmax, kurt, mad, max, mean, median, min, pct_change,
#    quantile, skew, std, sum, var
#
# 6. time series (PDA p. 289-328)
#


#
# Function c2max
#
    
def c2max(f, c1, c2):
    return max(f[c1], f[c2])


#
# Function c2min
#
    
def c2min(f, c1, c2):
    return min(f[c1], f[c2])


#
# Function pchange1
#
    
def pchange1(f, c, o = 1):
    return f[c] / f[c].shift(o) - 1.0


#
# Function pchange2
#

def pchange2(f, c1, c2):
    return f[c1] / f[c2] - 1.0


#
# Function diff
#

def diff(f, c, n = 1):
    return np.diff(f[c], n)


#
# Function down
#

def down(f, c):
    return f[c] < 0


#
# Function up
#

def up(f, c):
    return f[c] > 0


#
# Function higher
#

def higher(f, c, o = 1):
    return f[c] > f[c].shift(o)


#
# Function highest
#

def highest(f, c, p = 20):
    return f[c].rolling(p).max()


#
# Function lower
#

def lower(f, c, o = 1):
    return f[c] < f[c].shift(o)


#
# Function lowest
#

def lowest(f, c, p = 20):
    return f[c].rolling(p).min()


#
# Function ma
#

def ma(f, c, p = 20):
    return f[c].rolling(p).mean()


#
# Function ema
#

def ema(f, c, p = 20):
    return pd.ewma(f[c], span=p)


#
# Function maratio
#

def maratio(f, c, p1 = 1, p2 = 10):
    return ma(f, c, p1) / ma(f, c, p2)


#
# Function net
#

def net(f, c='close', o = 1):
    return f[c] - f[c].shift(o)


#
# Function gap
#

def gap(f):
    c1 = 'open'
    c2 = 'close[1]'
    vexec(f, c2)
    return 100 * pchange2(f, c1, c2)


#
# Function gapdown
#

def gapdown(f):
    return f['open'] < f['close'].shift(1)


#
# Function gapup
#

def gapup(f):
    return f['open'] > f['close'].shift(1)


#
# Function gapbadown
#

def gapbadown(f):
    return f['open'] < f['low'].shift(1)


#
# Function gapbaup
#

def gapbaup(f):
    return f['open'] > f['high'].shift(1)


#
# Function truehigh
#

def truehigh(f):
    c1 = 'low[1]'
    vexec(f, c1)
    c2 = 'high'
    return f.apply(c2max, axis=1, args=[c1, c2])


#
# Function truelow
#

def truelow(f):
    c1 = 'high[1]'
    vexec(f, c1)
    c2 = 'low'
    return f.apply(c2min, axis=1, args=[c1, c2])


#
# Function truerange
#

def truerange(f):
    return truehigh(f) - truelow(f)


#
# Function hlrange
#

def hlrange(f, p = 1):
    return highest(f, 'high', p) - lowest(f, 'low', p)


#
# Function netreturn
#

def netreturn(f, c, o = 1):
    return 100 * pchange1(f, c, o)


#
# Function rindex
#

def rindex(f, ci, ch, cl, p = 1):
    o = p-1 if f[ci].name == 'open' else 0
    hh = highest(f, ch, p)
    ll = lowest(f, cl, p)
    fn = f[ci].shift(o) - ll
    fd = hh - ll
    return 100 * fn / fd


#
# Function mval
#
   
def mval(f, c):
    return -f[c] if f[c] < 0 else 0


#
# Function pval
#
  
def pval(f, c):
    return f[c] if f[c] > 0 else 0


#
# Function dpc
#

def dpc(f, c):
    return f.apply(mval, axis=1, args=[c])


#
# Function upc
#

def upc(f, c):
    return f.apply(pval, axis=1, args=[c])


#
# Function rsi
#

def rsi(f, c, p = 14):
    cdiff = 'net'
    vexec(f, cdiff)
    f['pval'] = upc(f, cdiff)
    f['mval'] = dpc(f, cdiff)
    upcs = ma(f, 'pval', p)
    dpcs = ma(f, 'mval', p)
    return 100 - (100 / (1 + (upcs / dpcs)))


#
# Function gtval
#

def gtval(f, c1, c2):
    return f[c1] > f[c2]


#
# Function gtval0
#

def gtval0(f, c1, c2):
    if f[c1] > f[c2] and f[c1] > 0:
        return f[c1]
    else:
        return 0


#
# Function dmplus
#

def dmplus(f):
    c1 = 'upmove'
    f[c1] = net(f, 'high')
    c2 = 'downmove'
    f[c2] = -net(f, 'low')
    return f.apply(gtval0, axis=1, args=[c1, c2])


#
# Function dminus
#

def dminus(f):
    c1 = 'downmove'
    f[c1] = -net(f, 'low')
    c2 = 'upmove'
    f[c2] = net(f, 'high')
    return f.apply(gtval0, axis=1, args=[c1, c2])


#
# Function diplus
#

def diplus(f, p = 14):
    tr = 'truerange'
    vexec(f, tr)
    atr = USEP.join(['atr', str(p)])
    vexec(f, atr)
    dmp = 'dmplus'
    vexec(f, dmp)
    return 100 * f[dmp].ewm(span=p).mean() / f[atr]


#
# Function diminus
#

def diminus(f, p = 14):
    tr = 'truerange'
    vexec(f, tr)
    atr = USEP.join(['atr', str(p)])
    vexec(f, atr)
    dmm = 'dmminus'
    f[dmm] = dminus(f)
    return 100 * dminus(f).ewm(span=p).mean() / f[atr]


#
# Function adx
#

def adx(f, p = 14):
    c1 = 'diplus'
    vexec(f, c1)
    c2 = 'diminus'
    vexec(f, c2)
    # calculations
    dip = f[c1]
    dim = f[c2]
    didiff = abs(dip - dim)
    disum = dip + dim
    return 100 * didiff.ewm(span=p).mean() / disum


#
# Function abovema
#

def abovema(f, c, p = 50):
    return f[c] > ma(f, c, p)


#
# Function belowma
#

def belowma(f, c, p = 50):
    return f[c] < ma(f, c, p)


#
# Function xmaup
#

def xmaup(f, c, pshort = 20, plong = 50):
    sma = ma(f, c, pshort)
    sma_prev = sma.shift(1)
    lma = ma(f, c, plong)
    lma_prev = lma.shift(1)
    return sma > lma and sma_prev < lma_prev
