################################################################################
#
# Package   : AlphaPy
# Module    : market_variables
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
from alphapy.globals import BSEP, LOFF, ROFF, USEP
from alphapy.utilities import valid_name

from collections import OrderedDict
from importlib import import_module
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
    """Create a new variable as a key-value pair. All variables are stored
    in ``Variable.variables``. Duplicate keys or values are not allowed,
    unless the ``replace`` parameter is ``True``.

    Parameters
    ----------
    name : str
        Variable key.
    expr : str
        Variable value.
    replace : bool, optional
        Replace the current key-value pair if it already exists.

    Attributes
    ----------
    variables : dict
        Class variable for storing all known variables

    Examples
    --------
    
    >>> Variable('rrunder', 'rr_3_20 <= 0.9')
    >>> Variable('hc', 'higher_close')

    """

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
    r"""Parse a variable name into its respective components.

    Parameters
    ----------
    vname : str
        The name of the variable.

    Returns
    -------
    vxlag : str
        Variable name without the ``lag`` component.
    root : str
        The base variable name without the parameters.
    plist : list
        The parameter list.
    lag : int
        The offset starting with the current value [0]
        and counting back, e.g., an offset [1] means the
        previous value of the variable.

    Notes
    -----

    **AlphaPy** makes feature creation easy. The syntax
    of a variable name maps to a function call:

    xma_20_50 => xma(20, 50)

    Examples
    --------

    >>> vparse('xma_20_50[1]')
    # ('xma_20_50', 'xma', ['20', '50'], 1)

    """

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
    r"""Get the list of valid names in the expression.

    Parameters
    ----------
    expr : str
        A valid expression conforming to the Variable Definition Language.

    Returns
    -------
    vlist : list
        List of valid variable names.

    """
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
    r"""Get all of the antecedent variables. 

    Before applying a variable to a dataframe, we have to recursively
    get all of the child variables, beginning with the starting variable's
    expression. Then, we have to extract the variables from all the
    subsequent expressions. This process continues until all antecedent
    variables are obtained.

    Parameters
    ----------
    vname : str
        A valid variable stored in ``Variable.variables``.

    Returns
    -------
    all_variables : list
        The variables that need to be applied before ``vname``.

    Other Parameters
    ----------------
    Variable.variables : dict
        Global dictionary of variables

    """
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
    all_variables = list(OrderedDict.fromkeys(allv))
    return all_variables


#
# Function vsub
#

def vsub(v, expr):
    r"""Substitute the variable parameters into the expression.

    This function performs the parameter substitution when
    applying features to a dataframe. It is a mechanism for
    the user to override the default values in any given
    expression when defining a feature, instead of having
    to programmatically call a function with new values.  

    Parameters
    ----------
    v : str
        Variable name.
    expr : str
        The expression for substitution.

    Returns
    -------
    newexpr
        The expression with the new, substituted values.

    """
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
# Function vexec
#

def vexec(f, v, vfuncs=None):
    r"""Add a variable to the given dataframe.

    This is the core function for adding a variable to a dataframe.
    The default variable functions are already defined locally
    in ``alphapy.var``; however, you may want to define your
    own variable functions. If so, then the ``vfuncs`` parameter
    will contain the list of modules and functions to be imported
    and applied by the ``vexec`` function.

    To write your own variable function, your function must have
    a pandas *DataFrame* as an input parameter and must return
    a pandas *Series* that represents the new variable.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe to contain the new variable.
    v : str
        Variable to add to the dataframe.
    vfuncs : dict, optional
        Dictionary of external modules and functions.

    Returns
    -------
    f : pandas.DataFrame
        Dataframe with the new variable.

    Other Parameters
    ----------------
    Variable.variables : dict
        Global dictionary of variables

    """
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
            logger.debug("Expression: %s", estr)
            # pandas eval
            f[vxlag] = f.eval(estr)
        else:
            logger.debug("Did not find variable: %s", root)
            # Must be a function call
            func_name = root
            # Convert the parameter list and prepend the data frame
            newlist = []
            for p in plist:
                try:
                    newlist.append(int(p))
                except:
                    try:
                        newlist.append(float(p))
                    except:
                        newlist.append(p)
            newlist.insert(0, f)
            # Find the module and function
            module = None
            if vfuncs:
                for m in vfuncs:
                    funcs = vfuncs[m]
                    if func_name in funcs:
                        module = m
                        break
            # If the module was found, import the external treatment function,
            # else search the local namespace.
            if module:
                ext_module = import_module(module)
                func = getattr(my_module, func_name)
                # Create the variable by calling the function
                f[v] = func(*newlist)
            else:
                modname = globals()['__name__']
                module = sys.modules[modname]
                if func_name in dir(module):
                    func = getattr(module, func_name)
                    # Create the variable
                    f[v] = func(*newlist)
                else:
                    logger.debug("Could not find function %s", func_name)
    # if necessary, add the lagged variable
    if lag > 0 and vxlag in f.columns:
        f[v] = f[vxlag].shift(lag)
    # output frame
    return f


#
# Function vapply
#

def vapply(group, vname, vfuncs=None):
    r"""Apply a variable to multiple dataframes.

    Parameters
    ----------
    group : alphapy.Group
        The input group.
    vname : str
        The variable to apply to the ``group``.
    vfuncs : dict, optional
        Dictionary of external modules and functions.

    Returns
    -------
    None : None

    Other Parameters
    ----------------
    Frame.frames : dict
        Global dictionary of dataframes

    See Also
    --------
    vunapply

    """
    # get all frame names to apply variables
    gnames = [item.lower() for item in group.members]
    # get all the precedent variables
    allv = vtree(vname)
    # apply the variables to each frame
    for g in gnames:
        fname = frame_name(g, group.space)
        if fname in Frame.frames:
            f = Frame.frames[fname].df
            if not f.empty:
                for v in allv:
                    logger.debug("Applying variable %s to %s", v, g)
                    f = vexec(f, v, vfuncs)
            else:
                logger.debug("Frame for %s is empty", g)
        else:
            logger.debug("Frame not found: %s", fname)
                

#
# Function vmapply
#

def vmapply(group, vs, vfuncs=None):
    r"""Apply multiple variables to multiple dataframes.

    Parameters
    ----------
    group : alphapy.Group
        The input group.
    vs : list
        The list of variables to apply to the ``group``.
    vfuncs : dict, optional
        Dictionary of external modules and functions.

    Returns
    -------
    None : None

    See Also
    --------
    vmunapply

    """
    for v in vs:
        logger.info("Applying variable: %s", v)
        vapply(group, v, vfuncs)

        
#
# Function vunapply
#

def vunapply(group, vname):
    r"""Remove a variable from multiple dataframes.

    Parameters
    ----------
    group : alphapy.Group
        The input group.
    vname : str
        The variable to remove from the ``group``.

    Returns
    -------
    None : None

    Other Parameters
    ----------------
    Frame.frames : dict
        Global dictionary of dataframes

    See Also
    --------
    vapply

    """
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
    r"""Remove a list of variables from multiple dataframes.

    Parameters
    ----------
    group : alphapy.Group
        The input group.
    vs : list
        The list of variables to remove from the ``group``.

    Returns
    -------
    None : None

    See Also
    --------
    vmapply

    """
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
    r"""Take the maximum value between two columns in a dataframe.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``f``.
    c2 : str
        Name of the second column in the dataframe ``f``.

    Returns
    -------
    max_val : float
        The maximum value of the two columns.

    """
    max_val = max(f[c1], f[c2])
    return max_val


#
# Function c2min
#
    
def c2min(f, c1, c2):
    r"""Take the minimum value between two columns in a dataframe.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``f``.
    c2 : str
        Name of the second column in the dataframe ``f``.

    Returns
    -------
    min_val : float
        The minimum value of the two columns.

    """
    min_val = min(f[c1], f[c2])
    return min_val


#
# Function pchange1
#
    
def pchange1(f, c, o = 1):
    r"""Calculate the percentage change within the same variable.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    o : int
        Offset to the previous value.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = f[c] / f[c].shift(o) - 1.0
    return new_column


#
# Function pchange2
#

def pchange2(f, c1, c2):
    r"""Calculate the percentage change between two variables.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``f``.
    c2 : str
        Name of the second column in the dataframe ``f``.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = f[c1] / f[c2] - 1.0
    return new_column


#
# Function diff
#

def diff(f, c, n = 1):
    r"""Calculate the n-th order difference for the given variable.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    n : int
        The number of times that the values are differenced.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = np.diff(f[c], n)
    return new_column


#
# Function down
#

def down(f, c):
    r"""Find the negative values in the series.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = f[c] < 0
    return new_column


#
# Function up
#

def up(f, c):
    r"""Find the positive values in the series.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = f[c] > 0
    return new_column


#
# Function higher
#

def higher(f, c, o = 1):
    r"""Determine whether or not a series value is higher than
    the value ``o`` periods back.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    o : int, optional
        Offset value for shifting the series.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = f[c] > f[c].shift(o)
    return new_column


#
# Function highest
#

def highest(f, c, p = 20):
    r"""Calculate the highest value on a rolling basis.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    p : int
        The period over which to calculate the rolling maximum.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = f[c].rolling(p).max()
    return new_column


#
# Function lower
#

def lower(f, c, o = 1):
    r"""Determine whether or not a series value is lower than
    the value ``o`` periods back.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    o : int, optional
        Offset value for shifting the series.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = f[c] < f[c].shift(o)
    return new_column


#
# Function lowest
#

def lowest(f, c, p = 20):
    r"""Calculate the lowest value on a rolling basis.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    p : int
        The period over which to calculate the rolling minimum.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    return f[c].rolling(p).min()


#
# Function ma
#

def ma(f, c, p = 20):
    r"""Calculate the mean on a rolling basis.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    p : int
        The period over which to calculate the rolling mean.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *In statistics, a moving average (rolling average or running average)
    is a calculation to analyze data points by creating series of averages
    of different subsets of the full data set* [WIKI_MA]_.

    .. [WIKI_MA] https://en.wikipedia.org/wiki/Moving_average

    """
    new_column = f[c].rolling(p).mean()
    return new_column


#
# Function ema
#

def ema(f, c, p = 20):
    r"""Calculate the mean on a rolling basis.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    p : int
        The period over which to calculate the rolling mean.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *An exponential moving average (EMA) is a type of moving average
    that is similar to a simple moving average, except that more weight
    is given to the latest data* [IP_EMA]_.

    .. [IP_EMA] http://www.investopedia.com/terms/e/ema.asp

    """
    new_column = pd.ewma(f[c], span=p)
    return new_column


#
# Function maratio
#

def maratio(f, c, p1 = 1, p2 = 10):
    r"""Calculate the ratio of two moving averages.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    p1 : int
        The period of the first moving average.
    p2 : int
        The period of the second moving average.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = ma(f, c, p1) / ma(f, c, p2)
    return new_column


#
# Function net
#

def net(f, c='close', o = 1):
    r"""Calculate the net change of a given column.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    o : int, optional
        Offset value for shifting the series.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *Net change is the difference between the closing price of a security
    on the day's trading and the previous day's closing price. Net change
    can be positive or negative and is quoted in terms of dollars* [IP_NET]_.

    .. [IP_NET] http://www.investopedia.com/terms/n/netchange.asp

    """
    new_column = f[c] - f[c].shift(o)
    return new_column


#
# Function gap
#

def gap(f):
    r"""Calculate the gap percentage between the current open and
    the previous close.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``open`` and ``close``.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *A gap is a break between prices on a chart that occurs when the
    price of a stock makes a sharp move up or down with no trading
    occurring in between* [IP_GAP]_.

    .. [IP_GAP] http://www.investopedia.com/terms/g/gap.asp

    """
    c1 = 'open'
    c2 = 'close[1]'
    vexec(f, c2)
    new_column = 100 * pchange2(f, c1, c2)
    return new_column


#
# Function gapdown
#

def gapdown(f):
    r"""Determine whether or not there has been a gap down.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``open`` and ``close``.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    References
    ----------
    *A gap is a break between prices on a chart that occurs when the
    price of a stock makes a sharp move up or down with no trading
    occurring in between* [IP_GAP]_.

    """
    new_column = f['open'] < f['close'].shift(1)
    return new_column


#
# Function gapup
#

def gapup(f):
    r"""Determine whether or not there has been a gap up.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``open`` and ``close``.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    References
    ----------
    *A gap is a break between prices on a chart that occurs when the
    price of a stock makes a sharp move up or down with no trading
    occurring in between* [IP_GAP]_.

    """
    new_column = f['open'] > f['close'].shift(1)
    return new_column


#
# Function gapbadown
#

def gapbadown(f):
    r"""Determine whether or not there has been a breakaway gap down.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``open`` and ``low``.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    References
    ----------
    *A breakaway gap represents a gap in the movement of a stock price
    supported by levels of high volume* [IP_BAGAP]_.

    .. [IP_BAGAP] http://www.investopedia.com/terms/b/breakawaygap.asp

    """
    new_column = f['open'] < f['low'].shift(1)
    return new_column


#
# Function gapbaup
#

def gapbaup(f):
    r"""Determine whether or not there has been a breakaway gap up.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``open`` and ``high``.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    References
    ----------
    *A breakaway gap represents a gap in the movement of a stock price
    supported by levels of high volume* [IP_BAGAP]_.

    """
    new_column = f['open'] > f['high'].shift(1)
    return new_column


#
# Function truehigh
#

def truehigh(f):
    r"""Calculate the *True High* value.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *Today's high, or the previous close, whichever is higher* [TS_TR]_.

    .. [TS_TR] http://help.tradestation.com/09_01/tradestationhelp/charting_definitions/true_range.htm

    """
    c1 = 'low[1]'
    vexec(f, c1)
    c2 = 'high'
    new_column = f.apply(c2max, axis=1, args=[c1, c2])
    return new_column


#
# Function truelow
#

def truelow(f):
    r"""Calculate the *True Low* value.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *Today's low, or the previous close, whichever is lower* [TS_TR]_.

    """
    c1 = 'high[1]'
    vexec(f, c1)
    c2 = 'low'
    new_column = f.apply(c2min, axis=1, args=[c1, c2])
    return new_column


#
# Function truerange
#

def truerange(f):
    r"""Calculate the *True Range* value.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *True High - True Low* [TS_TR]_.

    """
    new_column = truehigh(f) - truelow(f)
    return new_column


#
# Function hlrange
#

def hlrange(f, p = 1):
    r"""Calculate the Range, the difference between High and Low.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.
    p : int
        The period over which the range is calculated.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = highest(f, 'high', p) - lowest(f, 'low', p)
    return new_column


#
# Function netreturn
#

def netreturn(f, c, o = 1):
    r"""Calculate the net return, or Return On Invesment (ROI)

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    o : int, optional
        Offset value for shifting the series.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *ROI measures the amount of return on an investment relative to the
    original cost. To calculate ROI, the benefit (or return) of an
    investment is divided by the cost of the investment, and the result
    is expressed as a percentage or a ratio* [IP_ROI]_.

    .. [IP_ROI] http://www.investopedia.com/terms/r/returnoninvestment.asp

    """
    new_column = 100 * pchange1(f, c, o)
    return new_column


#
# Function rindex
#

def rindex(f, ci, ch, cl, p = 1):
    r"""Calculate the *range index* spanning a given period ``p``.

    The **range index** is a number between 0 and 100 that
    relates the value of the index column ``ci`` to the
    high column ``ch`` and the low column ``cl``. For example,
    if the low value of the range is 10 and the high value
    is 20, then the range index for a value of 15 would be 50%.
    The range index for 18 would be 80%.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the columns ``ci``, ``ch``, and ``cl``.
    ci : str
        Name of the index column in the dataframe ``f``.
    ch : str
        Name of the high column in the dataframe ``f``.
    cl : str
        Name of the low column in the dataframe ``f``.
    p : int
        The period over which the range index of column ``ci``
        is calculated.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    o = p-1 if f[ci].name == 'open' else 0
    hh = highest(f, ch, p)
    ll = lowest(f, cl, p)
    fn = f[ci].shift(o) - ll
    fd = hh - ll
    new_column = 100 * fn / fd
    return new_column


#
# Function mval
#
   
def mval(f, c):
    r"""Get the negative value, otherwise zero.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.

    Returns
    -------
    new_val : float
        Negative value or zero.

    """
    new_val = -f[c] if f[c] < 0 else 0
    return new_val


#
# Function pval
#
  
def pval(f, c):
    r"""Get the positive value, otherwise zero.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.

    Returns
    -------
    new_val : float
        Positive value or zero.

    """
    new_val = f[c] if f[c] > 0 else 0
    return new_val


#
# Function dpc
#

def dpc(f, c):
    r"""Get the negative values, with positive values zeroed.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with column ``c``.
    c : str
        Name of the column.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = f.apply(mval, axis=1, args=[c])
    return new_column


#
# Function upc
#

def upc(f, c):
    r"""Get the positive values, with negative values zeroed.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with column ``c``.
    c : str
        Name of the column.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = f.apply(pval, axis=1, args=[c])
    return new_column


#
# Function rsi
#

def rsi(f, c, p = 14):
    r"""Calculate the Relative Strength Index (RSI).

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``net``.
    c : str
        Name of the column in the dataframe ``f``.
    p : int
        The period over which to calculate the RSI.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *Developed by J. Welles Wilder, the Relative Strength Index (RSI) is a momentum
    oscillator that measures the speed and change of price movements* [SC_RSI]_.

    .. [SC_RSI] http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:relative_strength_index_rsi

    """
    cdiff = 'net'
    vexec(f, cdiff)
    f['pval'] = upc(f, cdiff)
    f['mval'] = dpc(f, cdiff)
    upcs = ma(f, 'pval', p)
    dpcs = ma(f, 'mval', p)
    new_column = 100 - (100 / (1 + (upcs / dpcs)))
    return new_column


#
# Function gtval
#

def gtval(f, c1, c2):
    r"""Determine whether or not the first column of a dataframe
    is greater than the second.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``f``.
    c2 : str
        Name of the second column in the dataframe ``f``.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = f[c1] > f[c2]
    return new_column


#
# Function gtval0
#

def gtval0(f, c1, c2):
    r"""For positive values in the first column of the dataframe
    that are greater than the second column, get the value in
    the first column, otherwise return zero. 

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``f``.
    c2 : str
        Name of the second column in the dataframe ``f``.

    Returns
    -------
    new_val : float
        A positive value or zero.

    """
    if f[c1] > f[c2] and f[c1] > 0:
        new_val = f[c1]
    else:
        new_val = 0
    return new_val


#
# Function dmplus
#

def dmplus(f):
    r"""Calculate the Plus Directional Movement (+DM).

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *Directional movement is positive (plus) when the current high minus
    the prior high is greater than the prior low minus the current low.
    This so-called Plus Directional Movement (+DM) then equals the current
    high minus the prior high, provided it is positive. A negative value
    would simply be entered as zero* [SC_ADX]_.

    .. [SC_ADX] http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx

    """
    c1 = 'upmove'
    f[c1] = net(f, 'high')
    c2 = 'downmove'
    f[c2] = -net(f, 'low')
    new_column = f.apply(gtval0, axis=1, args=[c1, c2])
    return new_column


#
# Function dminus
#

def dminus(f):
    r"""Calculate the Minus Directional Movement (-DM).

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *Directional movement is negative (minus) when the prior low minus
    the current low is greater than the current high minus the prior high.
    This so-called Minus Directional Movement (-DM) equals the prior low
    minus the current low, provided it is positive. A negative value
    would simply be entered as zero* [SC_ADX]_.

    """
    c1 = 'downmove'
    f[c1] = -net(f, 'low')
    c2 = 'upmove'
    f[c2] = net(f, 'high')
    new_column = f.apply(gtval0, axis=1, args=[c1, c2])
    return new_column


#
# Function diplus
#

def diplus(f, p = 14):
    r"""Calculate the Plus Directional Indicator (+DI).

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.
    p : int
        The period over which to calculate the +DI.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *A component of the average directional index (ADX) that is used to
    measure the presence of an uptrend. When the +DI is sloping upward,
    it is a signal that the uptrend is getting stronger* [IP_PDI]_.

    .. [IP_PDI] http://www.investopedia.com/terms/p/positivedirectionalindicator.asp

    """
    tr = 'truerange'
    vexec(f, tr)
    atr = USEP.join(['atr', str(p)])
    vexec(f, atr)
    dmp = 'dmplus'
    vexec(f, dmp)
    new_column = 100 * f[dmp].ewm(span=p).mean() / f[atr]
    return new_column


#
# Function diminus
#

def diminus(f, p = 14):
    r"""Calculate the Minus Directional Indicator (-DI).

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.
    p : int
        The period over which to calculate the -DI.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *A component of the average directional index (ADX) that is used to
    measure the presence of a downtrend. When the -DI is sloping downward,
    it is a signal that the downtrend is getting stronger* [IP_NDI]_.

    .. [IP_NDI] http://www.investopedia.com/terms/n/negativedirectionalindicator.asp

    """
    tr = 'truerange'
    vexec(f, tr)
    atr = USEP.join(['atr', str(p)])
    vexec(f, atr)
    dmm = 'dmminus'
    f[dmm] = dminus(f)
    new_column = 100 * dminus(f).ewm(span=p).mean() / f[atr]
    return new_column


#
# Function adx
#

def adx(f, p = 14):
    r"""Calculate the Average Directional Index (ADX).

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with all columns required for calculation. If you
        are applying ADX through ``vapply``, then these columns are
        calculated automatically.
    p : int
        The period over which to calculate the ADX.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    The Average Directional Movement Index (ADX) was invented by J. Welles
    Wilder in 1978 [WIKI_ADX]_.  Its value reflects the strength of trend in any
    given instrument.

    .. [WIKI_ADX] https://en.wikipedia.org/wiki/Average_directional_movement_index

    """
    c1 = 'diplus'
    vexec(f, c1)
    c2 = 'diminus'
    vexec(f, c2)
    # calculations
    dip = f[c1]
    dim = f[c2]
    didiff = abs(dip - dim)
    disum = dip + dim
    new_column = 100 * didiff.ewm(span=p).mean() / disum
    return new_column


#
# Function abovema
#

def abovema(f, c, p = 50):
    r"""Determine those values of the dataframe that are above the
    moving average.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    p : int
        The period of the moving average.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = f[c] > ma(f, c, p)
    return new_column


#
# Function belowma
#

def belowma(f, c, p = 50):
    r"""Determine those values of the dataframe that are below the
    moving average.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    p : int
        The period of the moving average.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = f[c] < ma(f, c, p)
    return new_column


#
# Function xmadown
#

def xmadown(f, c='close', pfast = 20, pslow = 50):
    r"""Determine those values of the dataframe that are below the
    moving average.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str, optional
        Name of the column in the dataframe ``f``.
    pfast : int, optional
        The period of the fast moving average.
    pslow : int, optional
        The period of the slow moving average.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    References
    ----------
    *In the statistics of time series, and in particular the analysis
    of financial time series for stock trading purposes, a moving-average
    crossover occurs when, on plotting two moving averages each based
    on different degrees of smoothing, the traces of these moving averages
    cross* [WIKI_XMA]_.

    .. [WIKI_XMA] https://en.wikipedia.org/wiki/Moving_average_crossover

    """
    sma = ma(f, c, pfast)
    sma_prev = sma.shift(1)
    lma = ma(f, c, pslow)
    lma_prev = lma.shift(1)
    new_column = (sma < lma) & (sma_prev > lma_prev)
    return new_column


#
# Function xmaup
#

def xmaup(f, c='close', pfast = 20, pslow = 50):
    r"""Determine those values of the dataframe that are below the
    moving average.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str, optional
        Name of the column in the dataframe ``f``.
    pfast : int, optional
        The period of the fast moving average.
    pslow : int, optional
        The period of the slow moving average.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    References
    ----------
    *In the statistics of time series, and in particular the analysis
    of financial time series for stock trading purposes, a moving-average
    crossover occurs when, on plotting two moving averages each based
    on different degrees of smoothing, the traces of these moving averages
    cross* [WIKI_XMA]_.

    """
    sma = ma(f, c, pfast)
    sma_prev = sma.shift(1)
    lma = ma(f, c, pslow)
    lma_prev = lma.shift(1)
    new_column = (sma > lma) & (sma_prev < lma_prev)
    return new_column
