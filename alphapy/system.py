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
from alphapy.globs import SSEP
from alphapy.space import Space
from alphapy.portfolio import Trade
from alphapy.var import vexec

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
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Parameters
    ----------
    msg : str
        Human readable string describing the exception.
    code : :obj:`int`, optional
        Numeric error code.

    Attributes
    ----------
    msg : str
        Human readable string describing the exception.
    code : int
        Numeric error code.

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
# Class Orders
#

class Orders:
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Parameters
    ----------
    msg : str
        Human readable string describing the exception.
    code : :obj:`int`, optional
        Numeric error code.

    Attributes
    ----------
    msg : str
        Human readable string describing the exception.
    code : int
        Numeric error code.

    """
    le = 'le'
    se = 'se'
    lx = 'lx'
    sx = 'sx'
    lh = 'lh'
    sh = 'sh'


#
# Function long_short
#

def long_short(system, name, space, quantity):
    r"""Generate the list of trades based on the long and short events

    Several sentences providing an extended description. Refer to
    variables using back-ticks, e.g. `var`.

    Parameters
    ----------
    var1 : array_like
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.  We can also refer to
        variables like `var1`.
    var2 : int
        The type above can either refer to an actual Python type
        (e.g. ``int``), or describe the type of the variable in more
        detail, e.g. ``(N,) ndarray`` or ``array_like``.
    long_var_name : {'hi', 'ho'}, optional
        Choices in brackets, default first when optional.

    Returns
    -------
    type
        Explanation of anonymous return value of type ``type``.
    describe : type
        Explanation of return value named `describe`.
    out : type
        Explanation of `out`.

    Other Parameters
    ----------------
    only_seldom_used_keywords : type
        Explanation
    common_parameters_listed_above : type
        Explanation

    Raises
    ------
    BadException
        Because you shouldn't have done that.

    See Also
    --------
    otherfunc : relationship (optional)
    newfunc : Relationship (optional), which could be fairly long, in which
              case the line wraps here.
    thirdfunc, fourthfunc, fifthfunc

    Notes
    -----
    Notes about the implementation algorithm (if needed).

    This can have multiple paragraphs.

    You may include some math:

    .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}

    And even use a greek symbol like :math:`omega` inline.

    References
    ----------
    Cite the relevant literature, e.g. [1]_.  You may also cite these
    references in the notes section above.

    .. [1] O. McNoleg, "The integration of GIS, remote sensing,
       expert systems and adaptive co-kriging for environmental habitat
       modelling of the Highland Haggis using object-oriented, fuzzy-logic
       and neural-network techniques," Computers & Geosciences, vol. 22,
       pp. 585-588, 1996.

    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> a = [1, 2, 3]
    >>> print [x + 3 for x in a]
    [4, 5, 6]
    >>> print "a\n\nb"
    a
    b

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

def open_range_breakout(name, space, quantity):
    r"""Open Range Breakout

    Several sentences providing an extended description. Refer to
    variables using back-ticks, e.g. `var`.

    Parameters
    ----------
    var1 : array_like
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.  We can also refer to
        variables like `var1`.
    var2 : int
        The type above can either refer to an actual Python type
        (e.g. ``int``), or describe the type of the variable in more
        detail, e.g. ``(N,) ndarray`` or ``array_like``.
    long_var_name : {'hi', 'ho'}, optional
        Choices in brackets, default first when optional.

    Returns
    -------
    type
        Explanation of anonymous return value of type ``type``.
    describe : type
        Explanation of return value named `describe`.
    out : type
        Explanation of `out`.

    Other Parameters
    ----------------
    only_seldom_used_keywords : type
        Explanation
    common_parameters_listed_above : type
        Explanation

    Raises
    ------
    BadException
        Because you shouldn't have done that.

    See Also
    --------
    otherfunc : relationship (optional)
    newfunc : Relationship (optional), which could be fairly long, in which
              case the line wraps here.
    thirdfunc, fourthfunc, fifthfunc

    Notes
    -----
    Notes about the implementation algorithm (if needed).

    This can have multiple paragraphs.

    You may include some math:

    .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}

    And even use a greek symbol like :math:`omega` inline.

    References
    ----------
    Cite the relevant literature, e.g. [1]_.  You may also cite these
    references in the notes section above.

    .. [1] O. McNoleg, "The integration of GIS, remote sensing,
       expert systems and adaptive co-kriging for environmental habitat
       modelling of the Highland Haggis using object-oriented, fuzzy-logic
       and neural-network techniques," Computers & Geosciences, vol. 22,
       pp. 585-588, 1996.

    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> a = [1, 2, 3]
    >>> print [x + 3 for x in a]
    [4, 5, 6]
    >>> print "a\n\nb"
    a
    b

    """
    # system parameters
    trigger_first = 7
    trigger_last = 56
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
        elif bar_number < trigger_first:
            # set opening range
            if h > hh:
                hh = h
            if l < ll:
                ll = l
        else:
            if not traded and bar_number < trigger_last:
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
               quantity = 1):
    r"""Run a system for a given group, creating a trades frame

    Several sentences providing an extended description. Refer to
    variables using back-ticks, e.g. `var`.

    Parameters
    ----------
    var1 : array_like
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.  We can also refer to
        variables like `var1`.
    var2 : int
        The type above can either refer to an actual Python type
        (e.g. ``int``), or describe the type of the variable in more
        detail, e.g. ``(N,) ndarray`` or ``array_like``.
    long_var_name : {'hi', 'ho'}, optional
        Choices in brackets, default first when optional.

    Returns
    -------
    type
        Explanation of anonymous return value of type ``type``.
    describe : type
        Explanation of return value named `describe`.
    out : type
        Explanation of `out`.

    Other Parameters
    ----------------
    only_seldom_used_keywords : type
        Explanation
    common_parameters_listed_above : type
        Explanation

    Raises
    ------
    BadException
        Because you shouldn't have done that.

    See Also
    --------
    otherfunc : relationship (optional)
    newfunc : Relationship (optional), which could be fairly long, in which
              case the line wraps here.
    thirdfunc, fourthfunc, fifthfunc

    Notes
    -----
    Notes about the implementation algorithm (if needed).

    This can have multiple paragraphs.

    You may include some math:

    .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}

    And even use a greek symbol like :math:`omega` inline.

    References
    ----------
    Cite the relevant literature, e.g. [1]_.  You may also cite these
    references in the notes section above.

    .. [1] O. McNoleg, "The integration of GIS, remote sensing,
       expert systems and adaptive co-kriging for environmental habitat
       modelling of the Highland Haggis using object-oriented, fuzzy-logic
       and neural-network techniques," Computers & Geosciences, vol. 22,
       pp. 585-588, 1996.

    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> a = [1, 2, 3]
    >>> print [x + 3 for x in a]
    [4, 5, 6]
    >>> print "a\n\nb"
    a
    b

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
                tlist = globals()[system_name](symbol, gspace, quantity)
            except:
                logger.info('Could not execute system for %s', symbol)
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
