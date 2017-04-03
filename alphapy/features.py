################################################################################
#
# Package   : AlphaPy
# Module    : features
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

from alphapy.estimators import ModelType
from alphapy.globs import BSEP, NULLTEXT, USEP
from alphapy.var import Variable

import category_encoders as ce
from enum import Enum, unique
from gplearn.genetic import SymbolicTransformer
from importlib import import_module
from itertools import groupby
import logging
import math
import numpy as np
import pandas as pd
import re
from scipy import sparse
import scipy.stats as sps
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Encoder Types
#

@unique
class Encoders(Enum):
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
    backdiff = 1
    binary = 2
    factorize = 3
    helmert = 4
    onehot = 5
    ordinal = 6
    polynomial = 7
    sumcont = 8


#
# Scaler Types
#

@unique
class Scalers(Enum):
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
    minmax = 1
    standard = 2


#
# Define feature scoring functions
#

feature_scorers = {'f_classif'    : f_classif,
                   'chi2'         : chi2,
                   'f_regression' : f_regression,
                   'SelectKBest'  : SelectKBest,
                   'SelectFpr'    : SelectFpr,
                   'SelectFdr'    : SelectFdr,
                   'SelectFwe'    : SelectFwe}


#
# Function rtotal
#
# Example: rvec.rolling(window=20).apply(rtotal)
#

def rtotal(vec):
    r"""Read in data from the given directory in a given format.

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

    tcount = np.count_nonzero(vec)
    fcount = len(vec) - tcount
    return tcount - fcount


#
# Function runs
#
# Example: rvec.rolling(window=20).apply(runs)
#

def runs(vec):
    r"""Read in data from the given directory in a given format.

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

    return len(list(groupby(vec)))


#
# Function streak
#
# Example: rvec.rolling(window=20).apply(streak)
#

def streak(vec):
    r"""Read in data from the given directory in a given format.

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

    return [len(list(g)) for k, g in groupby(vec)][-1]


#
# Function zscore
#
# Example: rvec.rolling(window=20).apply(zscore)
#

def zscore(vec):
    r"""Read in data from the given directory in a given format.

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

    n1 = np.count_nonzero(vec)
    n2 = len(vec) - n1
    fac1 = float(2 * n1 * n2)
    fac2 = float(n1 + n2)
    rbar = fac1 / fac2 + 1
    sr2num = fac1 * (fac1 - n1 - n2)
    sr2den = math.pow(fac2, 2) * (fac2 - 1)
    sr = math.sqrt(sr2num / sr2den)
    if sr2den and sr:
        zscore = (runs(vec) - rbar) / sr
    else:
        zscore = 0
    return zscore

    
#
# Function runs_test
#

def runs_test(f, c, wfuncs, window):
    r"""Read in data from the given directory in a given format.

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

    fc = f[c]
    all_funcs = {'runs'   : runs,
                 'streak' : streak,
                 'rtotal' : rtotal,
                 'zscore' : zscore}
    # use all functions
    if 'all' in wfuncs:
        wfuncs = all_funcs.keys()
    # apply each of the runs functions
    new_features = pd.DataFrame()
    for w in wfuncs:
        if w in all_funcs:
            new_feature = fc.rolling(window=window).apply(all_funcs[w])
            new_feature.fillna(0, inplace=True)
            frames = [new_features, new_feature]
            new_features = pd.concat(frames, axis=1)
        else:
            logger.info("Runs Function %s not found", w)
    return new_features


#
# Function split_to_letters
#

def split_to_letters(f, c):
    r"""Read in data from the given directory in a given format.

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

    fc = f[c]
    new_feature = None
    dtype = fc.dtypes
    if dtype == 'object':
        fc.fillna(NULLTEXT, inplace=True)
        maxlen = fc.str.len().max()
        if maxlen > 1:
            new_feature = fc.apply(lambda x: BSEP.join(list(x)))
    return new_feature


#
# Function texplode
#

def texplode(f, c):
    r"""Read in data from the given directory in a given format.

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

    fc = f[c]
    maxlen = fc.str.len().max()
    fc.fillna(maxlen * BSEP, inplace=True)
    fpad = str().join(['{:', BSEP, '>', str(maxlen), '}'])
    fcpad = fc.apply(fpad.format)
    fcex = fcpad.apply(lambda x: pd.Series(list(x)))
    return pd.get_dummies(fcex)


#
# Function cvectorize
#

def cvectorize(f, c, n):
    r"""Read in data from the given directory in a given format.

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

    fc = f[c]
    fc.fillna(BSEP, inplace=True)
    cvect = CountVectorizer(ngram_range=[1, n], analyzer='char')
    cfeat = cvect.fit_transform(fc)
    tfidf_transformer = TfidfTransformer()
    return tfidf_transformer.fit_transform(cfeat).toarray()


#
# Function apply_treatment
#

def apply_treatment(fnum, fname, df, nvalues, fparams):
    r"""Process any special treatments from the configuration file.

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

    logger.info("Feature %d: %s is a special treatment with %d unique values",
                fnum, fname, nvalues)
    # Extract the treatment parameter list
    module = fparams[0]
    func_name = fparams[1]
    plist = fparams[2:]
    # Import the external treatment function
    ext_module = import_module(module)
    func = getattr(ext_module, func_name)
    # Prepend the parameter list with the data frame and feature name
    plist.insert(0, fname)
    plist.insert(0, df)
    # Apply the treatment
    logger.info("Applying function %s from module %s to feature %s",
                func_name, module, fname)
    return func(*plist)


#
# Function impute_values
#

def impute_values(features, dt):
    r"""Read in data from the given directory in a given format.

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

    try:
        nfeatures = features.shape[1]
    except:
        features = features.values.reshape(-1, 1)
    if dt == 'float64':
        imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    elif dt == 'int64' or dt == 'bool':
        imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    else:
        raise TypeError('Data Type %s is invalid for imputation', dt)
    imputed = imp.fit_transform(features)
    return imputed


#
# Function get_numerical_features
#

def get_numerical_features(fnum, fname, df, nvalues, dt, logt, plevel):
    r"""Get numerical features by looking for float and integer values.

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

    feature = df[fname]
    if len(feature) == nvalues:
        logger.info("Feature %d: %s is a numerical feature of type %s with maximum number of values %d",
                    fnum, fname, dt, nvalues)
    else:
        logger.info("Feature %d: %s is a numerical feature of type %s with %d unique values",
                    fnum, fname, dt, nvalues)
    # imputer for float, integer, or boolean data types
    new_values = impute_values(feature, dt)
    # log-transform any values that do not fit a normal distribution
    if logt and np.all(new_values > 0):
        stat, pvalue = sps.normaltest(new_values)
        if pvalue <= plevel:
            logger.info("Feature %d: %s is not normally distributed [p-value: %f]",
                        fnum, fname, pvalue)
            new_values = np.log(new_values)
    return new_values


#
# Function get_polynomials
#

def get_polynomials(features, poly_degree, input_names=None):
    r"""Get feature interactions and possibly polynomial interactions.

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

    polyf = PolynomialFeatures(interaction_only=True,
                               degree=poly_degree,
                               include_bias=False)
    poly_features = polyf.fit_transform(features)
    return poly_features


#
# Function get_text_features
#

def get_text_features(fnum, fname, df, nvalues, dummy_limit,
                      vectorize, ngrams_max):
    r"""Vectorize a text feature and transform to TF-IDF format.

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

    feature = df[fname]
    min_length = int(feature.str.len().min())
    max_length = int(feature.str.len().max())
    if len(feature) == nvalues:
        logger.info("Feature %d: %s is a text feature [%d:%d] with maximum number of values %d",
                    fnum, fname, min_length, max_length, nvalues)
    else:
        logger.info("Feature %d: %s is a text feature [%d:%d] with %d unique values",
                    fnum, fname, min_length, max_length, nvalues)
    # need a null text placeholder for vectorization
    feature.fillna(value=NULLTEXT, inplace=True)
    # vectorization creates many columns, otherwise just factorize
    if vectorize:
        logger.info("Feature %d: %s => Attempting Vectorization", fnum, fname)
        count_vect = CountVectorizer(ngram_range=[1, ngrams_max])
        try:
            count_feature = count_vect.fit_transform(feature)
            tfidf_transformer = TfidfTransformer()
            new_features = tfidf_transformer.fit_transform(count_feature).todense()
            logger.info("Feature %d: %s => Vectorization Succeeded", fnum, fname)
        except:
            logger.info("Feature %d: %s => Vectorization Failed", fnum, fname)
            new_features, uniques = pd.factorize(feature)
    else:
        logger.info("Feature %d: %s => Factorization", fnum, fname)
        new_features, uniques = pd.factorize(feature)
    return new_features


#
# Function float_factor
#

def float_factor(x, rounding):
    r"""Read in data from the given directory in a given format.

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

    num2str = '{0:.{1}f}'.format
    fstr = re.sub("[^0-9]", "", num2str(x, rounding))
    ffactor = int(fstr) if len(fstr) > 0 else 0
    return ffactor


#
# Function get_factors
#

def get_factors(fnum, fname, df, nvalues, dtype, encoder, rounding,
                sentinel, target_value, X, y, classify=False):
    r"""Factorize a feature.

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

    logger.info("Feature %d: %s is a factor of type %s with %d unique values",
                fnum, fname, dtype, nvalues)
    logger.info("Encoding: %s", encoder)
    feature = df[fname]
    # convert float to factor
    if dtype == 'float64':
        logger.info("Rounding: %d", rounding)
        feature = feature.apply(float_factor, args=[rounding])
    # encoders
    enc = None
    ef = pd.DataFrame(feature)
    if encoder == Encoders.factorize:
        pd_factors = pd.factorize(feature)[0]
        pd_features = pd.DataFrame(pd_factors)
    elif encoder == Encoders.onehot:
        pd_features = pd.get_dummies(feature)
    elif encoder == Encoders.ordinal:
        enc = ce.OrdinalEncoder(cols=[fname])
    elif encoder == Encoders.binary:
        enc = ce.BinaryEncoder(cols=[fname])
    elif encoder == Encoders.helmert:
        enc = ce.HelmertEncoder(cols=[fname])
    elif encoder == Encoders.sumcont:
        enc = ce.SumEncoder(cols=[fname])
    elif encoder == Encoders.polynomial:
        enc = ce.PolynomialEncoder(cols=[fname])
    elif encoder == Encoders.backdiff:
        enc = ce.BackwardDifferenceEncoder(cols=[fname])
    else:
        raise ValueError("Unknown Encoder %s", encoder)
    # If encoding worked, calculate target percentages for classifiers.
    pd_exists = not pd_features.empty
    enc_exists = enc is not None
    all_features = None
    if pd_exists or enc_exists:
        if pd_exists:
            all_features = pd_features
        elif enc_exists:
            all_features = enc.fit_transform(ef, None)
        # Calculate target percentages for classifiers
        if classify:
            # get the crosstab between feature labels and target
            logger.info("Calculating target percentages")
            ct = pd.crosstab(X[fname], y).apply(lambda r : r / r.sum(), axis=1)
            # map target percentages to the new feature
            ct_map = ct.to_dict()[target_value]
            ct_feature = df[[fname]].applymap(ct_map.get)
            # impute sentinel for any values that could not be mapped
            ct_feature.fillna(value=sentinel, inplace=True)
            # concatenate all generated features
            all_features = np.column_stack((all_features, ct_feature))
    else:
        raise RuntimeError("Encoding for feature %s failed", fname)
    return all_features


#
# Function create_numpy_features
#

def create_numpy_features(base_features):
    r"""Create NumPy features.

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

    logger.info("Creating NumPy Features")

    # Calculate the total, mean, standard deviation, and variance.

    logger.info("NumPy Feature: sum")
    row_sum = np.sum(base_features, axis=1)
    logger.info("NumPy Feature: mean")
    row_mean = np.mean(base_features, axis=1)
    logger.info("NumPy Feature: standard deviation")
    row_std = np.std(base_features, axis=1)
    logger.info("NumPy Feature: variance")
    row_var = np.var(base_features, axis=1)

    # Impute, scale, and stack all new features.

    np_features = np.column_stack((row_sum, row_mean, row_std, row_var))
    np_features = impute_values(np_features, 'float64')
    np_features = StandardScaler().fit_transform(np_features)

    # Return new NumPy features

    logger.info("NumPy Feature Count : %d", np_features.shape[1])
    return np_features


#
# Function create_scipy_features
#

def create_scipy_features(base_features):
    r"""Create SciPy features.

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

    logger.info("Creating SciPy Features")

    # Generate scipy features

    logger.info("SciPy Feature: geometric mean")
    row_gmean = sps.gmean(base_features, axis=1)
    logger.info("SciPy Feature: kurtosis")
    row_kurtosis = sps.kurtosis(base_features, axis=1)
    logger.info("SciPy Feature: kurtosis test")
    row_ktest, pvalue = sps.kurtosistest(base_features, axis=1)
    logger.info("SciPy Feature: normal test")
    row_normal, pvalue = sps.normaltest(base_features, axis=1)
    logger.info("SciPy Feature: skew")
    row_skew = sps.skew(base_features, axis=1)
    logger.info("SciPy Feature: skew test")
    row_stest, pvalue = sps.skewtest(base_features, axis=1)
    logger.info("SciPy Feature: variation")
    row_var = sps.variation(base_features, axis=1)
    logger.info("SciPy Feature: signal-to-noise ratio")
    row_stn = sps.signaltonoise(base_features, axis=1)
    logger.info("SciPy Feature: standard error of mean")
    row_sem = sps.sem(base_features, axis=1)

    sp_features = np.column_stack((row_gmean, row_kurtosis, row_ktest,
                                   row_normal, row_skew, row_stest,
                                   row_var, row_stn, row_sem))
    sp_features = impute_values(sp_features, 'float64')
    sp_features = StandardScaler().fit_transform(sp_features)

    # Return new SciPy features

    logger.info("SciPy Feature Count : %d", sp_features.shape[1])
    return sp_features


#
# Function create_clusters
#

def create_clusters(features, model):
    r"""Create clustering features.

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

    logger.info("Creating Clustering Features")

    # Extract model parameters

    cluster_inc = model.specs['cluster_inc']
    cluster_max = model.specs['cluster_max']
    cluster_min = model.specs['cluster_min']
    n_jobs = model.specs['n_jobs']
    seed = model.specs['seed']

    # Log model parameters

    logger.info("Cluster Minimum   : %d", cluster_min)
    logger.info("Cluster Maximum   : %d", cluster_max)
    logger.info("Cluster Increment : %d", cluster_inc)

    # Generate clustering features

    cfeatures = np.zeros((features.shape[0], 1))
    for i in range(cluster_min, cluster_max+1, cluster_inc):
        logger.info("k = %d", i)
        km = MiniBatchKMeans(n_clusters=i, random_state=seed)
        km.fit(features)
        labels = km.predict(features)
        labels = labels.reshape(-1, 1)
        cfeatures = np.column_stack((cfeatures, labels))
    cfeatures = np.delete(cfeatures, 0, axis=1)

    # Return new clustering features

    logger.info("Clustering Feature Count : %d", cfeatures.shape[1])
    return cfeatures


#
# Function create_pca_features
#

def create_pca_features(features, model):
    r"""Create PCA features.

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

    logger.info("Creating PCA Features")

    # Extract model parameters

    pca_inc = model.specs['pca_inc']
    pca_max = model.specs['pca_max']
    pca_min = model.specs['pca_min']
    pca_whiten = model.specs['pca_whiten']

    # Log model parameters

    logger.info("PCA Minimum   : %d", pca_min)
    logger.info("PCA Maximum   : %d", pca_max)
    logger.info("PCA Increment : %d", pca_inc)
    logger.info("PCA Whitening : %r", pca_whiten)

    # Generate clustering features

    pfeatures = np.zeros((features.shape[0], 1))
    for i in range(pca_min, pca_max+1, pca_inc):
        logger.info("n_components = %d", i)
        X_pca = PCA(n_components=i, whiten=pca_whiten).fit_transform(features)
        pfeatures = np.column_stack((pfeatures, X_pca))
    pfeatures = np.delete(pfeatures, 0, axis=1)

    # Return new clustering features

    logger.info("PCA Feature Count : %d", pfeatures.shape[1])
    return pfeatures


#
# Function create_isomap_features
#

def create_isomap_features(features, model):
    r"""Create Isomap features.

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

    logger.info("Creating Isomap Features")

    # Extract model parameters

    iso_components = model.specs['iso_components']
    iso_neighbors = model.specs['iso_neighbors']
    n_jobs = model.specs['n_jobs']

    # Log model parameters

    logger.info("Isomap Components : %d", iso_components)
    logger.info("Isomap Neighbors  : %d", iso_neighbors)

    # Generate Isomap features

    model = Isomap(n_neighbors=iso_neighbors, n_components=iso_components,
                   n_jobs=n_jobs)
    ifeatures = model.fit_transform(features)

    # Return new Isomap features

    logger.info("Isomap Feature Count : %d", ifeatures.shape[1])
    return ifeatures


#
# Function create_tsne_features
#

def create_tsne_features(features, model):
    r"""Create T-SNE features.

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

    logger.info("Creating T-SNE Features")

    # Extract model parameters

    seed = model.specs['seed']
    tsne_components = model.specs['tsne_components']
    tsne_learn_rate = model.specs['tsne_learn_rate']
    tsne_perplexity = model.specs['tsne_perplexity']

    # Log model parameters

    logger.info("T-SNE Components    : %d", tsne_components)
    logger.info("T-SNE Learning Rate : %d", tsne_learn_rate)
    logger.info("T-SNE Perplexity    : %d", tsne_perplexity)

    # Generate T-SNE features

    model = TSNE(n_components=tsne_components, perplexity=tsne_perplexity,
                 learning_rate=tsne_learn_rate, random_state=seed)
    tfeatures = model.fit_transform(features)

    # Return new T-SNE features

    logger.info("T-SNE Feature Count : %d", tfeatures.shape[1])
    return tfeatures


#
# Function create_features
#

def create_features(X, model, split_point, y_train):
    r"""Extract features from the training and test set.

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

    # Extract model parameters

    clustering = model.specs['clustering']
    counts_flag = model.specs['counts']
    dummy_limit = model.specs['dummy_limit']
    encoder = model.specs['encoder']
    isomap = model.specs['isomap']
    logtransform = model.specs['logtransform']
    model_type = model.specs['model_type']
    ngrams_max = model.specs['ngrams_max']
    numpy_flag = model.specs['numpy']
    pca = model.specs['pca']
    pvalue_level = model.specs['pvalue_level']
    rounding = model.specs['rounding']
    scaling = model.specs['scaler_option']
    scaler = model.specs['scaler_type']
    scipy_flag = model.specs['scipy']
    sentinel = model.specs['sentinel']
    target_value = model.specs['target_value']
    treatments = model.specs['treatments']
    tsne = model.specs['tsne']
    vectorize = model.specs['vectorize']

    # Log input parameters

    logger.info("Original Features : %s", X.columns)
    logger.info("Feature Count     : %d", X.shape[1])

    # Set classification flag

    classify = True if model_type == ModelType.classification else False

    # Count zero and NaN values

    if counts_flag:
        logger.info("Creating Count Features")
        logger.info("NA Counts")
        X['nan_count'] = X.count(axis=1)
        logger.info("Number Counts")
        for i in range(10):
            fc = USEP.join(['count', str(i)])
            X[fc] = (X == i).astype(int).sum(axis=1)
        logger.info("New Feature Count : %d", X.shape[1])

    # Iterate through columns, dispatching and transforming each feature.

    logger.info("Creating Base Features")

    X_train, X_test = np.array_split(X, [split_point])
    all_features = np.zeros((X.shape[0], 1))

    for i, fc in enumerate(X):
        fnum = i + 1
        dtype = X[fc].dtypes
        nunique = len(X[fc].unique())
        # treatments
        if treatments and fc in treatments:
            features = apply_treatment(fnum, fc, X, nunique, treatments[fc])
            all_features = np.column_stack((all_features, features))
        # standard processing of numerical, categorical, and text features
        if nunique <= dummy_limit:
            features = get_factors(fnum, fc, X, nunique, dtype, encoder, rounding,
                                   sentinel, target_value, X_train, y_train,
                                   classify)            
        elif dtype == 'float64' or dtype == 'int64' or dtype == 'bool':
            features = get_numerical_features(fnum, fc, X, nunique, dtype,
                                              logtransform, pvalue_level)
        elif dtype == 'object':
            features = get_text_features(fnum, fc, X, nunique, dummy_limit,
                                         vectorize, ngrams_max)
        else:
            raise TypeError("Base Feature Error with unrecognized type %s", dtype)
        if features is not None:
            all_features = np.column_stack((all_features, features))
    all_features = np.delete(all_features, 0, axis=1)

    logger.info("New Feature Count : %d", all_features.shape[1])

    # Call standard scaler for all features

    if scaling:
        logger.info("Scaling Base Features")
        if scaler == Scalers.standard:
            all_features = StandardScaler().fit_transform(all_features)
        elif scaler == Scalers.minmax:
            all_features = MinMaxScaler().fit_transform(all_features)
        else:
            logger.info("Unrecognized scaler: %s", scaler)
    else:
        logger.info("Skipping Scaling")

    # Perform dimensionality reduction only on base feature set

    base_features = all_features

    # Calculate the total, mean, standard deviation, and variance

    if numpy_flag:
        np_features = create_numpy_features(base_features)
        all_features = np.column_stack((all_features, np_features))
        logger.info("New Feature Count : %d", all_features.shape[1])

    # Generate scipy features

    if scipy_flag:
        sp_features = create_scipy_features(base_features)
        all_features = np.column_stack((all_features, sp_features))
        logger.info("New Feature Count : %d", all_features.shape[1])

    # Create clustering features

    if clustering:
        cfeatures = create_clusters(base_features, model)
        all_features = np.column_stack((all_features, cfeatures))
        logger.info("New Feature Count : %d", all_features.shape[1])

    # Create PCA features

    if pca:
        pfeatures = create_pca_features(base_features, model)
        all_features = np.column_stack((all_features, pfeatures))
        logger.info("New Feature Count : %d", all_features.shape[1])

    # Create Isomap features

    if isomap:
        ifeatures = create_isomap_features(base_features, model)
        all_features = np.column_stack((all_features, ifeatures))
        logger.info("New Feature Count : %d", all_features.shape[1])

    # Create T-SNE features

    if tsne:
        tfeatures = create_tsne_features(base_features, model)
        all_features = np.column_stack((all_features, tfeatures))
        logger.info("New Feature Count : %d", all_features.shape[1])

    # Return all transformed training and test features
    
    return all_features


#
# Function select_features
#

def select_features(model):
    r"""Select features with univariate selection.

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

    logger.info("Feature Selection")

    # Extract model data.

    X_train = model.X_train
    y_train = model.y_train

    # Extract model parameters.

    fs_percentage = model.specs['fs_percentage']
    fs_score_func = model.specs['fs_score_func']

    # Select top features based on percentile.

    fs = SelectPercentile(score_func=fs_score_func,
                          percentile=fs_percentage)

    # Perform feature selection and get the support mask

    fsfit = fs.fit(X_train, y_train)
    support = fsfit.get_support()

    # Record the support vector

    X_train_new = model.X_train[:, support]
    X_test_new = model.X_test[:, support]

    # Count the number of new features.

    logger.info("Old Feature Count : %d", X_train.shape[1])
    logger.info("New Feature Count : %d", X_train_new.shape[1])

    # Store the reduced features in the model.

    model.X_train = X_train_new
    model.X_test = X_test_new

    # Return the modified model

    return model


#
# Function save_features
#

def save_features(model, X_train, X_test, y_train=None, y_test=None):
    r"""Save new features in model.

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

    logger.info("Saving New Features in Model")

    model.X_train = X_train
    model.X_test = X_test
    if y_train is not None:
        model.y_train = y_train
    if y_test is not None:
        model.y_test = y_test

    return model


#
# Function create_interactions
#

def create_interactions(X, model):
    r"""Create feature interactions using the training data.

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

    logger.info("Creating Interactions")

    # Extract model parameters

    genetic = model.specs['genetic']
    gfeatures = model.specs['gfeatures']
    interactions = model.specs['interactions']
    isample_pct = model.specs['isample_pct']
    model_type = model.specs['model_type']
    n_jobs = model.specs['n_jobs']
    poly_degree = model.specs['poly_degree']
    seed = model.specs['seed']
    verbosity = model.specs['verbosity']

    # Extract model data

    X_train = model.X_train
    y_train = model.y_train

    # Log parameters

    logger.info("Initial Feature Count  : %d", X.shape[1])

    # Initialize all features

    all_features = X

    # Get polynomial features

    if interactions:
        logger.info("Generating Polynomial Features")
        logger.info("Interaction Percentage : %d", isample_pct)
        logger.info("Polynomial Degree      : %d", poly_degree)
        if model_type == ModelType.regression:
            selector = SelectPercentile(f_regression, percentile=isample_pct)
        elif model_type == ModelType.classification:
            selector = SelectPercentile(f_classif, percentile=isample_pct)
        else:
            raise TypeError("Unknown model type when creating interactions")
        selector.fit(X_train, y_train)
        support = selector.get_support()
        pfeatures = get_polynomials(X[:, support], poly_degree)
        logger.info("Polynomial Feature Count : %d", pfeatures.shape[1])
        pfeatures = StandardScaler().fit_transform(pfeatures)
        all_features = np.hstack((all_features, pfeatures))
        logger.info("New Total Feature Count  : %d", all_features.shape[1])
    else:
        logger.info("Skipping Interactions")

    # Generate genetic features

    if genetic:
        logger.info("Generating Genetic Features")
        logger.info("Genetic Features : %r", gfeatures)
        gp = SymbolicTransformer(generations=20, population_size=2000,
                                 hall_of_fame=100, n_components=gfeatures,
                                 parsimony_coefficient=0.0005,
                                 max_samples=0.9, verbose=verbosity,
                                 random_state=seed, n_jobs=n_jobs)
        gp.fit(X_train, y_train)
        gp_features = gp.transform(X)
        logger.info("Genetic Feature Count : %d", gp_features.shape[1])
        gp_features = StandardScaler().fit_transform(gp_features)
        all_features = np.hstack((all_features, gp_features))
        logger.info("New Total Feature Count : %d", all_features.shape[1])
    else:
        logger.info("Skipping Genetic Features")

    # Return all features

    return all_features


#
# Function drop_features
#

def drop_features(X, drop):
    r"""Drop any specified features.

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

    X.drop(drop, axis=1, inplace=True, errors='ignore')
    return X


#
# Function remove_lv_features
#

def remove_lv_features(X):
    r"""Remove low-variance features.

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

    logger.info("Removing Low-Variance Features")
    logger.info("Original Feature Count  : %d", X.shape[1])

    # Remove duplicated columns

    selector = VarianceThreshold()
    X_reduced = selector.fit_transform(X)
    logger.info("Reduced Feature Count   : %d", X_reduced.shape[1])

    return X_reduced
