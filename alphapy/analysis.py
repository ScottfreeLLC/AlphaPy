################################################################################
#
# Package   : AlphaPy
# Module    : analysis
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

from alphapy.__main__ import main_pipeline
from alphapy.frame import load_frames
from alphapy.frame import write_frame
from alphapy.globs import SSEP, USEP

import logging
import pandas as pd
from pandas.tseries.offsets import BDay


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function analysis_name
#

def analysis_name(gname, target):
    r"""Find an alias value with the given key.

    Parameters
    ----------
    subject : str
        Key for finding the alias value.
    schema : str
        Key for finding the alias value.
    fractal : str
        Key for finding the alias value.

    Returns
    -------
    name : str
        Value for the corresponding key.

    Examples
    --------

    >>> name = space_name('atr')
    >>> name = space_name('hc')

    """

    return USEP.join([gname, target])


#
# Class Analysis
#

class Analysis(object):
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

    analyses = {}

    # __new__
    
    def __new__(cls,
                model,
                group,
                train_date = pd.datetime(1900, 1, 1),
                predict_date = pd.datetime.today() - BDay(2)):
        # verify that dates are in sequence
        if train_date >= predict_date:
            raise ValueError("Training date must be before prediction date")
        # set analysis name
        name = model.specs['directory'].split(SSEP)[-1]
        target = model.specs['target']
        an = analysis_name(name, target)
        if not an in Analysis.analyses:
            return super(Analysis, cls).__new__(cls)
        else:
            logger.info("Analysis %s already exists", an)

    # function __init__

    def __init__(self,
                 model,
                 group,
                 train_date = pd.datetime(1900, 1, 1),
                 predict_date = pd.datetime.today() - BDay(2)):
        # set analysis name
        name = model.specs['directory'].split(SSEP)[-1]
        target = model.specs['target']
        an = analysis_name(name, target)
        # initialize analysis
        self.name = an
        self.model = model
        self.group = group
        self.train_date = train_date.strftime('%Y-%m-%d')
        self.predict_date = predict_date.strftime('%Y-%m-%d')
        self.target = target
        # add analysis to analyses list
        Analysis.analyses[an] = self
        
    # __str__

    def __str__(self):
        return self.name


#
# Function run_analysis
#

def run_analysis(analysis, forecast_period, leaders, splits=True):
    r"""Run an analysis for a given model and group

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

    name = analysis.name
    model = analysis.model
    group = analysis.group
    target = analysis.target
    train_date = analysis.train_date
    predict_date = analysis.predict_date

    # Unpack model data

    directory = model.specs['directory']
    extension = model.specs['extension']
    separator = model.specs['separator']
    test_file = model.specs['test_file']
    test_labels = model.specs['test_labels']
    train_file = model.specs['train_file']

    # Load the data frames

    data_frames = load_frames(group, directory, extension, separator, splits)
    if data_frames:
        # create training and test frames
        train_frame = pd.DataFrame()
        test_frame = pd.DataFrame()
        # Subset each frame and add to the model frame
        for df in data_frames:
            # shift the target for the forecast period
            if forecast_period > 0:
                df[target] = df[target].shift(-forecast_period)
            # shift any leading features if necessary
            if leaders:
                df[leaders] = df[leaders].shift(-1)
            # split data into train and test
            new_train = df.loc[(df.index >= train_date) & (df.index < predict_date)]
            if len(new_train) > 0:
                # train frame
                new_train = new_train.dropna()
                train_frame = train_frame.append(new_train)
                # test frame
                new_test = df.loc[df.index >= predict_date]
                if len(new_test) > 0:
                    if test_labels:
                        new_test = new_test.dropna()
                    test_frame = test_frame.append(new_test)
                else:
                    logger.info("A test frame has zero rows. Check prediction date.")
            else:
                logger.warning("A training frame has zero rows. Check data source.")
        # write out the training and test files
        if len(train_frame) > 0 and len(test_frame) > 0:
            directory = SSEP.join([directory, 'input'])
            write_frame(train_frame, directory, train_file, extension, separator,
                        index=True, index_label='date')
            write_frame(test_frame, directory, test_file, extension, separator,
                        index=True, index_label='date')
        # run the AlphaPy pipeline
        analysis.model = main_pipeline(model)
    else:
        # no frames found
        logger.info("No frames were found for analysis %s", name)
    # return the analysis
    return analysis
