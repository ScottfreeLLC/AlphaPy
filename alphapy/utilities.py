################################################################################
#
# Package   : AlphaPy
# Module    : utilities
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

from alphapy.globals import PSEP, SSEP, USEP

import argparse
from datetime import datetime, timedelta
import glob
import inspect
from itertools import groupby
import logging
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import re


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_datestamp
#

def get_datestamp():
    r"""Returns today's datestamp.

    Returns
    -------
    datestamp : str
        The valid date string in YYYY-mm-dd format.

    """
    d = datetime.now()
    f = "%Y%m%d"
    datestamp = d.strftime(f)
    return datestamp


#
# Function most_recent_file
#

def most_recent_file(directory, file_spec):
    r"""Find the most recent file in a directory.

    Parameters
    ----------
    directory : str
        Full directory specification.
    file_spec : str
        Wildcard search string for the file to locate.

    Returns
    -------
    file_name : str
        Name of the file to read, excluding the ``extension``.

    """
    # Create search path
    search_path = SSEP.join([directory, file_spec])
    # find the latest file
    file_name = max(glob.iglob(search_path), key=os.path.getctime)
    # load the model predictor
    return file_name


#
# Function np_store_data
#

def np_store_data(data, dir_name, file_name, extension, separator):
    r"""Store NumPy data in a file.

    Parameters
    ----------
    data : numpy array
        The model component to store
    dir_name : str
        Full directory specification.
    file_name : str
        Name of the file to read, excluding the ``extension``.
    extension : str
        File name extension, e.g., ``csv``.
    separator : str
        The delimiter between fields in the file.

    Returns
    -------
    None : None

    """
    output_file = PSEP.join([file_name, extension])
    output = SSEP.join([dir_name, output_file])
    logger.info("Storing output to %s", output)
    np.savetxt(output, data, delimiter=separator)


#
# Function remove_list_items
#

def remove_list_items(elements, alist):
    r"""Remove one or more items from the given list.

    Parameters
    ----------
    elements : list
        The items to remove from the list ``alist``.
    alist : list
        Any object of any type can be a list item.

    Returns
    -------
    sublist : list
        The subset of items after removal.

    Examples
    --------

    >>> test_list = ['a', 'b', 'c', test_func]
    >>> remove_list_items([test_func], test_list)  # ['a', 'b', 'c']

    """
    sublist = [x for x in alist if x not in elements]
    return sublist


#
# Function subtract_days
#

def subtract_days(date_string, ndays):
    r"""Subtract a number of days from a given date.

    Parameters
    ----------
    date_string : str
        An alphanumeric string in the format %Y-%m-%d.
    ndays : int
        Number of days to subtract.

    Returns
    -------
    new_date_string : str
        The adjusted date string in the format %Y-%m-%d.

    Examples
    --------

    >>> subtract_days('2017-11-10', 31)   # '2017-10-10'

    """
    new_date_string = None
    valid = valid_date(date_string)
    if valid:
        date_dt = datetime.strptime(date_string, "%Y-%m-%d")
        new_date = date_dt - timedelta(days=ndays)
        new_date_string = new_date.strftime("%Y-%m-%d")
    return new_date_string


#
# Function valid_date
#

def valid_date(date_string):
    r"""Determine whether or not the given string is a valid date.

    Parameters
    ----------
    date_string : str
        An alphanumeric string in the format %Y-%m-%d.

    Returns
    -------
    date_string : str
        The valid date string.

    Raises
    ------
    ValueError
        Not a valid date.

    Examples
    --------

    >>> valid_date('2016-7-1')   # datetime.datetime(2016, 7, 1, 0, 0)
    >>> valid_date('345')        # ValueError: Not a valid date

    """
    try:
        date_time = datetime.strptime(date_string, "%Y-%m-%d")
        return date_string
    except:
        message = "Not a valid date: '{0}'.".format(date_string)
        raise argparse.ArgumentTypeError(message)


#
# Function valid_name
#

def valid_name(name):
    r"""Determine whether or not the given string is a valid
    alphanumeric string.

    Parameters
    ----------
    name : str
        An alphanumeric identifier.

    Returns
    -------
    result : bool
        ``True`` if the name is valid, else ``False``.

    Examples
    --------

    >>> valid_name('alpha')   # True
    >>> valid_name('!alpha')  # False

    """
    identifier = re.compile(r"^[^\d\W]\w*\Z", re.UNICODE)
    result = re.match(identifier, name)
    return result is not None
