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

from alphapy.globals import PSEP, USEP

import inspect
from itertools import groupby
from os import listdir
from os.path import isfile, join
import re


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
