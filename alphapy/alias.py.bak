################################################################################
#
# Package   : AlphaPy
# Module    : alias
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

import logging
import parser
import re


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Class Alias
#

class Alias(object):
    """Create a new alias as a key-value pair. All aliases are stored
    in ``Alias.aliases``. Duplicate keys or values are not allowed,
    unless the ``replace`` parameter is ``True``.

    Parameters
    ----------
    name : str
        Alias key.
    expr : str
        Alias value.
    replace : bool, optional
        Replace the current key-value pair if it already exists.

    Attributes
    ----------
    Alias.aliases : dict
        Class variable for storing all known aliases

    Examples
    --------
    
    >>> Alias('atr', 'ma_truerange')
    >>> Alias('hc', 'higher_close')

    """

    # class variable to track all aliases

    aliases = {}

    # function __new__

    def __new__(cls,
                name,
                expr,
                replace = False):
        # code
        efound = expr in [Alias.aliases[key] for key in Alias.aliases]
        if efound == True:
            key = [key for key, aexpr in Alias.aliases.items() if aexpr == expr]
            logger.info("Expression %s already exists for key %s", expr, key)
            return
        else:
            if replace == True or not name in Alias.aliases:
                identifier = re.compile(r"^[^\d\W]\w*\Z", re.UNICODE)
                result1 = re.match(identifier, name)
                if result1 is None:
                    logger.info("Invalid alias key: %s", name)
                    return
                result2 = re.match(identifier, expr)
                if result2 is None:
                    logger.info("Invalid alias expression: %s", expr)
                    return
                return super(Alias, cls).__new__(cls)
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
        Alias.aliases[name] = expr
            
    # function __str__

    def __str__(self):
        return self.expr

    
#
# Function get_alias
#

def get_alias(alias):
    r"""Find an alias value with the given key.

    Parameters
    ----------
    alias : str
        Key for finding the alias value.

    Returns
    -------
    alias_value : str
        Value for the corresponding key.

    Examples
    --------

    >>> alias_value = get_alias('atr')
    >>> alias_value = get_alias('hc')

    """
    if alias in Alias.aliases:
        return Alias.aliases[alias]
    else:
        return None
