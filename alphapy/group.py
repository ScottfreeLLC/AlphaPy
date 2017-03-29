################################################################################
#
# Package   : AlphaPy
# Module    : group
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

from alphapy.globs import USEP
from alphapy.space import Space

import logging


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Class Group
#

class Group(object):
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

    # class variable to track all groups

    groups = {}
    
    # function __init__
    
    def __init__(self,
                 name,
                 space = Space(),
                 dynamic = True):
        # code
        if not name in Group.groups:
            self.name = name
            self.space = space
            self.dynamic =  dynamic
            self.members = set()
            if dynamic == False:
                # load members from fixed groups
                pass
                # dir = getdirectory(name)
                # members = with(dir, dir$key)
            # add group to groups list
            Group.groups[name] = self
        else:
            logger.info("Group already %s exists", name)
        
    # function __str__

    def __str__(self):
        return self.name
            
    # function add
            
    def add(self,
            newlist):
        r"""Read in data from the given directory in a given format.

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

        """
        if all([type(item) is str for item in newlist]):
            newset = set(newlist)
            if self.dynamic:
                if newset.issubset(self.members):
                    logger.info("New members already in set")
                else:
                    madd = newset - self.members
                    self.members = self.members | newset
                    logger.info("Added: %s", madd)
            else:
                logger.info("Cannot add members to a non-dynamic group")
        else:
            logger.info("All new members must be of type str")
            
    # function member
            
    def member(self, item):
        r"""Read in data from the given directory in a given format.

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

        """
        return item in self.members
        
    # function remove
    
    def remove(self, remlist):
        r"""Read in data from the given directory in a given format.

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

        """
        if self.dynamic:
            nonefound = not any([self.member(item) for item in remlist])
            if nonefound == True:
                logger.info("Members to remove not found")
            else:
                removed = []
                for item in remlist:
                    if self.member(item):
                        self.members.remove(item)
                        removed += [item]
                logger.info("Removed: %s", removed)
        else:
            logger.info("Cannot remove members from a non-dynamic group")
