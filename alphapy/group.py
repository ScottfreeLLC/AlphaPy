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

from alphapy.globals import USEP
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
    """Create a new Group that contains common members. All
    defined groups are stored in ``Group.groups``. Group
    names must be unique.

    Parameters
    ----------
    name : str
        Group name.
    space : alphapy.Space, optional
        Namespace for the given group.
    dynamic : bool, optional, default ``True``
        Flag for defining whether or not the group membership
        can change.
    members : set, optional
        The initial members of the group, especially if the
        new group is fixed, e.g., not ``dynamic``.

    Attributes
    ----------
    groups : dict
        Class variable for storing all known groups

    Examples
    --------
    
    >>> Group('tech')

    """

    # class variable to track all groups

    groups = {}
    
    # function __init__
    
    def __init__(self,
                 name,
                 space = Space(),
                 dynamic = True,
                 members = set()):
        # code
        if not name in Group.groups:
            self.name = name
            self.space = space
            self.dynamic =  dynamic
            self.members = members
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
        r"""Add new members to the group.

        Parameters
        ----------
        newlist : list
            New members or identifiers to add to the group.

        Returns
        -------
        None : None

        Notes
        -----

        New members cannot be added to a fixed or non-dynamic group.

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
        r"""Find a member in the group.

        Parameters
        ----------
        item : str
            The member to find the group.

        Returns
        -------
        member_exists : bool
            Flag indicating whether or not the member is in the group.

        """
        return item in self.members
        
    # function remove
    
    def remove(self, remlist):
        r"""Read in data from the given directory in a given format.

        Parameters
        ----------
        remlist : list
            The list of members to remove from the group.

        Returns
        -------
        None : None

        Notes
        -----

        Members cannot be removed from a fixed or non-dynamic group.

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
