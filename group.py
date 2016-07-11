##############################################################
#
# Package   : AlphaPy
# Module    : group
# Version   : 1.0
# Copyright : Mark Conway
# Date      : June 29, 2013
#
##############################################################


#
# Imports
#

from globs import USEP
import logging
from space import Space


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Class Group
#

class Group(object):

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
        # code
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
        # code
        return item in self.members
        
    # function remove
    
    def remove(self, remlist):
        # code
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
