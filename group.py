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
from space import Space


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
                 dynamic = True,
                 recursive = False):
        # code
        if not name in Group.groups:
            self.name = name
            self.space = space
            self.dynamic =  dynamic
            self.recursive = recursive
            self.members = set()
            if dynamic == False:
                # load members from fixed groups
                pass
                # dir = getdirectory(name)
                # members = with(dir, dir$key)
            # add group to groups list
            Group.groups[name] = self
        else:
            print "Group %s already exists" % name
        
    # function __repr__

    def __repr__(self):
        return self.name
        
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
                gflat = all({type(item) is str for item in self.members})
                if gflat == False and self.recursive == False:
                    print "Cannot add members to non-recursive groups"
                else:
                    if newset.issubset(self.members):
                        print "New members already in set"
                    else:
                        madd = newset - self.members
                        self.members = self.members | newset
                        print "Added: ", madd
            else:
                print "Cannot add members to a non-dynamic group"
        else:
            print "All new members must be of type str"
            
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
                print "Members to remove not found"
            else:
                removed = []
                for item in remlist:
                    if self.member(item):
                        self.members.remove(item)
                        removed += [item]
                print "Removed: ", removed
        else:
            print "Cannot remove members from a non-dynamic group"
            
    # function allgroups

    def all_groups(self):
        gs = [repr(self)]
        def group_tree(self):
            glist = {item for item in self.members if item in Group.groups}
            if len(glist) > 0:
                for item in glist:
                    gs.append(item)
                    gsub = group_tree(Group.groups[item])
            return gs
        gs = group_tree(self)    
        return sorted(gs)

    # function allmembers

    def all_members(self):
        members = []        
        subgs = self.all_groups()
        for i in range(0, len(subgs)):
            mlist = [item for item in Group.groups[subgs[i]].members
                     if not item in Group.groups]
            members.append(mlist)
        members = [item for sublist in members for item in sublist]
        return sorted(members)
