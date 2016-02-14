##############################################################
#
# Package   : AlphaPy
# Module    : alias
# Version   : 1.0
# Copyright : Mark Conway
# Date      : October 19, 2014
#
##############################################################


#
# Imports
#

import parser
import re


#
# Class Alias
#

class Alias(object):

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
            print "Expression %s already exists for key %s" % (expr, key)
            return
        else:
            if replace == True or not name in Alias.aliases:
                identifier = re.compile(r"^[^\d\W]\w*\Z", re.UNICODE)
                result1 = re.match(identifier, name)
                if result1 is None:
                    print "Invalid alias key: %s" % name
                    return
                result2 = re.match(identifier, expr)
                if result2 is None:
                    print "Invalid alias expression: %s" % expr
                    return
                return super(Alias, cls).__new__(cls)
            else:
                print "Key %s already exists" % name

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
    if alias in Alias.aliases:
        return Alias.aliases[alias]
    else:
        return None
