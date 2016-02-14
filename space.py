##############################################################
#
# Package   : AlphaPy
# Module    : space
# Version   : 1.0
# Copyright : Mark Conway
# Date      : June 29, 2013
#
##############################################################


#
# Imports
#

from globs import USEP


#
# Function space_name
#

def space_name(subject, schema, fractal):
    return USEP.join([subject, schema, fractal])
    

#
# Class Space
#

class Space:
    
    # __init__
    
    def __init__(self,
                 subject = "stock",
                 schema = "prices",
                 fractal = "1d"):
        # code
        self.subject = subject
        self.schema = schema
        self.fractal = fractal
        
    # __str__

    def __str__(self):
        return space_name(self.subject, self.schema, self.fractal)
