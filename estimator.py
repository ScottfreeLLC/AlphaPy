##############################################################
#
# Package   : AlphaPy
# Module    : estimator
# Version   : 1.0
# Copyright : Mark Conway
# Date      : June 29, 2013
#
##############################################################


#
# Class Estimator
#

class Estimator:

    # __new__
    
    def __new__(cls,
                algorithm,
                estimator,
                grid,
                scoring=False):
        return super(Estimator, cls).__new__(cls)
    
    # __init__
    
    def __init__(self,
                 algorithm,
                 estimator,
                 grid,
                 scoring=False):
        self.algorithm = algorithm.upper()
        self.estimator = estimator
        self.grid = grid
        self.scoring = scoring
        
    # __str__

    def __str__(self):
        return self.name
