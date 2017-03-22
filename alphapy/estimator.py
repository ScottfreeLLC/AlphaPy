################################################################################
#
# Package   : AlphaPy
# Module    : estimator
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
# Class Estimator
#

class Estimator:

    # __new__
    
    def __new__(cls,
                algorithm,
                model_type,
                estimator,
                grid,
                scoring=False):
        return super(Estimator, cls).__new__(cls)
    
    # __init__
    
    def __init__(self,
                 algorithm,
                 model_type,
                 estimator,
                 grid,
                 scoring=False):
        self.algorithm = algorithm.upper()
        self.model_type = model_type
        self.estimator = estimator
        self.grid = grid
        self.scoring = scoring
        
    # __str__

    def __str__(self):
        return self.name
