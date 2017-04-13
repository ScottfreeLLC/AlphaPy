################################################################################
#
# Package   : AlphaPy
# Module    : space
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


#
# Function space_name
#

def space_name(subject, schema, fractal):
    r"""Get the namespace string.

    Parameters
    ----------
    subject : str
        An identifier for a group of related items.
    schema : str
        The data related to the ``subject``.
    fractal : str
        The time fractal of the data, e.g., "5m" or "1d".

    Returns
    -------
    name : str
        The joined namespace string.

    """
    name = USEP.join([subject, schema, fractal])
    return name
    

#
# Class Space
#

class Space:
    """Create a new namespace.

    Parameters
    ----------
    subject : str
        An identifier for a group of related items.
    schema : str
        The data related to the ``subject``.
    fractal : str
        The time fractal of the data, e.g., "5m" or "1d".

    """
    
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
