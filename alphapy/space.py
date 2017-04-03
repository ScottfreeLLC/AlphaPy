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

from alphapy.globs import USEP


#
# Function space_name
#

def space_name(subject, schema, fractal):
    r"""Find an alias value with the given key.

    Parameters
    ----------
    subject : str
        Key for finding the alias value.
    schema : str
        Key for finding the alias value.
    fractal : str
        Key for finding the alias value.

    Returns
    -------
    name : str
        Value for the corresponding key.

    """
    return USEP.join([subject, schema, fractal])
    

#
# Class Space
#

class Space:
    """Create a new alias as a key-value pair. All aliases are stored
    in Alias.aliases. Duplicate keys or values are not allowed, unless
    the ``replace`` parameter is ``True``.

    Parameters
    ----------
    subject : str, optional
        Alias key.
    schema : str, optional
        Alias value.
    fractal : bool, optional
        Replace the current key-value pair if it already exists.

    Examples
    --------
    
    >>> Space('atr', 'ma_truerange')
    >>> Space('hc', 'higher_close')

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
