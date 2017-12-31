################################################################################
#
# Package   : AlphaPy
# Module    : globals
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

from enum import Enum, unique


#
# Global Variables
#

#
# Delimiters
#

BSEP = ' '
CSEP = ':'
PSEP = '.'
SSEP = '/'
USEP = '_'
LOFF = '['
ROFF = ']'

#
# Numerical Constants
#

Q1 = 0.25
Q2 = 0.50
Q3 = 0.75

#
# String Constants
#

NULLTEXT = 'NULLTEXT'
TAG_ID = 'tag'
WILDCARD = '*'

#
# Dictionaries
#

MULTIPLIERS = {'crypto' : 1.0,
               'stock' : 1.0}

#
# Pandas Time Offset Aliases
#

PD_INTRADAY_OFFSETS = ['H', 'T', 'min', 'S', 'L', 'ms', 'U', 'us', 'N']

#
# Pandas Web Reader Feeds
#

PD_WEB_DATA_FEEDS = ['google', 'quandl', 'yahoo']


#
# Encoder Types
#

@unique
class Encoders(Enum):
    """AlphaPy Encoders.

    These are the encoders used in AlphaPy, as configured in the
    ``model.yml`` file (features:encoding:type) You can learn more
    about encoders here [ENC]_.

    .. [ENC] https://github.com/scikit-learn-contrib/categorical-encoding

    """
    backdiff = 1
    binary = 2
    factorize = 3
    helmert = 4
    onehot = 5
    ordinal = 6
    polynomial = 7
    sumcont = 8


#
# Model Types
#

@unique
class ModelType(Enum):
    """AlphaPy Model Types.

    .. note:: One-Class Classification ``oneclass`` is not yet
       implemented.

    """
    classification = 1
    clustering = 2
    multiclass = 3
    oneclass = 4
    regression = 5


#
# Objective Functions
#

@unique
class Objective(Enum):
    """Scoring Function Objectives.

    Best model selection is based on the scoring or Objective
    function, which must be either maximized or minimized. For
    example, ``roc_auc`` is maximized, while ``neg_log_loss``
    is minimized.

    """
    maximize = 1
    minimize = 2


#
# Class Orders
#

class Orders:
    """System Order Types.

    Attributes
    ----------
    le : str
        long entry
    se : str
        short entry
    lx : str
        long exit
    sx : str
        short exit
    lh : str
        long exit at the end of the holding period
    sh : str
        short exit at the end of the holding period

    """
    le = 'le'
    se = 'se'
    lx = 'lx'
    sx = 'sx'
    lh = 'lh'
    sh = 'sh'


#
# Partition Types
#

@unique
class Partition(Enum):
    """AlphaPy Partitions.

    """
    predict = 1
    test = 2
    train = 3


#
# Sampling Methods
#

@unique
class SamplingMethod(Enum):
    """AlphaPy Sampling Methods.

    These are the data sampling methods used in AlphaPy, as configured
    in the ``model.yml`` file (data:sampling:method) You can learn more
    about resampling techniques here [IMB]_.

    .. [IMB] https://github.com/scikit-learn-contrib/imbalanced-learn

    """
    ensemble_bc = 1
    ensemble_easy = 2
    over_random = 3
    over_smote = 4
    over_smoteb = 5
    over_smotesv = 6
    overunder_smote_enn = 7
    overunder_smote_tomek = 8
    under_cluster = 9
    under_ncr = 10
    under_nearmiss = 11
    under_random = 12
    under_tomek = 13


#
# Scaler Types
#

@unique
class Scalers(Enum):
    """AlphaPy Scalers.

    These are the scaling methods used in AlphaPy, as configured in the
    ``model.yml`` file (features:scaling:type) You can learn more about
    feature scaling here [SCALE]_.

    .. [SCALE] http://scikit-learn.org/stable/modules/preprocessing.html

    """
    minmax = 1
    standard = 2


#
# Datasets
#

datasets = {Partition.train   : 'train',
            Partition.test    : 'test',
            Partition.predict : 'predict'}
