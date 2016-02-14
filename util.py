##############################################################
#
# Package  : AlphaPy
# Module   : util
# Version  : 1.0
# Copyright: Mark Conway
# Date     : July 20, 2015
#
##############################################################


#
# Imports
#

from globs import PSEP
from globs import USEP
from itertools import groupby
from os import listdir
from os.path import isfile, join
import re


#
# Function get_public_vars
#

def get_public_vars(obj):
    return [(name, value) for name, value 
            in inspect.getmembers(obj, lambda x: not callable(x))
            if not name.startswith(USEP)]


#
# Function get_public_names
#

def get_public_names(obj):
    return [item[0] for item in get_public_vars(obj)]


#
# Function get_public_vals
#

def get_public_vals(obj):
    return [item[1] for item in get_public_vars(obj)]


#
# Function list_files
#

def list_files(path, s, extension):
    just_files = [ f for f in listdir(path) if isfile(join(path, f)) ]
    ss = USEP.join([s.subject, s.schema])
    matching_files = [x for i, x in enumerate(just_files) if ss in just_files[i]]
    final_set = []
    if matching_files is not None:
        epat = PSEP + extension + '$'
        for f in matching_files:
            match = re.search(epat, f)
            if match:
                final_set.append(f)
    return final_set


#
# Function rtotal
#
# Example: pd.rolling_apply(rvec, 20, rtotal)
#

def rtotal(vec):
    tcount = np.count_nonzero(vec)
    fcount = len(vec) - tcount
    return tcount - fcount


#
# Function runs
#
# Example: pd.rolling_apply(rvec, 20, runs)
#

def runs(vec):
    return len(list(groupby(vec)))


#
# Function streak
#
# Example: pd.rolling_apply(rvec, 20, streak)
#

def streak(vec):
    return [len(list(g)) for k, g in groupby(vec)][-1]


#
# Function valid_name
#

def valid_name(name):
    identifier = re.compile(r"^[^\d\W]\w*\Z", re.UNICODE)
    result = re.match(identifier, name)
    return result is not None


#
# Function zscore
#
# Example: pd.rolling_apply(rvec, 20, zscore)
#

def zscore(vec):
    n1 = np.count_nonzero(vec)
    n2 = len(vec) - n1
    fac1 = float(2 * n1 * n2)
    fac2 = float(n1 + n2)
    rbar = fac1 / fac2 + 1
    sr2num = fac1 * (fac1 - n1 - n2)
    sr2den = math.pow(fac2, 2) * (fac2 - 1)
    sr = math.sqrt(sr2num / sr2den)
    if sr2den and sr:
        zscore = (runs(vec) - rbar) / sr
    else:
        zscore = 0
    return zscore
