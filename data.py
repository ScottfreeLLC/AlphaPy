##############################################################
#
# Package  : AlphaPy
# Module   : data
# Version  : 1.0
# Copyright: Mark Conway
# Date     : July 29, 2015
#
##############################################################


#
# Imports
#

import _pickle as pickle
from datetime import datetime
from datetime import timedelta
from frame import Frame
from frame import frame_name
from frame import read_frame
from globs import FEEDS
from globs import PSEP
from globs import SSEP
from globs import WILDCARD
import logging
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from scipy import sparse
from sklearn.preprocessing import LabelEncoder


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function load_data
#

def load_data(directory, filename, extension, separator,
              features, target, return_labels=True):
    """
    Read in data from the given directory in a given format.
    """
    # read in file
    df = read_frame(directory, filename, extension, separator)
    # assign target and drop it if necessary
    if target in df.columns:
        y = df[target].values
        y = LabelEncoder().fit_transform(y)
        df = df.drop([target], axis=1)
    elif return_labels:
        logger.info("Target ", target, " not found")
        raise Exception('Target not found')
    # extract features
    if features == WILDCARD:
        X = df
    else:
        X = df[features]
    # labels are returned usually only for training data
    if return_labels:
        return X, y
    else:
        return X


#
# Function get_remote_data
#

def get_remote_data(group,
                    start = datetime.now() - timedelta(365),
                    end = datetime.now()):
    gam = group.all_members()
    feed = FEEDS[group.space.subject]
    for item in gam:
        logger.info("Getting ", item, " data from ", start, " to ", end)
        df = web.DataReader(item, feed, start, end)
        df.reset_index(level=0, inplace=True)
        df = df.rename(columns = lambda x: x.lower().replace(' ',''))
        newf = Frame(item.lower(), group.space, df)
    return


#
# Function load_from_cache
#

def load_from_cache(filename, use_cache=True):
    """
    Attempt to load data from cache.
    """
    data = None
    read_mode = 'rb' if '.pkl' in filename else 'r'
    if use_cache:
        try:
            path = SSEP.join(["cache", filename])
            with open(path, read_mode) as f:
                data = pickle.load(f)
        except IOError:
            pass
    return data


#
# Function save_dataset
#

def save_dataset(filename, X, X_test, features=None, features_test=None):
    """
    Save the training and test sets augmented with the given features.
    """
    if features is not None:
        assert features.shape[1] == features_test.shape[1], "features mismatch"
        if sparse.issparse(X):
            features = sparse.lil_matrix(features)
            features_test = sparse.lil_matrix(features_test)
            X = sparse.hstack((X, features), 'csr')
            X_test = sparse.hstack((X_test, features_test), 'csr')
        else:
            X = np.hstack((X, features))
            X_test = np.hstack((X_test, features_test))
    # Save data to disk
    logger.info("> saving %s to disk", filename)
    pickle_file = PSEP.join([filename, "pkl"])
    pickle_path = SSEP.join(["cache", pickle_file])
    with open(pickle_path, 'wb') as f:
        pickle.dump((X, X_test), f, pickle.HIGHEST_PROTOCOL)
