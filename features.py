##############################################################
#
# Package  : AlphaPy
# Module   : features
# Version  : 1.0
# Copyright: Mark Conway
# Date     : July 29, 2015
#
##############################################################


#
# Imports
#

from __future__ import division
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_numerical_feature
#

def get_numerical_feature(fname, feature, dt):
    """
    Get numerical features by looking for float and integer values.
    """
    logger.info("Feature %s is a feature of type %s", fname, dt)
    if dt == 'int64':
        value = feature.value_counts().index[0]
    elif dt == 'float64':
        value = feature.mean()
    feature.fillna(value, inplace=True)
    return feature


#
# Function get_polynomials
#

def get_polynomials(features, interactions, poly_degree):
    """
    Get feature interactions and possibly polynomial interactions.
    """
    polyf = PolynomialFeatures(interaction_only=interactions,
                               degree=poly_degree,
                               include_bias=False)
    poly_features = polyf.fit_transform(features)
    return poly_features


#
# Function get_categorical
#

def get_categorical(fname, feature, nvalues):
    """
    Convert a categorical feature to one-hot encoding.
    """
    logger.info("Feature %s is a categorical feature with %d unique values", fname, nvalues)
    value = feature.value_counts().index[0]
    feature.fillna(value, inplace=True)
    dummies = pd.get_dummies(feature)
    return dummies


#
# Function get_text_feature
#

def get_text_feature(fname, feature, ngrams_max):
    """
    Vectorize a text feature and transform to TF-IDF format.
    """
    logger.info("Feature %s is a text feature", fname)
    feature.fillna('', inplace=True)
    hashed_feature = feature.apply(hash)
    return hashed_feature

#    count_vect = CountVectorizer(ngram_range=[1, ngrams_max])
#    count_feature = count_vect.fit_transform(feature)
#    tfidf_transformer = TfidfTransformer()
#    tfidf_feature = tfidf_transformer.fit_transform(count_feature)
#    return tfidf_feature


#
# Function create_features
#

def create_features(X, dummy_limit, ngrams_max):
    """
    Extract features from the training and test set.
    """

    logger.info("Original Features : %s", X.columns)
    logger.info("Feature Count     : %d", X.shape[1])
    logger.info("Dummy Limit       : %d", dummy_limit)
    logger.info("N-Grams           : %d", ngrams_max)

    # Iterate through columns, dispatching for each feature.

    logger.info("Extracting Features")

    all_features = pd.DataFrame()
    for fc in X:
        dtype = X[fc].dtypes
        if dtype == 'float64' or dtype == 'int64':
            feature = get_numerical_feature(fc, X[fc], dtype)
        elif dtype == 'object':
            nunique = len(X[fc].unique())
            if nunique <= dummy_limit:
                feature = get_categorical(fc, X[fc], nunique)
            else:
                feature = get_text_feature(fc, X[fc], ngrams_max)
        else:
            raise TypeError("The pandas column type %s is unrecognized", dtype)
        all_features = pd.concat([all_features, feature], axis=1)

    # Count zero and NaN values

    all_features['zero_count'] = (all_features == 0).astype(int).sum(axis=1)
    all_features['nan_count'] = all_features.count(axis=1)

    # standard scaler code for all features

    all_features = StandardScaler().fit_transform(all_features)

    # fully transformed training and test features

    logger.info("Extracted Feature Count : %d", all_features.shape[1])

    return all_features


#
# Function create_interactions
#

def create_interactions(X, X_train, y_train, fs_percent, interactions, poly_degree):
    """
    Create feature interactions using the training data.
    """

    logger.info("Original Features    : %s", X.columns)
    logger.info("Feature Count        : %d", X.shape[1])
    logger.info("Selection Percentage : %f", fs_percent)
    logger.info("Interactions         : %r", interactions)
    logger.info("Polynomial Degree    : %d", poly_degree)

    # get polynomial features

    if poly_degree > 0:
        selector = SelectPercentile(f_classif, percentile=10)
        selector.fit(X, y)
        pfeatures = get_polynomials(features, interactions, poly_degree)

    # fully transformed training and test features

    logger.info("Extracted Feature Count : %d", all_features.shape[1])

    return all_features


#
# Function drop_features
#

def drop_features(X, drop):
    """
    Drop the given features.
    """

    logger.info("Dropping Features")

    X.drop(drop, axis=1, inplace=True, errors='ignore')
    return X
