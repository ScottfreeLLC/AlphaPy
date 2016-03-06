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
from gplearn.genetic import SymbolicTransformer
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

def get_numerical_feature(fname, feature, dt, na_fill=0):
    """
    Get numerical features by looking for float and integer values.
    """
    logger.info("Feature %s is a feature of type %s", fname, dt)
    if na_fill == 0:
        if dt == 'int64':
            fill = feature.value_counts().index[0]
        elif dt == 'float64':
            fill = feature.mean()
    else:
        fill = na_fill 
    feature.fillna(fill, inplace=True)
    return feature


#
# Function get_polynomials
#

def get_polynomials(features, poly_degree):
    """
    Get feature interactions and possibly polynomial interactions.
    """
    polyf = PolynomialFeatures(interaction_only=True,
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

#   count_vect = CountVectorizer(ngram_range=[1, ngrams_max])
#   count_feature = count_vect.fit_transform(feature)
#   tfidf_transformer = TfidfTransformer()
#   tfidf_feature = tfidf_transformer.fit_transform(count_feature)
#   return tfidf_feature


#
# Function create_features
#

def create_features(X, model):
    """
    Extract features from the training and test set.
    """

    logger.info("Creating Features")

    # Extract model parameters

    dummy_limit = model.specs['dummy_limit']
    na_fill = model.specs['na_fill']
    ngrams_max = model.specs['ngrams_max']

    # Log input parameters

    logger.info("Original Features : %s", X.columns)
    logger.info("Feature Count     : %d", X.shape[1])
    logger.info("Dummy Limit       : %d", dummy_limit)
    logger.info("N-Grams           : %d", ngrams_max)

    # Count zero and NaN values

    X['zero_count'] = (X == 0).astype(int).sum(axis=1)
    X['nan_count'] = X.count(axis=1)

    # Iterate through columns, dispatching and transforming each feature.

    all_features = pd.DataFrame()
    for fc in X:
        dtype = X[fc].dtypes
        if dtype == 'float64' or dtype == 'int64':
            feature = get_numerical_feature(fc, X[fc], dtype, na_fill)
        elif dtype == 'object':
            nunique = len(X[fc].unique())
            if nunique <= dummy_limit:
                feature = get_categorical(fc, X[fc], nunique)
            else:
                feature = get_text_feature(fc, X[fc], ngrams_max)
        else:
            raise TypeError("The pandas column type %s is unrecognized", dtype)
        all_features = pd.concat([all_features, feature], axis=1)

    # Call standard scaler for all features

    all_features = StandardScaler().fit_transform(all_features)

    # Return all transformed training and test features

    logger.info("Extracted Feature Count : %d", all_features.shape[1])

    return all_features


#
# Function save_features
#

def save_features(model, X_train, X_test, y_train, y_test):
    """
    Save new features in model.
    """

    logger.info("Saving New Features in Model")

    model.X_train = X_train
    model.X_test = X_test
    model.y_train = y_train
    model.y_test = y_test

    return model


#
# Function create_interactions
#

def create_interactions(X, model):
    """
    Create feature interactions using the training data.
    """

    logger.info("Creating Interactions")

    # Extract model parameters

    fsample_pct = model.specs['fsample_pct']
    gp_learn = model.specs['gp_learn']
    n_jobs = model.specs['n_jobs']
    poly_degree = model.specs['poly_degree']
    regression = model.specs['regression']
    seed = model.specs['seed']
    verbosity = model.specs['verbosity']

    # Extract model data

    X_train = model.X_train
    y_train = model.y_train

    # Log parameters

    logger.info("Initial Feature Count : %d", X.shape[1])
    logger.info("Selection Percentage  : %d", fsample_pct)
    logger.info("Interactions          : %r", interactions)
    logger.info("Polynomial Degree     : %d", poly_degree)
    logger.info("Genetic Features      : %r", gp_learn)

    # Initialize all features

    all_features = X

    # Get polynomial features

    if poly_degree > 0:
        logger.info("Generating Polynomial Features")
        if regression:
            selector = SelectPercentile(f_regression, percentile=fsample_pct)
        else:
            selector = SelectPercentile(f_classif, percentile=fsample_pct)
        selector.fit(X_train, y_train)
        support = selector.get_support()
        pfeatures = get_polynomials(X[:, support], poly_degree)
        logger.info("Polynomial Feature Count : %d", pfeatures.shape[1])
        pfeatures = StandardScaler().fit_transform(pfeatures)
        all_features = np.hstack((all_features, pfeatures))
        logger.info("New Total Feature Count  : %d", all_features.shape[1])
    else:
        logger.info("Skipping Interactions")

    # Generate genetic features

    if gp_learn > 0:
        logger.info("Generating Genetic Features")
        gp = SymbolicTransformer(generations=20, population_size=2000,
                                 hall_of_fame=100, n_components=gp_learn,
                                 parsimony_coefficient=0.0005,
                                 max_samples=0.9, verbose=verbosity,
                                 random_state=seed, n_jobs=n_jobs)
        gp.fit(X_train, y_train)
        gp_features = gp.transform(X)
        logger.info("Genetic Feature Count : %d", gp_features.shape[1])
        gp_features = StandardScaler().fit_transform(gp_features)
        all_features = np.hstack((all_features, gp_features))
        logger.info("New Total Feature Count  : %d", all_features.shape[1])
    else:
        logger.info("Skipping Genetic Features")

    # Return all features

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
