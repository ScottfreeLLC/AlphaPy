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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_numerical_features
#

def get_numerical_features(features):
    """
    Get numerical features by looking for float and integer values.
    """
    numericals = []
    for i, f in enumerate(features.columns):
        dt = features.dtypes[i]
        if dt == 'int64' or dt == 'float64':
            if dt == 'int64':
                value = features[f].value_counts().index[0]
            elif dt == 'float64':
                value = features[f].mean()
            features[f].fillna(value, inplace=True)
            numericals.append(f)
    return features[numericals]


#
# Function get_polynomials
#

def get_polynomials(features, interactions, poly_degree):
    """
    Get feature interactions and possibly polynomial interactions.
    """
    # create polynomial features
    polyf = PolynomialFeatures(interaction_only=interactions,
                               degree=poly_degree,
                               include_bias=False)
    poly_features = polyf.fit_transform(features)
    return poly_features


#
# Function get_categoricals
#

def get_categoricals(features, categoricals, dummy_limit, separator):
    """
    Convert categorical features to one-hot encoded features.
    """
    # parse categorical features
    categoricals = categoricals.split(separator)
    # get dummy variables for each feature
    features = features[categoricals]
    for fc in categoricals:
        nunique = len(features[fc].unique())
        if nunique <= dummy_limit:
            value = features[fc].value_counts().index[0]
            features[fc].fillna(value, inplace=True)
            dummies = pd.get_dummies(features[fc])
            features = features.drop(fc, axis=1)
            features = pd.concat([features, dummies], axis=1)
        else:
            features = features.drop(fc, axis=1)
    return features


#
# Function get_text_features
#

def get_text_features(features, text_features, ngrams_max, separator):
    """
    Vectorize text features and transform to TF-IDF format.
    """
    # parse text features
    text_features = text_features.split(separator)
    # output is in NumPy format
    output_features = np.array([])
    # vectorize each of the text features
    for f in text_features:
        count_vect = CountVectorizer(ngram_range=[1, ngrams_max])
        count_features = count_vect.fit_transform(features[f])
        tfidf_transformer = TfidfTransformer()
        tfidf_features = tfidf_transformer.fit_transform(count_features)
        output_features = np.hstack((output_features, tfidf_features))
    return output_features


#
# Function create_features
#

def create_features(X, categoricals, dummy_limit, interactions, poly_degree,
                    text_features, ngrams_max, separator):
    """
    Extract features from the training and test set.
    """

    logger.info("Features          : %s", X.columns)
    logger.info("Categoricals      : %s", categoricals)
    logger.info("Dummy Limit       : %d", dummy_limit)
    logger.info("Interactions      : %r", interactions)
    logger.info("Polynomial Degree : %d", poly_degree)
    logger.info("Text Features     : %s", text_features)
    logger.info("N-Grams           : %d", ngrams_max)

    # get base features, which are either integer or float
    # categorical and text features are separately identified

    features = get_numerical_features(X)

    # get polynomial features

    if poly_degree > 0:
        pfeatures = get_polynomials(features, interactions, poly_degree)

    # process categorical features

    if categoricals:
        cfeatures = get_categoricals(X, categoricals, dummy_limit, separator)

    # text features code

    if text_features:
        tfeatures = get_text_features(X, text_features, ngrams_max, separator)

    # merge numerical and categorical features

    if categoricals:
        merge_set = [features, cfeatures]
        features = pd.concat(merge_set, axis=1)

    # merge numericals/categoricals, interactions/polynomials, and text

    all_features = features

    if poly_degree > 0:
        all_features = np.hstack((all_features, pfeatures))

    if text_features:
        all_features = np.hstack((all_features, tfeatures))

    # standard scaler code for all features

    all_features = StandardScaler().fit_transform(all_features)

    # fully transformed training and test features

    return all_features


#
# Function drop_features
#

def drop_features(X, drop):
    X.drop(drop, axis=1, inplace=True, errors='ignore')
    return X
