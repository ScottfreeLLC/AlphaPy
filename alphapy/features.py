################################################################################
#
# Package   : AlphaPy
# Module    : features
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

from alphapy.globals import BSEP, LOFF, NULLTEXT
from alphapy.globals import PSEP, SSEP, USEP
from alphapy.globals import Encoders
from alphapy.globals import ModelType
from alphapy.globals import Scalers
from alphapy.market_variables import Variable
from alphapy.market_variables import vparse

import category_encoders as ce
from importlib import import_module
from itertools import groupby
import logging
import math
import numpy as np
import os
import pandas as pd
import re
from scipy import sparse
import scipy.stats as sps
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import sys


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Define feature scoring functions
#

feature_scorers = {'f_classif'    : f_classif,
                   'chi2'         : chi2,
                   'f_regression' : f_regression,
                   'SelectKBest'  : SelectKBest,
                   'SelectFpr'    : SelectFpr,
                   'SelectFdr'    : SelectFdr,
                   'SelectFwe'    : SelectFwe}


#
# Function rtotal
#

def rtotal(vec):
    r"""Calculate the running total.

    Parameters
    ----------
    vec : pandas.Series
        The input array for calculating the running total.

    Returns
    -------
    running_total : int
        The final running total.

    Example
    -------

    >>> vec.rolling(window=20).apply(rtotal)

    """
    tcount = np.count_nonzero(vec)
    fcount = len(vec) - tcount
    running_total = tcount - fcount
    return running_total


#
# Function runs
#

def runs(vec):
    r"""Calculate the total number of runs.

    Parameters
    ----------
    vec : pandas.Series
        The input array for calculating the number of runs.

    Returns
    -------
    runs_value : int
        The total number of runs.

    Example
    -------

    >>> vec.rolling(window=20).apply(runs)

    """
    runs_value = len(list(groupby(vec)))
    return runs_value


#
# Function streak
#

def streak(vec):
    r"""Determine the length of the latest streak.

    Parameters
    ----------
    vec : pandas.Series
        The input array for calculating the latest streak.

    Returns
    -------
    latest_streak : int
        The length of the latest streak.

    Example
    -------

    >>> vec.rolling(window=20).apply(streak)

    """
    latest_streak = [len(list(g)) for k, g in groupby(vec)][-1]
    return latest_streak


#
# Function zscore
#

def zscore(vec):
    r"""Calculate the Z-Score.

    Parameters
    ----------
    vec : pandas.Series
        The input array for calculating the Z-Score.

    Returns
    -------
    zscore : float
        The value of the Z-Score.

    References
    ----------
    To calculate the Z-Score, you can find more information here [ZSCORE]_.

    .. [ZSCORE] https://en.wikipedia.org/wiki/Standard_score

    Example
    -------

    >>> vec.rolling(window=20).apply(zscore)

    """
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

    
#
# Function runs_test
#

def runs_test(f, c, wfuncs, window):
    r"""Perform a runs test on binary series.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``f``.
    wfuncs : list
        The set of runs test functions to apply to the column:

        ``'all'``:
            Run all of the functions below.
        ``'rtotal'``:
            The running total over the ``window`` period.
        ``'runs'``:
            Total number of runs in ``window``.
        ``'streak'``:
            The length of the latest streak.
        ``'zscore'``:
            The Z-Score over the ``window`` period.
    window : int
        The rolling period.

    Returns
    -------
    new_features : pandas.DataFrame
        The dataframe containing the runs test features.

    References
    ----------
    For more information about runs tests for detecting non-randomness,
    refer to [RUNS]_.

    .. [RUNS] http://www.itl.nist.gov/div898/handbook/eda/section3/eda35d.htm

    """

    fc = f[c]
    all_funcs = {'runs'   : runs,
                 'streak' : streak,
                 'rtotal' : rtotal,
                 'zscore' : zscore}
    # use all functions
    if 'all' in wfuncs:
        wfuncs = list(all_funcs.keys())
    # apply each of the runs functions
    new_features = pd.DataFrame()
    for w in wfuncs:
        if w in all_funcs:
            new_feature = fc.rolling(window=window).apply(all_funcs[w])
            new_feature.fillna(0, inplace=True)
            new_column_name = PSEP.join([c, w])
            new_feature = new_feature.rename(new_column_name)
            frames = [new_features, new_feature]
            new_features = pd.concat(frames, axis=1)
        else:
            logger.info("Runs Function %s not found", w)
    return new_features


#
# Function split_to_letters
#

def split_to_letters(f, c):
    r"""Separate text into distinct characters.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the text column in the dataframe ``f``.

    Returns
    -------
    new_feature : pandas.Series
        The array containing the new feature.

    Example
    -------
    The value 'abc' becomes 'a b c'.

    """
    fc = f[c]
    new_feature = None
    dtype = fc.dtypes
    if dtype == 'object':
        fc.fillna(NULLTEXT, inplace=True)
        maxlen = fc.str.len().max()
        if maxlen > 1:
            new_feature = fc.apply(lambda x: BSEP.join(list(x)))
    return new_feature


#
# Function texplode
#

def texplode(f, c):
    r"""Get dummy values for a text column.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the text column in the dataframe ``f``.

    Returns
    -------
    dummies : pandas.DataFrame
        The dataframe containing the dummy variables.

    Example
    -------

    This function is useful for columns that appear to
    have separate character codes but are consolidated
    into a single column. Here, the column ``c`` is
    transformed into five dummy variables.

    === === === === === ===
     c  0_a 1_x 1_b 2_x 2_z
    === === === === === ===
    abz   1   0   1   0   1
    abz   1   0   1   0   1
    axx   1   1   0   1   0
    abz   1   0   1   0   1
    axz   1   1   0   0   1
    === === === === === ===

    """
    fc = f[c]
    maxlen = fc.str.len().max()
    fc.fillna(maxlen * BSEP, inplace=True)
    fpad = str().join(['{:', BSEP, '>', str(maxlen), '}'])
    fcpad = fc.apply(fpad.format)
    fcex = fcpad.apply(lambda x: pd.Series(list(x)))
    dummies = pd.get_dummies(fcex)
    return dummies


#
# Function cvectorize
#

def cvectorize(f, c, n):
    r"""Use the Count Vectorizer and TF-IDF Transformer.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the text column in the dataframe ``f``.
    n : int
        The number of n-grams.

    Returns
    -------
    new_features : sparse matrix
        The transformed features.

    References
    ----------
    To use count vectorization and TF-IDF, you can find more
    information here [TFE]_.

    .. [TFE] http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

    """
    fc = f[c]
    fc.fillna(BSEP, inplace=True)
    cvect = CountVectorizer(ngram_range=[1, n], analyzer='char')
    cfeat = cvect.fit_transform(fc)
    tfidf_transformer = TfidfTransformer()
    new_features = tfidf_transformer.fit_transform(cfeat).toarray()
    return new_features


#
# Function apply_treatment
#

def apply_treatment(fname, df, fparams):
    r"""Apply a treatment function to a column of the dataframe.

    Parameters
    ----------
    fname : str
        Name of the column to be treated in the dataframe ``df``.
    df : pandas.DataFrame
        Dataframe containing the column ``fname``.
    fparams : list
        The module, function, and parameter list of the treatment
        function

    Returns
    -------
    new_features : pandas.DataFrame
        The set of features after applying a treatment function.

    """
    # Extract the treatment parameter list
    module = fparams[0]
    func_name = fparams[1]
    plist = fparams[2:]
    # Append to system path
    sys.path.append(os.getcwd())
    # Import the external treatment function
    ext_module = import_module(module)
    func = getattr(ext_module, func_name)
    # Prepend the parameter list with the data frame and feature name
    plist.insert(0, fname)
    plist.insert(0, df)
    # Apply the treatment
    logger.info("Applying function %s from module %s to feature %s",
                func_name, module, fname)
    return func(*plist)


#
# Function apply_treatments
#

def apply_treatments(model, X):
    r"""Apply special functions to the original features.

    Parameters
    ----------
    model : alphapy.Model
        Model specifications indicating any treatments.
    X : pandas.DataFrame
        Combined train and test data, or just prediction data.

    Returns
    -------
    all_features : pandas.DataFrame
        All features, including treatments.

    Raises
    ------
    IndexError
        The number of treatment rows must match the number of
        rows in ``X``.

    """

    # Extract model parameters
    treatments = model.specs['treatments']

    # Log input parameters

    logger.info("Original Features : %s", X.columns)
    logger.info("Feature Count     : %d", X.shape[1])

    # Iterate through columns, dispatching and transforming each feature.

    logger.info("Applying Treatments")
    all_features = X

    if treatments:
        for fname in treatments:
            # find feature series
            fcols = []
            for col in X.columns:
                if col.split(LOFF)[0] == fname:
                    fcols.append(col)
            # get lag values
            lag_values = []
            for item in fcols:
                _, _, _, lag = vparse(item)
                lag_values.append(lag)
            # apply treatment to the most recent value
            if lag_values:
                f_latest = fcols[lag_values.index(min(lag_values))]
                features = apply_treatment(f_latest, X, treatments[fname])
                if features is not None:
                    if features.shape[0] == X.shape[0]:
                        all_features = pd.concat([all_features, features], axis=1)
                    else:
                        raise IndexError("The number of treatment rows [%d] must match X [%d]" %
                                         (features.shape[0], X.shape[0]))
                else:
                    logger.info("Could not apply treatment for feature %s", fname)
            else:
                logger.info("Feature %s is missing for treatment", fname)
    else:
        logger.info("No Treatments Specified")

    logger.info("New Feature Count : %d", all_features.shape[1])

    # Return all transformed training and test features
    return all_features


#
# Function impute_values
#

def impute_values(features, dt, sentinel):
    r"""Impute values for a given data type. The *median* strategy
    is applied for floating point values, and the *most frequent*
    strategy is applied for integer or Boolean values.

    Parameters
    ----------
    features : pandas.DataFrame
        Dataframe containing the features for imputation.
    dt : str
        The values ``'float64'``, ``'int64'``, or ``'bool'``.
    sentinel : float
        The number to be imputed for NaN values.

    Returns
    -------
    imputed_features : numpy array
        The features after imputation.

    Raises
    ------
    TypeError
        Data type ``dt`` is invalid for imputation.

    References
    ----------
    You can find more information on feature imputation here [IMP]_.

    .. [IMP] http://scikit-learn.org/stable/modules/preprocessing.html#imputation

    """
    try:
        nfeatures = features.shape[1]
    except:
        features = features.values.reshape(-1, 1)
    if dt == 'float64':
        imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    elif dt == 'int64' or dt == 'bool':
        imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    else:
        raise TypeError("Data Type %s is invalid for imputation" % dt)
    imputed = imp.fit_transform(features)
    if imputed.shape[1] == 0:
        nans = np.isnan(features)
        features[nans] = sentinel
        imputed_features = features
    else:
        imputed_features = imputed
    return imputed_features


#
# Function get_numerical_features
#

def get_numerical_features(fnum, fname, df, nvalues, dt,
                           sentinel, logt, plevel):
    r"""Transform numerical features with imputation and possibly
    log-transformation.

    Parameters
    ----------
    fnum : int
        Feature number, strictly for logging purposes
    fname : str
        Name of the numerical column in the dataframe ``df``.
    df : pandas.DataFrame
        Dataframe containing the column ``fname``.
    nvalues : int
        The number of unique values.
    dt : str
        The values ``'float64'``, ``'int64'``, or ``'bool'``.
    sentinel : float
        The number to be imputed for NaN values.
    logt : bool
        If ``True``, then log-transform numerical values.
    plevel : float
        The p-value threshold to test if a feature is normally distributed.

    Returns
    -------
    new_values : numpy array
        The set of imputed and transformed features.

    """
    feature = df[fname]
    if len(feature) == nvalues:
        logger.info("Feature %d: %s is a numerical feature of type %s with maximum number of values %d",
                    fnum, fname, dt, nvalues)
    else:
        logger.info("Feature %d: %s is a numerical feature of type %s with %d unique values",
                    fnum, fname, dt, nvalues)
    # imputer for float, integer, or boolean data types
    new_values = impute_values(feature, dt, sentinel)
    # log-transform any values that do not fit a normal distribution
    if logt and np.all(new_values > 0):
        stat, pvalue = sps.normaltest(new_values)
        if pvalue <= plevel:
            logger.info("Feature %d: %s is not normally distributed [p-value: %f]",
                        fnum, fname, pvalue)
            new_values = np.log(new_values)
    return new_values


#
# Function get_polynomials
#

def get_polynomials(features, poly_degree):
    r"""Generate interactions that are products of distinct features.

    Parameters
    ----------
    features : pandas.DataFrame
        Dataframe containing the features for generating interactions.
    poly_degree : int
        The degree of the polynomial features.

    Returns
    -------
    poly_features : numpy array
        The interaction features only.

    References
    ----------
    You can find more information on polynomial interactions here [POLY]_.

    .. [POLY] http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

    """
    polyf = PolynomialFeatures(interaction_only=True,
                               degree=poly_degree,
                               include_bias=False)
    poly_features = polyf.fit_transform(features)
    return poly_features


#
# Function get_text_features
#

def get_text_features(fnum, fname, df, nvalues, vectorize, ngrams_max):
    r"""Transform text features with count vectorization and TF-IDF,
    or alternatively factorization.

    Parameters
    ----------
    fnum : int
        Feature number, strictly for logging purposes
    fname : str
        Name of the text column in the dataframe ``df``.
    df : pandas.DataFrame
        Dataframe containing the column ``fname``.
    nvalues : int
        The number of unique values.
    vectorize : bool
        If ``True``, then attempt count vectorization.
    ngrams_max : int
        The maximum number of n-grams for count vectorization.

    Returns
    -------
    new_features : numpy array
        The vectorized or factorized text features.

    References
    ----------
    To use count vectorization and TF-IDF, you can find more
    information here [TFE]_.

    """
    feature = df[fname]
    min_length = int(feature.str.len().min())
    max_length = int(feature.str.len().max())
    if len(feature) == nvalues:
        logger.info("Feature %d: %s is a text feature [%d:%d] with maximum number of values %d",
                    fnum, fname, min_length, max_length, nvalues)
    else:
        logger.info("Feature %d: %s is a text feature [%d:%d] with %d unique values",
                    fnum, fname, min_length, max_length, nvalues)
    # need a null text placeholder for vectorization
    feature.fillna(value=NULLTEXT, inplace=True)
    # vectorization creates many columns, otherwise just factorize
    if vectorize:
        logger.info("Feature %d: %s => Attempting Vectorization", fnum, fname)
        count_vect = CountVectorizer(ngram_range=[1, ngrams_max])
        try:
            count_feature = count_vect.fit_transform(feature)
            tfidf_transformer = TfidfTransformer()
            new_features = tfidf_transformer.fit_transform(count_feature).todense()
            logger.info("Feature %d: %s => Vectorization Succeeded", fnum, fname)
        except:
            logger.info("Feature %d: %s => Vectorization Failed", fnum, fname)
            new_features, uniques = pd.factorize(feature)
    else:
        logger.info("Feature %d: %s => Factorization", fnum, fname)
        new_features, uniques = pd.factorize(feature)
    return new_features


#
# Function float_factor
#

def float_factor(x, rounding):
    r"""Convert a floating point number to a factor.

    Parameters
    ----------
    x : float
        The value to convert to a factor.
    rounding : int
        The number of places to round.

    Returns
    -------
    ffactor : int
        The resulting factor.

    """
    num2str = '{0:.{1}f}'.format
    fstr = re.sub("[^0-9]", "", num2str(x, rounding))
    ffactor = int(fstr) if len(fstr) > 0 else 0
    return ffactor


#
# Function create_crosstabs
#

def create_crosstabs(model):
    r"""Create cross-tabulations for categorical variables.

    Parameters
    ----------
    model : alphapy.Model
        The model object containing the data.

    Returns
    -------
    model : alphapy.Model
        The model object with the updated feature map.

    """

    logger.info("Creating Cross-Tabulations")

    # Extract model data
    X = model.X_train
    y = model.y_train

    # Extract model parameters

    factors = model.specs['factors']
    target_value = model.specs['target_value']

    # Iterate through columns, dispatching and transforming each feature.

    crosstabs = {}
    for fname in X:
        if fname in factors:
            logger.info("Creating crosstabs for feature %s", fname)
            ct = pd.crosstab(X[fname], y).apply(lambda r : r / r.sum(), axis=1)
            crosstabs[fname] = ct

    # Save crosstabs to the feature map

    model.feature_map['crosstabs'] = crosstabs
    return model


#
# Function get_factors
#

def get_factors(model, df, fnum, fname, nvalues, dtype,
                encoder, rounding, sentinel):
    r"""Convert the original feature to a factor.

    Parameters
    ----------
    model : alphapy.Model
        Model object with the feature specifications.
    df : pandas.DataFrame
        Dataframe containing the column ``fname``.
    fnum : int
        Feature number, strictly for logging purposes
    fname : str
        Name of the text column in the dataframe ``df``.
    nvalues : int
        The number of unique values.
    dtype : str
        The values ``'float64'``, ``'int64'``, or ``'bool'``.
    encoder : alphapy.features.Encoders
        Type of encoder to apply.
    rounding : int
        Number of places to round.
    sentinel : float
        The number to be imputed for NaN values.

    Returns
    -------
    all_features : numpy array
        The features that have been transformed to factors.

    """

    logger.info("Feature %d: %s is a factor of type %s with %d unique values",
                fnum, fname, dtype, nvalues)
    logger.info("Encoding: %s", encoder)

    # Extract model data

    feature_map = model.feature_map
    model_type = model.specs['model_type']
    target_value = model.specs['target_value']

    # get feature
    feature = df[fname]
    # convert float to factor
    if dtype == 'float64':
        logger.info("Rounding: %d", rounding)
        feature = feature.apply(float_factor, args=[rounding])
    # encoders
    enc = None
    ef = pd.DataFrame(feature)
    if encoder == Encoders.factorize:
        pd_factors = pd.factorize(feature)[0]
        pd_features = pd.DataFrame(pd_factors)
    elif encoder == Encoders.onehot:
        pd_features = pd.get_dummies(feature)
    elif encoder == Encoders.ordinal:
        enc = ce.OrdinalEncoder(cols=[fname])
    elif encoder == Encoders.binary:
        enc = ce.BinaryEncoder(cols=[fname])
    elif encoder == Encoders.helmert:
        enc = ce.HelmertEncoder(cols=[fname])
    elif encoder == Encoders.sumcont:
        enc = ce.SumEncoder(cols=[fname])
    elif encoder == Encoders.polynomial:
        enc = ce.PolynomialEncoder(cols=[fname])
    elif encoder == Encoders.backdiff:
        enc = ce.BackwardDifferenceEncoder(cols=[fname])
    else:
        raise ValueError("Unknown Encoder %s" % encoder)
    # If encoding worked, calculate target percentages for classifiers.
    pd_exists = not pd_features.empty
    enc_exists = enc is not None
    all_features = None
    if pd_exists or enc_exists:
        if pd_exists:
            all_features = pd_features
        elif enc_exists:
            all_features = enc.fit_transform(ef, None)
        # Calculate target percentages for factors
        if (model_type == ModelType.classification and
           fname in feature_map['crosstabs']):
            # Get the crosstab for this feature
            ct = feature_map['crosstabs'][fname]
            # map target percentages to the new feature
            ct_map = ct.to_dict()[target_value]
            ct_feature = df[[fname]].applymap(ct_map.get)
            # impute sentinel for any values that could not be mapped
            ct_feature.fillna(value=sentinel, inplace=True)
            # concatenate all generated features
            all_features = np.column_stack((all_features, ct_feature))
            logger.info("Applied target percentages for %s", fname)
    else:
        raise RuntimeError("Encoding for feature %s failed" % fname)
    return all_features


#
# Function create_numpy_features
#

def create_numpy_features(base_features, sentinel):
    r"""Calculate the sum, mean, standard deviation, and variance
    of each row.

    Parameters
    ----------
    base_features : numpy array
        The feature dataframe.
    sentinel : float
        The number to be imputed for NaN values.

    Returns
    -------
    np_features : numpy array
        The calculated NumPy features.

    """

    logger.info("Creating NumPy Features")

    # Calculate the total, mean, standard deviation, and variance.

    logger.info("NumPy Feature: sum")
    row_sum = np.sum(base_features, axis=1)
    logger.info("NumPy Feature: mean")
    row_mean = np.mean(base_features, axis=1)
    logger.info("NumPy Feature: standard deviation")
    row_std = np.std(base_features, axis=1)
    logger.info("NumPy Feature: variance")
    row_var = np.var(base_features, axis=1)

    # Impute, scale, and stack all new features.

    np_features = np.column_stack((row_sum, row_mean, row_std, row_var))
    np_features = impute_values(np_features, 'float64', sentinel)
    np_features = StandardScaler().fit_transform(np_features)

    # Return new NumPy features

    logger.info("NumPy Feature Count : %d", np_features.shape[1])
    return np_features


#
# Function create_scipy_features
#

def create_scipy_features(base_features, sentinel):
    r"""Calculate the skew, kurtosis, and other statistical features
    for each row.

    Parameters
    ----------
    base_features : numpy array
        The feature dataframe.
    sentinel : float
        The number to be imputed for NaN values.

    Returns
    -------
    sp_features : numpy array
        The calculated SciPy features.

    """

    logger.info("Creating SciPy Features")

    # Generate scipy features

    logger.info("SciPy Feature: geometric mean")
    row_gmean = sps.gmean(base_features, axis=1)
    logger.info("SciPy Feature: kurtosis")
    row_kurtosis = sps.kurtosis(base_features, axis=1)
    logger.info("SciPy Feature: kurtosis test")
    row_ktest, pvalue = sps.kurtosistest(base_features, axis=1)
    logger.info("SciPy Feature: normal test")
    row_normal, pvalue = sps.normaltest(base_features, axis=1)
    logger.info("SciPy Feature: skew")
    row_skew = sps.skew(base_features, axis=1)
    logger.info("SciPy Feature: skew test")
    row_stest, pvalue = sps.skewtest(base_features, axis=1)
    logger.info("SciPy Feature: variation")
    row_var = sps.variation(base_features, axis=1)
    logger.info("SciPy Feature: signal-to-noise ratio")
    row_stn = sps.signaltonoise(base_features, axis=1)
    logger.info("SciPy Feature: standard error of mean")
    row_sem = sps.sem(base_features, axis=1)

    sp_features = np.column_stack((row_gmean, row_kurtosis, row_ktest,
                                   row_normal, row_skew, row_stest,
                                   row_var, row_stn, row_sem))
    sp_features = impute_values(sp_features, 'float64', sentinel)
    sp_features = StandardScaler().fit_transform(sp_features)

    # Return new SciPy features

    logger.info("SciPy Feature Count : %d", sp_features.shape[1])
    return sp_features


#
# Function create_clusters
#

def create_clusters(features, model):
    r"""Cluster the given features.

    Parameters
    ----------
    features : numpy array
        The features to cluster.
    model : alphapy.Model
        The model object with the clustering parameters.

    Returns
    -------
    cfeatures : numpy array
        The calculated clusters.

    References
    ----------
    You can find more information on clustering here [CLUS]_.

    .. [CLUS] http://scikit-learn.org/stable/modules/clustering.html

    """

    logger.info("Creating Clustering Features")

    # Extract model parameters

    cluster_inc = model.specs['cluster_inc']
    cluster_max = model.specs['cluster_max']
    cluster_min = model.specs['cluster_min']
    n_jobs = model.specs['n_jobs']
    seed = model.specs['seed']

    # Log model parameters

    logger.info("Cluster Minimum   : %d", cluster_min)
    logger.info("Cluster Maximum   : %d", cluster_max)
    logger.info("Cluster Increment : %d", cluster_inc)

    # Generate clustering features

    cfeatures = np.zeros((features.shape[0], 1))
    for i in range(cluster_min, cluster_max+1, cluster_inc):
        logger.info("k = %d", i)
        km = MiniBatchKMeans(n_clusters=i, random_state=seed)
        km.fit(features)
        labels = km.predict(features)
        labels = labels.reshape(-1, 1)
        cfeatures = np.column_stack((cfeatures, labels))
    cfeatures = np.delete(cfeatures, 0, axis=1)

    # Return new clustering features

    logger.info("Clustering Feature Count : %d", cfeatures.shape[1])
    return cfeatures


#
# Function create_pca_features
#

def create_pca_features(features, model):
    r"""Apply Principal Component Analysis (PCA) to the features.

    Parameters
    ----------
    features : numpy array
        The input features.
    model : alphapy.Model
        The model object with the PCA parameters.

    Returns
    -------
    pfeatures : numpy array
        The PCA features.

    References
    ----------
    You can find more information on Principal Component Analysis here [PCA]_.

    .. [PCA] http://scikit-learn.org/stable/modules/decomposition.html#pca

    """

    logger.info("Creating PCA Features")

    # Extract model parameters

    pca_inc = model.specs['pca_inc']
    pca_max = model.specs['pca_max']
    pca_min = model.specs['pca_min']
    pca_whiten = model.specs['pca_whiten']

    # Log model parameters

    logger.info("PCA Minimum   : %d", pca_min)
    logger.info("PCA Maximum   : %d", pca_max)
    logger.info("PCA Increment : %d", pca_inc)
    logger.info("PCA Whitening : %r", pca_whiten)

    # Generate clustering features

    pfeatures = np.zeros((features.shape[0], 1))
    for i in range(pca_min, pca_max+1, pca_inc):
        logger.info("n_components = %d", i)
        X_pca = PCA(n_components=i, whiten=pca_whiten).fit_transform(features)
        pfeatures = np.column_stack((pfeatures, X_pca))
    pfeatures = np.delete(pfeatures, 0, axis=1)

    # Return new clustering features

    logger.info("PCA Feature Count : %d", pfeatures.shape[1])
    return pfeatures


#
# Function create_isomap_features
#

def create_isomap_features(features, model):
    r"""Create Isomap features.

    Parameters
    ----------
    features : numpy array
        The input features.
    model : alphapy.Model
        The model object with the Isomap parameters.

    Returns
    -------
    ifeatures : numpy array
        The Isomap features.

    Notes
    -----

    Isomaps are very memory-intensive. Your process will be killed
    if you run out of memory.

    References
    ----------
    You can find more information on Principal Component Analysis here [ISO]_.

    .. [ISO] http://scikit-learn.org/stable/modules/manifold.html#isomap

    """

    logger.info("Creating Isomap Features")

    # Extract model parameters

    iso_components = model.specs['iso_components']
    iso_neighbors = model.specs['iso_neighbors']
    n_jobs = model.specs['n_jobs']

    # Log model parameters

    logger.info("Isomap Components : %d", iso_components)
    logger.info("Isomap Neighbors  : %d", iso_neighbors)

    # Generate Isomap features

    model = Isomap(n_neighbors=iso_neighbors, n_components=iso_components,
                   n_jobs=n_jobs)
    ifeatures = model.fit_transform(features)

    # Return new Isomap features

    logger.info("Isomap Feature Count : %d", ifeatures.shape[1])
    return ifeatures


#
# Function create_tsne_features
#

def create_tsne_features(features, model):
    r"""Create t-SNE features.

    Parameters
    ----------
    features : numpy array
        The input features.
    model : alphapy.Model
        The model object with the t-SNE parameters.

    Returns
    -------
    tfeatures : numpy array
        The t-SNE features.

    References
    ----------
    You can find more information on the t-SNE technique here [TSNE]_.

    .. [TSNE] http://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne

    """

    logger.info("Creating T-SNE Features")

    # Extract model parameters

    seed = model.specs['seed']
    tsne_components = model.specs['tsne_components']
    tsne_learn_rate = model.specs['tsne_learn_rate']
    tsne_perplexity = model.specs['tsne_perplexity']

    # Log model parameters

    logger.info("T-SNE Components    : %d", tsne_components)
    logger.info("T-SNE Learning Rate : %d", tsne_learn_rate)
    logger.info("T-SNE Perplexity    : %d", tsne_perplexity)

    # Generate T-SNE features

    model = TSNE(n_components=tsne_components, perplexity=tsne_perplexity,
                 learning_rate=tsne_learn_rate, random_state=seed)
    tfeatures = model.fit_transform(features)

    # Return new T-SNE features

    logger.info("T-SNE Feature Count : %d", tfeatures.shape[1])
    return tfeatures


#
# Function create_features
#

def create_features(model, X):
    r"""Create features for the train and test set.

    Parameters
    ----------
    model : alphapy.Model
        Model object with the feature specifications.
    X : pandas.DataFrame
        Combined train and test data.

    Returns
    -------
    all_features : numpy array
        The new features.

    Raises
    ------
    TypeError
        Unrecognized data type.

    """

    # Extract model parameters

    clustering = model.specs['clustering']
    counts_flag = model.specs['counts']
    encoder = model.specs['encoder']
    factors = model.specs['factors']
    isomap = model.specs['isomap']
    logtransform = model.specs['logtransform']
    model_type = model.specs['model_type']
    ngrams_max = model.specs['ngrams_max']
    numpy_flag = model.specs['numpy']
    pca = model.specs['pca']
    pvalue_level = model.specs['pvalue_level']
    rounding = model.specs['rounding']
    scaling = model.specs['scaler_option']
    scaler = model.specs['scaler_type']
    scipy_flag = model.specs['scipy']
    sentinel = model.specs['sentinel']
    target_value = model.specs['target_value']
    tsne = model.specs['tsne']
    vectorize = model.specs['vectorize']

    # Log input parameters

    logger.info("Original Features : %s", X.columns)
    logger.info("Feature Count     : %d", X.shape[1])

    # Set classification flag

    classify = True if model_type == ModelType.classification else False

    # Count zero and NaN values

    if counts_flag:
        logger.info("Creating Count Features")
        logger.info("NA Counts")
        X['nan_count'] = X.count(axis=1)
        logger.info("Number Counts")
        for i in range(10):
            fc = USEP.join(['count', str(i)])
            X[fc] = (X == i).astype(int).sum(axis=1)
        logger.info("New Feature Count : %d", X.shape[1])

    # Iterate through columns, dispatching and transforming each feature.

    logger.info("Creating Base Features")
    all_features = np.zeros((X.shape[0], 1))

    for i, fc in enumerate(X):
        fnum = i + 1
        dtype = X[fc].dtypes
        nunique = len(X[fc].unique())
        # standard processing of numerical, categorical, and text features
        if fc in factors:
            features = get_factors(model, X, fnum, fc, nunique, dtype,
                                   encoder, rounding, sentinel)            
        elif dtype == 'float64' or dtype == 'int64' or dtype == 'bool':
            features = get_numerical_features(fnum, fc, X, nunique, dtype,
                                              sentinel, logtransform, pvalue_level)
        elif dtype == 'object':
            features = get_text_features(fnum, fc, X, nunique, vectorize, ngrams_max)
        else:
            raise TypeError("Base Feature Error with unrecognized type %s" % dtype)
        if features.shape[0] == all_features.shape[0]:
            all_features = np.column_stack((all_features, features))
        else:
            logger.info("Feature %s has the wrong number of rows: %d",
                        fc, features.shape[0])
    all_features = np.delete(all_features, 0, axis=1)

    logger.info("New Feature Count : %d", all_features.shape[1])

    # Call standard scaler for all features

    if scaling:
        logger.info("Scaling Base Features")
        if scaler == Scalers.standard:
            all_features = StandardScaler().fit_transform(all_features)
        elif scaler == Scalers.minmax:
            all_features = MinMaxScaler().fit_transform(all_features)
        else:
            logger.info("Unrecognized scaler: %s", scaler)
    else:
        logger.info("Skipping Scaling")

    # Perform dimensionality reduction only on base feature set
    base_features = all_features

    # Calculate the total, mean, standard deviation, and variance

    if numpy_flag:
        np_features = create_numpy_features(base_features, sentinel)
        all_features = np.column_stack((all_features, np_features))
        logger.info("New Feature Count : %d", all_features.shape[1])

    # Generate scipy features

    if scipy_flag:
        sp_features = create_scipy_features(base_features, sentinel)
        all_features = np.column_stack((all_features, sp_features))
        logger.info("New Feature Count : %d", all_features.shape[1])

    # Create clustering features

    if clustering:
        cfeatures = create_clusters(base_features, model)
        all_features = np.column_stack((all_features, cfeatures))
        logger.info("New Feature Count : %d", all_features.shape[1])

    # Create PCA features

    if pca:
        pfeatures = create_pca_features(base_features, model)
        all_features = np.column_stack((all_features, pfeatures))
        logger.info("New Feature Count : %d", all_features.shape[1])

    # Create Isomap features

    if isomap:
        ifeatures = create_isomap_features(base_features, model)
        all_features = np.column_stack((all_features, ifeatures))
        logger.info("New Feature Count : %d", all_features.shape[1])

    # Create T-SNE features

    if tsne:
        tfeatures = create_tsne_features(base_features, model)
        all_features = np.column_stack((all_features, tfeatures))
        logger.info("New Feature Count : %d", all_features.shape[1])

    # Return all transformed training and test features
    return all_features


#
# Function select_features
#

def select_features(model):
    r"""Select features with univariate selection.

    Parameters
    ----------
    model : alphapy.Model
        Model object with the feature selection specifications.

    Returns
    -------
    model : alphapy.Model
        Model object with the revised number of features.

    References
    ----------
    You can find more information on univariate feature selection here [UNI]_.

    .. [UNI] http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection

    """

    logger.info("Feature Selection")

    # Extract model data.

    X_train = model.X_train
    y_train = model.y_train

    # Extract model parameters.

    fs_percentage = model.specs['fs_percentage']
    fs_score_func = model.specs['fs_score_func']

    # Select top features based on percentile.

    fs = SelectPercentile(score_func=fs_score_func,
                          percentile=fs_percentage)

    # Perform feature selection and get the support mask

    fsfit = fs.fit(X_train, y_train)
    support = fsfit.get_support()

    # Record the support vector

    logger.info("Saving Univariate Support")
    model.feature_map['uni_support'] = support

    # Record the support vector

    X_train_new = model.X_train[:, support]
    X_test_new = model.X_test[:, support]

    # Count the number of new features.

    logger.info("Old Feature Count : %d", X_train.shape[1])
    logger.info("New Feature Count : %d", X_train_new.shape[1])

    # Store the reduced features in the model.

    model.X_train = X_train_new
    model.X_test = X_test_new

    # Return the modified model
    return model


#
# Function save_features
#

def save_features(model, X_train, X_test, y_train=None, y_test=None):
    r"""Save new features to the model.

    Parameters
    ----------
    model : alphapy.Model
        Model object with train and test data.
    X_train : numpy array
        Training features.
    X_test : numpy array
        Testing features.
    y_train : numpy array
        Training labels.
    y_test : numpy array
        Testing labels.

    Returns
    -------
    model : alphapy.Model
        Model object with new train and test data.

    """

    logger.info("Saving New Features in Model")

    model.X_train = X_train
    model.X_test = X_test
    if y_train is not None:
        model.y_train = y_train
    if y_test is not None:
        model.y_test = y_test

    return model


#
# Function create_interactions
#

def create_interactions(model, X):
    r"""Create feature interactions based on the model specifications.

    Parameters
    ----------
    model : alphapy.Model
        Model object with train and test data.
    X : numpy array
        Feature Matrix.

    Returns
    -------
    all_features : numpy array
        The new interaction features.

    Raises
    ------
    TypeError
        Unknown model type when creating interactions.

    """

    logger.info("Creating Interactions")

    # Extract model parameters

    interactions = model.specs['interactions']
    isample_pct = model.specs['isample_pct']
    model_type = model.specs['model_type']
    n_jobs = model.specs['n_jobs']
    poly_degree = model.specs['poly_degree']
    predict_mode = model.specs['predict_mode']
    seed = model.specs['seed']
    verbosity = model.specs['verbosity']

    # Extract model data

    X_train = model.X_train
    y_train = model.y_train

    # Log parameters
    logger.info("Initial Feature Count  : %d", X.shape[1])

    # Initialize all features
    all_features = X

    # Get polynomial features

    if interactions:
        if not predict_mode:
            logger.info("Generating Polynomial Features")
            logger.info("Interaction Percentage : %d", isample_pct)
            logger.info("Polynomial Degree      : %d", poly_degree)
            if model_type == ModelType.regression:
                selector = SelectPercentile(f_regression, percentile=isample_pct)
            elif model_type == ModelType.classification:
                selector = SelectPercentile(f_classif, percentile=isample_pct)
            else:
                raise TypeError("Unknown model type when creating interactions")
            selector.fit(X_train, y_train)
            support = selector.get_support()
            model.feature_map['poly_support'] = support
        else:
            support = model.feature_map['poly_support']
        pfeatures = get_polynomials(X[:, support], poly_degree)
        logger.info("Polynomial Feature Count : %d", pfeatures.shape[1])
        pfeatures = StandardScaler().fit_transform(pfeatures)
        all_features = np.hstack((all_features, pfeatures))
        logger.info("New Total Feature Count  : %d", all_features.shape[1])
    else:
        logger.info("Skipping Interactions")

    # Return all features
    return all_features


#
# Function drop_features
#

def drop_features(X, drop):
    r"""Drop any specified features.

    Parameters
    ----------
    X : pandas.DataFrame
        The dataframe containing the features.
    drop : list
        The list of features to remove from ``X``.

    Returns
    -------
    X : pandas.DataFrame
        The dataframe without the dropped features.

    """
    drop_cols = []
    for d in drop:
        for col in X.columns:
            if col.split(LOFF)[0] == d:
                drop_cols.append(col)
    logger.info("Dropping Features: %s", drop_cols)
    logger.info("Original Feature Count : %d", X.shape[1])
    X.drop(drop_cols, axis=1, inplace=True, errors='ignore')
    logger.info("Reduced Feature Count  : %d", X.shape[1])
    return X


#
# Function remove_lv_features
#

def remove_lv_features(model, X):
    r"""Remove low-variance features.

    Parameters
    ----------
    model : alphapy.Model
        Model specifications for removing features.
    X : numpy array
        The feature matrix.

    Returns
    -------
    X_reduced : numpy array
        The reduced feature matrix.

    References
    ----------
    You can find more information on low-variance feature selection here [LV]_.

    .. [LV] http://scikit-learn.org/stable/modules/feature_selection.html#variance-threshold

    """

    logger.info("Removing Low-Variance Features")

    # Extract model parameters

    lv_remove = model.specs['lv_remove']
    lv_threshold = model.specs['lv_threshold']
    predict_mode = model.specs['predict_mode']

    # Remove low-variance features

    if lv_remove:
        logger.info("Low-Variance Threshold  : %.2f", lv_threshold)
        logger.info("Original Feature Count  : %d", X.shape[1])
        if not predict_mode:
            selector = VarianceThreshold(threshold=lv_threshold)
            selector.fit(X)
            support = selector.get_support()
            model.feature_map['lv_support'] = support
        else:
            support = model.feature_map['lv_support']
        X_reduced = X[:, support]
        logger.info("Reduced Feature Count   : %d", X_reduced.shape[1])
    else:
        X_reduced = X
        logger.info("Skipping Low-Variance Features")

    return X_reduced
