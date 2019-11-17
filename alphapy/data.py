################################################################################
#
# Package   : AlphaPy
# Module    : data
# Created   : July 11, 2013
#
# Copyright 2019 ScottFree Analytics LLC
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

from alphapy.frame import Frame
from alphapy.frame import frame_name
from alphapy.frame import read_frame
from alphapy.globals import ModelType
from alphapy.globals import Partition, datasets
from alphapy.globals import PSEP, SSEP, USEP
from alphapy.globals import SamplingMethod
from alphapy.globals import WILDCARD
from alphapy.space import Space

import arrow
from datetime import datetime
from datetime import timedelta
from iexfinance.stocks import get_historical_data
from iexfinance.stocks import get_historical_intraday
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalanceCascade
from imblearn.ensemble import EasyEnsemble
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import TomekLinks
import logging
import math
import numpy as np
import os
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import re
import requests
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
import sys


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_data
#

def get_data(model, partition):
    r"""Get data for the given partition.

    Parameters
    ----------
    model : alphapy.Model
        The model object describing the data.
    partition : alphapy.Partition
        Reference to the dataset.

    Returns
    -------
    X : pandas.DataFrame
        The feature set.
    y : pandas.Series
        The array of target values, if available.

    """

    logger.info("Loading Data")

    # Extract the model data

    directory = model.specs['directory']
    extension = model.specs['extension']
    features = model.specs['features']
    model_type = model.specs['model_type']
    separator = model.specs['separator']
    target = model.specs['target']

    # Initialize X and y

    X = pd.DataFrame()
    y = np.empty([0, 0])

    # Read in the file

    filename = datasets[partition]
    input_dir = SSEP.join([directory, 'input'])
    df = read_frame(input_dir, filename, extension, separator)

    # Get features and target

    if not df.empty:
        if target in df.columns:
            logger.info("Found target %s in data frame", target)
            # check if target column has NaN values
            nan_count = df[target].isnull().sum()
            if nan_count > 0:
                logger.info("Found %d records with NaN target values", nan_count)
                logger.info("Labels (y) for %s will not be used", partition)
            else:
                # assign the target column to y
                y = df[target]
                # encode label only for classification
                if model_type == ModelType.classification:
                    y = LabelEncoder().fit_transform(y)
                logger.info("Labels (y) found for %s", partition)
            # drop the target from the original frame
            df = df.drop([target], axis=1)
        else:
            logger.info("Target %s not found in %s", target, partition)
        # Extract features
        if features == WILDCARD:
            X = df
        else:
            X = df[features]

    # Labels are returned usually only for training data
    return X, y


#
# Function shuffle_data
#

def shuffle_data(model):
    r"""Randomly shuffle the training data.

    Parameters
    ----------
    model : alphapy.Model
        The model object describing the data.

    Returns
    -------
    model : alphapy.Model
        The model object with the shuffled data.

    """

    # Extract model parameters.

    seed = model.specs['seed']
    shuffle = model.specs['shuffle']

    # Extract model data.

    X_train = model.X_train
    y_train = model.y_train

    # Shuffle data

    if shuffle:
        logger.info("Shuffling Training Data")
        np.random.seed(seed)
        new_indices = np.random.permutation(y_train.size)
        model.X_train = X_train[new_indices]
        model.y_train = y_train[new_indices]
    else:
        logger.info("Skipping Shuffling")

    return model


#
# Function sample_data
#

def sample_data(model):
    r"""Sample the training data.

    Sampling is configured in the ``model.yml`` file (data:sampling:method)
    You can learn more about resampling techniques here [IMB]_.

    Parameters
    ----------
    model : alphapy.Model
        The model object describing the data.

    Returns
    -------
    model : alphapy.Model
        The model object with the sampled data.

    """

    logger.info("Sampling Data")

    # Extract model parameters.

    sampling_method = model.specs['sampling_method']
    sampling_ratio = model.specs['sampling_ratio']
    target = model.specs['target']
    target_value = model.specs['target_value']

    # Extract model data.

    X_train = model.X_train
    y_train = model.y_train

    # Calculate the sampling ratio if one is not provided.

    if sampling_ratio > 0.0:
        ratio = sampling_ratio
    else:
        uv, uc = np.unique(y_train, return_counts=True)
        target_index = np.where(uv == target_value)[0][0]
        nontarget_index = np.where(uv != target_value)[0][0]
        ratio = (uc[nontarget_index] / uc[target_index]) - 1.0
    logger.info("Sampling Ratio for target %s [%r]: %f",
                target, target_value, ratio)

    # Choose the sampling method.

    if sampling_method == SamplingMethod.under_random:
        sampler = RandomUnderSampler()
    elif sampling_method == SamplingMethod.under_tomek:
        sampler = TomekLinks()
    elif sampling_method == SamplingMethod.under_cluster:
        sampler = ClusterCentroids()
    elif sampling_method == SamplingMethod.under_nearmiss:
        sampler = NearMiss(version=1)
    elif sampling_method == SamplingMethod.under_ncr:
        sampler = NeighbourhoodCleaningRule()
    elif sampling_method == SamplingMethod.over_random:
        sampler = RandomOverSampler(ratio=ratio)
    elif sampling_method == SamplingMethod.over_smote:
        sampler = SMOTE(ratio=ratio, kind='regular')
    elif sampling_method == SamplingMethod.over_smoteb:
        sampler = SMOTE(ratio=ratio, kind='borderline1')
    elif sampling_method == SamplingMethod.over_smotesv:
        sampler = SMOTE(ratio=ratio, kind='svm')
    elif sampling_method == SamplingMethod.overunder_smote_tomek:
        sampler = SMOTETomek(ratio=ratio)
    elif sampling_method == SamplingMethod.overunder_smote_enn:
        sampler = SMOTEENN(ratio=ratio)
    elif sampling_method == SamplingMethod.ensemble_easy:
        sampler = EasyEnsemble()
    elif sampling_method == SamplingMethod.ensemble_bc:
        sampler = BalanceCascade()
    else:
        raise ValueError("Unknown Sampling Method %s" % sampling_method)

    # Get the newly sampled features.

    X, y = sampler.fit_sample(X_train, y_train)

    logger.info("Original Samples : %d", X_train.shape[0])
    logger.info("New Samples      : %d", X.shape[0])

    # Store the new features in the model.

    model.X_train = X
    model.y_train = y

    return model


#
# Function convert_data
#

def convert_data(df, index_column, intraday_data):
    r"""Convert the market data frame to canonical format.

    Parameters
    ----------
    df : pandas.DataFrame
        The intraday dataframe.
    index_column : str
        The name of the index column.
    intraday_data : bool
        Flag set to True if the frame contains intraday data.

    Returns
    -------
    df : pandas.DataFrame
        The canonical dataframe with date/time index.

    """

    # Standardize column names
    df = df.rename(columns = lambda x: x.lower().replace(' ',''))

    # Create the time/date index if not already done

    if not isinstance(df.index, pd.DatetimeIndex):
        df.reset_index(inplace=True)
        if intraday_data:
            dt_column = df['date'] + ' ' + df['time']
        else:
            dt_column = df['date']
        df[index_column] = pd.to_datetime(dt_column)
        df.set_index(pd.DatetimeIndex(df[index_column]),
                     drop=True, inplace=True)
        del df['date']
        if intraday_data:
            del df['time']

    # Make the remaining columns floating point

    cols_float = ['open', 'high', 'low', 'close', 'volume']
    df[cols_float] = df[cols_float].astype(float)

    # Order the frame by increasing date if necessary
    df = df.sort_index()

    return df


#
# Function enhance_intraday_data
#

def enhance_intraday_data(df):
    r"""Add columns to the intraday dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The intraday dataframe.

    Returns
    -------
    df : pandas.DataFrame
        The dataframe with bar number and end-of-day columns.

    """

    # Group by date first

    df['date'] = df.index.strftime('%Y-%m-%d')
    date_group = df.groupby('date')

    # Number the intraday bars
    df['bar_number'] = date_group.cumcount()

    # Mark the end of the trading day

    df['end_of_day'] = False
    df.loc[date_group.tail(1).index, 'end_of_day'] = True

    # Return the enhanced frame

    del df['date']
    return df


#
# Function get_google_intraday_data
#

def get_google_intraday_data(symbol, lookback_period, fractal):
    r"""Get Google Finance intraday data.

    We get intraday data from the Google Finance API, even though
    it is not officially supported. You can retrieve a maximum of
    50 days of history, so you may want to build your own database
    for more extensive backtesting.

    Parameters
    ----------
    symbol : str
        A valid stock symbol.
    lookback_period : int
        The number of days of intraday data to retrieve, capped at 50.
    fractal : str
        The intraday frequency, e.g., "5m" for 5-minute data.

    Returns
    -------
    df : pandas.DataFrame
        The dataframe containing the intraday data.

    """

    # Google requires upper-case symbol, otherwise not found
    symbol = symbol.upper()
    # Initialize data frame
    df = pd.DataFrame()
    # Convert fractal to interval
    interval = 60 * int(re.findall('\d+', fractal)[0])
    # Google has a 50-day limit
    max_days = 50
    if lookback_period > max_days:
        lookback_period = max_days
    # Set Google data constants
    toffset = 7
    line_length = 6
    # Make the request to Google
    base_url = 'https://finance.google.com/finance/getprices?q={}&i={}&p={}d&f=d,o,h,l,c,v'
    url = base_url.format(symbol, interval, lookback_period)
    response = requests.get(url)
    # Process the response
    text = response.text.split('\n')
    records = []
    for line in text[toffset:]:
        items = line.split(',')
        if len(items) == line_length:
            dt_item = items[0]
            close_item = items[1]
            high_item = items[2]
            low_item = items[3]
            open_item = items[4]
            volume_item = items[5]
            if dt_item[0] == 'a':
                day_item = float(dt_item[1:])
                offset = 0
            else:
                offset = float(dt_item)
            dt = datetime.fromtimestamp(day_item + (interval * offset))
            dt = pd.to_datetime(dt)
            dt_date = dt.strftime('%Y-%m-%d')
            dt_time = dt.strftime('%H:%M:%S')
            record = (dt_date, dt_time, open_item, high_item, low_item, close_item, volume_item)
            records.append(record)
    # Create data frame
    cols = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame.from_records(records, columns=cols)
    # Return the dataframe
    return df


#
# Function get_google_data
#

def get_google_data(schema, subschema, symbol, intraday_data, data_fractal,
                    from_date, to_date, lookback_period):
    r"""Get data from Google.

    Parameters
    ----------
    schema : str
        The schema (including any subschema) for this data feed.
    subschema : str
        Any subschema for this data feed.
    symbol : str
        A valid stock symbol.
    intraday_data : bool
        If True, then get intraday data.
    data_fractal : str
        Pandas offset alias.
    from_date : str
        Starting date for symbol retrieval.
    to_date : str
        Ending date for symbol retrieval.
    lookback_period : int
        The number of periods of data to retrieve.

    Returns
    -------
    df : pandas.DataFrame
        The dataframe containing the market data.

    """

    df = pd.DataFrame()
    if intraday_data:
        # use internal function
        # df = get_google_intraday_data(symbol, lookback_period, data_fractal)
        logger.info("Google Finance API for intraday data no longer available")
    else:
        # Google Finance API no longer available
        logger.info("Google Finance API for daily data no longer available")
    return df


#
# Function get_iex_data
#

def get_iex_data(schema, subschema, symbol, intraday_data, data_fractal,
                 from_date, to_date, lookback_period):
    r"""Get data from IEX.

    Parameters
    ----------
    schema : str
        The schema (including any subschema) for this data feed.
    subschema : str
        Any subschema for this data feed.
    symbol : str
        A valid stock symbol.
    intraday_data : bool
        If True, then get intraday data.
    data_fractal : str
        Pandas offset alias.
    from_date : str
        Starting date for symbol retrieval.
    to_date : str
        Ending date for symbol retrieval.
    lookback_period : int
        The number of periods of data to retrieve.

    Returns
    -------
    df : pandas.DataFrame
        The dataframe containing the market data.

    """

    symbol = symbol.upper()
    df = pd.DataFrame()

    if intraday_data:
        # use iexfinance function to get intraday data for each date
        df = pd.DataFrame()
        for d in pd.date_range(from_date, to_date):
            dstr = d.strftime('%Y-%m-%d')
            logger.info("%s Data for %s", symbol, dstr)
            try:
                df1 = get_historical_intraday(symbol, d, output_format="pandas")
                df1_len = len(df1)
                if df1_len > 0:
                    logger.info("%s: %d rows", symbol, df1_len)
                    df = df.append(df1)
                else:
                    logger.info("%s: No Trading Data for %s", symbol, dstr)
            except:
                iex_error = "*** IEX Intraday Data Error (check Quota) ***"
                logger.error(iex_error)
                sys.exit(iex_error)
    else:
        # use iexfinance function for historical daily data
        try:
            df = get_historical_data(symbol, from_date, to_date, output_format="pandas")
        except:
            iex_error = "*** IEX Daily Data Error (check Quota) ***"
            logger.error(iex_error)
            sys.exit(iex_error)
    return df


#
# Function get_pandas_data
#

def get_pandas_data(schema, subschema, symbol, intraday_data, data_fractal,
                    from_date, to_date, lookback_period):
    r"""Get Pandas Web Reader data.

    Parameters
    ----------
    schema : str
        The schema (including any subschema) for this data feed.
    subschema : str
        Any subschema for this data feed.
    symbol : str
        A valid stock symbol.
    intraday_data : bool
        If True, then get intraday data.
    data_fractal : str
        Pandas offset alias.
    from_date : str
        Starting date for symbol retrieval.
    to_date : str
        Ending date for symbol retrieval.
    lookback_period : int
        The number of periods of data to retrieve.

    Returns
    -------
    df : pandas.DataFrame
        The dataframe containing the market data.

    """

    # Call the Pandas Web data reader.

    try:
        df = web.DataReader(symbol, schema, from_date, to_date)
    except:
        df = pd.DataFrame()
        logger.info("Could not retrieve %s data with pandas-datareader", symbol.upper())

    return df


#
# Function get_quandl_data
#

def get_quandl_data(schema, subschema, symbol, intraday_data, data_fractal,
                    from_date, to_date, lookback_period):
    r"""Get Quandl data.

    Parameters
    ----------
    schema : str
        The schema for this data feed.
    subschema : str
        Any subschema for this data feed.
    symbol : str
        A valid stock symbol.
    intraday_data : bool
        If True, then get intraday data.
    data_fractal : str
        Pandas offset alias.
    from_date : str
        Starting date for symbol retrieval.
    to_date : str
        Ending date for symbol retrieval.
    lookback_period : int
        The number of periods of data to retrieve.

    Returns
    -------
    df : pandas.DataFrame
        The dataframe containing the market data.

    """

    # Quandl is a special case with subfeeds.

    symbol = SSEP.join([subschema.upper(), symbol.upper()])

    # Call the Pandas Web data reader.

    df = get_pandas_data(schema, subschema, symbol, intraday_data, data_fractal,
                         from_date, to_date, lookback_period)

    return df


#
# Function get_yahoo_data
#

def get_yahoo_data(schema, subschema, symbol, intraday_data, data_fractal,
                    from_date, to_date, lookback_period):
    r"""Get Yahoo data.

    Parameters
    ----------
    schema : str
        The schema (including any subschema) for this data feed.
    subschema : str
        Any subschema for this data feed.
    symbol : str
        A valid stock symbol.
    intraday_data : bool
        If True, then get intraday data.
    data_fractal : str
        Pandas offset alias.
    from_date : str
        Starting date for symbol retrieval.
    to_date : str
        Ending date for symbol retrieval.
    lookback_period : int
        The number of periods of data to retrieve.

    Returns
    -------
    df : pandas.DataFrame
        The dataframe containing the market data.

    """

    df = pd.DataFrame()
    if intraday_data:
        url = 'https://query1.finance.yahoo.com/v8/finance/chart/'
        data_range = ''.join([str(lookback_period), 'd'])
        interval = int(''.join(filter(str.isdigit, data_fractal)))
        fractal = re.sub(r'\d+', '', data_fractal)
        mapper = {'H': 60, 'T': 1, 'min':1, 'S': 1./60}
        interval = math.ceil(interval * mapper[fractal])
        data_interval = ''.join([str(interval), 'm'])
        qualifiers = '{}?range={}&interval={}'.format(symbol, data_range, data_interval)
        request = url + qualifiers
        logger.info(request)
        response = requests.get(request)
        response_json = response.json()['chart']
        if response_json['result']:
            body = response_json['result'][0]
            dt = pd.Series(map(lambda x: arrow.get(x).to('EST').datetime.replace(tzinfo=None), body['timestamp']), name='dt')
            df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
            df = df.loc[:, ('open', 'high', 'low', 'close', 'volume')]
        else:
            logger.info("Could not get data from %s", schema)
            logger.info(response_json['error']['code'])
            logger.info(response_json['error']['description'])
    else:
        # use pandas data reader
        df = get_pandas_data(schema, subschema, symbol, intraday_data, data_fractal,
                             from_date, to_date, lookback_period)

    return df


#
# Data Dispatch Tables
#

data_dispatch_table = {'google' : get_google_data,
                       'iex'    : get_iex_data,
                       'pandas' : get_pandas_data,
                       'quandl' : get_quandl_data,
                       'yahoo'  : get_yahoo_data}


#
# Function get_market_data
#

def get_market_data(model, market_specs, group, lookback_period, intraday_data=False):
    r"""Get data from an external feed.

    Parameters
    ----------
    model : alphapy.Model
        The model object describing the data.
    market_specs : dict
        The specifications for controlling the MarketFlow pipeline.
    group : alphapy.Group
        The group of symbols.
    lookback_period : int
        The number of periods of data to retrieve.
    intraday_data : bool
        If True, then get intraday data.

    Returns
    -------
    n_periods : int
        The maximum number of periods actually retrieved.

    """

    # Unpack market specifications

    data_fractal = market_specs['data_fractal']
    subschema = market_specs['subschema']

    # Unpack model specifications

    directory = model.specs['directory']
    extension = model.specs['extension']
    separator = model.specs['separator']

    # Unpack group elements

    gspace = group.space
    schema = gspace.schema
    fractal = gspace.fractal

    # Determine the feed source

    if intraday_data:
        # intraday data (date and time)
        logger.info("%s Intraday Data [%s] for %d periods",
                    schema, data_fractal, lookback_period)
        index_column = 'datetime'
    else:
        # daily data or higher (date only)
        logger.info("%s Daily Data [%s] for %d periods",
                    schema, data_fractal, lookback_period)
        index_column = 'date'

    # Get the data from the relevant feed

    data_dir = SSEP.join([directory, 'data'])
    n_periods = 0
    resample_data = True if fractal != data_fractal else False

    # Date Arithmetic

    to_date = pd.to_datetime('today')
    from_date = to_date - pd.to_timedelta(lookback_period, unit='d')
    to_date = to_date.strftime('%Y-%m-%d')
    from_date = from_date.strftime('%Y-%m-%d')

    # Get the data from the specified data feed

    df = pd.DataFrame()
    for symbol in group.members:
        logger.info("Getting %s data from %s to %s",
                    symbol.upper(), from_date, to_date)
        # Locate the data source
        if schema == 'data':
            # local intraday or daily
            dspace = Space(gspace.subject, gspace.schema, data_fractal)
            fname = frame_name(symbol.lower(), dspace)
            df = read_frame(data_dir, fname, extension, separator)
        elif schema in data_dispatch_table.keys():
            df = data_dispatch_table[schema](schema,
                                             subschema,
                                             symbol,
                                             intraday_data,
                                             data_fractal,
                                             from_date,
                                             to_date,
                                             lookback_period)
        else:
            logger.error("Unsupported Data Source: %s", schema)
        # Now that we have content, standardize the data
        if not df.empty:
            logger.info("Rows: %d [%s]", len(df), data_fractal)
            # convert data to canonical form
            df = convert_data(df, index_column, intraday_data)
            # resample data and forward fill any NA values
            if resample_data:
                df = df.resample(fractal).agg({'open'   : 'first',
                                               'high'   : 'max',
                                               'low'    : 'min',
                                               'close'  : 'last',
                                               'volume' : 'sum'})
                df.dropna(axis=0, how='any', inplace=True)
                logger.info("Rows after Resampling at %s: %d",
                            fractal, len(df))
            # add intraday columns if necessary
            if intraday_data:
                df = enhance_intraday_data(df)
            # allocate global Frame
            newf = Frame(symbol.lower(), gspace, df)
            if newf is None:
                logger.error("Could not allocate Frame for: %s", symbol.upper())
            # calculate maximum number of periods
            df_len = len(df)
            if df_len > n_periods:
                n_periods = df_len
        else:
            logger.info("No DataFrame for %s", symbol.upper())

    # The number of periods actually retrieved
    return n_periods
