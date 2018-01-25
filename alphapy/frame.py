################################################################################
#
# Package   : AlphaPy
# Module    : frame
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

from alphapy.globals import PSEP, SSEP, USEP
from alphapy.globals import TAG_ID

import logging
import pandas as pd


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function frame_name
#

def frame_name(name, space):
    r"""Get the frame name for the given name and space.

    Parameters
    ----------
    name : str
        Group name.
    space : alphapy.Space
        Context or namespace for the given group name.

    Returns
    -------
    fname : str
        Frame name.

    Examples
    --------

    >>> fname = frame_name('tech', Space('stock', 'prices', '1d'))
    # 'tech_stock_prices_1d'

    """
    return USEP.join([name, space.subject, space.schema, space.fractal])


#
# Class Frame
#

class Frame(object):
    """Create a new Frame that points to a dataframe in memory. All
    frames are stored in ``Frame.frames``. Names must be unique.

    Parameters
    ----------
    name : str
        Frame key.
    space : alphapy.Space
        Namespace of the given frame.
    df : pandas.DataFrame
        The contents of the actual dataframe.

    Attributes
    ----------
    frames : dict
        Class variable for storing all known frames

    Examples
    --------
    
    >>> Frame('tech', Space('stock', 'prices', '5m'), df)

    """

    # class variable to track all frames

    frames = {}

    # __init__

    def __init__(self,
                 name,
                 space,
                 df):
        # code
        if df.__class__.__name__ == 'DataFrame':
            fn = frame_name(name, space)
            if not fn in Frame.frames:
                self.name = name
                self.space = space
                self.df = df
                # add frame to frames list
                Frame.frames[fn] = self
            else:
                logger.info("Frame ", fn, " already exists")
        else:
            logger.info("df must be of type Pandas DataFrame")
        
    # __str__

    def __str__(self):
        return frame_name(self.name, self.space)


#
# Function read_frame
#

def read_frame(directory, filename, extension, separator,
               index_col=None, squeeze=False):
    r"""Read a delimiter-separated file into a data frame.

    Parameters
    ----------
    directory : str
        Full directory specification.
    filename : str
        Name of the file to read, excluding the ``extension``.
    extension : str
        File name extension, e.g., ``csv``.
    separator : str
        The delimiter between fields in the file.
    index_col : str, optional
        Column to use as the row labels in the dataframe.
    squeeze : bool, optional
        If the data contains only one column, then return a pandas Series.

    Returns
    -------
    df : pandas.DataFrame
        The pandas dataframe loaded from the file location. If the file
        cannot be located, then ``None`` is returned.

    """
    file_only = PSEP.join([filename, extension])
    file_all = SSEP.join([directory, file_only])
    logger.info("Loading data from %s", file_all)
    try:
        df = pd.read_csv(file_all, sep=separator, index_col=index_col,
                         squeeze=squeeze)
    except:
        df = None
        logger.info("Could not find or access %s", file_all)
    return df


#
# Function write_frame
#

def write_frame(df, directory, filename, extension, separator,
                index=False, index_label=None, columns=None):
    r"""Write a dataframe into a delimiter-separated file.

    Parameters
    ----------
    df : pandas.DataFrame
        The pandas dataframe to save to a file.
    directory : str
        Full directory specification.
    filename : str
        Name of the file to write, excluding the ``extension``.
    extension : str
        File name extension, e.g., ``csv``.
    separator : str
        The delimiter between fields in the file.
    index : bool, optional
        If ``True``, write the row names (index).
    index_label : str, optional
        A column label for the ``index``.
    columns : str, optional
        A list of column names.

    Returns
    -------
    None : None

    """
    file_only = PSEP.join([filename, extension])
    file_all = SSEP.join([directory, file_only])
    logger.info("Writing data frame to %s", file_all)
    try:
        df.to_csv(file_all, sep=separator, index=index,
                  index_label=index_label, columns=columns)
    except:
        logger.info("Could not write data frame to %s", file_all)


#
# Function load_frames
#

def load_frames(group, directory, extension, separator, splits=False):        
    r"""Read a group of dataframes into memory.

    Parameters
    ----------
    group : alphapy.Group
        The collection of frames to be read into memory.
    directory : str
        Full directory specification.
    extension : str
        File name extension, e.g., ``csv``.
    separator : str
        The delimiter between fields in the file.
    splits : bool, optional
        If ``True``, then all the members of the group are stored in
        separate files corresponding with each member. If ``False``,
        then the data are stored in a single file.

    Returns
    -------
    all_frames : list
        The list of pandas dataframes loaded from the file location. If
        the files cannot be located, then ``None`` is returned.

    """
    logger.info("Loading frames from %s", directory)
    gname = group.name
    gspace = group.space
    # If this is a group analysis, then consolidate the frames.
    # Otherwise, the frames are already aggregated.
    all_frames = []
    if splits:
        gnames = [item.lower() for item in group.members]
        for gn in gnames:
            fname = frame_name(gn, gspace)
            if fname in Frame.frames:
                logger.info("Joining Frame %s", fname)
                df = Frame.frames[fname].df
            else:
                logger.info("Data Frame for %s not found", fname)
                # read file for corresponding frame
                logger.info("Load Data Frame %s from file", fname)
                df = read_frame(directory, fname, extension, separator)
            # add this frame to the consolidated frame list
            if df is not None and not df.empty:
                # set the name
                df.insert(0, TAG_ID, gn)
                all_frames.append(df)
            else:
                logger.debug("Empty Data Frame for: %s", gn)
    else:
        # no splits, so use data from consolidated files
        fname = frame_name(gname, gspace)
        df = read_frame(directory, fname, extension, separator)
        if df is not None and not df.empty:
            all_frames.append(df)
    return all_frames


#
# Function dump_frames
#

def dump_frames(group, directory, extension, separator):        
    r"""Save a group of data frames to disk.

    Parameters
    ----------
    group : alphapy.Group
        The collection of frames to be saved to the file system.
    directory : str
        Full directory specification.
    extension : str
        File name extension, e.g., ``csv``.
    separator : str
        The delimiter between fields in the file.

    Returns
    -------
    None : None

    """
    logger.info("Dumping frames from %s", directory)
    gnames = [item.lower() for item in group.members]
    gspace = group.space
    for gn in gnames:
        fname = frame_name(gn, gspace)
        if fname in Frame.frames:
            logger.info("Writing Data Frame for %s", fname)
            df = Frame.frames[fname].df
            write_frame(df, directory, fname, extension, separator, index=True)
        else:
            logger.info("Data Frame for %s not found", fname)


#
# Function sequence_frame
#

def sequence_frame(df, target, leaders, lag_period=1,
                   forecast_period=1, exclude_cols=[]):
    r"""Run an analysis for a given model and group.

    Parameters
    ----------
    df : pandas.DataFrame
        The original dataframe.
    target : str
        The target variable for prediction.
    leaders : list
        The features that are contemporaneous with the target.
    lag_period : int
        The number of lagged rows for prediction.
    forecast_period : int
        The period for forecasting the target of the analysis.

    Returns
    -------
    new_frame : pandas.DataFrame
        The transformed dataframe with variable sequences.

    """

    # Set Leading and Lagging Columns
    le_cols = sorted(leaders)
    le_len = len(le_cols)
    df_cols = sorted(list(set(df.columns) - set(le_cols)))
    for c in exclude_cols:
        df_cols.remove(c)
    df_len = len(df_cols)

    # Excluded Columns
    new_cols, new_names = list(), list()
    for c in exclude_cols:
        new_cols.append(pd.DataFrame(df[c]))
        new_names.append(c)

    # Lag Features
    for i in range(lag_period, 0, -1):
        new_cols.append(df[df_cols].shift(i))
        new_names += ['%s[%d]' % (df_cols[j], i) for j in range(df_len)]

    # Lag Leaders
    for i in range(lag_period-1, -1, -1):
        new_cols.append(df[le_cols].shift(i))
        if i == 0:
            new_names += [le_cols[j] for j in range(le_len)]
        else:
            new_names += ['%s[%d]' % (le_cols[j], i) for j in range(le_len)]

    # Forecast Target(s)
    new_cols.append(pd.DataFrame(df[target].shift(1-forecast_period)))
    new_names.append(target)

    # Collect all columns into new frame
    new_frame = pd.concat(new_cols, axis=1)
    new_frame.columns = new_names
    return new_frame
