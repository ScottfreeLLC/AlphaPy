################################################################################
#
# Package   : AlphaPy
# Module    : frame
# Created   : July 11, 2013
#
# Copyright 2017 @ Alpha314
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

from alphapy.globs import PSEP, SSEP, USEP

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
    return USEP.join([name, space.subject, space.schema, space.fractal])


#
# Class Frame
#

class Frame(object):

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
    """
    Read from a file into a data frame.
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
                index=False, index_label=None):
    """
    Write to a file from a data frame.
    """
    file_only = PSEP.join([filename, extension])
    file_all = SSEP.join([directory, file_only])
    logger.info("Writing data frame to %s", file_all)
    try:
        df.to_csv(file_all, sep=separator, index=index, index_label=index_label)
    except:
        logger.info("Could not write data frame to %s", file_all)


#
# Function load_frames
#

def load_frames(group, directory, extension, separator, splits=False):        
    """
    Load data into frames unless they are already cached.
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
            if len(df) > 0:
                # set the name
                df.insert(0, 'tag', gn)
                all_frames.append(df)
            else:
                logger.info("Empty Data Frame for: %s", gn)
    else:
        # no splits, so use data from consolidated files
        fname = frame_name(gname, gspace)
        df = read_frame(directory, fname, extension, separator)
        if df is not None:
            all_frames.append(df)
    return all_frames


#
# Function dump_frames
#

def dump_frames(group, directory, extension, separator):        
    """
    Dump frames to disk.
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
