##############################################################
#
# Package   : AlphaPy
# Module    : alpha_game
# Version   : 1.0
# Copyright : Mark Conway
# Date      : October 26, 2015
#
##############################################################


#
# Imports
#

print(__doc__)

from alpha import pipeline
import argparse
import datetime
from frame import read_frame
from frame import write_frame
from globs import PSEP
from globs import SSEP
from globs import USEP
from globs import WILDCARD
from itertools import groupby
import logging
import math
from model import Model
import numpy as np
from numpy.random import randn
import pandas as pd
from space import Space


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Sports Fields
#
# The following fields are repeated for:
#     1. 'home'
#     2. 'away'
#     3. 'delta'
#
# Note that [Target]s will not be merged into the Game table;
# these targets will be predictors in the Game table that are
# generated after each game result. All of the fields below
# are predictors and are generated a priori, i.e., we calculate
# deltas from the last previously played game for each team and
# these data go into the row for the next game to be played.
#

sports_dict = {'wins' : int,
               'losses' : int,
               'ties' : int,
               'days_since_first_game' : int,
               'days_since_previous_game' : int,
               'won_on_points' : bool,
               'lost_on_points' : bool,
               'won_on_spread' : bool,
               'lost_on_spread' : bool,
               'point_win_streak' : int,
               'point_loss_streak' : int,
               'point_margin_game' : int,
               'point_margin_season' : int,
               'point_margin_season_avg' : float,
               'point_margin_streak' : int,
               'point_margin_streak_avg' : float,
               'point_margin_ngames' : int,
               'point_margin_ngames_avg' : float,
               'cover_win_streak' : int,
               'cover_loss_streak' : int,
               'cover_margin_game' : float,
               'cover_margin_season' : float, 
               'cover_margin_season_avg' : float,
               'cover_margin_streak' : float,
               'cover_margin_streak_avg' : float,
               'cover_margin_ngames' : float,
               'cover_margin_ngames_avg' : float,
               'total_points' : int,
               'overunder_margin' : float,
               'over' : bool,
               'under' : bool,
               'over_streak' : int,
               'under_streak' : int,
               'overunder_season' : float,
               'overunder_season_avg' : float,
               'overunder_streak' : float,
               'overunder_streak_avg' : float,
               'overunder_ngames' : float,
               'overunder_ngames_avg' : float}


#
# These are the "leakers". Generally, we try to predict one of these
# variables as the target and lag the remaining ones.
#

game_dict = {'point_margin_game' : int,
             'won_on_points' : bool,
             'lost_on_points' : bool,
             'cover_margin_game' : float,
             'won_on_spread' : bool,
             'lost_on_spread' : bool,
             'overunder_margin' : float,
             'over' : bool,
             'under' : bool}


#
# Function get_point_margin
#

def get_point_margin(row, score, opponent_score):
    point_margin = 0
    nans = math.isnan(row[score]) or math.isnan(row[opponent_score])
    if not nans:
        point_margin = row[score] - row[opponent_score]
    return point_margin


#
# Function get_wins
#

def get_wins(point_margin):
    return 1 if point_margin > 0 else 0


#
# Function get_losses
#

def get_losses(point_margin):
    return 1 if point_margin < 0 else 0


#
# Function get_ties
#

def get_ties(point_margin):
    return 1 if point_margin == 0 else 0


#
# Function get_day_offset
#

def get_day_offset(date_vector):
    dv = pd.to_datetime(date_vector)
    offsets = pd.to_datetime(dv) - pd.to_datetime(dv[0])
    return offsets.astype('timedelta64[D]').astype(int)


#
# Function get_series_diff
#

def get_series_diff(series):
    new_series = pd.Series(len(series))
    new_series = series.diff()
    new_series[0] = 0
    return new_series


#
# Function get_streak
#

def get_streak(series, start_index, window):
    if window <= 0:
        window = len(series)
    i = start_index
    streak = 0
    while i >= 0 and (start_index-i+1) < window and series[i]:
        streak += 1
        i -= 1
    return streak


#
# Function add_features
#

def add_features(frame, fdict, flen, prefix=''):
    # generate sequences
    seqint = [0] * flen
    seqfloat = [0.0] * flen
    seqbool = [False] * flen
    # initialize new fields in frame
    for key, value in fdict.iteritems():
        newkey = key
        if prefix:
            newkey = PSEP.join([prefix, newkey])
        if value == int:
            frame[newkey] = pd.Series(seqint)
        elif value == float:
            frame[newkey] = pd.Series(seqfloat)
        elif value == bool:
            frame[newkey] = pd.Series(seqbool)
        else:
            raise ValueError("Type to generate feature series not found")
    return frame


#
# Function generate_team_frame
#

def generate_team_frame(team, tf, tdict):
    # Initialize new features
    tf = add_features(tf, tdict, len(tf))
    # Daily Offsets
    tf['days_since_first_game'] = get_day_offset(tf['date'])
    tf['days_since_previous_game'] = get_series_diff(tf['days_since_first_game'])
    # Team Loop
    for index, row in tf.iterrows():
        if team == row[home_team]:
            tf['point_margin_game'].at[index] = get_point_margin(row, 'home.score', 'away.score')
            line = row['line']
        elif team == row[away_team]:
            tf['point_margin_game'].at[index] = get_point_margin(row, 'away.score', 'home.score')
            line = -row['line']
        else:
            raise KeyError("Team not found in Team Frame")
        if index == 0:
            tf['wins'].at[index] = get_wins(tf['point_margin_game'].at[index])
            tf['losses'].at[index] = get_losses(tf['point_margin_game'].at[index])
            tf['ties'].at[index] = get_ties(tf['point_margin_game'].at[index])
        else:
            tf['wins'].at[index] = tf['wins'].at[index-1] + get_wins(tf['point_margin_game'].at[index])
            tf['losses'].at[index] = tf['losses'].at[index-1] + get_losses(tf['point_margin_game'].at[index])
            tf['ties'].at[index] = tf['ties'].at[index-1] + get_ties(tf['point_margin_game'].at[index])
        tf['won_on_points'].at[index] = True if tf['point_margin_game'].at[index] > 0 else False
        tf['lost_on_points'].at[index] = True if tf['point_margin_game'].at[index] < 0 else False
        tf['cover_margin_game'].at[index] = tf['point_margin_game'].at[index] + line
        tf['won_on_spread'].at[index] = True if tf['cover_margin_game'].at[index] > 0 else False
        tf['lost_on_spread'].at[index] = True if tf['cover_margin_game'].at[index] <= 0 else False
        nans = math.isnan(row['home.score']) or math.isnan(row['away.score'])
        if not nans:
            tf['total_points'].at[index] = row['home.score'] + row['away.score']
        nans = math.isnan(row['over_under'])
        if not nans:
            tf['overunder_margin'].at[index] = tf['total_points'].at[index] - row['over_under']
        tf['over'].at[index] = True if tf['overunder_margin'].at[index] > 0 else False
        tf['under'].at[index] = True if tf['overunder_margin'].at[index] < 0 else False
        tf['point_win_streak'].at[index] = get_streak(tf['won_on_points'], index, 0)
        tf['point_loss_streak'].at[index] = get_streak(tf['lost_on_points'], index, 0)
        tf['cover_win_streak'].at[index] = get_streak(tf['won_on_spread'], index, 0)
        tf['cover_loss_streak'].at[index] = get_streak(tf['lost_on_spread'], index, 0)
        tf['over_streak'].at[index] = get_streak(tf['over'], index, 0)
        tf['under_streak'].at[index] = get_streak(tf['under'], index, 0)
        # Handle the streaks
        if tf['point_win_streak'].at[index] > 0:
            streak = tf['point_win_streak'].at[index]
        elif tf['point_loss_streak'].at[index] > 0:
            streak = tf['point_loss_streak'].at[index]
        else:
            streak = 1
        tf['point_margin_streak'].at[index] = tf['point_margin_game'][index-streak+1:index+1].sum()
        tf['point_margin_streak_avg'].at[index] = tf['point_margin_game'][index-streak+1:index+1].mean()
        if tf['cover_win_streak'].at[index] > 0:
            streak = tf['cover_win_streak'].at[index]
        elif tf['cover_loss_streak'].at[index] > 0:
            streak = tf['cover_loss_streak'].at[index]
        else:
            streak = 1
        tf['cover_margin_streak'].at[index] = tf['cover_margin_game'][index-streak+1:index+1].sum()
        tf['cover_margin_streak_avg'].at[index] = tf['cover_margin_game'][index-streak+1:index+1].mean()
        if tf['over_streak'].at[index] > 0:
            streak = tf['over_streak'].at[index]
        elif tf['under_streak'].at[index] > 0:
            streak = tf['under_streak'].at[index]
        else:
            streak = 1
        tf['overunder_streak'].at[index] = tf['overunder_margin'][index-streak+1:index+1].sum()
        tf['overunder_streak_avg'].at[index] = tf['overunder_margin'][index-streak+1:index+1].mean()
    # Rolling and Expanding Variables
    tf['point_margin_season'] = tf['point_margin_game'].cumsum()
    tf['point_margin_season_avg'] = pd.expanding_mean(tf['point_margin_game'])
    tf['point_margin_ngames'] = pd.rolling_sum(tf['point_margin_game'], window=window, min_periods=1)
    tf['point_margin_ngames_avg'] = pd.rolling_mean(tf['point_margin_game'], window=window, min_periods=1)
    tf['cover_margin_season'] = tf['cover_margin_game'].cumsum()
    tf['cover_margin_season_avg'] = pd.expanding_mean(tf['cover_margin_game'])
    tf['cover_margin_ngames'] = pd.rolling_sum(tf['cover_margin_game'], window=window, min_periods=1)
    tf['cover_margin_ngames_avg'] = pd.rolling_mean(tf['cover_margin_game'], window=window, min_periods=1)
    tf['overunder_season'] = tf['overunder_margin'].cumsum()
    tf['overunder_season_avg'] = pd.expanding_mean(tf['overunder_margin'])
    tf['overunder_ngames'] = pd.rolling_sum(tf['overunder_margin'], window=window, min_periods=1)
    tf['overunder_ngames_avg'] = pd.rolling_mean(tf['overunder_margin'], window=window, min_periods=1)
    return tf


#
# Function insert_model_data
#

def insert_model_data(mf, mpos, mdict, tf, tpos, prefix):
    team_row = tf.iloc[tpos]
    for key, value in mdict.iteritems():
        newkey = key
        if prefix:
            newkey = PSEP.join([prefix, newkey])
        mf.at[mpos, newkey] = team_row[key]
    return mf


#
# Function generate_delta_data
#

def generate_delta_data(frame, fdict, prefix1, prefix2):
    for key, value in fdict.iteritems():
        newkey = PSEP.join(['delta', key])
        key1 = PSEP.join([prefix1, key])
        key2 = PSEP.join([prefix2, key])
        frame[newkey] = frame[key1] - frame[key2]
    return frame


#
# MAIN PROGRAM
#

if __name__ == '__main__':

    # Logging

    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="alpha314_game.log", filemode='a', level=logging.DEBUG,
                        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    logger = logging.getLogger(__name__)

    # Argument Parsing

    algos = 'XGB'

    parser = argparse.ArgumentParser(description="Alpha314 Game Parser")
    parser.add_argument('-algos', dest="algorithms", action='store', default=algos,
                        help='algorithms for either classification or regression')
    parser.add_argument('-base', dest="base_dir", default="/Users/markconway/Projects/AlphaPy_Projects",
                        help="base directory location")
    parser.add_argument('-grid', dest="grid_search", action="store_true",
                        help="perform a grid search [False]")
    parser.add_argument("-name", dest="project", default="NBA",
                        help="unique project name")
    parser.add_argument('-nest', dest="n_estimators", type=int, default=201,
                        help="default number of estimators [201]")
    parser.add_argument('-nfold', dest="n_folds", type=int, default=5,
                        help="number of folds for cross-validation")
    parser.add_argument('-nstep', dest="n_step", type=int, default=5,
                        help="step increment for recursive feature elimination")
    parser.add_argument('-plots', dest="plots", action="store_true",
                        help="show plots [False]")
    parser.add_argument('-pmax', dest="points_max", type=int, default=45,
                        help="maximum points for simulation")
    parser.add_argument('-pmin', dest="points_min", type=int, default=3,
                        help="minimum points for simulation")
    parser.add_argument('-rand', dest="random_scoring", action="store_true",
                        help="generate random scores for simulation [False]")
    parser.add_argument('-reg', dest="regression", action="store_true",
                        help="classification [default] or regression")
    parser.add_argument('-rfe', dest="rfe", action="store_true",
                        help="recursive feature elimination [False]")
    parser.add_argument('-seas', dest="season", type=int, default=0,
                        help="season [default is all seasons]")
    parser.add_argument('-split', dest="split", type=float, default=0.3,
                        help="percentage of data withheld for testing")
    parser.add_argument('-win', dest="window", type=int, default=3,
                        help="sliding window length for rolling calculations")
    parser.add_argument('-v', dest="verbosity", type=int, default=2,
                        help="verbosity level")
    parser.add_argument("-y", dest="target", action='store', default='won_on_spread',
                        help="target variable [y]")

    # Print the arguments

    args = parser.parse_args()

    logger.info('\nPARAMETERS:\n')
    logger.info('algorithms      = %s', args.algorithms)
    logger.info('base_dir        = %s', args.base_dir)
    logger.info('grid_search     = %r', args.grid_search)
    logger.info('n_estimators    = %d', args.n_estimators)
    logger.info('n_folds         = %d', args.n_folds)
    logger.info('n_step          = %d', args.n_step)
    logger.info('plots           = %r', args.plots)
    logger.info('points_max      = %d', args.points_max)
    logger.info('points_min      = %d', args.points_min)
    logger.info('project         = %s', args.project)
    logger.info('random_scoring  = %r', args.random_scoring)
    logger.info('regression      = %r', args.regression)
    logger.info('rfe             = %r', args.rfe)
    logger.info('season          = %d', args.season)
    logger.info('split           = %f', args.split)
    logger.info('target [y]      = %s', args.target)
    logger.info('verbosity       = %d', args.verbosity)
    logger.info('window          = %d', args.window)
    logger.info('\n')

    # Unpack arguments

    base_dir = args.base_dir
    points_max = args.points_max
    points_min = args.points_min
    organization = args.project
    random_scoring = args.random_scoring
    season = args.season
    window = args.window    

    # Debug the program

    logger.debug('\n' + '='*50 + '\n')

    # Call the pipeline

    logger.info("Starting Game Pipeline")

    # Create the game scores space

    space = Space('game', 'scores', '1g')

    #
    # Derived Variables
    #

    series = space.schema
    team1_prefix = 'home'
    team2_prefix = 'away'
    home_team = PSEP.join([team1_prefix, 'team'])
    away_team = PSEP.join([team2_prefix, 'team'])

    #
    # Create the model from specifications
    #

    specs = {}
    specs['algorithms'] = args.algorithms
    specs['base_dir'] = base_dir
    specs['calibration'] = 'isotonic'
    specs['drop'] = ['Unnamed: 0', 'index', 'season', 'date', 'away.team', 'away.score', 'home.team', 'home.score', 'total_points', 'over', 'point_margin_game', 'cover_margin_game', 'lost_on_spread', 'under', 'overunder_margin', 'lost_on_points', 'won_on_points']
    specs['dummy_limit'] = 100
    specs['esr'] = 30
    specs['extension'] = 'csv'
    specs['features'] = WILDCARD
    specs['fsample_pct'] = 10
    specs['gp_learn'] = 0
    specs['grid_search'] = args.grid_search
    specs['gs_iters'] = 100
    specs['na_fill'] = 0
    specs['n_estimators'] = args.n_estimators
    specs['n_folds'] = args.n_folds
    specs['n_jobs'] = -1
    specs['n_step'] = args.n_step
    specs['ngrams_max'] = 2
    specs['plots'] = args.plots
    specs['poly_degree'] = 2
    specs['project'] = organization
    specs['regression'] = args.regression
    specs['rfe'] = args.rfe
    specs['scorer'] = 'roc_auc'
    specs['seed'] = 13201
    specs['separator'] = ','
    specs['shuffle'] = False
    specs['split'] = args.split
    specs['subsample'] = False
    specs['subsample_pct'] = 0.2
    specs['test_file'] = 'test'
    specs['test_labels'] = True
    specs['train_file'] = 'train'
    specs['target'] = args.target
    specs['verbosity'] = args.verbosity

    #
    # Read in the game frame. This is the feature generation phase.
    #

    logger.info("Reading Game Data")

    directory = SSEP.join([base_dir, organization])
    file_base = USEP.join([organization, space.subject, space.schema, space.fractal])
    df = read_frame(directory, file_base, specs['extension'], specs['separator'])
    logger.info("Total Game Records: %d", df.shape[0])

    #
    # Locate any rows with null values
    #

    null_rows = df.isnull().any(axis=1)
    null_indices = [i for i, val in enumerate(null_rows.tolist()) if val == True]
    for i in null_indices:
        logger.info("Null Record: %d on Date: %s", i, df.date[i])

    #
    # Set the training date and prediction date
    #

    train_date = datetime.date(1900, 1, 1)
    train_date = train_date.strftime('%Y-%m-%d')
    predict_date = datetime.datetime.now()
    predict_date = '2016-01-01' # predict_date.strftime("%Y-%m-%d")


    #
    # Run the game pipeline on a seasonal loop
    #

    if season:
        # run model for a specific season
        seasons = [season]
    else:
        # run model on all seasons
        seasons = df['season'].unique().tolist()

    #
    # Initialize the final frame
    #

    ff = pd.DataFrame()

    #
    # Iterate through each season of the game frame
    #

    for season in seasons:

        # Generate a frame for each season

        gf = df[df['season'] == season]
        gf = gf.reset_index(level=0)

        # Generate derived variables for the game frame

        total_games = gf.shape[0]
        if random_scoring:
            gf['home.score'] = np.random.randint(points_min, points_max, total_games)
            gf['away.score'] = np.random.randint(points_min, points_max, total_games)
        gf['total_points'] = gf['home.score'] + gf['away.score']

        gf = add_features(gf, game_dict, gf.shape[0])
        for index, row in gf.iterrows():
            gf['point_margin_game'].at[index] = get_point_margin(row, 'home.score', 'away.score')
            gf['won_on_points'].at[index] = True if gf['point_margin_game'].at[index] > 0 else False
            gf['lost_on_points'].at[index] = True if gf['point_margin_game'].at[index] < 0 else False
            gf['cover_margin_game'].at[index] = gf['point_margin_game'].at[index] + row['line']
            gf['won_on_spread'].at[index] = True if gf['cover_margin_game'].at[index] > 0 else False
            gf['lost_on_spread'].at[index] = True if gf['cover_margin_game'].at[index] <= 0 else False
            gf['overunder_margin'].at[index] = gf['total_points'].at[index] - row['over_under']
            gf['over'].at[index] = True if gf['overunder_margin'].at[index] > 0 else False
            gf['under'].at[index] = True if gf['overunder_margin'].at[index] < 0 else False

        # Generate each team frame

        teams = gf.groupby([home_team])
        team_frames = []
        for team, data in teams:
            team_frame = USEP.join([organization, team.lower(), series, str(season)])
            print "Generating team frame: %s" % team_frame
            command = "tf = gf[(gf[home_team] == '%s') | (gf[away_team] == '%s')]" % (team, team)
            exec(command)
            tf = tf.reset_index(level=0)
            tf = generate_team_frame(team, tf, sports_dict)
            command = "%s = tf" % team_frame
            exec(command)
            team_frames.append(team_frame)

        # Create the model frame, initializing the home and away frames

        mdict = {k:v for (k,v) in sports_dict.iteritems() if v != bool}
        team1_frame = pd.DataFrame()
        team1_frame = add_features(team1_frame, mdict, gf.shape[0], prefix=team1_prefix)
        team2_frame = pd.DataFrame()
        team2_frame = add_features(team2_frame, mdict, gf.shape[0], prefix=team2_prefix)
        frames = [gf, team1_frame, team2_frame]
        mf = pd.concat(frames, axis=1)

        # Loop through each team frame, inserting data into the model frame row
        #     get index+1 [if valid]
        #     determine if team is home or away to get prefix
        #     try: np.where((gf[home_team] == 'PHI') & (gf['date'] == '09/07/14'))[0][0]
        #     Assign team frame fields to respective model frame fields: set gf.at(pos, field)

        for team, data in teams:
            team_frame = USEP.join([organization, team.lower(), series, str(season)])
            print "Merging team frame %s into model frame" % team_frame
            command = "tf = %s" % team_frame
            exec(command)
            for index in range(0, tf.shape[0]-1):
                gindex = index + 1
                model_row = tf.iloc[gindex]
                key_date = model_row['date']
                at_home = False
                if team == model_row[home_team]:
                    at_home = True
                    key_team = model_row[home_team]
                elif team == model_row[away_team]:
                    key_team = model_row[away_team]
                else:
                    raise KeyError("Team not found in Team Frame")            
                try:
                    if at_home:
                        mpos = np.where((mf[home_team] == key_team) & (mf['date'] == key_date))[0][0]
                    else:
                        mpos = np.where((mf[away_team] == key_team) & (mf['date'] == key_date))[0][0]
                except:
                    raise IndexError("Team/Date Key not found in Model Frame")
                # print team, gindex, mpos
                # insert team data into model row
                mf = insert_model_data(mf, mpos, mdict, tf, index, team1_prefix if at_home else team2_prefix)

        # Compute delta data 'home' - 'away'

        mf = generate_delta_data(mf, mdict, team1_prefix, team2_prefix)

        # Append this to final frame

        frames = [ff, mf]
        ff = pd.concat(frames)

    #
    # Split data into training and test data
    #

    new_train_frame = ff.loc[(ff.date >= train_date) & (ff.date < predict_date)]
    if len(new_train_frame) <= 1:
        raise ValueError("Training frame has length 1 or less")

    new_test_frame = ff.loc[ff.date >= predict_date]
    if len(new_test_frame) <= 1:
        raise ValueError("Test frame has length 1 or less")

    #
    # Rewrite with all the features to the train and test files
    #

    write_frame(new_train_frame, directory, specs['train_file'],
                specs['extension'], specs['separator'])
    write_frame(new_test_frame, directory, specs['test_file'],
                specs['extension'], specs['separator'])

    #
    # Create the model from specs, and run the pipeline
    #

    logger.info("Running Model")

    model = Model(specs)
    model = pipeline(model)

    #
    # Log the predictions
    #

    for algo in model.algolist:
        print "Algorithm: %s" % algo
        print "Predictions:"
        print model.preds[(algo, 'test')]
        print "Probabilities:"
        print model.probas[(algo, 'test')]

    #
    # End of Program
    #

    logger.info("Completed Game Pipeline")
