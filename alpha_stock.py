##############################################################
#
# Package   : AlphaPy
# Module    : alpha_stock
# Version   : 1.0
# Copyright : Mark Conway
# Date      : September 13, 2015
#
##############################################################


#
# Imports
#

from alias import Alias
from alpha import pipeline
from analysis import Analysis
from analysis import run_analysis
from datetime import datetime
from datetime import timedelta
from data import get_remote_data
from frame import Frame
from globs import WILDCARD
from group import Group
import logging
from model import Model
from space import Space
from system import System
from var import Variable
from var import vmapply


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Define aliases
#

Alias('atr', 'ma_truerange')
Alias('cma', 'ma_close')
Alias('cmax', 'highest_close')
Alias('cmin', 'lowest_close')
Alias('hc', 'higher_close')
Alias('hh', 'higher_high')
Alias('hl', 'higher_low')
Alias('ho', 'higher_open')
Alias('hmax', 'highest_high')
Alias('hmin', 'lowest_high')
Alias('lc', 'lower_close')
Alias('lh', 'lower_high')
Alias('ll', 'lower_low')
Alias('lo', 'lower_open')
Alias('lmax', 'highest_low')
Alias('lmin', 'lowest_low')
Alias('net', 'net_close')
Alias('netdown', 'down_net')
Alias('netup', 'up_net')
Alias('omax', 'highest_open')
Alias('omin', 'lowest_open')
Alias('rmax', 'highest_hlrange')
Alias('rmin', 'lowest_hlrange')
Alias('rr', 'maratio_hlrange')
Alias('rixc', 'rindex_close_high_low')
Alias('rixo', 'rindex_open_high_low')
Alias('roi', 'netreturn_close')
Alias('rsi', 'rsi_close')
Alias('sepma', 'ma_sep')
Alias('vma', 'ma_volume')
Alias('vmratio', 'maratio_volume')
Alias('upmove', 'net_high')


#
# Define variables
#
#
# Events
# ------
#
# Numeric substitution is allowed for any number in the expression.
# Offsets are allowed in event expressions but cannot be substituted.
#
# Examples
# --------
#
# Event('rrunder', 'rr_3_20 <= 0.9')
#
# 'rrunder_2_10_0.7'
# 'rrunder_2_10_0.9'
# 'xmaup_20_50_20_200'
# 'xmaup_10_50_20_50'
#
# Event('xmaup', 'cma_20 > cma_50', 'runs', 50) generates:
#     xmaup_20_50
#     xmaup_20_50_runs_50
#
# Event('xmaup', 'cma_20 > cma_50', ['streak', 'zscore'], 50) generates:
#     xmaup_20_50
#     xmaup_20_50_streak_50
#     xmaup_20_50_zscore_50
#
# Event('xmaup', 'cma_20 > cma_50', 'rtotal', 50) generates:
#     xmaup_20_50
#     xmaup_20_50_rtotal_50
#
# Event('xmaup', 'cma_20 > cma_50', 'all', 50) generates:
#     xmaup_20_50
#     xmaup_20_50_runs_50
#     xmaup_20_50_rtotal_50
#     xmaup_20_50_streak_50
#     xmaup_20_50_zscore_50
#

Variable('abovema', 'close > cma_50')
Variable('belowma', 'close < cma_50')
Variable('bigup', 'rrover & sepwide & netup')
Variable('bigdown', 'rrover & sepwide & netdown')
Variable('madelta', 'close - cma_50')
Variable('nr', 'hlrange == rmin_7')    # nr_7
Variable('roihigh', 'roi >= 10')
Variable('roilow', 'roi < 10')
Variable('rrunder', 'rr_1_10 < 1')
Variable('rrover', 'rr_1_10 >= 1')
Variable('sep', 'rixc - rixo')
Variable('sephigh', 'sep >= 70')
Variable('seplow', 'sep <= -70')
Variable('sepwide', 'sephigh | seplow')
Variable('sepnothigh', 'sep <= 30')
Variable('sepnotlow', 'sep >= -30')
Variable('sepnarrow', 'sepnothigh & sepnotlow')
Variable('trend', 'rrover & sepwide')
Variable('vmover', 'vmratio >= 1')
Variable('vmunder', 'vmratio < 1')
Variable('volatility', 'close / atr_10')
Variable('wr', 'hlrange == rmax_4')    # wr_4
Variable('xmadown', 'cma_20 < cma_50 & cma_20[1] > cma_50[1]')
Variable('xmaup', 'cma_20 > cma_50 & cma_20[1] < cma_50[1]')


#
# Create default space stock_prices_1d
#

space = Space()


#
# Create groups for each genre of stock
#

gs = Group('my', space)
ge = Group('psp', space)
gt = Group('tech', space)


#
# Populate groups with members and other groups
#

gs.add([repr(ge), repr(gt)])
ge.add(['qqq', 'spy', 'tna', 'tza', 'nugt', 'dust', 'fas', 'faz', 'eem', 'iwm', \
        'tvix', 'vxx', 'tlt', 'tbt', 'edc', 'edz'])
gt.add(['aapl', 'amzn', 'fb', 'goog', 'lnkd', 'nflx', 'yhoo'])


#
# Display all members
#

print gs.all_members()


#
# Get stock data
#

get_remote_data(gs, datetime.now() - timedelta(1000))


#
# Define feature sets
#

features_gap = ['gap', 'gapbadown', 'gapbaup', 'gapdown', 'gapup']
features_ma = ['cma_10', 'cma_20', 'cma_50']
features_range = ['net', 'netup', 'netdown', 'rr', 'rr_2', 'rr_3', 'rr_4', 'rr_5', \
                  'rr_6', 'rr_7', 'rrunder', 'rrover']
features_roi = ['roi', 'roi_2', 'roi_3', 'roi_4', 'roi_5', 'roi_10', 'roi_20']
features_sep = ['sepa', 'sepa_2', 'sepa_3', 'sepa_4', 'sepa_5', 'sepa_6', 'sepa_7', \
                'sephigh', 'seplow', 'sepover', 'sepunder']
features_simple = ['hc', 'hh', 'ho', 'hl', 'lc', 'lh', 'll', 'lo']
features_trend = ['adx', 'diplus', 'diminus', 'trend', 'rsi_8', 'rsi_14', 'bigdown', 'bigup']
features_volatility = ['atr', 'volatility', 'nr_4', 'nr_7', 'nr_10', 'wr_4', 'wr_7', 'wr_10']
features_volume = ['vmover', 'vmunder', 'vma', 'vmratio']
features_all = features_gap + features_ma + features_range + features_roi + \
               features_sep + features_simple + features_trend + features_volatility + \
               features_volume


#
# Apply the features to all of the frames
#

vmapply(gs, features_simple)
vmapply(gs, features_gap)
vmapply(gs, ['sephigh'])


#
# Create the model from specifications
#

specs = {}

specs['algorithms'] = 'RF,XGB'
specs['base_dir'] = '/Users/markconway/Projects/AlphaPy_Projects'
specs['blend'] = False
specs['categoricals'] = None
specs['drop'] = ['Unnamed: 0', 'date', 'open', 'high', 'low', 'close', 'adjclose']
specs['dummy_limit'] = 100
specs['extension'] = 'csv'
specs['features'] = WILDCARD
specs['interactions'] = True
specs['n_folds'] = 3
specs['n_iters'] = 100
specs['n_jobs'] = -1
specs['n_step'] = 1
specs['ngrams_max'] = 2
specs['plots'] = True
specs['poly_degree'] = 2
specs['project'] = 'Trend Days'
specs['regression'] = False
specs['scorer'] = 'roc_auc'
specs['seed'] = 10231
specs['separator'] = ','
specs['shuffle'] = False
specs['split'] = 0.3
specs['subsample'] = False
specs['subsample_pct'] = 0.2
specs['test_file'] = 'test'
specs['test_labels'] = True
specs['text_features'] = None
specs['train_file'] = 'train'
specs['target'] = 'sephigh'
specs['verbosity'] = 1

m = Model(specs)


#
# Run the analysis, including the model pipeline
#

a = Analysis(m, gs)

forecast_period = 1
leaders = ['open', 'gap', 'gapbadown', 'gapbaup', 'gapdown', 'gapup']
a = run_analysis(a, forecast_period, leaders)


#
# Create and run systems
#

pass


# ts = System('trend', 'bigup', 'bigdown')
# run_system(ts, gs)
# gen_portfolio(ts, gs)

# cs = System('closer', 'hc', 'lc')
# run_system(cs, gs)
# gen_portfolio(cs, gs)
