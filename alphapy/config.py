################################################################################
#
# Package   : AlphaPy
# Module    : config
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

from alphapy.alias import Alias
from alphapy.data import SamplingMethod
from alphapy.estimators import ModelType
from alphapy.features import Encoders
from alphapy.features import feature_scorers
from alphapy.features import Scalers
from alphapy.globs import SSEP
from alphapy.group import Group
from alphapy.space import Space
from alphapy.var import Variable

from datetime import datetime
import logging
import os
import sys
import yaml


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_model_config
#

def get_model_config(cfg_dir):

    logger.info("Model Configuration")

    # Read the configuration file

    full_path = SSEP.join([cfg_dir, 'model.yml'])
    with open(full_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Store configuration parameters in dictionary

    specs = {}

    # Section: project [this section must be first]

    specs['directory'] = cfg['project']['directory']
    specs['extension'] = cfg['project']['file_extension']
    specs['sample_submission'] = cfg['project']['sample_submission']
    specs['scoring_mode'] = cfg['project']['scoring_mode']
    specs['submission_file'] = cfg['project']['submission_file']

    # Section: data

    specs['drop'] = cfg['data']['drop']
    specs['dummy_limit'] = cfg['data']['dummy_limit']
    specs['features'] = cfg['data']['features']
    specs['sentinel'] = cfg['data']['sentinel']
    specs['separator'] = cfg['data']['separator']
    specs['shuffle'] = cfg['data']['shuffle']
    specs['split'] = cfg['data']['split']
    specs['target'] = cfg['data']['target']
    specs['target_value'] = cfg['data']['target_value']
    specs['test_file'] = cfg['data']['test']
    specs['test_labels'] = cfg['data']['test_labels']
    specs['train_file'] = cfg['data']['train']
    # sampling
    specs['sampling'] = cfg['data']['sampling']['option']
    # determine whether or not sampling method is valid
    samplers = {x.name: x.value for x in SamplingMethod}
    sampling_method = cfg['data']['sampling']['method']
    if sampling_method in samplers:
        specs['sampling_method'] = SamplingMethod(samplers[sampling_method])
    else:
        raise ValueError(".yml data:sampling:method %s unrecognized",
                         sampling_method)
    # end of sampling method
    specs['sampling_ratio'] = cfg['data']['sampling']['ratio']

    # Section: features

    # clustering
    specs['clustering'] = cfg['features']['clustering']['option']
    specs['cluster_min'] = cfg['features']['clustering']['minimum']
    specs['cluster_max'] = cfg['features']['clustering']['maximum']
    specs['cluster_inc'] = cfg['features']['clustering']['increment']
    # counts
    specs['counts'] = cfg['features']['counts']['option']
    # encoding
    specs['rounding'] = cfg['features']['encoding']['rounding']
    # determine whether or not encoder is valid
    encoders = {x.name: x.value for x in Encoders}
    encoder = cfg['features']['encoding']['type']
    if encoder in encoders:
        specs['encoder'] = Encoders(encoders[encoder])
    else:
        raise ValueError(".yml features:encoding:type %s unrecognized", encoder)
    # genetic
    specs['genetic'] = cfg['features']['genetic']['option']
    specs['gfeatures'] = cfg['features']['genetic']['features']
    # interactions
    specs['interactions'] = cfg['features']['interactions']['option']
    specs['isample_pct'] = cfg['features']['interactions']['sampling_pct']
    specs['poly_degree'] = cfg['features']['interactions']['poly_degree']
    # isomap
    specs['isomap'] = cfg['features']['isomap']['option']
    specs['iso_components'] = cfg['features']['isomap']['components']
    specs['iso_neighbors'] = cfg['features']['isomap']['neighbors']
    # log transformation
    specs['logtransform'] = cfg['features']['logtransform']['option']
    # NumPy
    specs['numpy'] = cfg['features']['numpy']['option']
    # pca
    specs['pca'] = cfg['features']['pca']['option']
    specs['pca_min'] = cfg['features']['pca']['minimum']
    specs['pca_max'] = cfg['features']['pca']['maximum']
    specs['pca_inc'] = cfg['features']['pca']['increment']
    specs['pca_whiten'] = cfg['features']['pca']['whiten']
    # Scaling
    specs['scaler_option'] = cfg['features']['scaling']['option']
    # determine whether or not scaling type is valid
    scaler_types = {x.name: x.value for x in Scalers}
    scaler_type = cfg['features']['scaling']['type']
    if scaler_type in scaler_types:
        specs['scaler_type'] = Scalers(scaler_types[scaler_type])
    else:
        raise ValueError(".yml features:scaling:type %s unrecognized", scaler_type)
    # SciPy
    specs['scipy'] = cfg['features']['scipy']['option']
    # text
    specs['ngrams_max'] = cfg['features']['text']['ngrams']
    specs['vectorize'] = cfg['features']['text']['vectorize']
    # t-sne
    specs['tsne'] = cfg['features']['tsne']['option']
    specs['tsne_components'] = cfg['features']['tsne']['components']
    specs['tsne_learn_rate'] = cfg['features']['tsne']['learning_rate']
    specs['tsne_perplexity'] = cfg['features']['tsne']['perplexity']

    # Section: model

    specs['algorithms'] = cfg['model']['algorithms']
    specs['balance_classes'] = cfg['model']['balance_classes']
    specs['cv_folds'] = cfg['model']['cv_folds']
    # determine whether or not model type is valid
    model_types = {x.name: x.value for x in ModelType}
    model_type = cfg['model']['type']
    if model_type in model_types:
        specs['model_type'] = ModelType(model_types[model_type])
    else:
        raise ValueError(".yml model:type %s unrecognized", model_type)
    # end of model type
    specs['n_estimators'] = cfg['model']['estimators']
    specs['pvalue_level'] = cfg['model']['pvalue_level']
    specs['scorer'] = cfg['model']['scoring_function']
    # calibration
    specs['calibration'] = cfg['model']['calibration']['option']
    specs['cal_type'] = cfg['model']['calibration']['type']
    # feature selection
    specs['feature_selection'] = cfg['model']['feature_selection']['option']
    specs['fs_percentage'] = cfg['model']['feature_selection']['percentage']
    specs['fs_uni_grid'] = cfg['model']['feature_selection']['uni_grid']
    score_func = cfg['model']['feature_selection']['score_func']
    if score_func in feature_scorers:
        specs['fs_score_func'] = feature_scorers[score_func]
    else:
        raise ValueError('.yml model:feature_selection:score_func %s unrecognized',
                         score_func)
    # grid search
    specs['grid_search'] = cfg['model']['grid_search']['option']
    specs['gs_iters'] = cfg['model']['grid_search']['iterations']
    specs['gs_random'] = cfg['model']['grid_search']['random']
    specs['gs_sample'] = cfg['model']['grid_search']['subsample']
    specs['gs_sample_pct'] = cfg['model']['grid_search']['sampling_pct']
    # rfe
    specs['rfe'] = cfg['model']['rfe']['option']
    specs['rfe_step'] = cfg['model']['rfe']['step']

    # Section: pipeline

    specs['n_jobs'] = cfg['pipeline']['number_jobs']
    specs['seed'] = cfg['pipeline']['seed']
    specs['verbosity'] = cfg['pipeline']['verbosity']

    # Section: plots

    specs['calibration_plot'] = cfg['plots']['calibration']
    specs['confusion_matrix'] = cfg['plots']['confusion_matrix']
    specs['importances'] = cfg['plots']['importances']
    specs['learning_curve'] = cfg['plots']['learning_curve']
    specs['roc_curve'] = cfg['plots']['roc_curve']

    # Section: treatments

    try:
        specs['treatments'] = cfg['treatments']
    except:
        specs['treatments'] = None
        logger.info("No Treatments Found")

    # Section: xgboost

    specs['esr'] = cfg['xgboost']['stopping_rounds']

    # Log the configuration parameters

    logger.info('MODEL PARAMETERS:')
    logger.info('algorithms        = %s', specs['algorithms'])
    logger.info('balance_classes   = %s', specs['balance_classes'])
    logger.info('calibration       = %r', specs['calibration'])
    logger.info('cal_type          = %s', specs['cal_type'])
    logger.info('calibration_plot  = %r', specs['calibration'])
    logger.info('clustering        = %r', specs['clustering'])
    logger.info('cluster_inc       = %d', specs['cluster_inc'])
    logger.info('cluster_max       = %d', specs['cluster_max'])
    logger.info('cluster_min       = %d', specs['cluster_min'])
    logger.info('confusion_matrix  = %r', specs['confusion_matrix'])
    logger.info('counts            = %r', specs['counts'])
    logger.info('cv_folds          = %d', specs['cv_folds'])
    logger.info('directory         = %s', specs['directory'])
    logger.info('extension         = %s', specs['extension'])
    logger.info('drop              = %s', specs['drop'])
    logger.info('dummy_limit       = %d', specs['dummy_limit'])
    logger.info('encoder           = %r', specs['encoder'])
    logger.info('esr               = %d', specs['esr'])
    logger.info('features [X]      = %s', specs['features'])
    logger.info('feature_selection = %r', specs['feature_selection'])
    logger.info('fs_percentage     = %d', specs['fs_percentage'])
    logger.info('fs_score_func     = %s', specs['fs_score_func'])
    logger.info('fs_uni_grid       = %s', specs['fs_uni_grid'])
    logger.info('genetic           = %r', specs['genetic'])
    logger.info('gfeatures         = %d', specs['gfeatures'])
    logger.info('grid_search       = %r', specs['grid_search'])
    logger.info('gs_iters          = %d', specs['gs_iters'])
    logger.info('gs_random         = %r', specs['gs_random'])
    logger.info('gs_sample         = %r', specs['gs_sample'])
    logger.info('gs_sample_pct     = %f', specs['gs_sample_pct'])
    logger.info('importances       = %r', specs['importances'])
    logger.info('interactions      = %r', specs['interactions'])
    logger.info('isomap            = %r', specs['isomap'])
    logger.info('iso_components    = %d', specs['iso_components'])
    logger.info('iso_neighbors     = %d', specs['iso_neighbors'])
    logger.info('isample_pct       = %d', specs['isample_pct'])
    logger.info('learning_curve    = %r', specs['learning_curve'])
    logger.info('logtransform      = %r', specs['logtransform'])
    logger.info('model_type        = %r', specs['model_type'])
    logger.info('n_estimators      = %d', specs['n_estimators'])
    logger.info('n_jobs            = %d', specs['n_jobs'])
    logger.info('ngrams_max        = %d', specs['ngrams_max'])
    logger.info('numpy             = %r', specs['numpy'])
    logger.info('pca               = %r', specs['pca'])
    logger.info('pca_inc           = %d', specs['pca_inc'])
    logger.info('pca_max           = %d', specs['pca_max'])
    logger.info('pca_min           = %d', specs['pca_min'])
    logger.info('pca_whiten        = %r', specs['pca_whiten'])
    logger.info('poly_degree       = %d', specs['poly_degree'])
    logger.info('pvalue_level      = %f', specs['pvalue_level'])
    logger.info('rfe               = %r', specs['rfe'])
    logger.info('rfe_step          = %d', specs['rfe_step'])
    logger.info('roc_curve         = %r', specs['roc_curve'])
    logger.info('rounding          = %d', specs['rounding'])
    logger.info('sample_submission = %r', specs['sample_submission'])
    logger.info('sampling          = %r', specs['sampling'])
    logger.info('sampling_method   = %r', specs['sampling_method'])
    logger.info('sampling_ratio    = %f', specs['sampling_ratio'])
    logger.info('scaler_option     = %r', specs['scaler_option'])
    logger.info('scaler_type       = %r', specs['scaler_type'])
    logger.info('scipy             = %r', specs['scipy'])
    logger.info('scorer            = %s', specs['scorer'])
    logger.info('scoring_mode      = %r', specs['scoring_mode'])
    logger.info('seed              = %d', specs['seed'])
    logger.info('sentinel          = %d', specs['sentinel'])
    logger.info('separator         = %s', specs['separator'])
    logger.info('shuffle           = %r', specs['shuffle'])
    logger.info('split             = %f', specs['split'])
    logger.info('submission_file   = %s', specs['submission_file'])
    logger.info('target [y]        = %s', specs['target'])
    logger.info('target_value      = %d', specs['target_value'])
    logger.info('test_file         = %s', specs['test_file'])
    logger.info('test_labels       = %r', specs['test_labels'])
    logger.info('train_file        = %s', specs['train_file'])
    logger.info('treatments        = %s', specs['treatments'])
    logger.info('tsne              = %r', specs['tsne'])
    logger.info('tsne_components   = %d', specs['tsne_components'])
    logger.info('tsne_learn_rate   = %f', specs['tsne_learn_rate'])
    logger.info('tsne_perplexity   = %f', specs['tsne_perplexity'])
    logger.info('vectorize         = %r', specs['vectorize'])
    logger.info('verbosity         = %d', specs['verbosity'])

    # Specifications to create the model

    return specs


#
# Function get_market_config
#

def get_market_config(cfg_dir):

    logger.info("Market Configuration")

    # Read the configuration file

    full_path = SSEP.join([cfg_dir, 'market.yml'])
    with open(full_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Store configuration parameters in dictionary

    specs = {}

    # Section: market [this section must be first]

    specs['forecast_period'] = cfg['market']['forecast_period']
    specs['fractal'] = cfg['market']['fractal']
    specs['leaders'] = cfg['market']['leaders']
    specs['lookback_period'] = cfg['market']['lookback_period']
    specs['predict_date'] = cfg['market']['predict_date']
    specs['schema'] = cfg['market']['schema']
    specs['target_group'] = cfg['market']['target_group']
    specs['train_date'] = cfg['market']['train_date']

    # Create the subject/schema/fractal namespace

    sspecs = ['stock', specs['schema'], specs['fractal']]    
    space = Space(*sspecs)

    # Section: features

    try:
        logger.info("Getting Features")
        specs['features'] = cfg['features']
    except:
        logger.info("No Features Found")

    # Section: groups

    try:
        logger.info("Defining Groups")
        for g, m in cfg['groups'].items():
            command = 'Group(\'' + g + '\', space)'
            exec(command)
            Group.groups[g].add(m)
    except:
        logger.info("No Groups Found")

    # Section: aliases

    try:
        logger.info("Defining Aliases")
        for k, v in cfg['aliases'].items():
            Alias(k, v)
    except:
        logger.info("No Aliases Found")

    # Section: variables

    try:
        logger.info("Defining Variables")
        for k, v in cfg['variables'].items():
            Variable(k, v)
    except:
        logger.info("No Variables Found")

    # Log the stock parameters

    logger.info('MARKET PARAMETERS:')
    logger.info('features        = %s', specs['features'])
    logger.info('forecast_period = %d', specs['forecast_period'])
    logger.info('fractal         = %s', specs['fractal'])
    logger.info('leaders         = %s', specs['leaders'])
    logger.info('lookback_period = %d', specs['lookback_period'])
    logger.info('predict_date    = %s', specs['predict_date'])
    logger.info('schema          = %s', specs['schema'])
    logger.info('target_group    = %s', specs['target_group'])
    logger.info('train_date      = %s', specs['train_date'])

    # Market Specifications
    return specs


#
# Function get_game_config
#

def get_game_config(cfg_dir):

    # Read the configuration file

    full_path = SSEP.join([cfg_dir, 'game.yml'])
    with open(full_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Store configuration parameters in dictionary

    specs = {}

    # Section: game

    specs['points_max'] = cfg['game']['points_max']
    specs['points_min'] = cfg['game']['points_min']
    specs['predict_date'] = cfg['game']['predict_date']
    specs['random_scoring'] = cfg['game']['random_scoring']
    specs['rolling_window'] = cfg['game']['rolling_window']   
    specs['seasons'] = cfg['game']['seasons']
    specs['train_date'] = cfg['game']['train_date']

    # Log the game parameters

    logger.info('GAME PARAMETERS:')
    logger.info('points_max       = %d', specs['points_max'])
    logger.info('points_min       = %d', specs['points_min'])
    logger.info('predict_date     = %s', specs['predict_date'])
    logger.info('random_scoring   = %r', specs['random_scoring'])
    logger.info('rolling_window   = %d', specs['rolling_window'])
    logger.info('seasons          = %s', specs['seasons'])
    logger.info('train_date       = %s', specs['train_date'])

    # Game Specifications
    return specs
