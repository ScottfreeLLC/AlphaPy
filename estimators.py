##############################################################
#
# Package  : AlphaPy
# Module   : estimators
# Version  : 1.0
# Copyright: Mark Conway
# Date     : June 29, 2013
#
##############################################################


#
# Imports
#

from estimator import Estimator
import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLasso
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import xgboost as xgb


#
# Classes
#

class AdaBoostClassifierCoef(AdaBoostClassifier):
    def fit(self, *args, **kwargs):
        super(AdaBoostClassifierCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

class ExtraTreesClassifierCoef(ExtraTreesClassifier):
    def fit(self, *args, **kwargs):
        super(ExtraTreesClassifierCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

class RandomForestClassifierCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

class GradientBoostingClassifierCoef(GradientBoostingClassifier):
    def fit(self, *args, **kwargs):
        super(GradientBoostingClassifierCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

#
# Enumeration Type
#

class Enum(set):
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError


#
# Model Types
#

model_types = Enum(["classification", "clustering", "multiclass", "regression"])


#
# Define scorers
#

scorers = {'accuracy' : model_types.classification,
           'average_precision' : model_types.classification,
           'f1' : model_types.classification,
           'f1_macro' : model_types.classification,
           'f1_micro' : model_types.classification,
           'f1_samples' : model_types.classification,
           'f1_weighted' : model_types.classification,
           'log_loss' : model_types.classification,
           'precision' : model_types.classification,
           'recall' : model_types.classification,
           'roc_auc' : model_types.classification,
           'adjusted_rand_score' : model_types.clustering,
           'mean_absolute_error' : model_types.regression,
           'mean_squared_error' : model_types.regression,
           'median_absolute_error' : model_types.regression,
           'r2' : model_types.regression}


#
# Function get_scorers
#

def get_scorers():
    return scorers


#
# Function get_estimators
#

# AdaBoost (feature_importances_)
# Gradient Boosting (feature_importances_)
# K-Nearest Neighbors (NA)
# Linear Regression (coef_)
# Linear Support Vector Machine (coef_)
# Logistic Regression (coef_)
# Naive Bayes (coef_)
# Radial Basis Function (NA)
# Random Forest (feature_importances_)
# Support Vector Machine (NA)
# XGBoost Binary (NA)
# XGBoost Multi (NA)
# Extra Trees (feature_importances_)
# Random Forest (feature_importances_)
# Randomized Lasso

def get_estimators(n_estimators, seed, n_jobs, verbosity):
    # initialize estimator dictionary
    estimators = {}
    # AdaBoost
    algo = 'AB'
    model_type = model_types.classification
    params = {"n_estimators" : n_estimators,
              "random_state" : seed}
    est = AdaBoostClassifierCoef(**params)
    grid = {"n_estimators" : [10, 50, 100, 150, 200],
            "learning_rate" : [0.2, 0.5, 0.7, 1.0, 1.5, 2.0],
            "algorithm" : ['SAMME', 'SAMME.R']}
    scoring = True
    estimators[algo] = Estimator(algo, model_type, est, grid, scoring)
    # Gradient Boosting
    algo = 'GB'
    model_type = model_types.classification
    params = {"n_estimators" : n_estimators,
              "max_depth" : 3,
              "random_state" : seed,
              "verbose" : verbosity}
    est = GradientBoostingClassifierCoef(**params)
    grid = {"loss" : ['deviance', 'exponential'],
            "learning_rate" : [0.05, 0.1, 0.15],
            "n_estimators" : [50, 100, 200],
            "max_depth" : [3, None],
            "min_samples_split" : [1, 2, 3],
            "min_samples_leaf" : [1, 2]
            }
    scoring = True
    estimators[algo] = Estimator(algo, model_type, est, grid, scoring)
    # Gradient Boosting Regression
    algo = 'GBR'
    model_type = model_types.regression
    params = {"n_estimators" : n_estimators,
              "random_state" : seed,
              "verbose" : verbosity}
    est = GradientBoostingRegressor()
    estimators[algo] = Estimator(algo, model_type, est, grid)
    # K-Nearest Neighbors
    algo = 'KNN'
    model_type = model_types.classification
    params = {"n_jobs" : n_jobs}
    est = KNeighborsClassifier(**params)
    grid = {"n_neighbors" : [3, 5, 7, 10],
            "weights" : ['uniform', 'distance'],
            "algorithm" : ['ball_tree', 'kd_tree', 'brute', 'auto'],
            "leaf_size" : [10, 20, 30, 40, 50]}
    scoring = False
    estimators[algo] = Estimator(algo, model_type, est, grid, scoring)
    # K-Nearest Neighbor Regression
    algo = 'KNR'
    model_type = model_types.regression
    params = {"n_jobs" : n_jobs}
    est = KNeighborsRegressor(**params)
    estimators[algo] = Estimator(algo, model_type, est, grid)
    # Linear Support Vector Classification
    algo = 'LSVC'
    model_type = model_types.classification
    params = {"C" : 0.01,
              "max_iter" : 2000,
              "penalty" : 'l1',
              "dual" : False,
              "random_state" : seed,
              "verbose" : verbosity}
    est = LinearSVC(**params)
    grid = {"C" : np.logspace(-2, 10, 13),
            "penalty" : ['l1', 'l2'],
            "dual" : [True, False],
            "tol" : [0.0005, 0.001, 0.005],
            "max_iter" : [500, 1000, 2000]}
    scoring = False
    estimators[algo] = Estimator(algo, model_type, est, grid, scoring)
    # Linear Support Vector Machine
    algo = 'LSVM'
    model_type = model_types.classification
    params = {"kernel" : 'linear',
              "probability" : True,
              "random_state" : seed,
              "verbose" : verbosity}
    est = SVC(**params)
    grid = {"C" : np.logspace(-2, 10, 13),
            "gamma" : np.logspace(-9, 3, 13),
            "shrinking" : [True, False],
            "tol" : [0.0005, 0.001, 0.005],
            "decision_function_shape" : ['ovo', 'ovr', None]}
    scoring = False
    estimators[algo] = Estimator(algo, model_type, est, grid, scoring)
    # Logistic Regression
    algo = 'LOGR'
    model_type = model_types.classification
    params = {"random_state" : seed,
              "n_jobs" : n_jobs,
              "verbose" : verbosity}
    est = LogisticRegression(**params)
    grid = {"penalty" : ['l2'],
            "C" : [0.1, 1, 10, 100, 1000, 1e4, 1e5],
            "fit_intercept" : [True, False],
            "solver" : ['newton-cg', 'lbfgs', 'liblinear', 'sag']}
    scoring = True
    estimators[algo] = Estimator(algo, model_type, est, grid, scoring)
    # Linear Regression
    algo = 'LR'
    model_type = model_types.regression
    params = {"n_jobs" : n_jobs}
    est = LinearRegression()
    grid = {"fit_intercept" : [True, False],
            "normalize" : [True, False],
            "copy_X" : [True, False]}
    estimators[algo] = Estimator(algo, model_type, est, grid)
    # Naive Bayes
    algo = 'NB'
    model_type = model_types.classification
    est = MultinomialNB()
    grid = {"alpha" : [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "fit_prior" : [True, False]}
    scoring = True
    estimators[algo] = Estimator(algo, model_type, est, grid, scoring)
    # Radial Basis Function
    algo = 'RBF'
    model_type = model_types.classification
    params = {"kernel" : 'rbf',
              "probability" : True,
              "random_state" : seed,
              "verbose" : verbosity}
    est = SVC(**params)
    grid = {"C" : np.logspace(-2, 10, 13),
            "gamma" : np.logspace(-9, 3, 13),
            "shrinking" : [True, False],
            "tol" : [0.0005, 0.001, 0.005],
            "decision_function_shape" : ['ovo', 'ovr', None]}
    scoring = False
    estimators[algo] = Estimator(algo, model_type, est, grid, scoring)
    # Random Forest
    algo = 'RF'
    model_type = model_types.classification
    params = {"n_estimators" : n_estimators,
              "max_depth" : 10,
              "min_samples_split" : 2,
              "min_samples_leaf" : 2,
              "bootstrap" : True,
              "criterion" : 'gini',
              "random_state" : seed,
              "n_jobs" : n_jobs,
              "verbose" : verbosity}
    est = RandomForestClassifierCoef(**params)
    grid = {"n_estimators" : [21, 51, 101, 201, 501],
            "max_depth" : [5, 10, None],
            "min_samples_split" : [1, 3, 5, 10],
            "min_samples_leaf" : [1, 2, 3],
            "bootstrap" : [True, False],
            "criterion" : ['gini', 'entropy']}
    scoring = True
    estimators[algo] = Estimator(algo, model_type, est, grid, scoring)
    # Random Forest Regression
    algo = 'RFR'
    model_type = model_types.regression
    params = {"n_estimators" : n_estimators,
              "random_state" : seed,
              "n_jobs" : n_jobs,
              "verbose" : verbosity}
    est = RandomForestRegressor(**params)
    estimators[algo] = Estimator(algo, model_type, est, grid)
    # Support Vector Machine
    algo = 'SVM'
    model_type = model_types.classification
    params = {"probability" : True,
              "random_state" : seed,
              "verbose" : verbosity}
    est = SVC(**params)
    grid = {"C" : np.logspace(-2, 10, 13),
            "gamma" : np.logspace(-9, 3, 13),
            "shrinking" : [True, False],
            "tol" : [0.0005, 0.001, 0.005],
            "decision_function_shape" : ['ovo', 'ovr', None]}
    scoring = False
    estimators[algo] = Estimator(algo, model_type, est, grid, scoring)
    # XGBoost Binary
    algo = 'XGB'
    model_type = model_types.classification
    params = {"objective" : 'binary:logistic',
              "n_estimators" : n_estimators,
              "max_depth" : 10,
              "learning_rate" : 0.1,
              "min_child_weight" : 1.05,
              "subsample" : 0.95,
              "colsample_bytree" : 0.8,
              "nthread" : n_jobs,
              "silent" : False}
    est = xgb.XGBClassifier(**params)
    grid = {"n_estimators" : [21, 51, 101, 201, 501],
            "max_depth" : [None, 6, 8, 10, 20],
            "learning_rate" : [0.02, 0.05, 0.1],
            "min_child_weight" : [1.0, 1.1],
            "subsample" : [0.8, 0.9, 1.0],
            "colsample_bytree" : [0.8, 0.9, 1.0]}
    scoring = False
    estimators[algo] = Estimator(algo, model_type, est, grid, scoring)
    # XGBoost Multiclass
    algo = 'XGBM'
    model_type = model_types.multiclass
    params = {"objective" : 'multi:softmax',
              "n_estimators" : n_estimators,
              "max_depth" : 10,
              "learning_rate" : 0.1,
              "min_child_weight" : 1.05,
              "subsample" : 0.85,
              "colsample_bytree" : 0.8,
              "nthread" : n_jobs,
              "silent" : False}
    est = xgb.XGBClassifier(**params)
    estimators[algo] = Estimator(algo, model_type, est, grid)
    # XGBoost Regression
    algo = 'XGBR'
    model_type = model_types.regression
    params = {"objective" : 'reg:linear',
              "n_estimators" : n_estimators,
              "max_depth" : 10,
              "learning_rate" : 0.1,
              "min_child_weight" : 1.05,
              "subsample" : 0.85,
              "colsample_bytree" : 0.8,
              "seed" : seed,
              "nthread" : n_jobs,
              "silent" : False}
    est = xgb.XGBRegressor(**params)
    estimators[algo] = Estimator(algo, model_type, est, grid)
    # Extra Trees
    algo = 'XT'
    model_type = model_types.classification
    params = {"n_estimators" : n_estimators,
              "random_state" : seed,
              "n_jobs" : n_jobs,
              "verbose" : verbosity}
    est = ExtraTreesClassifierCoef(**params)
    grid = {"n_estimators" : [10, 50, 100, 150, 200],
            "max_features" : ['auto', 'sqrt', 'log2', None],
            "max_depth" : [3, 5, None],
            "min_samples_split" : [1, 2, 3],
            "min_samples_leaf" : [1, 2],
            "bootstrap" : [True, False],
            "warm_start" : [True, False]}
    scoring = True
    estimators[algo] = Estimator(algo, model_type, est, grid, scoring)
    # Extra Trees Regression
    algo = 'XTR'
    model_type = model_types.regression
    params = {"n_estimators" : n_estimators,
              "random_state" : seed,
              "n_jobs" : n_jobs,
              "verbose" : verbosity}
    est = ExtraTreesRegressor(**params)
    estimators[algo] = Estimator(algo, model_type, est, grid)
    # return the entire classifier list
    return estimators
