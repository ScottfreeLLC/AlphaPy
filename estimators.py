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
# Function get_classifiers
#

# AdaBoost (feature_importances_)
# Gradient Boosting (feature_importances_)
# K-Nearest Neighbors (NA)
# Linear Support Vector Machine (coef_)
# Logistic Regression (coef_)
# Naive Bayes (coef_)
# Radial Basis Function (NA)
# Random Forest (feature_importances_)
# Support Vector Machine (NA)
# XGBoost Binary (NA)
# XGBoost Multi (NA)
# Extra Trees (feature_importances_)


def get_classifiers(n_estimators, seed, n_jobs, verbosity):
    # initialize classifier dictionary
    classifiers = {}
    # AdaBoost
    algo = 'AB'
    params = {"n_estimators" : n_estimators,
              "random_state" : seed}
    est = AdaBoostClassifierCoef(**params)
    grid = {"n_estimators" : [10, 50, 100, 150, 200],
            "learning_rate" : [0.2, 0.5, 0.7, 1.0, 1.5, 2.0],
            "algorithm" : ['SAMME', 'SAMME.R']}
    scoring = True
    classifiers[algo] = Estimator(algo, est, grid, scoring)
    # Gradient Boosting
    algo = 'GB'
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
    classifiers[algo] = Estimator(algo, est, grid, scoring)
    # K-Nearest Neighbors
    algo = 'KNN'
    params = {"n_jobs" : n_jobs}
    est = KNeighborsClassifier(**params)
    grid = {"n_neighbors" : [3, 5, 7, 10],
            "weights" : ['uniform', 'distance'],
            "algorithm" : ['ball_tree', 'kd_tree', 'brute', 'auto'],
            "leaf_size" : [10, 20, 30, 40, 50]}
    scoring = False
    classifiers[algo] = Estimator(algo, est, grid, scoring)
    # Linear Support Vector Classification
    algo = 'LSVC'
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
    classifiers[algo] = Estimator(algo, est, grid, scoring)
    # Linear Support Vector Machine
    algo = 'LSVM'
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
    classifiers[algo] = Estimator(algo, est, grid, scoring)
    # Logistic Regression
    algo = 'LOGR'
    params = {"random_state" : seed,
              "n_jobs" : n_jobs,
              "verbose" : verbosity}
    est = LogisticRegression(**params)
    grid = {"penalty" : ['l2'],
            "C" : [0.1, 1, 10, 100, 1000, 1e4, 1e5],
            "fit_intercept" : [True, False],
            "solver" : ['newton-cg', 'lbfgs', 'liblinear', 'sag']}
    scoring = True
    classifiers[algo] = Estimator(algo, est, grid, scoring)
    # Naive Bayes
    algo = 'NB'
    est = MultinomialNB()
    grid = {"alpha" : [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "fit_prior" : [True, False]}
    scoring = True
    classifiers[algo] = Estimator(algo, est, grid, scoring)
    # Radial Basis Function
    algo = 'RBF'
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
    classifiers[algo] = Estimator(algo, est, grid, scoring)
    # Random Forest
    algo = 'RF'
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
    classifiers[algo] = Estimator(algo, est, grid, scoring)
    # Randomized Logistic Regression
    algo = 'RLOGR'
    params = {"n_resampling" : n_estimators,
              "random_state" : seed,
              "verbose" : verbosity}
    est = RandomizedLogisticRegression(**params)
    grid = {"n_resampling" : [100, 200, 500],
            "C" : [0.1, 1, 10, 100, 1000, 1e4, 1e5],
            "scaling" : [0.3, 0.5, 0.7],
            "sample_fraction" : [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "selection_threshold" : [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]}
    scoring = False
    classifiers[algo] = Estimator(algo, est, grid, scoring)
    # Support Vector Machine
    algo = 'SVM'
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
    classifiers[algo] = Estimator(algo, est, grid, scoring)
    # XGBoost Binary
    algo = 'XGB'
    params = {"objective" : 'binary:logistic',
              "n_estimators" : n_estimators,
              "max_depth" : 10,
              "learning_rate" : 0.1,
              "min_child_weight" : 1.05,
              "subsample" : 0.95,
              "colsample_bytree" : 0.8,
              "nthread" : n_jobs,
              "silent" : True}
    est = xgb.XGBClassifier(**params)
    grid = {"n_estimators" : [21, 51, 101, 201, 501, 1001],
            "max_depth" : [None, 6, 8, 10, 20],
            "learning_rate" : [0.02, 0.05, 0.1],
            "min_child_weight" : [1.0, 1.1],
            "subsample" : [0.8, 0.9, 1.0],
            "colsample_bytree" : [0.8, 0.9, 1.0]}
    scoring = False
    classifiers[algo] = Estimator(algo, est, grid, scoring)
    # Extra Trees
    algo = 'XT'
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
    classifiers[algo] = Estimator(algo, est, grid, scoring)
    # return the entire classifier list
    return classifiers


#
# Function get_class_scorers
#

def get_class_scorers():
    scorers = ['accuracy',
               'f1',
               'log_loss',
               'precision',
               'recall',
               'roc_auc']
    return scorers


#
# Function get_regressors
#

# Decision Tree (feature_importances_)
# Gradient Boosting (feature_importances_)
# K-Nearest Neighbors (NA)
# Linear Regression (coef_)
# Random Forest (feature_importances_)
# Randomized Lasso
# XG Boost (NA)

def get_regressors(n_estimators, seed, n_jobs, verbosity):
    # initialize regressor list
    regressors = {}
    # Gradient Boosting
    algo = 'GBR'
    params = {"n_estimators" : n_estimators,
              "random_state" : seed,
              "verbose" : verbosity}
    est = GradientBoostingRegressor()
    grid = {"loss" : ['deviance', 'exponential'],
            "learning_rate" : [0.05, 0.1, 0.15],
            "n_estimators" : [50, 100, 200],
            "max_depth" : [3, None],
            "min_samples_split" : [1, 2, 3],
            "min_samples_leaf" : [1, 2]}
    regressors[algo] = Estimator(algo, est, grid)
    # K-Nearest Neighbor
    algo = 'KNR'
    params = {"n_jobs" : n_jobs}
    est = KNeighborsRegressor(**params)
    grid = {"n_neighbors" : [3, 5, 7, 10],
            "weights" : ['uniform', 'distance'],
            "algorithm" : ['ball_tree', 'kd_tree', 'brute', 'auto'],
            "leaf_size" : [10, 20, 30, 40, 50]}
    regressors[algo] = Estimator(algo, est, grid)
    # Linear Regression
    algo = 'LR'
    params = {"n_jobs" : n_jobs}
    est = LinearRegression()
    grid = {"fit_intercept" : [True, False],
            "normalize" : [True, False],
            "copy_X" : [True, False]}
    regressors[algo] = Estimator(algo, est, grid)
    # Random Forest
    algo = 'RFR'
    params = {"n_estimators" : n_estimators,
              "random_state" : seed,
              "n_jobs" : n_jobs,
              "verbose" : verbosity}
    est = RandomForestRegressor(**params)
    grid = {"n_estimators" : [50, 100, 150, 200],
            "max_depth" : [3, 10, None],
            "min_samples_split" : [1, 2, 3, 5, 10],
            "min_samples_leaf" : [1, 2, 3],
            "bootstrap" : [True, False],
            "criterion" : ['gini', 'entropy']}
    regressors[algo] = Estimator(algo, est, grid)
    # Randomized Lasso
    algo = 'RLASS'
    params = {"n_resampling" : n_estimators,
              "random_state" : seed,
              "n_jobs" : n_jobs,
              "verbose" : verbosity}
    est = RandomizedLasso(**params)
    grid = {"n_resampling" : [100, 200, 500],
            "max_iter" : [300, 500, 700, 1000],
            "sample_fraction" : [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "selection_threshold" : [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]}
    regressors[algo] = Estimator(algo, est, grid)
    # XGBoost Multi
    algo = 'XGBM'
    params = {"objective" : 'multi:softmax',
              "n_estimators" : n_estimators,
              "max_depth" : 10,
              "learning_rate" : 0.1,
              "min_child_weight" : 1.05,
              "subsample" : 0.85,
              "colsample_bytree" : 0.8,
              "nthread" : n_jobs,
              "silent" : True}
    est = xgb.XGBClassifier(**params)
    grid = {"n_estimators" : [10, 21, 31, 51, 101, 201, 501, 1001],
            "max_depth" : [None, 3, 6, 7, 8, 10, 20],
            "learning_rate" : [0.01, 0.02, 0.05, 0.1, 0.2],
            "min_child_weight" : [1.0, 1.05, 1.1, 1.2],
            "subsample" : [0.8, 0.85, 0.9, 0.95],
            "colsample_bytree" : [0.6, 0.7, 0.8, 0.9]}
    regressors[algo] = Estimator(algo, est, grid)
    # XGBoost
    algo = 'XGBR'
    params = {"objective" : 'reg:linear',
              "n_estimators" : n_estimators,
              "max_depth" : 10,
              "learning_rate" : 0.1,
              "min_child_weight" : 1.05,
              "subsample" : 0.85,
              "colsample_bytree" : 0.8,
              "seed" : seed,
              "nthread" : n_jobs,
              "silent" : True}
    est = xgb.XGBRegressor(**params)
    grid = {"n_estimators" : [10, 20, 30, 50, 100, 200],
            "max_depth" : [None, 3, 6, 7, 8, 10, 20],
            "learning_rate" : [0.01, 0.02, 0.05, 0.1, 0.2],
            "min_child_weight" : [1.0, 1.05, 1.1, 1.2],
            "subsample" : [0.8, 0.85, 0.9, 0.95],
            "colsample_bytree" : [0.6, 0.7, 0.8, 0.9]}
    regressors[algo] = Estimator(algo, est, grid)
    # Extra Trees
    algo = 'XTR'
    params = {"n_estimators" : n_estimators,
              "random_state" : seed,
              "n_jobs" : n_jobs,
              "verbose" : verbosity}
    est = ExtraTreesRegressor(**params)
    grid = {"n_estimators" : [10, 50, 100, 150, 200],
            "max_features" : ['auto', 'sqrt', 'log2', None],
            "max_depth" : [3, 5, None],
            "min_samples_split" : [1, 2, 3],
            "min_samples_leaf" : [1, 2],
            "bootstrap" : [True, False],
            "warm_start" : [True, False]}
    regressors[algo] = Estimator(algo, est, grid)
    # return the entire regressor list
    return regressors


#
# Function get_regr_scorers
#

def get_regr_scorers():
    scorers = ['mean_absolute_error',
               'mean_squared_error',
               'median_absolute_error',
               'r2']
    return scorers
