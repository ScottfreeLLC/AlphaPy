##############################################################
#
# Package  : AlphaPy
# Module   : scoring
# Version  : 1.0
# Copyright: Mark Conway
# Date     : August 5, 2015
#
##############################################################


#
# Imports
#

import numpy as np
from operator import itemgetter
from sklearn import metrics


#
# Function compute_auc
#

def compute_auc(y, y_pred):
    fpr, tpr, _ = metrics.roc_curve(y, y_pred)
    return metrics.auc(fpr, tpr)


#
# Function gini
#

def gini(actual, pred, cmpcol= 0, sortcol=1):
     assert( len(actual) == len(pred) )
     all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
     all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
     totalLosses = all[:,0].sum()
     giniSum = all[:,0].cumsum().sum() / totalLosses
     giniSum -= (len(actual) + 1) / 2.
     return giniSum / len(actual)


#
# Function normalized_gini
#
 
def normalized_gini(a, p):
     return gini(a, p) / gini(a, a)


#
# Function report_scores
#

def report_scores(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.5f} (std: {1:.5f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
    print("Top Score: {0}".format(top_scores[0].mean_validation_score))
    return top_scores[0].mean_validation_score
