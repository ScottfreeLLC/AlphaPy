##############################################################
#
# Package   : AlphaPy
# Module    : plots
# Version   : 1.0
# Copyright : Mark Conway
# Date      : July 15, 2015
#
##############################################################


#
# Plots
#
#     1. Calibration
#     2. Feature Importances
#     3. Learning Curve
#     4. ROC Curve
#     5. Confusion Matrix
#     6. Classifier Comparison
#     7. Scatter Plot
#     8. Validation Curve
#     9. Partial Dependence
#

print(__doc__)


#
# Imports
#

from globs import PSEP, SSEP, USEP
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cross_validation
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.learning_curve import validation_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_partition_data
#

def get_partition_data(model, partition):
    """
    Get the X, y pair for a given model and partition
    """
    if partition == 'train':
        X = model.X_train
        y = model.y_train
    elif partition == 'test':
        X = model.X_test
        y = model.y_test
    else:
        raise TypeError('Partition must be train or test')
    return X, y


#
# Function write_plot
#

def write_plot(model, plot_type, partition):
    """
    Save plot to a file.
    """

    # Extract model parameters
    base_dir = model.specs['base_dir']
    project = model.specs['project']

    # Create output file specification
    file_only = ''.join([plot_type, USEP, partition, '.png'])
    file_all = SSEP.join([base_dir, project, file_only])

    # Save plot    
    logger.info("Writing plot to %s", file_all)
    plt.tight_layout()
    plt.savefig(file_all)


#
# Function plot_calibration
#

def plot_calibration(model, partition):

    # For classification only

    if model.specs['regression']:
        raise TypeError('Calibration plot is for classification only')

    # Get X, Y for correct partition

    X, y = get_partition_data(model, partition)

    # Excerpts from:
    #
    # Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
    # License: BSD Style.

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    for algo in model.algolist:
        clf = model.estimators[algo]
        if hasattr(clf, "predict_proba"):
            prob_pos = model.probas[(algo, partition)]
        else:  # use decision function
            prob_pos = clf.decision_function(X)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % (algo, ))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=algo,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of Positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration Plots [Reliability Curve]')

    ax2.set_xlabel("Mean Predicted Value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    write_plot(model, 'calibration', partition)


#
# Function plot_importances
#

def plot_importance(model, partition):
    """
    Display feature importances
    """

    # forest was input parameter
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(10):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(10), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(10), indices)
    plt.xlim([-1, 10])

    write_plot(model, 'importance', partition)


#
# Function plot_learning_curve
#

def plot_learning_curve(model, partition):
    """
    Display learning curve
    """

    # Get X, Y for correct partition

    X, y = get_partition_data(model, partition)

    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        test_score[i] = clf.loss_(y_test, y_pred)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
            label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
            label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')

    write_plot(model, 'learning_curve', partition)


#
# Function plot_partial_dependence
#


def plot_partial_dependence(model, partition):
    """
    Plot partial dependence
    """
    # Get X, Y for correct partition

    X, y = get_partition_data(model, partition)

    # fetch California housing dataset
    cal_housing = fetch_california_housing()

    # split 80/20 train-test
    X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                        cal_housing.target,
                                                        test_size=0.2,
                                                        random_state=1)
    names = cal_housing.feature_names

    print('_' * 80)
    print("Training GBRT...")
    clf = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                    learning_rate=0.1, loss='huber',
                                    random_state=1)
    clf.fit(X_train, y_train)
    print("done.")

    print('_' * 80)
    print('Convenience plot with ``partial_dependence_plots``')
    print

    features = [0, 5, 1, 2, (5, 1)]
    fig, axs = plot_partial_dependence(clf, X_train, features, feature_names=names,
                                       n_jobs=3, grid_resolution=50)
    fig.suptitle('Partial dependence of house value on nonlocation features\n'
                 'for the California housing dataset')
    plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle

    print('_' * 80)
    print('Custom 3d plot via ``partial_dependence``')
    print
    fig = plt.figure()

    target_feature = (1, 5)
    pdp, (x_axis, y_axis) = partial_dependence(clf, target_feature,
                                               X=X_train, grid_resolution=50)
    XX, YY = np.meshgrid(x_axis, y_axis)
    Z = pdp.T.reshape(XX.shape).T
    ax = Axes3D(fig)
    surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu)
    ax.set_xlabel(names[target_feature[0]])
    ax.set_ylabel(names[target_feature[1]])
    ax.set_zlabel('Partial dependence')
    #  pretty init view
    ax.view_init(elev=22, azim=122)
    plt.colorbar(surf)
    plt.suptitle('Partial dependence of house value on median age and '
                'average occupancy')
    plt.subplots_adjust(top=0.9)

    plt.show()


#
# Function plot_roc_curve
#

def plot_roc_curve(model, partition):
    """
    Display feature importances
    """

    # Get X, Y for correct partition

    X, y = get_partition_data(model, partition)

    cv = StratifiedKFold(y, n_folds=6)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    write_plot(model, 'roc_curve', partition)


#
# Function plot_confusion_matrix
#

def plot_confusion_matrix(model, partition):
    """
    Display the confusion matrix
    """

    write_plot(model, 'confusion', partition)


#
# Function plot_boundary
#

def plot_boundary(model):
    """
    Display a comparison of classifiers
    """

    # Extract model parameters

    # Get X, Y for correct partition

    X, y = get_partition_data(model, partition)

    h = .02  # step size in the mesh

    f1 = 3 * (len(classifiers) + 1)
    f2 = f1 / len(classifiers)
    figure = plt.figure(figsize=(f1, f2))

    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the dataset first
    i = 1
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(1, len(classifiers) + 1, i)
    # Plot the training and testing points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(1, len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].

        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training and testing points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

    figure.subplots_adjust(left=.02, right=.98)
    write_plot(model, 'boundary', partition)


#
# Function plot_scatterplot
#

def plot_scatterplot(model, partition, feature1, feature2):
    """
    Plot a scatterplot of two variables
    """

    # Get X, Y for correct partition

    X, y = get_partition_data(model, partition)

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    Y = iris.target

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    # To getter a better understanding of interaction of the dimensions
    # plot the first three PCA dimensions
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(iris.data)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,
               cmap=plt.cm.Paired)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    write_plot(model, 'scatter', partition)


#
# Function plot_validation
#

def plot_validation(model, partition):
    """
    Plot a cross-validation curve
    """

    # Get X, Y for correct partition

    X, y = get_partition_data(model, partition)

    param_range = np.logspace(-6, -1, 5)
    train_scores, test_scores = validation_curve(
        SVC(), X, y, param_name="gamma", param_range=param_range,
        cv=10, scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with SVM")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")

    write_plot(model, 'validation', partition)
