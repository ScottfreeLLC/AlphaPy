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
# Model Plots
#
#     1. Calibration
#     2. Feature Importance
#     3. Learning Curve
#     4. ROC Curve
#     5. Confusion Matrix
#     6. Validation Curve
#     7. Partial Dependence
#     8. Decision Boundary
#
# EDA Plots
#
#     1. Scatter Plot Matrix
#     2. Facet Grid
#     3. Distribution Plot
#     4. Box Plot
#     5. Swarm Plot
#
# Time Series
#
#     1. Time Series
#     2. Candlestick
#

print(__doc__)


#
# Imports
#

from bokeh.plotting import figure, show, output_file
from estimators import get_estimators
from estimators import ModelType
from globs import BSEP, PSEP, SSEP, USEP
from globs import Q1, Q3
from itertools import cycle
from itertools import product
import logging
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy import interp
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.learning_curve import validation_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from util import remove_list_items


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
# Function generate_plots
#

def generate_plots(model, partition):
    """
    Save plot to a file.
    """

    logger.info("Generating Plots for Partition: %s", partition)

    # Extract model parameters

    calibration_plot = model.specs['calibration_plot']
    confusion_matrix = model.specs['confusion_matrix']
    importances = model.specs['importances']
    learning_curve = model.specs['learning_curve']
    roc_curve = model.specs['roc_curve']

    # Generate plots

    if calibration_plot:
        plot_calibration(model, partition)
    if confusion_matrix:
        plot_confusion_matrix(model, partition)
    if roc_curve:
        plot_roc_curve(model, partition)
    if partition == 'train':
        if learning_curve:
            plot_learning_curve(model, partition)
        if importances:
            plot_importance(model, partition)


#
# Function write_plot
#

def write_plot(model, vizlib, plot, plot_type, tag, display=False):
    """
    Save plot to a file.
    """

    # Extract model parameters

    base_dir = model.specs['base_dir']
    project = model.specs['project']

    # Create output file specification

    file_only = ''.join([plot_type, USEP, tag, '.png'])
    file_all = SSEP.join([base_dir, project, file_only])

    # Show or write plot

    if display:
        plot.plot()
    else:
        logger.info("Writing plot to %s", file_all)
        if vizlib == 'matplotlib':
            plot.tight_layout()
            plot.savefig(file_all)
        elif vizlib == 'seaborn':
            plot.savefig(file_all)
        elif vizlib == 'bokeh':
            plot.save(file_all)
        elif vizlib == 'plotly':
            raise ValueError("Unsupported data visualization library: %s", vizlib)
        else:
            raise ValueError("Unrecognized data visualization library: %s", vizlib)


#
# Function plot_calibration
#

def plot_calibration(model, partition):
    """
    Display calibration plots

    Parameters
    ----------

    model : object that encapsulates all of the model parameters
    partition : 'train' or 'test'

    Excerpts from:
    
    Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
    License: BSD Style.

    """

    logger.info("Generating Calibration Plot")

    # For classification only

    if model.specs['model_type'] != ModelType.classification:
        logger.info('Calibration plot is for classification only')
        return None

    # Get X, Y for correct partition

    X, y = get_partition_data(model, partition)

    plt.style.use('classic')
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    for algo in model.algolist:
        logger.info("Calibration for Algorithm: %s", algo)
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

    write_plot(model, 'matplotlib', plt, 'calibration', partition)


#
# Function plot_importances
#

def plot_importance(model, partition):
    """
    Display feature importances

    Parameters
    ----------

    model : object that encapsulates all of the model parameters
    partition : 'train' or 'test'

    """

    logger.info("Generating Feature Importance Plots")

    # Get X, Y for correct partition

    X, y = get_partition_data(model, partition)

    # For each algorithm that has importances, generate the plot.

    n_top = 10
    for algo in model.algolist:
        logger.info("Feature Importances for Algorithm: %s", algo)
        try:
            importances = model.importances[algo]
            # forest was input parameter
            indices = np.argsort(importances)[::-1]
            # log the feature ranking
            logger.info("Feature Ranking:")
            for f in range(n_top):
                logger.info("%d. Feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
            # plot the feature importances
            title = BSEP.join([algo, "Feature Importances [", partition, "]"])
            plt.style.use('classic')
            plt.figure()
            plt.title(title)
            plt.bar(range(n_top), importances[indices][:n_top], color="b", align="center")
            plt.xticks(range(n_top), indices[:n_top])
            plt.xlim([-1, n_top])
            # save the plot
            tag = USEP.join([partition, algo])
            write_plot(model, 'matplotlib', plt, 'feature_importance', tag)
        except:
            logger.info("%s does not have feature importances", algo)


#
# Function plot_learning_curve
#

def plot_learning_curve(model, partition):
    """
    Generate learning curves for a given partition.

    Parameters
    ----------

    model : object that encapsulates all of the model parameters
    partition : 'train' or 'test'

    """

    logger.info("Generating Learning Curves")

    # Extract model parameters.

    cv_folds = model.specs['cv_folds']
    n_jobs = model.specs['n_jobs']
    seed = model.specs['seed']
    shuffle = model.specs['shuffle']
    verbosity = model.specs['verbosity']

    # Get original estimators

    estimators = get_estimators(model)

    # Get X, Y for correct partition.

    X, y = get_partition_data(model, partition)

    # Set cross-validation parameters to get mean train and test curves.

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=shuffle, random_state=seed)

    # Plot a learning curve for each algorithm.   

    ylim = (0.4, 1.01)

    for algo in model.algolist:
        logger.info("Learning Curve for Algorithm: %s", algo)
        # get estimator
        est = estimators[algo].estimator
        # plot learning curve
        title = BSEP.join([algo, "Learning Curve [", partition, "]"])
        # set up plot
        plt.style.use('classic')
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training Examples")
        plt.ylabel("Score")
        # call learning curve function
        train_sizes=np.linspace(0.1, 1.0, cv_folds)
        train_sizes, train_scores, test_scores = \
            learning_curve(est, X, y, train_sizes=train_sizes, cv=cv,
                           n_jobs=n_jobs, verbose=verbosity)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        # plot data
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training Score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-Validation Score")
        plt.legend(loc="lower right")
        # save the plot
        tag = USEP.join([partition, algo])
        write_plot(model, 'matplotlib', plt, 'learning_curve', tag)


#
# Function plot_roc_curve
#

def plot_roc_curve(model, partition):
    """
    Display ROC Curves with Cross-Validation
    """

    logger.info("Generating ROC Curves")

    # For classification only

    if model.specs['model_type'] != ModelType.classification:
        logger.info('ROC Curves are for classification only')
        return None

    # Get X, Y for correct partition.

    X, y = get_partition_data(model, partition)

    # Initialize plot parameters.

    plt.style.use('classic')
    plt.figure()
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2

    # Plot a ROC Curve for each algorithm.

    for algo in model.algolist:
        logger.info("ROC Curve for Algorithm: %s", algo)
        # get estimator
        estimator = model.estimators[algo]
        # compute ROC curve and ROC area for each class
        probas = model.probas[(algo, partition)]
        fpr, tpr, _ = roc_curve(y, probas)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, label='%s (area = %0.2f)' % (algo, roc_auc))

    # draw the luck line
    plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='Luck')
    # define plot characteristics
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = BSEP.join([algo, "ROC Curve [", partition, "]"])
    plt.title(title)
    plt.legend(loc="lower right")
    # save chart
    write_plot(model, 'matplotlib', plt, 'roc_curve', partition)


#
# Function plot_confusion_matrix
#

def plot_confusion_matrix(model, partition):
    """
    Display the confusion matrix
    """

    logger.info("Generating Confusion Matrices")

    # Get X, Y for correct partition.

    X, y = get_partition_data(model, partition)

    for algo in model.algolist:
        logger.info("Confusion Matrix for Algorithm: %s", algo)
        # get predictions for this partition
        y_pred = model.preds[(algo, partition)]
        # compute confusion matrix
        cm = confusion_matrix(y, y_pred)
        logger.info('Confusion Matrix:')
        logger.info('%s', cm)
        # initialize plot
        np.set_printoptions(precision=2)
        plt.style.use('classic')
        plt.figure()
        # plot the confusion matrix
        cmap = plt.cm.Blues
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        title = BSEP.join([algo, "Confusion Matrix [", partition, "]"])
        plt.title(title)
        plt.colorbar()
        # set up x and y axes
        y_values, y_counts = np.unique(y, return_counts=True)
        tick_marks = np.arange(len(y_values))
        plt.xticks(tick_marks, y_values, rotation=45)
        plt.yticks(tick_marks, y_values)
        # normalize confusion matrix
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm.max() / 2.0
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            cmr = round(cmn[i, j], 3)
            plt.text(j, i, cmr,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        # labels
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        # save the chart
        tag = USEP.join([partition, algo])
        write_plot(model, 'matplotlib', plt, 'confusion', tag)


#
# Function plot_validation_curve
#

def plot_validation_curve(model, partition, pname, prange):
    """
    Generate validation curves.

    Parameters
    ----------

    model     : object that encapsulates all of the model parameters
    partition : data subset ['train' or 'test']
    pname     : hyperparameter name ['gamma']
    prange    : hyperparameter values [np.logspace(-6, -1, 5)]

    """

    logger.info("Generating Validation Curves")

    # Extract model parameters.

    cv_folds = model.specs['cv_folds']
    n_jobs = model.specs['n_jobs']
    scorer = model.specs['scorer']
    verbosity = model.specs['verbosity']

    # Get X, Y for correct partition.

    X, y = get_partition_data(model, partition)

    # Define plotting constants.

    spacing = 0.5
    alpha = 0.2

    # Calculate a validation curve for each algorithm.
    
    for algo in model.algolist:
        logger.info("Algorithm: %s", algo)
        # get estimator
        estimator = model.estimators[algo]
        # set up plot
        train_scores, test_scores = validation_curve(
            estimator, X, y, param_name=pname, param_range=prange,
            cv=cv_folds, scoring=scorer, n_jobs=n_jobs)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        # set up figure
        plt.style.use('classic')
        plt.figure()
        # plot learning curves
        title = BSEP.join([algo, "Validation Curve [", partition, "]"])
        plt.title(title)
        # x-axis
        x_min, x_max = min(prange) - spacing, max(prange) + spacing
        plt.xlabel(pname)
        plt.xlim(x_min, x_max)
        # y-axis
        plt.ylabel("Score")
        plt.ylim(0.0, 1.1)
        # plot scores
        plt.plot(prange, train_scores_mean, label="Training Score", color="r")
        plt.fill_between(prange, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=alpha, color="r")
        plt.plot(prange, test_scores_mean, label="Cross-Validation Score",
                 color="g")
        plt.fill_between(prange, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=alpha, color="g")
        plt.legend(loc="best")        # save the plot
        tag = USEP.join([partition, algo])
        write_plot(model, 'matplotlib', plt, 'validation_curve', tag)


#
# Function plot_partial_dependence
#

def plot_partial_dependence(model, partition, algo, features, names=None):
    """
    Plot partial dependence
    """

    logger.info("Generating Partial Dependence Plot")

    # Extract model parameters.

    est = model.estimators[algo]
    n_jobs = model.specs['n_jobs']
    verbosity = model.specs['verbosity']

    # Get X, Y for correct partition

    X, y = get_partition_data(model, partition)

    # Plot partial dependence

    fig, axs = plot_partial_dependence(est, X, features, feature_names=names,
                                       grid_resolution=50, n_jobs=n_jobs,
                                       verbose=verbosity)
    title = BSEP.join([algo, "Partial Dependence [", partition, "]"])
    fig.suptitle(title)
    plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle

    # Save the plot

    tag = USEP.join([partition, algo])
    write_plot(model, 'matplotlib', plt, 'partial_dependence', tag)


#
# Function plot_boundary
#

def plot_boundary(model, partition, f1=0, f2=1):
    """
    Display a comparison of classifiers
    """

    logger.info("Generating Boundary Plots")

    # For classification only

    if model.specs['model_type'] != ModelType.classification:
        logger.info('Boundary Plots are for classification only')
        return None

    # Get X, Y for correct partition

    X, y = get_partition_data(model, partition)

    # Subset for the two boundary features

    X = X[[f1, f2]]

    # Initialize plot

    n_classifiers = len(model.algolist)
    plt.figure(figsize=(3 * 2, n_classifiers * 2))
    plt.subplots_adjust(bottom=.2, top=.95)

    xx = np.linspace(3, 9, 100)
    yy = np.linspace(1, 5, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]

    # Plot each classification probability

    for index, name in enumerate(model.algolist):
        # predictions
        y_pred = model.preds[(algo, partition)]
        classif_rate = np.mean(y_pred.ravel() == y.ravel()) * 100
        logger.info("Classification Rate for %s : %f " % (name, classif_rate))
        # probabilities
        probas = model.probas[(algo, partition)]
        n_classes = np.unique(y_pred).size
        # plot each class
        for k in range(n_classes):
            plt.subplot(n_classifiers, n_classes, index * n_classes + k + 1)
            plt.title("Class %d" % k)
            if k == 0:
                plt.ylabel(name)
            imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
                                       extent=(3, 9, 1, 5), origin='lower')
            plt.xticks(())
            plt.yticks(())
            idx = (y_pred == k)
            if idx.any():
                plt.scatter(X[idx, 0], X[idx, 1], marker='o', c='k')

    # Plot the probability color bar

    ax = plt.axes([0.15, 0.04, 0.7, 0.05])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')

    # Save the plot

    write_plot(model, 'matplotlib', figure, 'boundary', partition)


#
# EDA Plots
#


#
# Function plot_scatter
#

def plot_scatter(model, data, features, target, tag='eda'):
    """
    Plot a scatterplot matrix
    """

    logger.info("Generating Scatter Plot")

    # Get the feature subset

    features.append(target)
    df = data[features]

    # Generate the pair plot

    sns.set()
    sns_plot = sns.pairplot(df, hue=target)

    # Save the plot

    write_plot(model, 'seaborn', sns_plot, 'scatter_plot', tag)


#
# Function plot_facet_grid
#

def plot_facet_grid(model, data, target, frow, fcol, tag='eda'):
    """
    Plot a Seaborn faceted histogram grid
    """

    logger.info("Generating Facet Grid")

    # Calculate the number of bins using the Freedman-Diaconis rule.

    tlen = len(data[target])
    tmax = data[target].max()
    tmin = data[target].min()
    trange = tmax - tmin
    iqr = data[target].quantile(Q3) - data[target].quantile(Q1)
    h = 2 * iqr * (tlen ** (-1/3))
    nbins = math.ceil(trange / h)

    # Generate the pair plot

    sns.set(style="darkgrid")

    fg = sns.FacetGrid(data, row=frow, col=fcol, margin_titles=True)
    bins = np.linspace(tmin, tmax, nbins)
    fg.map(plt.hist, target, color="steelblue", bins=bins, lw=0)

    # Save the plot

    write_plot(model, 'seaborn', fg, 'facet_grid', tag)


#
# Function plot_distribution
#

def plot_distribution(model, data, target, tag='eda'):
    """
    Distribution Plot
    """

    logger.info("Generating Distribution Plot")

    # Generate the distribution plot

    dist_plot = sns.distplot(data[target])
    dist_fig = dist_plot.get_figure()

    # Save the plot

    write_plot(model, 'seaborn', dist_fig, 'distribution_plot', tag)


#
# Function plot_box
#

def plot_box(model, data, x, y, hue, tag='eda'):
    """
    Box Plot
    """

    logger.info("Generating Box Plot")

    # Generate the box plot

    box_plot = sns.boxplot(x=x, y=y, hue=hue, data=data)
    sns.despine(offset=10, trim=True)
    box_fig = box_plot.get_figure()

    # Save the plot

    write_plot(model, 'seaborn', box_fig, 'box_plot', tag)


#
# Function plot_swarm
#

def plot_swarm(model, data, x, y, hue, tag='eda'):
    """
    Swarm Plot
    """

    logger.info("Generating Swarm Plot")

    # Generate the swarm plot

    swarm_plot = sns.swarmplot(x=x, y=y, hue=hue, data=data)
    swarm_fig = swarm_plot.get_figure()

    # Save the plot

    write_plot(model, 'seaborn', swarm_fig, 'swarm_plot', tag)


#
# Time Series Plots
#


#
# Function plot_time_series
#

def plot_time_series(model, data, target, tag='eda'):
    """
    Time Series Plot
    """

    logger.info("Generating Time Series Plot")

    # Generate the time series plot

    ts_plot = sns.tsplot(data=data[target])
    ts_fig = ts_plot.get_figure()

    # Save the plot

    write_plot(model, 'seaborn', ts_fig, 'time_series_plot', tag)


#
# Function plot_candlestick
#

def plot_candlestick(df, symbol, cols=[], model=None):
    """
    Candlestick Charts
    """

    df["date"] = pd.to_datetime(df["date"])

    mids = (df.open + df.close) / 2
    spans = abs(df.close - df.open)

    inc = df.close > df.open
    dec = df.open > df.close
    w = 12 * 60 * 60 * 1000 # half day in ms

    TOOLS = "pan, wheel_zoom, box_zoom, reset, save"

    p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1000, toolbar_location="left")

    p.title = BSEP.join([symbol.upper(), "Candlestick"])
    p.xaxis.major_label_orientation = math.pi / 4
    p.grid.grid_line_alpha = 0.3

    p.segment(df.date, df.high, df.date, df.low, color="black")
    p.rect(df.date[inc], mids[inc], w, spans[inc], fill_color="#D5E1DD", line_color="black")
    p.rect(df.date[dec], mids[dec], w, spans[dec], fill_color="#F2583E", line_color="black")

    if model is not None:
        # save the plot
        write_plot(model, 'bokeh', p, 'candlestick_chart', symbol)
    else:
        show(p)
