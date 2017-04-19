SportFlow
=========

.. image:: sports_pipeline.png
   :alt: Sports Pipeline
   :width: 100%
   :align: center

SportFlow applies machine learning algorithms to predict game
outcomes for matches in any team sport. We created binary features
(for classification) to determine whether or not a team will
win the game or even more importantly, cover the spread. We
also try to predict whether or not a game's total points will
exceed the *over/under*.

Of course, there are practical matters to predicting a game's
outcome. The strength of supervised learning is to improve
an algorithm's performance with lots of data. While major-league
baseball has a total of 2,430 games per year, pro football has
only 256 games per year. College football and basketball are
somewhere in the middle of this range.

The other complication is determining whether or not a model
for one sport can be used for another. The advantage is that
combining sports gives us more data. The disadvantage is that
each sport has unique characteristics that could make a unified
model infeasible. Still, we can combine the game data to test
an overall model.

Data Sources
------------

SportFlow starts with minimal game data (lines and scores) and
expands these data into temporal features such as runs and
streaks for all of the features. Currently, we do not incorporate
player data or other external factors, but there are some
excellent open-source packages such as BurntSushi's *nflgame*
Python code. For its initial version, SportFlow game data
must be in the format below:

.. csv-table:: NCAA Basketball Data
   :file: ncaa.csv

The SportFlow logic is split-apply-combine, as the data are first
split along team lines, then team statistics are calculated and
applied, and finally the team data are inserted into the overall
model frame.

Domain Configuration
--------------------

The SportFlow configuration file is minimal. You can simulate random
scoring to compare with a real model. Further, you can experiment
with the rolling window for run and streak calculations.

.. literalinclude:: sport.yml
   :language: yaml
   :caption: **sport.yml**

``points_max``:
    Maximum number of simulated points to assign to any single team.

``points_min``: 
    Minimum number of simulated points to assign to any single team.

``random_scoring``: 
    If ``True``, assign random point values to games [Default: ``False``].

``seasons``: 
    The yearly list of seasons to evaluate.

``rolling_window``: 
    The period over which streaks are calculated.

Model Configuration
-------------------

SportFlow runs on top of AlphaPy, so the ``model.yml`` file has
the same format.

.. literalinclude:: sport_model.yml
   :language: text
   :caption: **model.yml**

Creating the Model
------------------

First, change the directory to your project location,
where you have already followed the :doc:`../user_guide/project`
specifications::

    cd path/to/project

Run this command to train a model::

    sflow

Usage::

    sflow [--train | --predict] [--tdate yyyy-mm-dd] [--pdate yyyy-mm-dd]

--train     Train a new model and make predictions (Default)
--predict   Make predictions from a saved model
--tdate     The training date in format YYYY-MM-DD (Default: Earliest Date in the Data)
--pdate     The prediction date in format YYYY-MM-DD (Default: Today's Date)

Running the Model
-----------------

In the project location, run ``sflow`` with the ``predict`` flag.
SportFlow will automatically create the ``predict.csv`` file using
the ``pdate`` option::

    sflow --predict [--pdate yyyy-mm-dd]
