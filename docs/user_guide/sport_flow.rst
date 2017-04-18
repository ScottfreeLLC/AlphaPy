SportFlow
=========

.. image:: sports_pipeline.png
   :alt: Sports Pipeline
   :width: 100%
   :align: center

Keeping general strategy in mind, apply machine learning algorithms to predict game outcomes using supervised learning, i.e., classification.
We will create binary features to determine whether or not a team will win the game or cover the spread.
•
We can also try to predict whether or not the total 11
•
score will be over or under.

Sports data are typically structured into a match or game format
•
after gathering team and player data.

Data Sources
------------

.. csv-table:: NCAA Basketball Data
   :file: ncaa.csv

Configuration
-------------

Here is an example of a model configuration file. It is written in YAML
and is divided into logical sections reflecting different parts of the
pipeline.

.. literalinclude:: sport.yml
   :language: yaml
   :caption: **sport.yml**


Model Configuration
-------------------

SportFlow runs on top of AlphaPy, so the ``model.yml`` file has
the same format.

.. literalinclude:: sport_model.yml
   :language: text
   :caption: **model.yml**
