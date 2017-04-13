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

.. csv-table:: Amazon Daily Stock Prices (Source: Yahoo)
   :file: amzn_daily.csv

Configuration
-------------

Here is an example of a model configuration file. It is written in YAML
and is divided into logical sections reflecting different parts of the
pipeline.

.. literalinclude:: game.yml
   :language: yaml
   :caption: **game.yml**
