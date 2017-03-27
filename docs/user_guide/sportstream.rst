SportStream
===========

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

.. csv-table:: Frozen Delights!
   :header: "Treat", "Quantity", "Description"
   :widths: 15, 10, 30

   "Albatross", 2.99, "On a stick!"
   "Crunchy Frog", 1.49, "If we took the bones out, it wouldn't be
   crunchy, now would it?"
   "Gannet Ripple", 1.99, "On a stick!"

Configuration
-------------

Here is an example of a model configuration file. It is written in YAML
and is divided into logical sections reflecting different parts of the
pipeline.

.. literalinclude:: game.yml
   :language: yaml
   :caption: **game.yml**
