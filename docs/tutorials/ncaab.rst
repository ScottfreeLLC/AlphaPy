NCAA Basketball Tutorial
========================

*SportFlow Running Time: Approximately 15 minutes*

.. image:: ncaab_roc_curve.png
   :alt: NCAAB ROC Curve
   :width: 80%
   :align: center

x

**Step 1**: From the ``examples`` directory, change your
directory::

    cd NCAAB

Before running SportFlow, let's briefly review the ``model.yml``
file in the ``config`` directory. We will submit the actual predictions instead of the
probabilities, so ``submit_probas`` is set to ``False``. All
features will be included except for the ``PassengerId``. The
target variable is ``Survived``, the label we are trying to
accurately predict.

We'll compare random forests and XGBoost, run recursive
feature elimination and a grid search, and select the best
model. Note that a blended model of all the algorithms is
a candidate for best model. The details of each algorithm
are located in the ``algos.yml`` file.

.. literalinclude:: ncaab.yml
   :language: yaml
   :caption: **model.yml**

**Step 2**: Now, we are ready to run SportFlow::

    sflow

From the ``examples`` directory, run the following commands::

    jupyter notebook
