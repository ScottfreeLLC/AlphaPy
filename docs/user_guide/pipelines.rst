AlphaPy Pipeline
================

.. image:: model_pipeline.png
   :height:  500 px
   :width:  1000 px
   :alt: AlphaPy Model Pipeline
   :align: center

Model Object Creation
---------------------

x

.. literalinclude:: alphapy.log
   :language: text
   :caption: **alphapy.log**
   :lines: 1-87

Data Ingestion
--------------

x

.. literalinclude:: alphapy.log
   :language: text
   :caption: **alphapy.log**
   :lines: 89-103

Feature Pipeline
----------------

.. image:: features.png
   :height:  500 px
   :width:  1000 px
   :alt: Feature Flowchart
   :align: center

Feature Encoding
~~~~~~~~~~~~~~~~

* Different techniques to handle unbalanced classes Undersampling
* Oversampling
* Combined Sampling (SMOTE)
* Ensemble Sampling

.. literalinclude:: alphapy.log
   :language: text
   :caption: **alphapy.log**
   :lines: 104-173

Feature Extraction
~~~~~~~~~~~~~~~~~~

* Imputation
* Row Statistics and Distributions
* Clustering and PCA
* Standard Scaling (e.g., mean-centering)
* Interactions (n-way)

.. literalinclude:: alphapy.log
   :language: text
   :caption: **alphapy.log**
   :lines: 174-210

Model Estimation
----------------

x

.. literalinclude:: alphapy.log
   :language: text
   :caption: **alphapy.log**
   :lines: 211-271

Feature Selection
-----------------

x

Univariate selection based on the percentile of highest feature scores
Scoring functions for both classification and regression, e.g., ANOVA F-value or chi-squared statistic
Recursive Feature Elimination (RFE) with Cross- Validation (CV) with configurable scoring function and step size

.. literalinclude:: alphapy.log
   :language: text
   :caption: **alphapy.log**
   :lines: 227-233

Grid Search
-----------

Full or Randomized Distributed Grid Search with subsampling (Spark if available)

.. literalinclude:: alphapy.log
   :language: text
   :caption: **alphapy.log**
   :lines: 234-245

Model Evaluation
----------------

Metrics
Calibration Plot
Confusion Matrix
Learning Curve
ROC Curve

.. literalinclude:: alphapy.log
   :language: text
   :caption: **alphapy.log**
   :lines: 272-311

Model Selection
---------------

x

.. literalinclude:: alphapy.log
   :language: text
   :caption: **alphapy.log**
   :lines: 312-318

Best Model
~~~~~~~~~~

.. image:: model_best.png
   :height:  500 px
   :width:  1000 px
   :alt: Best Model Selection
   :align: center

Blended Model
~~~~~~~~~~~~~

.. image:: model_blend.png
   :height:  500 px
   :width:  1000 px
   :alt: Blended Model Creation
   :align: center

Plot Generation
---------------

.. literalinclude:: alphapy.log
   :language: text
   :caption: **alphapy.log**
   :lines: 319-378

Recording
---------

.. literalinclude:: alphapy.log
   :language: text
   :caption: **alphapy.log**
   :lines: 379-389
