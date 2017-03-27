AlphaPy Pipeline
================

.. image:: model_pipeline.png
   :height:  500 px
   :width:  1000 px
   :alt: AlphaPy Model Pipeline
   :align: center

Data Ingestion
--------------

Data Sampling
-------------

Different techniques to handle unbalanced classes Undersampling
Oversampling
Combined Sampling (SMOTE)
Ensemble Sampling

Feature Extraction
------------------

Imputation
Row Statistics and Distributions
Clustering and PCA
Standard Scaling (e.g., mean-centering)
Interactions (n-way)

.. image:: features.png
   :height:  500 px
   :width:  1000 px
   :alt: Feature Pipeline
   :align: center

Encoders
~~~~~~~~

• Factorization • One-Hot
• Ordinal
• Binary
• Helmert Contrast
• Sum Contrast
• Polynomial Contrast
• Backward Difference Contrast
• Simple Hashing

Treatments
~~~~~~~~~~

Some features require special treatment, for example, a date column
•
that is split into separate columns for month, day, and year.
Treatments are specified in the configuration file with the feature
•
name, the treatment function, and its parameters.
In the following example, we apply a runs test to 6 features in the
•
YAML file:

Feature Selection
-----------------

Univariate selection based on the percentile of
•
highest feature scores
•
Scoring functions for both classification and regression, e.g., ANOVA F-value or chi-squared statistic
Recursive Feature Elimination (RFE) with Cross- Validation (CV) with configurable scoring function and step size

Grid Search
-----------

Full or Randomized Distributed Grid Search with
•
subsampling (Spark if available)

Model Selection
---------------

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

Model Evaluation
----------------

Metrics
Calibration Plot
Confusion Matrix
Learning Curve
ROC Curve

Logging
-------

Pipeline Start
~~~~~~~~~~~~~~

.. image:: mp_start.png
   :height:  500 px
   :width:  1000 px
   :alt: Pipeline Start
   :align: center

Data Ingestion
~~~~~~~~~~~~~~

.. image:: mp_data.png
   :height:  500 px
   :width:  1000 px
   :alt: Data Ingestion
   :align: center

Feature Analysis
~~~~~~~~~~~~~~~~

.. image:: mp_features.png
   :height:  500 px
   :width:  1000 px
   :alt: Feature Analysis
   :align: center

Treatment Application
~~~~~~~~~~~~~~~~~~~~~

.. image:: mp_treatments.png
   :height:  500 px
   :width:  1000 px
   :alt: Treatment Application
   :align: center

Model Fitting
~~~~~~~~~~~~~

.. image:: mp_fit.png
   :height:  500 px
   :width:  1000 px
   :alt: Model Fitting
   :align: center

Cross-Validation
~~~~~~~~~~~~~~~~

.. image:: mp_cv.png
   :height:  500 px
   :width:  1000 px
   :alt: Cross-Validation
   :align: center

Model Metrics
~~~~~~~~~~~~~

.. image:: mp_metrics.png
   :height:  500 px
   :width:  1000 px
   :alt: Model Metrics
   :align: center
