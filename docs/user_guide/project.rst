Project Structure
=================

Setup
-----

Your file system should now look similar to this::

    project
    ├── config
    └── data
        ├── index.rst
        ├── conf.py
        ├── Makefile
        ├── make.bat
        ├── _build
        ├── _static
        ├── _templates
    └── input
    └── model
    └── output
    └── plots

We have a top-level ``docs`` directory in the main project directory.
Inside of this is:

``index.rst``:
    This is the index file for the documentation, or what lives at ``/``.
    It normally contains a *Table of Contents* that will link to all other
    pages of the documentation.

``conf.py``: 
    Allows for customization of Sphinx.
    You won't need to use this too much yet,
    but it's good to be familiar with this file.

``Makefile`` & ``make.bat``: 
    This is the main interface for local development,
    and shouldn't be changed.

``_build``:  
    The directory that your output files go into.

``_static``: 
    The directory to include all your static files, like images.

``_templates``: 
    Allows you to override Sphinx templates to customize look and feel.

Configuration
-------------

My output looks like this:

.. literalinclude:: quickstart-output.txt
   :language: text
   :linenos:

Data Section
~~~~~~~~~~~~

.. image:: config_data.png
   :height:  500 px
   :width:  1000 px
   :alt: Data Section
   :align: center

Model Section
~~~~~~~~~~~~~

.. image:: config_model.png
   :height:  500 px
   :width:  1000 px
   :alt: Model Section
   :align: center

Features Section
~~~~~~~~~~~~~~~~

.. image:: config_features.png
   :height:  500 px
   :width:  1000 px
   :alt: Features Section
   :align: center

Treatments Section
~~~~~~~~~~~~~~~~~~

.. image:: config_treatments.png
   :height:  500 px
   :width:  1000 px
   :alt: Treatments Section
   :align: center

Other Sections
~~~~~~~~~~~~~~

.. image:: config_others.png
   :height:  500 px
   :width:  1000 px
   :alt: Treatments Section
   :align: center
