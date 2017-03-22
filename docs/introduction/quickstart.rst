Quick Start
===========

Install alphapy::

    pip install alphapy

.. note:: Please refer to Installation for further details.

From your command line application, clone the repository::

    git clone https://github.com/Alpha314/AlphaPy

Change your directory to the examples location::
  
    cd AlphaPy/alphapy/examples

Currently, there are four examples, two from the command line
and two from a Jupyter Notebook:

* Kaggle
* NCAA Basketball
* Trading Model
* Trading System

To run the Kaggle command line example::

    cd Kaggle
    python ../../alpha.py -d './config'

To run the NCAA Basketball command line example::

    cd NCAAB
    python ../../alpha.py -d './config'

To run the Trading Model notebook example::

    cd "Trading Model"
    python ../../alpha.py -d './config'

To run the Trading System notebook example::

    cd "Trading System"
    python ../../alpha.py -d './config'

Building docs
-------------

Let's build our docs into HTML to see how it works.
Simply run:

.. code-block:: python

    # Inside top-level docs/ directory.
    make html

This should run Sphinx in your shell, and output HTML.
At the end, it should say something about the documents being ready in
``_build/html``.
You can now open them in your browser by typing::

    open _build/html/index.html

You can also view it by running a web server in that directory::

    # Inside docs/_build/html directory.
    python -m SimpleHTTPServer

Then open your browser to http://localhost:8000.

This should display a rendered HTML page that says **Welcome to Crawler’s documentation!** at the top.

.. note:: ``make html`` is the main way you will build HTML documentation locally.
            It is simply a wrapper around a more complex call to Sphinx,
            which you can see as the first line of output.

Custom Theme
------------

You'll notice your docs look a bit different than mine.
You can change this by setting the ``html_theme`` setting in your ``conf.py``.
Go ahead and set it like this::

    html_theme = 'sphinx_rtd_theme'

If you rebuild your documentation,
you will see the new theme::

    make html

.. warning:: Didn't see your new theme?
             That's because Sphinx is smart,
             and only rebuilds pages that have changed.
             It might have thought none of your pages changed,
             so it didn't rebuild anything.
             Fix this by running a ``make clean html``,
             which will force a full rebuild.

Extra Credit
************

Have some extra time left?
Check out these other cool things you can do with Sphinx.

Understanding ``conf.py``
-------------------------

Sphinx is quite configurable,
which can be a bit overwhelming.
However, 
the ``conf.py`` file is quite well docuemnted.
You can read through it and get some ideas about what all it can do.

A few of the more useful settings are:

* project
* html_theme
* extensions
* exclude_patterns

This is all well documented in the Sphinx :ref:`sphinx:build-config` doc.

Moving on
---------

Now it is time to move on to :doc:`step-1`.