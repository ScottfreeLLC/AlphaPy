 #!/usr/bin/env python
from setuptools import find_packages
from setuptools import setup

DISTNAME = 'alphapy'
DESCRIPTION = "AlphaPy: A Machine Learning Pipeline for Speculators"
LONG_DESCRIPTION = "alphapy is a Python library for machine learning using scikit-learn. We have a stock market pipeline and a sports pipeline so that speculators can test predictive models, along with functions for trading systems and portfolio management."

MAINTAINER = 'ScottFree LLC [Mark Conway, Robert D. Scott II]'
MAINTAINER_EMAIL = 'alphapy@scottfreellc.com'
URL = "https://github.com/Alpha314/AlphaPy"
LICENSE = "Apache License, Version 2.0"
VERSION = "0.1.7"

classifiers = ['Development Status :: 4 - Beta',
               'Programming Language :: Python',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.4',
               'Programming Language :: Python :: 3.5',
               'License :: OSI Approved :: Apache Software License',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Operating System :: OS Independent']

install_reqs = [
    'bokeh>=0.12',
    'category_encoders>=1.2.0',
    'gplearn>=0.1.0',
    'imblearn>=0.0',
    'ipython>=3.2.3',
    'matplotlib>=2.0.0',
    'numpy>=1.9.1',
    'pandas>=0.19.0',
    'pandas-datareader>=0.3',
    'pyfolio>=0.7',
    'pyyaml>=3.12',
    'scikit-learn>=0.17.1',
    'scipy>=0.18.1',
    'seaborn>=0.7.1',
    'tensorflow>=1.0.0',
    'xgboost>=0.6',
]

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=VERSION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        classifiers=classifiers,
        install_requires=install_reqs,
    )
