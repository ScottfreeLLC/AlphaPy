 #!/usr/bin/env python
from setuptools import find_packages
from setuptools import setup

DISTNAME = 'alphapy'
DESCRIPTION = "AlphaPy: A Machine Learning Pipeline for Speculators"
LONG_DESCRIPTION = "alphapy is a Python library for machine learning using scikit-learn. We have a stock market pipeline and a sports pipeline so that speculators can test predictive models, along with functions for trading systems and portfolio management."

MAINTAINER = 'ScottFree LLC [Robert D. Scott II, Mark Conway]'
MAINTAINER_EMAIL = 'mark.conway@scottfreellc.com'
URL = "https://github.com/ScottFreeLLC/AlphaPy"
LICENSE = "Apache License, Version 2.0.1"
VERSION = "2.0.1"

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
    'imbalanced-learn>=0.2.1',
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
    'xgboost>=0.6a2',
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
        entry_points={
            'console_scripts': [
                'alphapy = alphapy.__main__:main',
                'mflow = alphapy.market_flow:main',
                'sflow = alphapy.sport_flow:main',
            ],
        }
    )
