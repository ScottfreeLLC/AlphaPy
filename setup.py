 #!/usr/bin/env python
from setuptools import find_packages
from setuptools import setup

DISTNAME = 'alphapy'
DESCRIPTION = "AlphaPy: A Machine Learning Pipeline for Speculators"
LONG_DESCRIPTION = "alphapy is a Python library for machine learning using scikit-learn. We have a stock market pipeline and a sports pipeline so that speculators can test predictive models, along with functions for trading systems and portfolio management."

MAINTAINER = 'ScottFree LLC [Robert D. Scott II, Mark Conway]'
MAINTAINER_EMAIL = 'scottfree.analytics@scottfreellc.com'
URL = "https://github.com/ScottFreeLLC/AlphaPy"
LICENSE = "Apache License, Version 2"
VERSION = "2.4.1"

classifiers = ['Development Status :: 4 - Beta',
               'Programming Language :: Python',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'License :: OSI Approved :: Apache Software License',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Operating System :: OS Independent']

install_reqs = [
    'arrow>=0.13',
    'bokeh>=1.3',
    'category_encoders>=2.1',
    'iexfinance>=0.4.3',
    'imbalanced-learn>=0.5',
    'ipython>=7.2',
    'keras>=2.2',
    'matplotlib>=3.0',
    'numpy>=1.17',
    'pandas>=1.0',
    'pandas-datareader>=0.8',
    'pyfolio>=0.9',
    'pyyaml>=5.0',
    'scikit-learn>=0.22',
    'scipy>=1.1',
    'seaborn>=0.9',
    'tensorflow>=1.15',
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
