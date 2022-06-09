"""
This is the initialization script.
"""
# flake8: noqa

# imports for preparations later
import sys, os
import multiprocessing
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams

# import submodules
#from . import data
#from . import plot
#from . import geod
#from . import math


# provide shortcuts for commonly used classes by importing them here

# package version
__version__ = '0.0'

# preparational steps
multiprocessing.set_start_method('spawn', True)
register_matplotlib_converters()
rcParams['figure.constrained_layout.use'] = "True"