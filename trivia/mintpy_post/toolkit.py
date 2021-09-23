

import os
import numpy as np
from copy import copy
from scipy import linalg
import glob
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from mintpy import view, tsview, plot_network, plot_transection, plot_coherence_matrix
from mintpy.objects import timeseries
from mintpy.utils import readfile
from mintpy.tsview import timeseriesViewer
from datetime import datetime
from mintpy.utils import utils as ut, plot as pp


# Inscrese matplotlib font size when plotting
plt.rcParams.update({'font.size': 16})