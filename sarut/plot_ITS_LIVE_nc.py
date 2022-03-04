#!/usr/bin/env python3
#
#   Read and plot ITS_LIVE data
#   Download data from : https://its-live.jpl.nasa.gov/
#   May need to read the documentation there
#
#   Only simple functionalities
#       - plot the CDF files (parallel plotting)
#       - exclude ploting certain frames


import glob
import matplotlib
import multiprocessing
from functools import partial
import sarut.tools.plot as sarplt

matplotlib.rcParams.update({'font.size': 16})


## grab all the netcdf files
path = './NetCDF/*.nc'
files = sorted(glob.glob(path))

## exclude frames
exc = ['064114','065113','065114','066113','066114','067113','222131']

## parallel the plotting
num_threads = 8
pool = multiprocessing.Pool(num_threads)
pool.map(partial(sarplt.plot_ITS_LIVE_nc, exclist=exc), files)

print('Complte plotting the .nc files')
