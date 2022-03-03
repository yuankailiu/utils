#!/usr/bin/env python3

import os
import glob
import numpy as np
import netCDF4 as nc
import matplotlib
import matplotlib.pyplot as plt
from functools import partial
import multiprocessing
matplotlib.rcParams.update({'font.size': 16})

## plotting function
def plot_nc(ncfile, exclist):
    fstr    = ncfile.split('/')[-1].split('_')
    mission = fstr[0]
    level   = fstr[1]
    pathrow = fstr[2]
    date1   = fstr[3]
    date2   = fstr[11]
    titstr  = '{}-{}'.format(date1, date2)
    fout    = '{}_{}_{}_{}-{}'.format(mission, level, pathrow, date1, date2)

    if pathrow in exclist:
        print('Skip frame: {}_{}_{}'.format(mission, level, pathrow))

    else:
        ## make output dir if not there
        outdir = './pic/{}_{}_{}'.format(mission, level, pathrow)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass

        ## read data
        ds = nc.Dataset(ncfile)
        v  = np.array(ds['v'][:]).astype(float)
        v[v==-32767] = np.nan       # masking out the NaN pixels
        v = v/365.25                # convert velocity to m/day

        ## plot it and save the fig
        plt.figure(figsize=[10,10])
        im = plt.imshow(v, vmin=0, vmax=2.4, cmap='RdYlBu_r')
        plt.colorbar(im)
        plt.title(titstr)
        plt.savefig('{}/{}.png'.format(outdir, fout), bbox_inches='tight')
        print('plotting', fout)
        plt.close()



## grab all the netcdf files
files = sorted(glob.glob('./NetCDF/*.nc'))

## exclude frames
exc = ['064114','065113','065114','066113','066114','067113','222131']

## parallel the plotting
pool = multiprocessing.Pool(10)
pool.map(partial(plot_nc, exclist=exc), files)

print('Complte plotting the .nc files')
