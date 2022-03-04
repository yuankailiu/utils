#!/usr/bin/env python3
# --------------------------------------------------------------
# Post-processing for MintPy output results
#
# Yuan-Kai Liu, 2022-3-3
# --------------------------------------------------------------

# Recommended usage:
#   import sarut.tools.geod as sargeo


import os
import sys
import glob
import h5py
import string
import warnings
import argparse
import numpy      as np
import matplotlib as mpl
from   scipy      import linalg
from   matplotlib import colors
from   datetime   import datetime
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
from   mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.linear_model import LinearRegression


from mintpy import (
    view,
    tsview,
    plot_network,
    plot_transection,
    plot_coherence_matrix,
    solid_earth_tides,
)
from mintpy.view import viewer

from mintpy.objects import timeseries
from mintpy.objects import coord
from mintpy.objects.coord import coordinate
from mintpy.tsview import timeseriesViewer
from mintpy.utils import (
    ptime,
    readfile,
    writefile,
    utils as ut,
    plot as pp,
    attribute as attr,
)

# Inscrese matplotlib font size when plotting
plt.rcParams.update({'font.size': 16})


#########################################################

def lalo2yx(lalo):
    la = float(lalo[0])
    lo = float(lalo[1])
    y = coord.lalo2yx(la, coord_type='lat')
    x = coord.lalo2yx(lo, coord_type='lon')
    return [y, x]

def yx2lalo(yx):
    y = int(yx[0])
    x = int(yx[1])
    lat = coord.yx2lalo(y, coord_type='az')
    lon = coord.yx2lalo(x, coord_type='rg')
    return [lat, lon]

def line_azimuth(yx_start, yx_end, yflip=True):
    # get the azimuth clockwise wrt north (positive y-axis)
    dy = yx_end[0] - yx_start[0]
    dx = yx_end[1] - yx_start[1]
    if yflip is False:
        azimuth = np.rad2deg(np.arctan2(dx, dy)) % 360
    elif yflip is True:
        azimuth = np.rad2deg(np.arctan2(dx, -dy)) % 360
    return azimuth

def pt_projline(lalo, start_lalo, end_lalo):
    la = lalo[0]                    # lalo = array(N, 2)
    lo = lalo[1]                    # lalo = array(N, 2)
    start_lalo = np.array(start_lalo) # start_lalo = np.array([lat, lon])
    end_lalo   = np.array(end_lalo)   # end_lalo   = np.array([lat, lon])
    lon12 = [end_lalo[1], start_lalo[1]]
    lat12 = [end_lalo[0], start_lalo[0]]
    u    = (end_lalo-start_lalo)  # u = np.array([lat, lon])
    v    = (lalo-start_lalo)      # v = np.array([lat, lon])
    un   = np.linalg.norm(u)
    vn   = np.linalg.norm(v)
    cos  = u.dot(v.T) / (un*vn).flatten()
    dpar = vn*cos                 # distance parallel to line
    new_lalo = start_lalo.reshape(1,2) + (dpar*((u/un).reshape(2,1))).T
    dper = np.linalg.norm(new_lalo-lalo, axis=1) # distance perpendicular to line
    return new_lalo, dpar, dper

def pts_projline(lalo, start_lalo, end_lalo):
    la = lalo[:,0]                    # lalo = array(N, 2)
    lo = lalo[:,1]                    # lalo = array(N, 2)
    start_lalo = np.array(start_lalo) # start_lalo = np.array([lat, lon])
    end_lalo   = np.array(end_lalo)   # end_lalo   = np.array([lat, lon])
    lon12 = [end_lalo[1], start_lalo[1]]
    lat12 = [end_lalo[0], start_lalo[0]]
    u    = (end_lalo-start_lalo)  # u = np.array([lat, lon])
    v    = (lalo-start_lalo)      # v = np.array([lat, lon])
    un   = np.linalg.norm(u)
    vn   = np.linalg.norm(v, axis=1)
    cos  = u.dot(v.T) / (un*vn).flatten()
    dpar = vn*cos                 # distance parallel to line
    new_lalo = start_lalo.reshape(1,2) + (dpar*((u/un).reshape(2,1))).T
    dper = np.linalg.norm(new_lalo-lalo, axis=1) # distance perpendicular to line
    return new_lalo, dpar, dper

def parallel_line(start_yx, end_yx, angle, space, yflip=True):
    # parallel line with space clockwise wrt to north (positive y-axis)
    dx = space * np.sin(np.deg2rad(angle))
    dy = space * np.cos(np.deg2rad(angle))
    if yflip is True:
        dy = -dy
    start_yx2 = np.zeros_like(start_yx)
    end_yx2 = np.zeros_like(end_yx)
    start_yx2[0] = start_yx[0] + dy
    start_yx2[1] = start_yx[1] + dx
    end_yx2[0]   = end_yx[0]   + dy
    end_yx2[1]   = end_yx[1]   + dx
    return list(start_yx2), list(end_yx2)

def make_transec_swath(start_yx1, end_yx1, start_yx2, end_yx2, n, outfile='prof_tmp.txt'):
    start_xs = np.linspace(start_yx1[1], start_yx2[1], n)
    start_ys = np.linspace(start_yx1[0], start_yx2[0], n)
    end_xs   = np.linspace(  end_yx1[1],   end_yx2[1], n)
    end_ys   = np.linspace(  end_yx1[0],   end_yx2[0], n)
    res = []
    f = open(outfile, '+w')
    for i in range(n):
        strng  = '{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(start_ys[i], start_xs[i], end_ys[i], end_xs[i])
        f.write(strng)
        res.append([start_ys[i], start_xs[i], end_ys[i], end_xs[i]])
    f.close()
    f = open('{}_pts.txt'.format(outfile.split('.')[0]), '+w')
    for i in range(n):
        strng  = '{:.4f}\t{:.4f}\n'.format(start_ys[i], start_xs[i])
        f.write(strng)
        strng  = '{:.4f}\t{:.4f}\n'.format( end_ys[i], end_xs[i]  )
        f.write(strng)
    f.close()
    print('profiles coord saved into {}'.format(outfile))
    return res

def transec_pick(file, dset, transec_txt, fmt='yx', mask_file='maskTempCoh.h5'): # return default unit, usaully meter
    # read transect file in (starty, startx, endy, endx) order
    starts = []
    ends = []
    with open(transec_txt) as f:
        for line in f:
            starts.append([line.split()[0], line.split()[1]])
            ends.append(  [line.split()[2], line.split()[3]])
    starts = np.array(starts).astype('float')
    ends   = np.array(ends).astype('float')

    data  = readfile.read(file, datasetName=dset)[0]  # data
    meta  = readfile.read(file, datasetName=dset)[1]  # metadata

    # Read mask and mask the dataset
    mask = readfile.read(mask_file)[0]
    data[mask==0] = np.nan

    # Extract velocity transection
    res = []
    X   = []
    Z   = []
    for i in range(len(starts)):
        if fmt == 'lalo':
            data_line = ut.transect_lalo(data , meta, starts[i], ends[i], interpolation='nearest')
        elif fmt == 'yx':
            data_line = ut.transect_yx(data , meta, starts[i], ends[i], interpolation='nearest')
        x = data_line['distance']/ 1000   # in km
        z = data_line['value']   * 1000   # in mm
        res.append(data_line)
        X = X + list(x)
        Z = Z + list(z)
    return X, Z, res
