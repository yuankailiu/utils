#!/usr/bin/env python3
############################################################
# This code it meant to examine the products from MintPy
# YKL @ 2021-05-19
############################################################

# This code is not complete, need to work on it...

import os
import sys
import glob
import h5py
import time
import warnings
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


from copy import copy
from scipy import linalg
from datetime import datetime, timedelta
from mintpy.objects import timeseries
from mintpy.tsview import timeseriesViewer
from mintpy.utils import readfile, utils as ut, plot as pp
from mintpy import view, tsview, plot_network, plot_transection, plot_coherence_matrix
matplotlib.rcParams.update({'font.size': 16})


from sklearn.linear_model import LinearRegression

from sarut.tools.geod import line_azimuth, lalo2yx


plt.rcParams.update({'font.size': 16})


##################################################################

def est_ramp(data, ramp_type='linear', mask='none'):
    width, length = data.shape
    # design matrix
    xx, yy = np.meshgrid(np.arange(0, width),
                         np.arange(0, length))
    xx = np.array(xx, dtype=np.float32).reshape(-1, 1)
    yy = np.array(yy, dtype=np.float32).reshape(-1, 1)
    ones = np.ones(xx.shape, dtype=np.float32)

    if ramp_type == 'linear':
        G = np.hstack((yy, xx, ones))
    elif ramp_type == 'quadratic':
        G = np.hstack((yy**2, xx**2, yy*xx, yy, xx, ones))
    elif ramp_type == 'linear_range':
        G = np.hstack((xx, ones))
    elif ramp_type == 'linear_azimuth':
        G = np.hstack((yy, ones))
    elif ramp_type == 'quadratic_range':
        G = np.hstack((xx**2, xx, ones))
    elif ramp_type == 'quadratic_azimuth':
        G = np.hstack((yy**2, yy, ones))
    else:
        raise ValueError('un-recognized ramp type: {}'.format(ramp_type))

    # estimate ramp
    mask = mask.flatten()
    X = np.dot(np.linalg.pinv(G[mask, :], rcond=1e-15), data[mask, :])
    ramp = np.dot(G, X)
    ramp = np.array(ramp, dtype=data.dtype)

    data_out = data - ramp
    return data_out, ramp

def linear_fit(x, y, report=False):
    # Create an instance of a linear regression model and fit it to the data with the fit() function:
    model = LinearRegression().fit(x, y)
    # Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
    r_sq = model.score(x, y)
    y_pred = model.predict(x)
    if report:
        print('coefficient of determination:', r_sq)
        print('slope:', model.coef_[0])
        print('intercept:', model.intercept_)
    return model, y_pred

def flatten_isnotnan(x):
    x = x.flatten()[~np.isnan(x.flatten())]
    return x

def dem_shading(dem, shade_azdeg=315, shade_altdeg=45, shade_exag=0.5, shade_min=-2e3, shade_max=3e3):
    # prepare shade relief
    import warnings
    from matplotlib.colors import LightSource
    from mintpy.objects.colors import ColormapExt

    ls = LightSource(azdeg=shade_azdeg, altdeg=shade_altdeg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dem_shade = ls.shade(dem, vert_exag=shade_exag, cmap=ColormapExt('gray').colormap, vmin=shade_min, vmax=shade_max)
    dem_shade[np.isnan(dem_shade[:, :, 0])] = np.nan
    return dem_shade


def wrapper(v, wraprange=np.pi):
    # wrap values around `wraprange`
    wrap_v = (v + wraprange) % (2 * wraprange) - wraprange
    return wrap_v


def scatter_fields(data1, data2, labels=['data1','data2'], vlim=[-20,20], title='', savedir=False):
    ## linear fit to the trend
    x = flatten_isnotnan(data1)
    y = flatten_isnotnan(data2)
    model, y_pred = linear_fit(x.reshape(-1, 1), y)

    # plot
    plt.figure(figsize=[6,6])
    plt.scatter(x, y)
    plt.scatter(x, y_pred, s=0.3, label='y=ax+b \n a={:.3f}, b={:.3f}'.format(model.coef_[0], model.intercept_))
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.ylim(vlim[0], vlim[1])
    plt.legend(loc='upper left')
    plt.title(title)
    if savedir is not False:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        # output
        out_file = f'{savedir}/{title}.png'
        plt.savefig(out_file, bbox_inches='tight', transparent=True, dpi=300)
        print('save to file: '+out_file)
        plt.close()
    else:
        plt.show()


def plot_range_var(data, titstr='', vlim=[None, None], option='plot'):

    ## flatten the field for scatter plot
    length, width = data.shape
    rbins = np.tile(np.arange(width), length)
    abins = np.tile(np.arange(length), (width, 1)).T
    vflat = data.flatten()

    ## linear fit to the trend
    x = np.tile(np.arange(width), (length, 1)).flatten()
    x = x[~np.isnan(data.flatten())]
    y = flatten_isnotnan(data)
    model, y_pred = linear_fit(x.reshape(-1, 1), y)
    params_label = 'y=ax+b \n a={:.3f}, b={:.3f}'.format(model.coef_[0], model.intercept_)
    params_label += '\n slope = {:.3f} mm/yr/track'.format(model.coef_[0]*width)

    if option == 'plot':
        plt.figure(figsize=[8,6])
        sc = plt.scatter(rbins, vflat, s=0.1, c=abins)
        plt.plot(x, y_pred, lw=2, label=params_label, c='r')
        plt.legend(loc='upper left')
        plt.xlabel('Range bin')
        plt.ylabel('LOS velocity [mm/yr]')
        plt.colorbar(sc, label='Azimuth bin')
        plt.title(titstr)
        plt.ylim(vlim[0], vlim[1])
        plt.show()
    elif option == 'report':
        return model.coef_[0]*width


def plot_dem_var(data, dem, titstr='', vlim=[None, None], option='plot'):

    ## flatten the field for scatter plot
    length, width = data.shape
    abins = np.tile(np.arange(length), (width, 1)).T
    vflat = data.flatten()

    ## linear fit to the trend
    x = dem.flatten()
    x = x[~np.isnan(data.flatten())]
    y = flatten_isnotnan(data)
    model, y_pred = linear_fit(x.reshape(-1, 1), y)
    params_label = 'y=ax+b \n a={:.3f}, b={:.3f}'.format(model.coef_[0], model.intercept_)

    if option == 'plot':
        plt.figure(figsize=[12,8])
        sc = plt.scatter(dem, vflat, s=0.1, c=abins)
        plt.plot(x, y_pred, lw=2, label=params_label, c='r')
        plt.legend(loc='upper left')
        plt.xlabel('Elevation from DEM [meter]')
        plt.ylabel('LOS velocity [mm/yr]')
        plt.colorbar(sc, label='Azimuth bin')
        plt.title(titstr)
        plt.ylim(vlim[0], vlim[1])
        plt.show()
    elif option == 'report':
        return model.coef_[0]*width



def prepare_los_geometry(geom_file):
    """Prepare LOS geometry data/info in geo-coordinates
    Parameters: geom_file  - str, path of geometry file
    Returns:    inc_angle  - 2D np.ndarray, incidence angle in radians
                head_angle - 2D np.ndarray, heading   angle in radians
                atr        - dict, metadata in geo-coordinate
    """

    print('prepare LOS geometry in geo-coordinates from file: {}'.format(geom_file))
    atr = readfile.read_attribute(geom_file)

    print('read incidenceAngle from file: {}'.format(geom_file))
    inc_angle = readfile.read(geom_file, datasetName='incidenceAngle')[0]

    if 'azimuthAngle' in readfile.get_dataset_list(geom_file):
        print('read azimuthAngle   from file: {}'.format(geom_file))
        print('convert azimuth angle to heading angle')
        az_angle  = readfile.read(geom_file, datasetName='azimuthAngle')[0]
        head_angle = ut.azimuth2heading_angle(az_angle)
    else:
        print('use the HEADING attribute as the mean heading angle')
        head_angle = np.ones(inc_angle.shape, dtype=np.float32) * float(atr['HEADING'])

    # turn default null value to nan
    inc_angle[inc_angle==0] = np.nan
    head_angle[head_angle==90] = np.nan

    # unit: degree to radian
    inc_angle *= np.pi / 180.
    head_angle *= np.pi / 180.

    return inc_angle, head_angle, atr


def plot_sin_cos(inc_angle, head_angle, in_tris=None):
    # Make a plot trigonometry of inc_angle and head_angle (in radians)

    if in_tris is not None:
        tris = np.array(in_tris)
    else:
        tris = np.array([np.sin(inc_angle), np.sin(head_angle), -1*np.sin(inc_angle)*np.cos(head_angle),
                        np.sin(inc_angle)*np.sin(head_angle), np.cos(inc_angle), np.cos(head_angle)])

    tstrs = ['sin(inc)', 'sin(head)', '-sin(inc) cos(head)',
             'sin(inc) sin(head)', 'cos(inc)', 'cos(head)']

    fig, axs = plt.subplots(nrows=1, ncols=6, figsize=[24,10], sharey=True, gridspec_kw={'wspace':0.4})
    for i, (tri, tstr) in enumerate(zip(tris, tstrs)):
        im = axs[i].imshow(tri)
        fig.colorbar(im, ax=axs[i], fraction=0.05)
        axs[i].set_title(tstr)
    plt.show()
    return tris


def enu2los(v3comp, inc_angle, head_angle, ref=False, display=True, display_more=False):
    # get LOS unit vector
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        unit_vec = [
            np.sin(inc_angle) * np.cos(head_angle) * -1,
            np.sin(inc_angle) * np.sin(head_angle),
            np.cos(inc_angle),
        ]

    # Three-component model motion (ENU); unit=mm/yr
    ve, vn, vu = v3comp
    disp = dict()
    disp['inc']  = inc_angle  * (180./np.pi)
    disp['head'] = head_angle * (180./np.pi)
    disp['ve']   = ve * np.ones_like(inc_angle)
    disp['vn']   = vn * np.ones_like(inc_angle)
    disp['vu']   = vu * np.ones_like(inc_angle)

    # convert ENU to LOS direction
    # sign convention: positive for motion towards satellite
    disp['v_los']= (disp['ve'] * unit_vec[0] + disp['vn'] * unit_vec[1] + disp['vu'] * unit_vec[2])

    # Take the reference pixel from the middle of the map
    if ref:
        disp['v_los'] -= disp['v_los'][length//2, width//2]

    if display:
        ## Make a quick plot
        fig, axs = plt.subplots(nrows=1, ncols=6, figsize=[24,10], sharey=True, gridspec_kw={'wspace':0.4})
        for i, key in enumerate(disp):
            im = axs[i].imshow(disp[key], cmap='RdYlBu_r')
            fig.colorbar(im, ax=axs[i], fraction=0.05)
            axs[i].set_title(key)
        plt.show()

        # report
        print('Min. LOS motion = {:.3f}'.format(np.nanmin(disp['v_los'])))
        print('Max. LOS motion = {:.3f}'.format(np.nanmax(disp['v_los'])))
        print('Dynamic range of LOS motion = {:.3f}'.format(np.nanmax(disp['v_los'])-np.nanmin(disp['v_los'])))
    if display_more:
        ## Make a plot about sin, cos
        plot_sin_cos(inc_angle, head_angle)
    return disp


def simple_orbit_geom(orbit, length, width):
    if orbit == 'A':
        # Ascending (geometry variation from near_range to far_range)
        inc_vary = (30.68, 46.24)
        azi_vary = (-258.93, -260.52)
    elif orbit == 'D':
        # Descending (geometry variation from near_range to far_range)
        inc_vary = (30.78, 46.28)
        azi_vary = (-101.06, -99.43)
        azi_vary = (ut.heading2azimuth_angle(-166.47261), ut.heading2azimuth_angle(-164.69743))
    inc_angle  = np.tile(np.linspace(*inc_vary, width), (length,1))
    azi_angle  = np.tile(np.linspace(*azi_vary, width), (length,1))
    #head_angle = -(azi_angle + 270.)  # this is essentially the same
    head_angle = ut.azimuth2heading_angle(azi_angle)
    inc_angle  *= np.pi/180
    azi_angle  *= np.pi/180
    head_angle *= np.pi/180
    return inc_angle, head_angle