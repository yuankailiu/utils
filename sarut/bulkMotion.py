#!/usr/bin/env python3
############################################################
# This code it meant to examine the products from MintPy
# YKL @ 2021-05-19
############################################################

# Usage:
#   from sarut import bulkMotion

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress


## Load from MintPy
from mintpy.utils import readfile, utils as ut

## Load my codes
import sarut.tools.data as sardata
import sarut.tools.plot as sarplt

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


def sklean_linear_model(x, y, report=False):
    # x need to be reshape(-1,1)
    # Create an instance of a linear regression model
    # and fit it to the data with the fit() function
    fit = LinearRegression().fit(x, y)
    # Obtain the coefficient of determination by calling the model
    # with the score() function, then print the coefficient:
    r_sq = fit.score(x, y)
    y_pred = fit.predict(x)
    if report:
        print('coefficient of determination:', r_sq)
        print('slope:', fit.coef_[0])
        print('intercept:', fit.intercept_)
    return fit, y_pred


def scipy_linear(x, y, report=False):
    fit = linregress(x, y)
    y_pred = fit.intercept + fit.slope * x
    if report:
        print('slope:', fit.slope)
        print('intercept:', fit.intercept)
    return fit, y_pred


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
    fit, y_pred = scipy_linear(x, y)

    # plot
    plt.figure(figsize=[6,6])
    plt.scatter(x, y)
    plt.scatter(x, y_pred, s=0.3, label='y=ax+b \n a={:.3f}±{:.3f}, b={:.3f}'.format(fit.slope, fit.stderr, fit.intercept))
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


def plot_range_ramp(data, slantrange=None, titstr='', vlim=[None, None], plot=True):
    # get range and azimuth bins
    length, width = data.shape
    rbins = np.tile(np.arange(width), (length, 1))
    abins = np.tile(np.arange(length), (width, 1)).T

    # range_bin or slantRangeDistance for x-axis
    if slantrange is None:
        x          = rbins.flatten()
        xlabel     = 'Range bin'
        slope_unit = 'mm/yr/track'
        factor     = float(width)
    else:
        x          = np.array(slantrange).flatten()
        xlabel     = 'Slant range distance [km]'
        slope_unit = 'mm/yr/km'
        factor     = 1.0

    # get flatten data array; non-NaN data and the x-axis
    vflat = data.flatten()
    x     =     x[~np.isnan(vflat)]
    y     = vflat[~np.isnan(vflat)]
    aflat = abins.flatten()[~np.isnan(vflat)]

    # linear fit to the trend
    fit, y_pred  = scipy_linear(x, y)
    params_legend = 'y=ax+b, a={:.3f} ± {:.2e}\n'.format(fit.slope, fit.stderr)

    if slantrange is None:
        params_legend += 'slope = {:.3f} {:s}'.format(fit.slope*factor, slope_unit)
    else:
        range_span = np.max(x) - np.min(x)
        print('Slant range distance spans {:.1f} km'.format(range_span))
        params_legend += 'slope = {:.3f} {:s}'.format(fit.slope*factor, slope_unit)
        params_legend += '\n ({:.3f} mm/yr/track)'.format(fit.slope * range_span)

    if plot:
        plt.figure(figsize=[8,6])
        sc = plt.scatter(x, y, s=0.1, c=aflat)
        plt.plot(x, y_pred, lw=2, label=params_legend, c='r')
        plt.legend(loc='upper left')
        plt.xlabel(xlabel)
        plt.ylabel('LOS velocity [mm/yr]')
        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel(ylabel='Azimuth', rotation=270, labelpad=20)
        plt.title(titstr)
        plt.ylim(vlim[0], vlim[1])
        plt.show()
    return fit.slope*factor


def plot_dem_var(data, dem, titstr='', vlim=[None, None], plot=True):
    # flatten the field for scatter plot
    length, width = data.shape
    abins = np.tile(np.arange(length), (width, 1)).T
    vflat = data.flatten()

    # linear fit to the trend
    x = dem.flatten()
    x = x[~np.isnan(data.flatten())]
    y = flatten_isnotnan(data)
    fit, y_pred = scipy_linear(x, y)
    params_legend = 'y=ax+b \n a={:.3f}±{:.3f}, b={:.3f}'.format(fit.slope, fit.stderr, fit.intercept)

    if plot:
        plt.figure(figsize=[12,8])
        sc = plt.scatter(dem, vflat, s=0.1, c=abins)
        plt.plot(x, y_pred, lw=2, label=params_legend, c='r')
        plt.legend(loc='upper left')
        plt.xlabel('Elevation from DEM [meter]')
        plt.ylabel('LOS velocity [mm/yr]')
        plt.colorbar(sc, label='Azimuth bin')
        plt.title(titstr)
        plt.ylim(vlim[0], vlim[1])
        plt.show()
    return fit.slope * width


def prepare_los_geometry(geom_file):
    """Prepare LOS geometry data/info in geo-coordinates
    Parameters: geom_file  - str, path of geometry file
    Returns:    inc_rad    - 2D np.ndarray, incidence angle in radians
                head_rad   - 2D np.ndarray, heading   angle in radians
                atr        - dict, metadata in geo-coordinate
    """

    print('prepare LOS geometry in geo-coordinates from file: {}'.format(geom_file))
    atr = readfile.read_attribute(geom_file)

    print('read incidenceAngle from file: {}'.format(geom_file))
    inc_deg = readfile.read(geom_file, datasetName='incidenceAngle')[0]

    if 'azimuthAngle' in readfile.get_dataset_list(geom_file):
        print('read azimuthAngle   from file: {}'.format(geom_file))
        print('convert azimuth angle to heading angle')
        azi_deg  = readfile.read(geom_file, datasetName='azimuthAngle')[0]
        head_deg = ut.azimuth2heading_angle(azi_deg)
    if 'slantRangeDistance' in readfile.get_dataset_list(geom_file):
        print('read slantRangeDistance   from file: {}'.format(geom_file))
        print('convert slantRangeDistance from meter to kilometer')
        slantrange  = readfile.read(geom_file, datasetName='slantRangeDistance')[0]
        slantrange /= 1e3
    else:
        print('use the HEADING attribute as the mean heading angle')
        head_deg = np.ones(inc_deg.shape, dtype=np.float32) * float(atr['HEADING'])

    # turn default null value to nan
    inc_deg[inc_deg==0]    = np.nan
    head_deg[head_deg==90] = np.nan
    # unit: degree to radian
    inc_rad  = np.deg2rad(inc_deg)
    head_rad = np.deg2rad(head_deg)
    return inc_rad, head_rad, slantrange, atr


def los_unit_vector(inc_rad, head_rad, in_unit=None):
    """
    Make a plot trigonometry of inc angle and head angle (in radians)

    inc_rad      inc_angle in radians
    head_rad     head_angle in radians
    in_unit      given input unit vector, for direct plotting

    """
    if in_unit is not None:
        unitv = np.array(in_unit)
    else:
        unitv = np.array([-1*np.sin(inc_rad)*np.cos(head_rad),
                             np.sin(inc_rad)*np.sin(head_rad),
                             np.cos(inc_rad)                  ])

    tstrs = ['for E\n-sin(inc)*cos(head)', 'for N\nsin(inc)*sin(head)', 'for U\ncos(inc)']

    if in_unit is not None:
        fig, axs = plt.subplots(nrows=1, ncols=len(unitv), figsize=[12,10], sharey=True, gridspec_kw={'wspace':0.14})
        for i, (u, tstr) in enumerate(zip(unitv, tstrs)):
            im = axs[i].imshow(u)
            fig.colorbar(im, ax=axs[i], fraction=0.05)
            axs[i].set_title(tstr)
        plt.show()
    return unitv


def plot_enulos(v, inc_deg, head_deg, ref=False, display=True, display_more=False):
    # v            [E, N, U] floats; Three-component model motion (ENU); unit: mm/yr
    # inc_deg      an array of floats (length * width); unit: degrees
    # head_deg     an array of floats (length * width); unit: degrees
    # ref          reference pixel for v_los
    v_los = ut.enu2los(v[0], v[1], v[2], inc_angle=inc_deg, head_angle=head_deg)
    disp         = dict()
    disp['inc']  = inc_deg
    disp['head'] = head_deg
    disp['ve']   = v[0] * np.ones_like(inc_deg)
    disp['vn']   = v[1] * np.ones_like(inc_deg)
    disp['vu']   = v[2] * np.ones_like(inc_deg)
    disp['v_los']= v_los    # sign convention: positive for motion towards satellite

    # Take the reference pixel from the middle of the map (for v_los only)
    if ref:
        if ref is True:
            idx = np.array(v_los.shape) // 2
        else:
            idx = ref
        V_ref = disp['v_los'][idx[0], idx[1]]
        if np.isnan(V_ref):
            print('Reference point is NaN, choose another point!')
            sys.exit(1)
        disp['v_los'] -= V_ref

    if display:
        ## Make a quick plot
        fig, axs = plt.subplots(nrows=1, ncols=6, figsize=[24,10], sharey=True, gridspec_kw={'wspace':0.14})
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
        unit_vector = los_unit_vector(np.deg2rad(inc_deg), np.deg2rad(head_deg))


def simple_orbit_geom(orbit, length, width, given_inc='table', given_head='table'):
    # Default table from Aqaba
    # Caveat: heading angle varies a lot! e.g. Australia heading angle = -166.47261 to -164.69743 deg
    table = dict()
    table['ASCENDING'] = {}
    table['ASCENDING']['inc'] = (30.68, 46.24)     # (degrees vary from near_range to far_range)
    table['ASCENDING']['azi'] = (-258.93, -260.52)
    table['DESCENDING'] = {}
    table['DESCENDING']['inc'] = (30.78, 46.28)    # (degrees vary from near_range to far_range)
    table['DESCENDING']['azi'] = (-101.06, -99.43)

    if given_inc == 'table':
        inc_vary = table[orbit]['inc']
    else:
        inc_vary = given_inc

    if given_head == 'table':
        azi_vary = table[orbit]['azi']
    else:
        azi_vary  = (ut.heading2azimuth_angle(given_head[0]), ut.heading2azimuth_angle(given_head[1]))

    inc_deg  = np.tile(np.linspace(*inc_vary, width), (length,1))
    azi_deg  = np.tile(np.linspace(*azi_vary, width), (length,1))
    head_deg = ut.azimuth2heading_angle(azi_deg)   #  literally: 90.0 - azi_deg
    inc_rad  = np.deg2rad(inc_deg)
    azi_rad  = np.deg2rad(azi_deg)
    head_rad = np.deg2rad(head_deg)
    return inc_rad, head_rad


def create_v(v_hor=1, theta=0, v_ver=0):
    """
    v_hor      horizontal absolute motion
    theta      angle of motion vector, clockwise from East is positive
    v_hor      vertical   absolute motion (default=0)
    """
    v = np.array([v_hor*np.cos(np.deg2rad(theta)), v_hor*np.sin(np.deg2rad(theta)), v_ver])
    return v



############################

class bulkMotion():
    """
    Classs to compute the bulk motion on a given geometry
    """

    def __init__(self, geom_file=None):
        ## geom_file: path to geometryGeo.h5 or geometryRadar.h5
        if geom_file is not None:
            # Strongly recommended !!
            # prepare LOS geometry: need to be in radar coord
            self.inc_rad, self.head_rad, self.slantrange, self.atr_geo = prepare_los_geometry(geom_file)
            self.width  = int(self.atr_geo['WIDTH'])
            self.length = int(self.atr_geo['LENGTH'])
            self.orbit  = self.atr_geo['ORBIT_DIRECTION']
        else:
            # Not recommend
            self.given_inc  = True
            self.given_head = True
            self.length     = 500
            self.width      = 300
            self.orbit      = 'Ascending'
            self.guess_geom(self)
        return


    def guess_geom(self, given_inc=False, given_head=False, length=False, width=False):
        # Not recommend
        # given_inc       Fasle, 'table' or a custom tuple (inc0, inc1); assume inc angle from simple guess
        # given_head      False, 'table' or a custom tuple (head0, head1); assume head angle from simple guess
        if given_inc:
            self.given_inc  = given_inc
        if given_head:
            self.given_head = given_head
        if length:
            self.length     = length
        if width:
            self.width      = width
        if self.given_inc:     # try to simply guess the inc
            if self.given_inc is True:
                self.given_inc = 'table'
                print('Use incidence angle from a table...')
            self.inc_rad  = simple_orbit_geom(self.orbit, self.length, self.width, self.given_inc, 'table')[0]
        if self.given_head:    # try to simply guess the head
            if self.given_head is True:
                self.given_head = 'table'
                print('Use head angle from a table...')
            self.head_rad = simple_orbit_geom(self.orbit, self.length, self.width, 'table', self.given_head)[1]


    def enu2los(self, V, ref=False):
        ## V               [ve, vn, vu]; floats; three component motions
        ## ref             reference the vlos to the middle pixel, or a custom pixel
        self.head_deg = np.rad2deg(self.head_rad)
        self.azi_deg  = ut.heading2azimuth_angle(self.head_deg)
        self.inc_deg  = np.rad2deg(self.inc_rad)
        #print('Dynamic range of head_angle:', np.nanmax(self.head_deg), np.nanmin(self.head_deg))

        self.V = V
        self.V_los = ut.enu2los(V[0], V[1], V[2], inc_angle=self.inc_deg, head_angle=self.head_deg)

        self.ref = ref
        if ref:
            if ref is True:
                idx = self.length//2, self.width//2
            else:
                idx = ref[0], ref[1]
            V_ref = self.V_los[idx[0], idx[1]]
            if np.isnan(V_ref):
                print('Reference point is NaN, choose another point!')
                sys.exit(1)
            self.V_los -= V_ref
            self.ref = idx

        res = dict()
        res['inc']   = self.inc_deg
        res['head']  = self.head_deg
        res['ve']    = self.V[0]
        res['vn']    = self.V[1]
        res['vu']    = self.V[2]
        res['v_los'] = self.V_los
        return res


    def plot_inputs(self, ref=False, display=True):
        ## Plot {e, n, u, inc, head, vlos}
        if ref:
            self.ref = ref
        plot_enulos(self.V, self.inc_deg, self.head_deg, self.ref, display=display)


    def plot_ramp(self):
        ## Plot forward model range ramp
        titstr = '{} modeled range ramp\n({}E, {}N mm/y bulk motion)'.format(self.orbit, self.V[0], self.V[1])
        self.rangeslope = plot_range_ramp(self.V_los, self.slantrange, titstr)


    def plot_demshade(self, dem_shade, picdir='./pic', outf='los_test'):
        ## Plot with DEM overlaid (need to have dem_shade input)
        if dem_shade and self.atr_geo:
            sarplt.plot_imgs(v=self.res, meta=self.atr_geo, dem=dem_shade, coord='rdr', picdir=picdir, outf=outf)



############################

# res_d021 = res1 = bulk_motion_los(v3comp, orbit='D', geom_file=geofile, given_inc=False, given_head=False, ref=True)
