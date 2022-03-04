#!/usr/bin/env python3
# --------------------------------------------------------------
# Post-processing for MintPy output results
#
# Yuan-Kai Liu, 2020-3-3
# --------------------------------------------------------------

# Recommended usage:
#   import sarut.tools.plot as sarplt

import os
import sys
import glob
import h5py
import time
import numpy as np
import netCDF4 as nc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


from copy import copy
from scipy import linalg
from datetime import datetime, timedelta
from mintpy.objects import timeseries
from mintpy.tsview import timeseriesViewer
from mintpy.utils import readfile, utils as ut, plot as pp
from mintpy import view #, tsview, plot_network, plot_transection, plot_coherence_matrix


# plot_network code
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mintpy.objects.colors import ColormapExt
from mintpy.utils import ptime

# Inscrese matplotlib font size when plotting
plt.rcParams.update({'font.size': 16})


#############################################################

# ----------------------------------------------
# Path settings
# ----------------------------------------------
if False:
    # where my MintPy dataset stored (my project directory)
    proj_dir = os.path.expanduser(os.getcwd())

    # where to store generated figures
    pic_dir = os.path.expanduser(f'{proj_dir}/pic_pproc')

    # time-series files to be loaded
    ts_file = os.path.expanduser(f'{proj_dir}/timeseries.h5')

    # read water mask file
    mask_file = os.path.expanduser(f'{proj_dir}/waterMask.h5')

    # go to project directory
    os.chdir(proj_dir)

    # create pic directory if not exist
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)

    # print paths
    print(f'MintPy project directory:\t{proj_dir}')
    print(f'Pictures will be saved to:\t{pic_dir}')
    print(f'Your current directory:\t\t{os.getcwd()}')


#################################################################

## modified from mintpy plot_network function
## now can plot network and mark the gaps. see `find_network_gap.py`
## to-do: combine plotting funcs down below to this func (2022-3-3 ykl)
def plot_network(ax, date12List, dateList, pbaseList, p_dict={}, date12List_drop=[], print_msg=True):
    """Plot Temporal-Perp baseline Network
    Inputs
        ax : matplotlib axes object
        date12List : list of string for date12 in YYYYMMDD_YYYYMMDD format
        dateList   : list of string, for date in YYYYMMDD format
        pbaseList  : list of float, perp baseline, len=number of acquisition
        p_dict   : dictionary with the following items:
                      fontsize
                      linewidth
                      markercolor
                      markersize

                      cohList : list of float, coherence value of each interferogram, len = number of ifgrams
                      colormap : string, colormap name
                      disp_title : bool, show figure title or not, default: True
                      disp_drop: bool, show dropped interferograms or not, default: True
    Output
        ax : matplotlib axes object
    """

    # Figure Setting
    if 'fontsize'    not in p_dict.keys():  p_dict['fontsize']    = 12
    if 'linewidth'   not in p_dict.keys():  p_dict['linewidth']   = 2
    if 'markercolor' not in p_dict.keys():  p_dict['markercolor'] = 'orange'
    if 'markersize'  not in p_dict.keys():  p_dict['markersize']  = 12

    # For colorful display of coherence
    if 'cohList'     not in p_dict.keys():  p_dict['cohList']     = None
    if 'xlabel'      not in p_dict.keys():  p_dict['xlabel']      = None #'Time [years]'
    if 'ylabel'      not in p_dict.keys():  p_dict['ylabel']      = 'Perp Baseline [m]'
    if 'cbar_label'  not in p_dict.keys():  p_dict['cbar_label']  = 'Average Spatial Coherence'
    if 'cbar_size'   not in p_dict.keys():  p_dict['cbar_size']   = '3%'
    if 'disp_cbar'   not in p_dict.keys():  p_dict['disp_cbar']   = True
    if 'colormap'    not in p_dict.keys():  p_dict['colormap']    = 'RdBu'
    if 'vlim'        not in p_dict.keys():  p_dict['vlim']        = [0.2, 1.0]
    if 'disp_title'  not in p_dict.keys():  p_dict['disp_title']  = True
    if 'disp_drop'   not in p_dict.keys():  p_dict['disp_drop']   = True
    if 'disp_legend' not in p_dict.keys():  p_dict['disp_legend'] = True
    if 'every_year'  not in p_dict.keys():  p_dict['every_year']  = 1
    if 'number'      not in p_dict.keys():  p_dict['number']      = None

    # support input colormap: string for colormap name, or colormap object directly
    if isinstance(p_dict['colormap'], str):
        cmap = ColormapExt(p_dict['colormap']).colormap
    elif isinstance(p_dict['colormap'], mpl.colors.LinearSegmentedColormap):
        cmap = p_dict['colormap']
    else:
        raise ValueError('unrecognized colormap input: {}'.format(p_dict['colormap']))

    cohList = p_dict['cohList']
    transparency = 0.7

    # Date Convert
    dateList = ptime.yyyymmdd(sorted(dateList))
    dates, datevector = ptime.date_list2vector(dateList)
    tbaseList = ptime.date_list2tbase(dateList)[0]

    ## maxBperp and maxBtemp
    date12List = ptime.yyyymmdd_date12(date12List)
    ifgram_num = len(date12List)
    pbase12 = np.zeros(ifgram_num)
    tbase12 = np.zeros(ifgram_num)
    for i in range(ifgram_num):
        m_date, s_date = date12List[i].split('_')
        m_idx = dateList.index(m_date)
        s_idx = dateList.index(s_date)
        pbase12[i] = pbaseList[s_idx] - pbaseList[m_idx]
        tbase12[i] = tbaseList[s_idx] - tbaseList[m_idx]
    if print_msg:
        print('max perpendicular baseline: {:.2f} m'.format(np.max(np.abs(pbase12))))
        print('max temporal      baseline: {} days'.format(np.max(tbase12)))

    ## Keep/Drop - date12
    date12List_keep = sorted(list(set(date12List) - set(date12List_drop)))
    if not date12List_drop:
        p_dict['disp_drop'] = False

    ## Keep/Drop - date
    m_dates = [i.split('_')[0] for i in date12List_keep]
    s_dates = [i.split('_')[1] for i in date12List_keep]
    dateList_keep = ptime.yyyymmdd(sorted(list(set(m_dates + s_dates))))
    dateList_drop = sorted(list(set(dateList) - set(dateList_keep)))
    idx_date_keep = [dateList.index(i) for i in dateList_keep]
    idx_date_drop = [dateList.index(i) for i in dateList_drop]

    # Ploting
    disp_min = p_dict['vlim'][0]
    disp_max = p_dict['vlim'][1]
    if cohList is not None:
        data_min = min(cohList)
        data_max = max(cohList)
        if print_msg:
            print('showing coherence')
            print('data range: {}'.format([data_min, data_max]))
            print('display range: {}'.format(p_dict['vlim']))

        # plot low coherent ifgram first and high coherence ifgram later
        cohList_keep = [cohList[date12List.index(i)] for i in date12List_keep]
        date12List_keep = [x for _, x in sorted(zip(cohList_keep, date12List_keep))]

    if p_dict['disp_cbar']:
        cax = make_axes_locatable(ax).append_axes("right", p_dict['cbar_size'], pad=p_dict['cbar_size'])
        norm = mpl.colors.Normalize(vmin=disp_min, vmax=disp_max)
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cbar.ax.tick_params(labelsize=p_dict['fontsize'])
        cbar.set_label(p_dict['cbar_label'], fontsize=p_dict['fontsize'], rotation=270, labelpad=25)

    # Dot - SAR Acquisition
    if idx_date_keep:
        x_list = [dates[i] for i in idx_date_keep]
        y_list = [pbaseList[i] for i in idx_date_keep]
        if isinstance(p_dict['markercolor'], str):
            ax.plot(x_list, y_list, 'ko', alpha=0.7, ms=p_dict['markersize'], mfc=p_dict['markercolor'])
        elif isinstance(p_dict['markercolor'], (list, tuple, np.ndarray)):
            ax.scatter(x_list, y_list, s=p_dict['markersize']**2, marker='o', c=p_dict['markercolor'], cmap=p_dict['colormap'], alpha=0.7, edgecolors='k')

    if idx_date_drop:
        x_list = [dates[i] for i in idx_date_drop]
        y_list = [pbaseList[i] for i in idx_date_drop]
        ax.plot(x_list, y_list, 'ko', alpha=0.7, ms=p_dict['markersize'], mfc='r')

    ## Line - Pair/Interferogram
    # interferograms dropped
    if p_dict['disp_drop']:
        for date12 in date12List_drop:
            date1, date2 = date12.split('_')
            idx1 = dateList.index(date1)
            idx2 = dateList.index(date2)
            x = np.array([dates[idx1], dates[idx2]])
            y = np.array([pbaseList[idx1], pbaseList[idx2]])
            if cohList is not None:
                val = cohList[date12List.index(date12)]
                val_norm = (val - disp_min) / (disp_max - disp_min)
                ax.plot(x, y, '--', lw=p_dict['linewidth'], alpha=transparency, c=cmap(val_norm))
            else:
                ax.plot(x, y, '--', lw=p_dict['linewidth'], alpha=transparency, c='r')

    # interferograms kept
    for date12 in date12List_keep:
        date1, date2 = date12.split('_')
        idx1 = dateList.index(date1)
        idx2 = dateList.index(date2)
        x = np.array([dates[idx1], dates[idx2]])
        y = np.array([pbaseList[idx1], pbaseList[idx2]])
        if cohList is not None:
            val = cohList[date12List.index(date12)]
            val_norm = (val - disp_min) / (disp_max - disp_min)
            ax.plot(x, y, '-', lw=p_dict['linewidth'], alpha=transparency, c=cmap(val_norm))
        else:
            ax.plot(x, y, '-', lw=p_dict['linewidth'], alpha=transparency, c='k')

    if p_dict['disp_title']:
        ax.set_title('Interferogram Network', fontsize=p_dict['fontsize'])

    # axis format
    ax = pp.auto_adjust_xaxis_date(ax, datevector, fontsize=p_dict['fontsize'],
                                every_year=p_dict['every_year'])[0]
    ax = pp.auto_adjust_yaxis(ax, pbaseList, fontsize=p_dict['fontsize'])
    ax.set_xlabel(p_dict['xlabel'], fontsize=p_dict['fontsize'])
    ax.set_ylabel(p_dict['ylabel'], fontsize=p_dict['fontsize'])
    ax.tick_params(which='both', direction='in', labelsize=p_dict['fontsize'],
                   bottom=True, top=True, left=True, right=True)

    if p_dict['number'] is not None:
        ax.annotate(p_dict['number'], xy=(0.03, 0.92), color='k',
                    xycoords='axes fraction', fontsize=p_dict['fontsize'])

    # Legend
    if p_dict['disp_drop'] and p_dict['disp_legend']:
        solid_line = mpl.lines.Line2D([], [], color='k', ls='solid',  label='Ifgram used')
        dash_line  = mpl.lines.Line2D([], [], color='k', ls='dashed', label='Ifgram dropped')
        ax.legend(handles=[solid_line, dash_line])

    return ax, cbar


## plot ITS_LIVE NetCDF files
## https://its-live.jpl.nasa.gov/
def plot_ITS_LIVE_nc(ncfile, exclist):
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



#################################################################

# ----------------------------------------------
# Define miscelaneous functions
# ----------------------------------------------
def deciyr2date(start):
    # Convert decimal years to python date
    year = int(start)
    rem  = start - year
    base = datetime(year, 1, 1)
    result = base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)
    return result

def date2deciyr(date):
    # Convert python date to decimal years
    yr_list = date.year + (date.timetuple().tm_yday - 1) / 365.25
    deciyr  = np.array(yr_list)
    return deciyr

# Read time-series geometry
def read_geo(fname, verbose=False):
    print('reading {}'.format(fname))
    with h5py.File(fname, 'r') as f:
        meta = dict(f.attrs)
        heading   = float(meta['HEADING'])
        incidence = float(meta['incidenceAngle'])
        orbit     = meta['ORBIT_DIRECTION']
        wl        = float(meta['WAVELENGTH']) # meter
        length    = int(meta['LENGTH'])
        width     = int(meta['WIDTH'])
        x_min     = float(meta['X_FIRST'])
        x_step    = float(meta['X_STEP'])
        y_min     = float(meta['Y_FIRST'])
        y_step    = float(meta['Y_STEP'])
        lats      = np.arange(y_min,length*y_step+y_min, y_step)
        lons      = np.arange(x_min, width*x_step+x_min, x_step)
        ref_lat   = float(meta['REF_LAT'])
        ref_lon   = float(meta['REF_LON'])
    if verbose is True:
        print(f'\n\tSentinel-1: {orbit}')
        print('\tWavelength = {:.4f} cm'.format(wl*100))
        print('\tHeading = {:.4f} deg\n\tIncidence Angle = {:.4f} deg'.format(heading, incidence))
        print('\tLength = {:}\n\tWidth = {:}'.format(length, width))
        print('\tReference lat \t= {:.4f}\n\tReference lon = {:.4f}'.format(ref_lat, ref_lon))
    return lats, lons, length, width, ref_lat, ref_lon


def get_POI(points, geom_file, out_file, vfile, maskfile, coord=0, reverse=0):
    # Sort the pixels by coordinate
    # coord = 0  # 0 for y-coordinate; 1 for x corrdinate
    # reverse = 0  # 0 for non-reversed; 1 for reversed order

    if reverse == 0:
        points = points[np.argsort(points[:,coord]),:]
    elif reverse == 1:
        points = points[np.argsort(points[:,coord]),:][::-1]

    # get geometry info
    lats, lons, length, width, ref_lat, ref_lon = read_geo(geom_file, verbose=False)

    # Save user selected pixels to txt files
    plats = lats[points[:,0]]
    plons = lons[points[:,1]]
    txtfile = out_file

    tmp=[]
    with open(txtfile, 'w+') as file:
        for i in range(len(plats)):
            tmp = [plats[i], plons[i]]
            file.write('%s  '*2%(tmp[0],tmp[1])+'\n')
        print(f'Complete writing to file: {txtfile}')

    file = [f'{vfile}', 'velocity']
    opt  = ['--pts-marker', 'r^', '--pts-ms', '6', '--mask', f'{maskfile}', '-v', '-3', '3', '-c', 'RdBu_r']
    pts  = ['--pts-file', f'{txtfile}']

    view.main(file+opt+pts)

# ---------------------------------------------------------
# Below are functions not being used, for tsview files I/O
# ---------------------------------------------------------

# Visualization functions
def tsview(fname, yx=None, ptsfile=None):
    """Plot input file using tsview.py"""
    cmd = 'tsview.py {} --ms 4 --ylim -20 20 --multilook-num 10 --save'.format(fname)
    if yx is not None:
        cmd += ' --yx {} {}'.format(yx[0], yx[1])
    if ptsfile is not None:
        cmd += ' --pts-file {} --pts-ms 4'.format(ptsfile)
    #print(cmd)
    obj = timeseriesViewer(cmd)
    obj.configure()
    obj.figsize_img = [5, 4]
    obj.figsize_pts = [5, 2]
    obj.plot()


# Read from tsview profiles output files
def read_tsprofs(profDir):
    files = sorted(glob.glob(f'{profDir}/*.txt'))
    time = []
    disp = []
    py   = []
    px   = []
    lat  = []
    lon  = []
    for i in range(len(files)):
        file = open(files[i],'r')
        lines = file.readlines()
        t = []
        d = []
        for x in lines:
            if x[0] == '#':
                if x[2:5]=='Y/X':
                    tmp = x.split()
                    py.append(tmp[3][:-1])
                    px.append(tmp[4][:-1])
                    lat.append(tmp[7][:-1])
                    lon.append(tmp[8])
                else:
                    continue
            else:
                tmp = x.split()
                t.append(tmp[0])
                d.append(tmp[1])
        if i == 0:
            time = np.array(t)
            disp = np.array(d)
        else:
            time = np.vstack((np.array(time), np.array(t)))
            disp = np.vstack((np.array(disp), np.array(d)))
    disp = np.array(disp).astype(np.float)
    py = np.array(py).astype(np.float)
    px = np.array(px).astype(np.float)
    lat = np.array(lat).astype(np.float)
    lon = np.array(lon).astype(np.float)
    return time, disp, py, px, lat, lon


# Jointly plot all the time-series 1D profiles
def plot_tsprofs(profDir):
    time, disp, py, px, lat, lon = read_tsprofs(profDir)
    # Make plots
    from matplotlib.dates import (YEARLY, MONTHLY, DateFormatter, rrulewrapper, RRuleLocator)
    import datetime
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(nrows=6, ncols=2, figsize=[12,16])
    j=0
    for row in ax:
        for col in row:
            try:
                t = time[j,:]
                y = disp[j,:]
                dates = []
                days = np.zeros(len(t))
                for i in range(len(t)):
                    dates.append(datetime.date(np.int(t[i][:4]), np.int(t[i][4:6]), np.int(t[i][6:8])))
                    days[i] = dates[i].toordinal()
                import string
                alphabet = string.ascii_uppercase
                pname = list(alphabet)
                yloc = RRuleLocator(rrulewrapper(YEARLY, byyearday=1))
                mloc = RRuleLocator(rrulewrapper(MONTHLY, bymonthday=1))
                formatter = DateFormatter('%Y')
                col.scatter(days, y, s=10, label=pname[j])
                col.set_ylim(np.mean(y)-25, np.mean(y)+25)
                col.xaxis.set_major_locator(yloc)
                col.xaxis.set_minor_locator(mloc)
                col.xaxis.set_major_formatter(formatter)
                col.legend(loc='upper right')
                col.set_ylabel('Displacement [cm]')
                col.set_title(f'lat={lat[j]}, lon={lon[j]}')
            except:
                col.axis('off')
            j+=1

    plt.tight_layout()
    plt.show()


# Plot one slice of the time-series
def imageSlider(ts_file, dset='timeseries-20150304', mask=mask_file, points=None, return_v=False):
    import matplotlib.gridspec as gridspec
    plt.rcParams.update({'font.size': 16})

    # read ts_file
    ts_obj = timeseries(ts_file)
    ts_obj.open()
    ts_data = readfile.read(ts_file, datasetName=dset)[0].astype("float") * 100. #cm
    numDate = ts_obj.numDate
    dateList = ts_obj.dateList

    # read geometry
    lats, lons, length, width, ref_lat, ref_lon = read_geo(ts_file)

    # read mask
    if mask is not None:
        mask_data = readfile.read(mask)[0]
        ts_data[mask_data==0] = np.nan

    # plot
    vmin = -10
    vmax =  10
    cmap=copy(plt.cm.jet)
    cmap.set_bad(color='white')
    fig = plt.figure(figsize=[20, 10])
    ax0 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=2)
    im = ax0.imshow(ts_data[:, :], extent=[lons[0],lons[-1],lats[-1],lats[0]], vmin=vmin, vmax=vmax, cmap=cmap)
    ax0.set_title('{} residual'.format(dset))
    plt.colorbar(im, ax=ax0, cmap=cmap, extend='both', aspect=50, shrink=0.8, pad=0.01, label='Displacement [cm]')
    if points is not None: # read tsview output files for lon lat of custom pixels
        _, _, _, _, plat, plon = read_tsprofs(points)
        import string
        alphabet = string.ascii_uppercase
        pname = list(alphabet)
        for i in range(len(plat)):
            ax0.scatter(plon[i], plat[i], s=40, marker='^', c='red', edgecolor='k')
            ax0.text(plon[i], plat[i], pname[i], fontsize=16)
    ax0.scatter(ref_lon, ref_lat, marker='s', s=50, c='k')
    # plot scale bar
    sblon = 34 # deg, lon location
    sblat = 27 # deg, lat location
    sbr   = 99 # km, distance for 1 longitude deg at sblat
    sblen = 80 # km, total length
    ax0.plot(np.array([sblon, sblon+sblen/sbr]), np.array([sblat, sblat]), linewidth=4, c='k')
    ax0.text(0.5*(sblon+sblon+sblen/sbr), sblat-0.3, f'{sblen} km', ha='center')

    data = ts_data[:, :]
    data = data[~np.isnan(data)]
    data_rms = np.sqrt(np.sum(data**2)/len(data))
    ax1 = plt.subplot2grid((2,2), (0,1))
    bins = np.linspace(np.min(data), np.max(data), 500)
    ax1.hist(data, bins, density=True)
    ax1.set_xlim([-10,10])
    ax1.set_ylim([0,0.6])
    ax1.set_xlabel('Residual [cm]')
    ax1.set_ylabel('Density')
    ax1.set_title('Residual distribution, RMS={:.2f}'.format(data_rms))
    plt.show()
    if return_v is True:
        return ts_data



# ---------------------------------------------------------
# Time function fitting (for time-series data)
# ---------------------------------------------------------
# FILL IN (only inside the functions)

# columns of the G matrix for a polynomial model
def get_polynomial(dt, degree=1):
    # input:
    #  - dt of shape (nt, ) in years since refDate
    # output:
    #  - Gcols of shape (nt, degree + 1) of the linear coefficients
    Gcols = np.empty((dt.size, degree + 1))
    for i in range(degree+1):
        Gcols[:,i] = dt**i
    return Gcols

# columns of the G matrix for a bilinear model (planar model)
def get_bilinear(dx, dy):
    # input:
    #  - dx of shape (nt, ) in length
    #  - dy of shape (nt, ) in length
    # output:
    #  - Gcols of shape (nt, 3) of the bilinear coefficients
    Gcols = np.empty((dx.size, 3))
    Gcols[:,0] = np.ones_like(dx)
    Gcols[:,1] = np.array(dx)
    Gcols[:,2] = np.array(dy)
    return Gcols

# columns of the G matrix for a sinusoid model
def get_sinusoidal(dt, period=1):
    # input:
    #  - dt of shape (nt, ) in years since refDate
    #  - period in years
    # output:
    #  - Gcols of shape (nt, 2) of the sin and cos coefficients
    Gcols = np.empty((dt.size, 2))
    Gcols[:,0] = np.cos(dt*2*np.pi/period)
    Gcols[:,1] = np.sin(dt*2*np.pi/period)
    return Gcols

# columns of the G matrix for a step function
def get_step(dt, tstep):
    # input:
    #  - dt of shape (nt, ) in years since refDate
    #  - step time tstep in years since refDate
    # output:
    #  - Gcols of shape (nt, 1) of the step coefficients (zeros and ones)
    Gcols = np.zeros(dt.size)
    idx = (np.abs(dt-tstep)).argmin()
    Gcols[idx:] = 1
    return Gcols

# here, we combine all the individual models
def get_G(timevector, poly_deg, periods=[], steps=[], bilinear=False, refDate=None, eventimev=False):
    # input:
    #  - timevector of shape (nt, ) in years
    #  - integer polynomial degree (0 for a constant, 1 for linear, 2 for quadratic, etc.)
    #  - a list of sinusoidal periods (for example [1, 0.5] for annual and semiannual)
    #  - a list of step times (for example [2006.0, 2007.0] for the 2006 and 2007 New Year's) in decimal years
    # output:
    #  - G matrix of shape (nt, polynomial degree+1 + 2*number of seasonals + number of logarithms + number of steps)


    if bilinear is False:
        if refDate is None:
            refDate = timevector[0]
        if eventimev is True:
            dts = []
            new_dates = []
            for i in range(len(timevector)-1):
                dts.append(timevector[i+1]-timevector[i])
            dtt = np.min(np.array(dts))
            ndt = int((timevector[-1] - timevector[0])/dtt)
            timevector = np.linspace(timevector[0], timevector[-1], ndt)
            for i in range(len(timevector)):
                new_dates.append(deciyr2date(timevector[i]))
        dt = timevector - refDate

        num_params = (poly_deg + 1) + 2*len(periods) + len(steps)
        G = np.empty((dt.size, num_params))
        # insert linear part
        G[:, 0:poly_deg+1] = get_polynomial(dt, degree=poly_deg)
        # insert all seasonal parts
        icol = poly_deg + 1
        for per in periods:
            G[:, icol:icol+2] = get_sinusoidal(dt, period=per)
            icol += 2
        # insert all steps
        for tstep in steps:
            G[:, icol] = get_step(dt, tstep=tstep-refDate)
            icol += 1

    if bilinear is True:
        dx = timevector[0]
        dy = timevector[1]
        G =  get_bilinear(dx, dy)
    # return
    if eventimev is True:
        return G, new_dates
    else:
        return G

def inv_param(t, dis, model, bilinear=False, eventimev=False):
    # set G matrix, linear (degree=1)
    if bilinear is True:
        G = get_G(t, [], [], [], bilinear=bilinear, eventimev=eventimev)
    else:
        poly_deg = model['polynomial']
        periods  = model['periodic']
        steps    = model['step']
        G = get_G(t, poly_deg, periods, steps, bilinear=bilinear, eventimev=eventimev)

    # solve for model parameters (intercept, slope, periodic coefficients, step func amp, etc.)
    # m, e2 = np.linalg.lstsq(G, dis, rcond=None)    # numpy
    m, e2 = linalg.lstsq(G, dis)[:2]             # scipy
    return G, m, e2

def get_model(poly_deg, periods, steps):
    model = dict()
    model['polynomial'] = poly_deg
    model['periodic']   = periods
    model['step']       = steps
    num_period = len(periods)
    num_step   = len(steps)
    num_param  = (poly_deg + 1) + (2 * num_period) + num_step
    return model, num_param



# ---------------------------------------------------------
# Time-series cleanup plot
# ---------------------------------------------------------

def date2decimalyear(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch
    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)
    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration
    return date.year + fraction

def plot_ts_step(ts_file, ts_file_tro, ts_file_set, ts_file_dem, pts_file, pts, fig_dpi=300):
    ts_file = os.path.expanduser(f'{proj_dir}/{ts_file}')
    ts_file_tro = os.path.expanduser(f'{proj_dir}/{ts_file_tro}')
    ts_file_dem = os.path.expanduser(f'{proj_dir}/{ts_file_dem}')

    # give a point to plot
    import string
    pix_name = list(string.ascii_uppercase)
    idx = pix_name.index(pts)

    # read points file
    file = open(f'{pts_file}','r')
    lines = file.readlines()
    i=0
    for x in lines:
        if i==idx:
            tmp = x.split()
            plat, plon = float(tmp[0]), float(tmp[1])
        i+=1
    print('Latitude %.02f   Longitude %.02f' % (plat, plon))


    # read data
    dates, dis_raw = ut.read_timeseries_lalo(lat=plat, lon=plon, ts_file=ts_file, win_size=1, unit='cm', print_msg=False)
    dates, dis_raw_tro = ut.read_timeseries_lalo(lat=plat, lon=plon, ts_file=ts_file_tro, win_size=1, unit='cm', print_msg=False)
    dates, dis_raw_tro_set = ut.read_timeseries_lalo(lat=plat, lon=plon, ts_file=ts_file_set, win_size=1, unit='cm', print_msg=False)
    dates, dis_raw_tro_set_dem = ut.read_timeseries_lalo(lat=plat, lon=plon, ts_file=ts_file_dem, win_size=1, unit='cm', print_msg=False)

    dis_raw *= 10               # mm
    dis_raw_tro *= 10           # mm
    dis_raw_tro_set *= 10       # mm
    dis_raw_tro_set_dem *= 10   # mm

    dis_final   = dis_raw_tro_set_dem
    delay_dem   = dis_raw_tro_set - dis_raw_tro_set_dem
    delay_set   = dis_raw_tro     - dis_raw_tro_set
    delay_tro   = dis_raw         - dis_raw_tro

    test         = dis_final+delay_dem+delay_set+delay_tro - dis_raw
    print('Testing: ', np.sum(test**2))

    # Convert decimal years
    yr_list = [i.year + (i.timetuple().tm_yday - 1) / 365.25 for i in dates]
    yr_diff = np.array(yr_list)

    # Get line fits for raw timeseries and tropo-corrected timeseries
    poly_degree = 1
    periods = [1.0, 0.5]
    steps       = []
    model, num_param = get_model(poly_degree, periods, steps)
    num_obs = len(yr_diff)

    ## raw timeseries
    G, m, e2 = inv_param(yr_diff, dis_raw, model)
    v0 = m[1]
    G, new_dates = get_G(yr_diff, poly_degree, periods, steps, eventimev=True)
    dis_est0 = np.dot(G, m)
    rms0 = np.sqrt(e2/num_obs)
    G_inv = linalg.inv(np.dot(G.T, G))
    var_param = e2.reshape(1, -1) / (num_obs - num_param)
    m_std0 = np.sqrt(np.dot(np.diag(G_inv).reshape(-1, 1), var_param))

    ## tropo-corrected timeseries
    G, m, e2 = inv_param(yr_diff, dis_raw_tro, model)
    v1 = m[1]
    G, new_dates = get_G(yr_diff, poly_degree, periods, steps, eventimev=True)
    dis_est1 = np.dot(G, m)
    rms1 = np.sqrt(e2/num_obs)
    G_inv = linalg.inv(np.dot(G.T, G))
    var_param = e2.reshape(1, -1) / (num_obs - num_param)
    m_std1 = np.sqrt(np.dot(np.diag(G_inv).reshape(-1, 1), var_param))

    ## tropo-corrected, SET corrected timeseries
    G, m, e2 = inv_param(yr_diff, dis_raw_tro_set, model)
    v2 = m[1]
    G, new_dates = get_G(yr_diff, poly_degree, periods, steps, eventimev=True)
    dis_est2 = np.dot(G, m)
    rms2 = np.sqrt(e2/num_obs)
    G_inv = linalg.inv(np.dot(G.T, G))
    var_param = e2.reshape(1, -1) / (num_obs - num_param)
    m_std2 = np.sqrt(np.dot(np.diag(G_inv).reshape(-1, 1), var_param))

    ## tropo-corrected, SET corrected, dem corrected timeseries (final ts)
    G, m, e2 = inv_param(yr_diff, dis_final, model)
    resid = dis_final - np.dot(G, m) # get residual at each sample
    v3 = m[1]
    G, new_dates = get_G(yr_diff, poly_degree, periods, steps, eventimev=True)
    dis_est3 = np.dot(G, m)
    rms3 = np.sqrt(e2/num_obs)
    G_inv = linalg.inv(np.dot(G.T, G))
    var_param = e2.reshape(1, -1) / (num_obs - num_param)
    m_std3 = np.sqrt(np.dot(np.diag(G_inv).reshape(-1, 1), var_param))


    ## plot
    font_size=16
    plt.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots(figsize=[12,24])
    s  = 120
    lw = .8
    marker = '^'
    space = 120
    ax.scatter(dates, dis_raw          +space*7  , marker='^', s=s, ec='k', linewidth=lw, label=f'Raw time series')
    ax.scatter(dates, delay_tro        +space*6  , marker='.', s=s,         linewidth=lw, label=f'delay of ERA5')
    ax.scatter(dates, dis_raw_tro      +space*5  , marker='^', s=s, ec='k', linewidth=lw, label=f'Corrected 1 (Raw $-$ ERA5)')
    ax.scatter(dates, delay_set        +space*4  , marker='.', s=s,         linewidth=lw, label=f'delay of SET')
    ax.scatter(dates, dis_raw_tro_set  +space*3  , marker='^', s=s, ec='k', linewidth=lw, label=f'Corrected 2 (Corrected 1 $-$ SET)')
    ax.scatter(dates, delay_dem        +space*2  , marker='.', s=s,         linewidth=lw, label=f'delay of DEM error')
    ax.scatter(dates, dis_final        +space*1  , marker='^', s=s, ec='k', linewidth=lw, label=f'Corrected 3 (Corrected 3 $-$ DEM error)')
    ax.scatter(dates, resid            +space*0  , marker='s', s=s, ec='k', linewidth=lw, label=f'Final residuals')

    ax.plot(new_dates, [+15]*len(new_dates), '--', lw=1, c='k')
    ax.plot(new_dates, [-15]*len(new_dates), '--', lw=1, c='k')

    #ax.plot(new_dates, dis_est0 +space*7, '-', lw=2, c='k')
    #ax.plot(new_dates, dis_est1 +space*5, '-', lw=2, c='k')
    #ax.plot(new_dates, dis_est2 +space*3, '-', lw=2, c='k')
    #ax.plot(new_dates, dis_est3 +space*1, '-', lw=2, c='k')
    txtloc = 10

    #ax.text(dates[txtloc], dis_est0[txtloc]+space*7+space/3, 'V={:.2f}±{:.2f} mm/yr, RMSR={:.2f} mm'.format(v0, m_std0[1,0], rms0))
    #ax.text(dates[txtloc], dis_est1[txtloc]+space*5+space/3, 'V={:.2f}±{:.2f} mm/yr, RMSR={:.2f} mm'.format(v1, m_std1[1,0], rms1))
    #ax.text(dates[txtloc], dis_est2[txtloc]+space*3+space/3, 'V={:.2f}±{:.2f} mm/yr, RMSR={:.2f} mm'.format(v2, m_std2[1,0], rms2))
    #ax.text(dates[txtloc], dis_est3[txtloc]+space*1+space/3, 'V={:.2f}±{:.2f} mm/yr, RMSR={:.2f} mm'.format(v3, m_std3[1,0], rms3))

    # axis format
    pp.auto_adjust_xaxis_date(ax, dates, fontsize=font_size)
    ax.tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True)
    #ax.set_ylim(np.min(dis_final)-50, np.max(dis_raw)+100*6)
    ax.set_xlim(dates[0], dates[-1])
    ax.set_ylim(-space/2, 8*space)
    ax.legend(bbox_to_anchor=(1,1), loc='upper left')

    ax.set_xlabel('Time [year]', fontsize=20)
    ax.set_ylabel('Relative displacement of each time series [mm]', fontsize=20)
    ax.set_title(f'Time-series displacement at pixel {pts}')

    # output
    out_file = f'{pic_dir}/pixel{pts}_ts_processing.png'
    plt.savefig(out_file, bbox_inches='tight', transparent=True, dpi=fig_dpi)
    print('save to file: '+out_file)
    plt.show()



# ---------------------------------------------------------
# Time-series velocity estimation from given points
# ---------------------------------------------------------

def dtstr(datestr):
    if isinstance(datestr, list) == False:
        if len(datestr) != 8:
            print('Date input error: enter YYYYMMDD string format')
            sys.exit()
        y = int(datestr[:4])
        m = int(datestr[4:6])
        d = int(datestr[6:8])
        dt = datetime(y,m,d)
    elif isinstance(datestr, list) == True:
        dt = []
        for i in range(len(datestr)):
            if len(datestr[i]) != 8:
                print('Date input error: enter YYYYMMDD string format')
                sys.exit()
            y = int(datestr[i][:4])
            m = int(datestr[i][4:6])
            d = int(datestr[i][6:8])
            dt.append(datetime(y,m,d))
    return dt


def checkdt(dates, dis, startdt=None, enddt=None, excdt=[]):
    yr_list = []
    new_dis = []
    new_dates = []
    for i in range(len(dates)):
        dd = dates[i]
        if startdt is not None:
            if dd<startdt:
                #print('exclude date {:04d}{:02d}{:02d}'.format(dd.year,dd.month,dd.day))
                continue
        if enddt is not None:
            if dd>enddt:
                #print('exclude date {:04d}{:02d}{:02d}'.format(dd.year,dd.month,dd.day))
                continue
        if len(excdt) != 0:
            if dd in excdt:
                #print('exclude date {:04d}{:02d}{:02d}'.format(dd.year,dd.month,dd.day))
                continue
        yr_list.append(dd.year + (dd.timetuple().tm_yday - 1) / 365.25)
        new_dis.append(dis[i])
        new_dates.append(dates[i])
    yr_diff = np.array(yr_list)
    new_dis = np.array(new_dis)
    return yr_diff, new_dis, new_dates


def plot_timeseries(ts_file, pts, folder='linear', figtitle='Time-series', startDate=None, endDate=None, excDate=[], fig_dpi=300):
    ts_file = os.path.expanduser(f'{proj_dir}/{ts_file}')
    pts_file = f'pixels_{pts}.txt'

    # read points file
    file = open(f'{pts_file}','r')
    lines = file.readlines()
    plats = []
    plons = []
    for x in lines:
        tmp = x.split()
        plats.append(float(tmp[0]))
        plons.append(float(tmp[1]))

    # points name
    import string
    alphabet = string.ascii_uppercase
    pix_name = list(alphabet)

    # plotting setup
    plot_fitting = 1
    plot_residual = 1
    font_size=16
    plt.rcParams.update({'font.size': font_size})
    nrows, ncols = int(np.round(len(plats)/2)), 2

    # save values for the residual plot
    dis_resids=[]
    rmsrs=[]

    # models (default: linear fitting)
    poly_degree=1
    periods=[]
    steps=[]
    if folder == 'linear':
        print('Linear fitting')
    elif folder == 'periodic':
        print('Linear + periodic fitting')
        periods=[1.0, 0.5]
    else:
        print('Erorr: wrong model input')

    # get velocity model
    model, num_param = get_model(poly_degree, periods, steps)

    # Dates modification
    if startDate is not None:
        print(f'Start date: {startDate}')
        startdt = dtstr(startDate)
    else:
        startdt = None
    if endDate is not None:
        print(f'End date: {endDate}')
        enddt = dtstr(endDate)
    else:
        enddt = None
    if excDate != []:
        print(f'Exclude date(s): {excDate}')
        excdt = dtstr(excDate)
    else:
        excdt = []


    # plots
    vmin = -12
    vmax =  12
    if plot_fitting == 1:
        fname = f'ts2velo_{pts}_{folder}'
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, gridspec_kw={'hspace':0.1, 'wspace':0.02}, figsize=[ncols*8,nrows*2])
        i=0
        for row in ax:
            for col in row:
                lat = plats[i]
                lon = plons[i]
                Dates, Dis = ut.read_timeseries_lalo(lat=lat, lon=lon, ts_file=ts_file, win_size=1, unit='cm', print_msg=False)

                yr_diff, dis, dates = checkdt(Dates, Dis)
                num_obs = len(yr_diff)
                yr_diff_up, dis_up, dates_up = checkdt(Dates, Dis, startdt, enddt, excdt)


                # Inversion for velocity
                G, m, e2  = inv_param(yr_diff, dis, model)
                vel_est   = m[1]
                dis_est   = np.dot(G, m)
                rmsr      = np.sqrt(e2/num_obs)
                G_inv     = linalg.inv(np.dot(G.T, G))
                var_param = e2.reshape(1, -1) / (num_obs - num_param)
                m_std     = np.sqrt(np.dot(np.diag(G_inv).reshape(-1, 1), var_param))
                vel_std   = m_std[1,0]
                dis_resid = dis_est - dis

                # fine time resolution estimate
                G, new_dates = get_G(yr_diff, poly_degree, periods, steps, eventimev=True)
                dis_fine = np.dot(G, m)
                # ------------------------------

                dis_resids.append(dis_resid)
                rmsrs.append(rmsr)

                plotUpdate = 0
                if dates_up != dates:
                    plotUpdate = 1
                    # Inversion for velocity
                    G, m, e2     = inv_param(yr_diff_up, dis_up, model)
                    vel_est_up   = m[1]
                    dis_est_up   = np.dot(G, m)
                    rmsr_up      = np.sqrt(e2/num_obs)
                    G_inv        = linalg.inv(np.dot(G.T, G))
                    var_param    = e2.reshape(1, -1) / (num_obs - num_param)
                    m_std        = np.sqrt(np.dot(np.diag(G_inv).reshape(-1, 1), var_param))
                    vel_std_up   = m_std[1,0]
                    dis_resid_up = dis_est - dis


                # plot
                col.scatter(dates, dis-np.mean(dis), marker='^', s=6**2, facecolors='none', edgecolors='k', linewidth=1.)
                col.plot(new_dates, dis_fine-np.mean(dis_est), '-', lw=2, c='crimson', label='{} ({:.2f}±{:.2f} cm/y, RMSR={:.2f} \
                    cm)'.format(pix_name[i],vel_est,vel_std,rmsr))
                if plotUpdate == 1:
                    col.plot(dates_up, dis_est_up-np.mean(dis_est_up), '-', lw=2, c='slateblue', label='{} ({:.2f}±{:.2f} cm/y, RMSR={:.2f} \
                        cm)'.format(pix_name[i],vel_est_up,vel_std_up,rmsr_up))

                # axis format
                pp.auto_adjust_xaxis_date(col, dates, fontsize=font_size)
                col.tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True)
                col.set_ylim(-10.01, 10.01)
                #col.text(dates[0], 8, 'RMSR={:.2f} cm'.format(rmsr))
                #col.text(dates[0],-8, 'VelStd={:.2f} cm/yr'.format(vel_std))
                col.legend(loc='upper right', fontsize=12, frameon=False)
                i+=1
        fig.text(0.50, 0.06, 'Time [year]', ha='center', va='center', fontsize=20)
        fig.text(0.06, 0.50, 'LOS displacement [cm]', ha='center', va='center', rotation='vertical', fontsize=20)
        fig.text(0.50, 0.90, figtitle, ha='center', va='center', fontsize=20)

        # output
        out_file = f'{pic_dir}/{fname}.png'
        plt.savefig(out_file, bbox_inches='tight', transparent=True, dpi=fig_dpi)
        print('save to file: '+out_file)
        plt.show()

    if plot_residual == 1:
        fname = f'ts2velo_{pts}_{folder}_resid'
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, gridspec_kw={'hspace':0.1, 'wspace':0.02}, figsize=[ncols*8,nrows*2])
        i=0
        for row in ax:
            for col in row:
                try:
                    # plot
                    col.scatter(dates, dis_resids[i], marker='.', label='{} (RMSR={:.2f})'.format(pix_name[i], rmsrs[i]))

                    # axis format
                    pp.auto_adjust_xaxis_date(col, dates, fontsize=font_size)
                    col.tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True)
                    col.set_ylim(-5.01, 5.01)
                    col.legend(loc='upper right', fontsize=12, frameon=True)
                except:
                    col.axis('off')
                i+=1

        figtitle = f'{figtitle}, residuals'
        fig.text(0.50, 0.06, 'Time [year]', ha='center', va='center', fontsize=20)
        fig.text(0.06, 0.50, 'LOS Residual [cm]', ha='center', va='center', rotation='vertical', fontsize=20)
        fig.text(0.50, 0.90, figtitle, ha='center', va='center', fontsize=20)

        # output
        out_file = f'{pic_dir}/{fname}.png'
        plt.savefig(out_file, bbox_inches='tight', transparent=True, dpi=fig_dpi)
        print('save to file: '+out_file)
        plt.show()



# ---------------------------------------------------------
# Plot velocity files
# ---------------------------------------------------------

def plot_velocitymap(data, pts=None, folder='velocity', title='Velocity', fig_dpi=300,
                     vmin=-20, vmax=20, stdmin=0.0, stdmax=4.0, ampmax=40, plot_seasonalAmp=False, gv=False):
    from matplotlib import colors

    velo_file = f'{folder}/{data}.h5'
    fname = '_'.join([data]+folder.split('_')[1:])
    #fname = '_'.join([data]+[pts]+folder.split('_')[1:])

    # print folder and dataset
    print(f'{folder}/{data}.h5')

    # look for DEM file from ARIA path
    demfile = '../DEM/SRTM_3arcsec.dem'

    # The dataset unit is meter
    v     = readfile.read(velo_file, datasetName='velocity')[0]*1000     # Unit: mm/y
    meta  = readfile.read(velo_file, datasetName='velocity')[1]          # metadata
    try:
        v_std = readfile.read(velo_file, datasetName='velocityStd')[0] *1000  # Unit: mm/y
    except:
        print('No STD for deramp velocity, read the non-deramp velocity STD instead')
        std_file = data.split('_')[0]
        std_file = f'{folder}/{std_file}.h5'
        v_std = readfile.read(std_file, datasetName='velocityStd')[0] *1000  # Unit: mm/y

    if plot_seasonalAmp == True:
        try:
            annualAmp = readfile.read(velo_file, datasetName='annualAmp')[0]*1000                 # Unit: mm
        except:
            print('No seasonal amp for deramp velocity, read the non-deramp velocity seasonal amp instead')
            annualAmp = readfile.read(std_file,  datasetName='annualAmp')[0]*1000  # Unit: mm/y
    else:
        try:
            dem = np.array(readfile.read(demfile)[0], dtype=float)      # Unit: m
        except:
            print('Error: DEM file not found.')


    # read lat/lon info
    try:
        length    = int(meta['LENGTH'])
        width     = int(meta['WIDTH'])
        x_min     = float(meta['X_FIRST'])
        x_step    = float(meta['X_STEP'])
        y_min     = float(meta['Y_FIRST'])
        y_step    = float(meta['Y_STEP'])
        lats      = np.arange(y_min,length*y_step+y_min, y_step)
        lons      = np.arange(x_min, width*x_step+x_min, x_step)
        ref_lat   = float(meta['REF_LAT'])
        ref_lon   = float(meta['REF_LON'])
    except:
        pass

    # read mask and mask the dataset
    mask_file = 'maskTempCoh095.h5'   # 'waterMask.h5' or 'maskTempCoh.h5'
    mask_data = readfile.read(mask_file)[0]
    v[mask_data==0] = np.nan
    v_std[mask_data==0] = np.nan
    if plot_seasonalAmp == True:
        annualAmp[mask_data==0] = np.nan
    else:
        water_mask = readfile.read('waterMask.h5')[0]
        dem[water_mask==0] = np.nan

    if pts is not None:
        # read points file
        pts_file = f'pixels_{pts}.txt'
        file = open(f'{pts_file}','r')
        lines = file.readlines()
        plats = []
        plons = []
        for x in lines:
            tmp = x.split()
            plats.append(float(tmp[0]))
            plons.append(float(tmp[1]))

        # points name
        import string
        alphabet = string.ascii_uppercase
        pix_name = list(alphabet)

    # plot
    font_size=16
    plt.rcParams.update({'font.size': font_size})
    cmap='jet'
    demcmap='terrain'
    nrows, ncols = 1, 3

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=[ncols*8,12])
    divcmap = plt.get_cmap(cmap)
    #divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
    im0 = ax[0].imshow(v, extent=[lons[0],lons[-1],lats[-1],lats[0]], cmap=divcmap, vmin=vmin, vmax=vmax) #, norm=divnorm)
    im1 = ax[1].imshow(v_std, extent=[lons[0],lons[-1],lats[-1],lats[0]], vmin=stdmin, vmax=stdmax, cmap=cmap)
    if plot_seasonalAmp == True:
        im2 = ax[2].imshow(annualAmp, extent=[lons[0],lons[-1],lats[-1],lats[0]],vmin=0, vmax=ampmax, cmap=cmap)
    else:
        im2 = ax[2].imshow(dem,       extent=[lons[0],lons[-1],lats[-1],lats[0]],vmin=-200, vmax=2000, cmap=demcmap)

    # axis format
    ax[0].set_title(f'{title}')
    first_line = title.split('\n')[0]
    ax[1].set_title(f'Std of {first_line}')
    cbar=fig.colorbar(im0, ax=ax[0], cmap=cmap, extend='both', aspect=50, shrink=0.8, pad=0.01)
    cbar.ax.set_ylabel('mm/year', rotation=270, labelpad=20)
    cbar=fig.colorbar(im1, ax=ax[1], cmap=cmap, extend='both', aspect=50, shrink=0.8, pad=0.01)
    cbar.ax.set_ylabel('mm/year', rotation=270, labelpad=20)

    if plot_seasonalAmp == True:
        ax[2].set_title(f'Absolute annual\namplitude')
        cbar=fig.colorbar(im2, ax=ax[2], cmap=cmap, extend='both', aspect=50, shrink=0.8, pad=0.01)
        cbar.ax.set_ylabel('mm', rotation=270, labelpad=20)
    else:
        ax[2].set_title(f'Topography\n(DEM)')
        cbar=fig.colorbar(im2, ax=ax[2], cmap=demcmap, extend='both', aspect=50, shrink=0.8, pad=0.01)
        cbar.ax.set_ylabel('m', rotation=270, labelpad=20)

    # set scale bar
    sblon = 34.1 # deg, lon location
    sblat = 27 # deg, lat location
    sbr   = 99 # km, distance for 1 longitude deg at sblat
    sblen = 80 # km, total length

    # plot other info
    for j in range(ncols):
        if pts is not None:
            for i in range(len(plats)):
                ax[j].scatter(plons[i], plats[i], s=40, marker='^', c='red', edgecolor='k')
                ax[j].text(plons[i], plats[i], pix_name[i], fontsize=16)
        try:
            ax[j].scatter(ref_lon, ref_lat, marker='s', s=50, c='k')
        except:
            pass
        ax[j].plot(np.array([sblon, sblon+sblen/sbr]), np.array([sblat, sblat]), linewidth=5, c='k')
        ax[j].text(0.5*(sblon+sblon+sblen/sbr), sblat-0.3, f'{sblen} km', ha='center')
        ax[j].set_xlabel('Longitude', fontsize=font_size+4)
        ax[j].set_ylabel('Latitude', fontsize=font_size+4)
        ax[j].set_xlim(34.0,37.5)
        ax[j].set_ylim(26.2,32.5)

    # output
    out_file = f'{pic_dir}/{fname}.png'
    plt.savefig(out_file, bbox_inches='tight', transparent=True, dpi=fig_dpi)
    print('save to file: '+out_file)
    plt.show()

    if gv is True:
        return v, v_std, meta



# ---------------------------------------------------------
# Plot transections
# ---------------------------------------------------------

def plot_transec(data, start_lalo, end_lalo, pts=None, mask_file=mask_file, folder='velocity', title='Velocity',
                 fig_dpi=300, vmin=-20, vmax=20, tvmin=-5, tvmax=5, svmin=-0.5, svmax=0.5):
    import matplotlib.gridspec as gridspec
    from matplotlib import colors

    velo_file = f'{folder}/{data}.h5'
    fname = '_'.join([data]+folder.split('_')[1:])
    #fname = '_'.join([data]+[pts]+folder.split('_')[1:])

    # print folder and dataset
    print(f'{folder}/{data}.h5')


    ## Look for DEM file from ARIA path
    demfile = '../DEM/SRTM_3arcsec.dem'

    v     = readfile.read(velo_file, datasetName='velocity')[0]    *1000  # Unit: mm/y
    try:
        v_std = readfile.read(velo_file, datasetName='velocityStd')[0] *1000  # Unit: mm/y
    except:
        print('No STD for deramp velocity, read the non-deramp velocity STD instead')
        std_file = data.split('_')[0]
        std_file = f'{folder}/{std_file}.h5'
        v_std = readfile.read(std_file, datasetName='velocityStd')[0] *1000  # Unit: mm/y

    meta  = readfile.read(velo_file, datasetName='velocity')[1]           # metadata


    ## Read lat/lon info
    try:
        length    = int(meta['LENGTH'])
        width     = int(meta['WIDTH'])
        x_min     = float(meta['X_FIRST'])
        x_step    = float(meta['X_STEP'])
        y_min     = float(meta['Y_FIRST'])
        y_step    = float(meta['Y_STEP'])
        lats      = np.arange(y_min,length*y_step+y_min, y_step)
        lons      = np.arange(x_min, width*x_step+x_min, x_step)
        ref_lat   = float(meta['REF_LAT'])
        ref_lon   = float(meta['REF_LON'])
    except:
        pass

    ## Read mask and mask the dataset
    mask_data = readfile.read(mask_file)[0]
    v[mask_data==0] = np.nan
    v_std[mask_data==0] = np.nan

    plats = []
    plons = []
    if pts is not None:
        # read points file
        pts_file = f'pixels_{pts}.txt'
        file = open(f'{pts_file}','r')
        lines = file.readlines()

        for x in lines:
            tmp = x.split()
            plats.append(float(tmp[0]))
            plons.append(float(tmp[1]))

    ## Points/profiles name
    import string
    alphabet = string.ascii_uppercase
    pix_name = list(alphabet)


    ## Set scale bar
    sblon = 34.2 # deg, lon location
    sblat = 26.5 # deg, lat location
    sbr   = 99   # km, distance for 1 longitude deg at sblat
    sblen = 100  # km, total length


    ## Plot settings
    font_size=16
    plt.rcParams.update({'font.size': font_size})
    cmap='jet'
    demcmap='terrain'


    ## Plot figure with subplots of different sizes
    fig = plt.figure(1, figsize=[24,12])
    # set up subplot grid
    gridspec.GridSpec(5,4)

    ## Large map plot on the left
    plt.subplot2grid((5,4), (0,0), colspan=3, rowspan=5)
    plt.title(f'{title}')
    plt.xlabel('Longitude', fontsize=font_size+4)
    plt.ylabel('Latitude', fontsize=font_size+4)
    #divcmap = plt.get_cmap(cmap)
    #divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
    im0 = plt.imshow(v, extent=[lons[0],lons[-1],lats[-1],lats[0]], cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xticks(np.arange(34,39))
    plt.xlim(34.0,37.5)
    plt.ylim(26.2,32.5)
    cbar=plt.colorbar(im0, cmap=cmap, aspect=30, shrink=0.8, pad=0.01)
    cbar.ax.set_ylabel('mm/year', rotation=270, labelpad=20)
    if vmax - vmin >=10:
        cbar.set_ticks(np.arange(vmin,vmax+0.1,3))
    # plot other info
    for i in range(len(plats)):
        plt.scatter(plons[i], plats[i], s=5, marker='^', c='red', edgecolor='k')
        plt.text(plons[i], plats[i]+0.05, pix_name[i], fontsize=14)
    try:
        plt.scatter(ref_lon, ref_lat, marker='s', s=50, c='k')
    except:
        pass
    plt.plot(np.array([sblon, sblon+sblen/sbr]), np.array([sblat, sblat]), linewidth=8, c='k')
    plt.text(0.5*(sblon+sblon+sblen/sbr), sblat-0.2, f'{sblen} km', ha='center')

    ## Plot city
    #plt.scatter(36.5662, 28.3835, marker='v', s=60, c='k')    # Tabuk

    ## Plot the transection locations on the map
    if True:
        for i in range(len(start_lalo)):
            plt.plot(np.array([start_lalo[i][1], end_lalo[i][1]]), np.array([start_lalo[i][0], end_lalo[i][0]]), 'k--', lw=1)
            plt.text(36.5, 0.1+float(end_lalo[i][0]), s=f'{i+1}', bbox={"boxstyle" : "circle", "color":"white"}, fontsize=font_size-2)

    if tvmin is None:
        tvmin = vmin
    if tvmax is None:
        tvmax = vmax

    ## Extract velocity transection
    dis_max = 0
    for i in range(len(start_lalo)):
        tvel = ut.transect_lalo(v    , meta, np.array(start_lalo[i]), np.array(end_lalo[i]), interpolation='nearest')
        terr = ut.transect_lalo(v_std, meta, np.array(start_lalo[i]), np.array(end_lalo[i]), interpolation='nearest')

        if np.max(tvel['distance']/1000.) >= dis_max:
            dis_max = np.max(tvel['distance']/1000.)


        ## Calculate the derivatives of velocity on the transection:
        chp   = int(9)             # chip length for derivative, must be odd interger
        hl    = int((chp-1)/2)      # half_length = (chp-1)/2
        Deri  = np.zeros(len(tvel['value']))
        Dstd  = np.zeros(len(tvel['value']))
        for j in range(len(tvel['value'])):
            sta = np.clip(j-hl,0,len(tvel['value']))
            end = np.clip(j+hl,0,len(tvel['value']))
            data = tvel['value'][sta:end]             # unit: mm/year
            dist = tvel['distance'][sta:end]*1000.    # unit: mm
            num_obs = len(dist)

            ## Set the  fitting model (linear slope)
            poly_degree=1
            periods=[]
            steps=[]

            ## Get velocity model
            model, num_param = get_model(poly_degree, periods, steps)

            ## Inversion for velocity
                # G      num_obs * num_param
                # m      num_param * 1
                # e2     scalar (square sum of misfit)
                # G_inv  num_param * num_param
            G, m, e2  = inv_param(dist, data, model)
            G_inv     = linalg.inv(np.dot(G.T, G))

            ## Data covariance matrix, C_d
            # Cd     num_obs * num_obs
            Cd        = np.diag(terr['value'][sta:end]**2)

            ## Model covariance matrix, C_m
            # Cm     num_param * num_param
            Cm        = G_inv @ G.T @ Cd @ G @ G_inv

            ## Get estimated strain rate and uncertainty
            Deri[j] = m[1]                    * 1e6   # nano strain rate; index 1 for intercept, 2 for slope
            Dstd[j] = np.sqrt(np.diag(Cm)[1]) * 1e6   # nano strain rate (uncertainty); index 1 for intercept, 2 for slope

        xvec = tvel['distance']/1000.
        yvec = tvel['value'] - np.mean(tvel['value'])
        yerr = terr['value']

        ## Make small transection profiles in subplots
        ax1 = plt.subplot2grid((5,4), (i,3), colspan=1, rowspan=1)
        ax2 = ax1.twinx()

        ax1.scatter(xvec, yvec, s=1, c='b',      label='velocity')
        ax2.scatter(xvec, Deri, s=1, c='tomato', label='Strain')
        ax1.fill_between(xvec, yvec-yerr, yvec+yerr, alpha=0.15, fc='b',      ec='b',      lw=0.5, label=r'±$\sigma$')
        ax2.fill_between(xvec, Deri-Dstd, Deri+Dstd, alpha=0.25, fc='tomato', ec='tomato', lw=0.5, label=r'±$\sigma$')
        ax1.text(240, tvmax*0.7, s=f'{i+1}', bbox={"boxstyle" : "circle", "facecolor":"white", "edgecolor":"black"}, fontsize=font_size)
        ax1.set_ylim(tvmin, tvmax)
        ax1.set_xlim(0,260)
        #ax1.legend(loc='lower left', frameon=False)
        ax2.set_ylim(svmin, svmax)
        ax2.yaxis.set_label_position('right')
        ax2.tick_params(axis='y', left=False, right=True, labelleft=False, labelright=True, direction='in', pad=8, labelcolor='tomato')
        ax1.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False, direction='in', pad=8, labelcolor='blue')

        if i+1 == 1:
            ax1.set_title(f'Profiles')
        if i+1 == 3:
            ax1.set_ylabel('Velocity [mm/year]', c='blue', fontsize=font_size+4, rotation=90, labelpad=20)
            ax2.set_ylabel(f'$\mu$strain rate [1/year]', c='tomato', fontsize=font_size+4, rotation=270, labelpad=40)
        if i+1 == len(start_lalo):
            ax1.set_xlabel('Distance [km]', fontsize=font_size+4)
            plt.setp(ax1.get_xticklabels(), visible=True)
        else:
            plt.setp(ax1.get_xticklabels(), visible=False)

        plt.subplots_adjust(hspace = 0.14)


    ## Save the figure
    out_file = f'{pic_dir}/prof_{fname}.png'
    plt.savefig(out_file, bbox_inches='tight', transparent=True, dpi=fig_dpi)
    print('save to file: '+out_file)
    plt.show()



# ---------------------------------------------------------
# Plot strain map
# ---------------------------------------------------------
def plot_strainmap(data, pts, folder='linear', title='Strain rate', fig_dpi=300,
                 vmin=-1, vmax=1, chip_size=11):
    import matplotlib.gridspec as gridspec

    velo_file = f'velocity_{folder}/{data}.h5'
    fname = f'deriv_{data}_{pts}_{folder}'

    # folder (default: plot the linear model results)
    if 'linear' in folder:
        print('Linear fitting')
    elif 'periodic' in folder:
        print('Linear + periodic fitting (including seasonal terms)')
    else:
        print('Erorr: wrong velocity folder input')

    # look for DEM file from ARIA path
    demfile = '../DEM/SRTM_3arcsec.dem'

    v     = readfile.read(velo_file, datasetName='velocity')[0]*1000     # Unit: mm/y
    v_std = readfile.read(velo_file, datasetName='velocityStd')[0]*1000  # Unit: mm/y
    meta  = readfile.read(velo_file, datasetName='velocity')[1]          # metadata


    # read lat/lon info
    try:
        length    = int(meta['LENGTH'])
        width     = int(meta['WIDTH'])
        x_min     = float(meta['X_FIRST'])
        x_step    = float(meta['X_STEP'])
        y_min     = float(meta['Y_FIRST'])
        y_step    = float(meta['Y_STEP'])
        lats      = np.arange(y_min,length*y_step+y_min, y_step)
        lons      = np.arange(x_min, width*x_step+x_min, x_step)
        ref_lat   = float(meta['REF_LAT'])
        ref_lon   = float(meta['REF_LON'])
    except:
        pass


    # Cartesian coordinate: ground distance for one pixel
    unitDx = 90 * (x_step/0.000833)   # 90 m (0.00083 deg) is original pixel size
    unitDy = 90 * (y_step/0.000833)   # 90 m (0.00083 deg) is original pixel size


    # calculate the derivatives of velocity field:
    chp   = int(chip_size)            # chip length for derivative, must be odd interger
    hl    = int((chp-1)/2)            # half_length = (chp-1)/2
    Deri  = np.zeros_like(v)
    Dstd  = np.zeros_like(v)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            nb = np.clip(i-hl,0,v.shape[0])
            sb = np.clip(i+hl,0,v.shape[0])
            wb = np.clip(j-hl,0,v.shape[1])
            eb = np.clip(j+hl,0,v.shape[1])
            data = v[nb:sb, wb:eb]              # unit: mm/year (velocity)
            px   = eb-wb
            py   = sb-nb
            dx   = unitDx*np.arange(px)*1000.   # unit: mm      (distance)
            dx   = np.tile(dx,(py,1))
            dy   = unitDy*np.arange(py)*1000.   # unit: mm      (distance)
            dy   = np.tile(dy,(px,1)).T
            X = inv_param(t=np.array([dx.flatten(),dy.flatten()]), dis=data.flatten(), model=[], bilinear=True)

            # micor strain rate: Velo/Dist = 1/year
            print(X.size)
            Deri[i,j] = np.sqrt(X[0][1]**2 + X[0][2]**2)*1e6


    # read mask and mask the dataset
    mask_file = 'maskTempCoh.h5'   # 'waterMask.h5' or 'maskTempCoh.h5'
    mask_data = readfile.read(mask_file)[0]
    v[mask_data==0] = np.nan
    Deri[mask_data==0] = np.nan

    # read points file
    pts_file = f'pixels_{pts}.txt'
    file = open(f'{pts_file}','r')
    lines = file.readlines()
    plats = []
    plons = []
    for x in lines:
        tmp = x.split()
        plats.append(float(tmp[0]))
        plons.append(float(tmp[1]))

    # points name
    import string
    alphabet = string.ascii_uppercase
    pix_name = list(alphabet)


    # set scale bar
    sblon = 34 # deg, lon location
    sblat = 27 # deg, lat location
    sbr   = 99 # km, distance for 1 longitude deg at sblat
    sblen = 80 # km, total length


    # plot
    font_size=16
    plt.rcParams.update({'font.size': font_size})
    cmap='rainbow'
    demcmap='terrain'

    # Plot figure with subplots of different sizes
    fig = plt.figure(1, figsize=[10,12])

    # Plot strain rate map
    plt.title(f'{title}')
    plt.xlabel('Longitude', fontsize=font_size+4)
    plt.ylabel('Latitude', fontsize=font_size+4)
    im0 = plt.imshow(Deri, extent=[lons[0],lons[-1],lats[-1],lats[0]],vmin=vmin, vmax=vmax, cmap=cmap)
    plt.xlim(34.0,37.5)
    plt.ylim(26.2,32.5)
    cbar=plt.colorbar(im0, cmap=cmap, extend='both', aspect=50, shrink=0.8, pad=0.01)
    cbar.ax.set_ylabel('micro strain rate [1/year]', rotation=270, labelpad=20)

    # plot other info
    for i in range(len(plats)):
        plt.scatter(plons[i], plats[i], s=40, marker='^', c='red', edgecolor='k')
        plt.text(plons[i], plats[i], pix_name[i], fontsize=16)
    try:
        plt.scatter(ref_lon, ref_lat, marker='s', s=50, c='k')
    except:
        pass
    plt.plot(np.array([sblon, sblon+sblen/sbr]), np.array([sblat, sblat]), linewidth=4, c='k')
    plt.text(0.5*(sblon+sblon+sblen/sbr), sblat-0.3, f'{sblen} km', ha='center')


    # output
    out_file = f'{pic_dir}/strain_{fname}.png'
    plt.savefig(out_file, bbox_inches='tight', transparent=True, dpi=fig_dpi)
    print('save to file: '+out_file)
    plt.show()



# ---------------------------------------------------------
# Residual RMS
# ---------------------------------------------------------
class tsRMS():
    def __init__(self, rms_file, k=1, cutoff=3.0, ifgramStack='ifgramStack_coherence_spatialAvg.txt'):
        self.rms_file = rms_file
        self.ifgramStack = ifgramStack

        try:
            in_file = open(f'./{ifgramStack}','r')
        except:
            in_file = open(f'./coherenceSpatialAvg.txt','r')
        lines = in_file.readlines()
        date12=[]
        bperp=[]
        for x in lines:
            if x[0] == '#':
                continue
            else:
                tmp = x.split()
                date12.append(tmp[0])       # date1 and date2 of an ifgram
                bperp.append(float(tmp[3])) # Unit: meter

        in_file = open(f'./{rms_file}','r')
        lines = in_file.readlines()
        dates=[]
        datestr=[]
        rms=[]
        for x in lines:
            if x[0] == '#':
                continue
            else:
                tmp = x.split()
                datestr.append(tmp[0])
                dates.append(datetime(np.int(tmp[0][:4]), np.int(tmp[0][4:6]), np.int(tmp[0][6:8]), 0, 0))
                rms.append(float(tmp[1])*1000)
        rms = np.array(rms)

        # Get noise level threshold
        rms_med   = np.median(rms)
        center    = k * rms_med
        MAD       = 1.4826 * np.median(np.abs(rms-rms_med))
        threshold = cutoff*MAD + center

        # Print median, MAD about the residual RMS
        print('Statistics:')
        print('\t Median of RMS \t\t= {:.2f} mm'.format(rms_med))
        print('\t MAD \t\t\t= {:.2f} mm \t (centered at {:d} * RMS median)'.format(MAD, k))
        print('\t {:.1f} * MAD + center \t= {:.2f} mm \t (MintPy default threshold)'.format(cutoff, threshold))


        G = np.zeros([len(date12),len(datestr)])
        for i in range(len(date12)):
            Grow=np.zeros(len(datestr))
            try:
                id1=datestr.index(date12[i].split('_')[0])
                id2=datestr.index(date12[i].split('_')[1])
            except:
                continue
            Grow[id1]=-1
            Grow[id2]=1
            G[i,:]=Grow

        bperp_est = np.linalg.lstsq(G, bperp, rcond=None)[0]
        bperp_est = bperp_est-bperp_est[0]

        self.dates      = dates
        self.datestr    = datestr
        self.date12     = date12
        self.bperp_est  = bperp_est

        self.rms        = rms
        self.rms_med    = rms_med

        self.center     = center
        self.MAD        = MAD
        self.cutoff     = cutoff
        self.threshold  = threshold

    # ----------------------------------------------------------------------------------
    def plot_rmsHist(self, bin_size=2, supp='', threshold=False):
        fig, ax = plt.subplots(ncols=2, sharey=True, gridspec_kw={'wspace':0.02}, figsize=[14,5])
        fig.text(0.42, 0.92, f'Time-series residual RMS {supp}')

        vmin = np.min(self.rms)
        vmax = np.max(self.rms)
        bins = np.linspace(vmin,vmax,int((vmax-vmin)/bin_size))
        ax[0].hist(self.rms, bins, edgecolor='k')
        ax[0].axvline(x=self.rms_med   , color='k', linestyle='--', label='median')
        if threshold == True:
            ax[0].axvline(x=self.MAD       , color='b', linestyle='--', label='MAD')
            ax[0].axvline(x=self.threshold , color='r', linestyle='--', label=f'{self.cutoff}*MAD')
        ax[0].axvline(x=np.percentile(self.rms,90)  , color=0.3*np.ones(3), lw=0.5, label='90%')
        ax[0].axvline(x=np.percentile(self.rms,95)  , color=0.5*np.ones(3), lw=0.5, label='95%')
        ax[0].axvline(x=np.percentile(self.rms,99.7), color=0.7*np.ones(3), lw=0.5, label='99.7%')
        ax[0].set_xlabel('RMS [mm]')
        ax[0].set_ylabel('Num of scenes')
        ax[0].legend(frameon=False, loc='upper right')

        vmin = np.min(self.rms)
        vmax = np.max(self.rms)
        bins = np.linspace(np.log10(vmin),np.log10(vmax),int((vmax-vmin)/bin_size))
        ax[1].hist(np.log10(self.rms), bins, edgecolor='k')
        ax[1].axvline(x=np.log10(self.rms_med)   , color='k', linestyle='--', label='median')
        if threshold == True:
            ax[1].axvline(x=np.log10(self.MAD), color='b', linestyle='--', label='MAD')
            ax[1].axvline(x=np.log10(self.threshold), color='r', linestyle='--', label=f'{self.cutoff}*MAD')
        ax[1].axvline(x=np.percentile(np.log10(self.rms),90)  , color=0.3*np.ones(3), lw=0.5, label='90%')
        ax[1].axvline(x=np.percentile(np.log10(self.rms),95)  , color=0.5*np.ones(3), lw=0.5, label='95%')
        ax[1].axvline(x=np.percentile(np.log10(self.rms),99.7), color=0.7*np.ones(3), lw=0.5, label='99.7%')
        ax[1].set_xlabel('log(RMS) [-]')
        ax[1].legend(frameon=False, loc='upper right')
        plt.show()


    # ----------------------------------------------------------------------------------
    # Read SLC filename to get Mission Identifier (Sen1-A or Sen1-B)
    def MI_info(self, SLC_path, colorA='dodgerblue', colorB='tomato'):
        fns = glob.glob(f'{SLC_path}/*.zip')
        for i in range(len(fns)):
            fns[i] = fns[i].split('/')[-1]
        acqs = []
        for i in range(len(self.datestr)):
            ymd = self.datestr[i]
            for j in range(len(fns)):
                if ymd == fns[j].split('1SDV_')[1][:8]:
                    mi = fns[j].split('1SDV_')[0][:3]
                    acqs.append((ymd, mi))
        acqs = sorted(list(set(acqs)))
        if len(acqs) > len(self.datestr):
            print('SLC filename reading rrror: Acqusition dates have non-unique mission identifier')
        if len(acqs) < len(self.datestr):
            lacks = sorted(list(set(self.datestr) - set(np.array(acqs)[:,0])))
            print(f'Number of dates from downloaded Sen1 SLCs = {len(acqs)}\nNumber of ARIA acquisitions = {len(self.datestr)}')
            print(f'Lack SLCs from the following {len(self.datestr)-len(acqs)} dates. Update the following dates from ASF vertex:')
            print(f'{lacks}')

        bcolor=[]
        colorA = 'dodgerblue'
        colorB = 'tomato'
        for i in range(len(self.datestr)):
            if i < len(acqs):
                if acqs[i][1] == 'S1A':
                    bcolor.append(colorA)
                elif acqs[i][1] == 'S1B':
                    bcolor.append(colorB)
            else:
                bcolor.append('grey')

        return bcolor, acqs

    # ----------------------------------------------------------------------------------
    def plot_network(self, SLC_path=None, return_values=False, plotTEC=False, pic_dir='./pic_supp'):
        from matplotlib import colors
        from matplotlib.lines import Line2D
        print('#################################')
        if plotTEC is False:
            filename = 'rms_ts_resid_ramp_network'
            title = 'Interferogram network and Residual RMS'
            cbarstr = 'Residual phase RMS [mm]'
        elif plotTEC is True:
            filename = 'tec_ts_network'
            title = 'Interferogram network and TEC delay'
            cbarstr = 'Estimate TEC delay [mm]'

        fig_dpi=150
        font_size=16
        plt.rcParams.update({'font.size': font_size})

        dates     = self.dates
        datestr   = self.datestr
        date12    = self.date12
        rms       = self.rms
        bperp_est = self.bperp_est

        if SLC_path is None: # now not in used
            colors_over = plt.cm.Reds_r(np.linspace(0, 0.50, 256))
            colors_okay = plt.cm.YlOrRd_r(np.linspace(0.70, 1, 256))
            all_colors = np.vstack((colors_okay[::-1], colors_over[::-1]))
            rms_color = colors.LinearSegmentedColormap.from_list('rms_color',all_colors)
            cmap = rms_color
            divnorm = colors.TwoSlopeNorm(vmin=self.rms_med, vcenter=1.75*self.rms_med, vmax=2.5*self.rms_med)
            figsize = [14,6]
        elif SLC_path is not None:
            colorA = 'dodgerblue'
            colorB = 'grey'
            bcolor, _= self.MI_info(SLC_path, colorA, colorB)
            figsize = [12,6]

        fig, ax = plt.subplots(figsize=figsize)
        for i in range(len(date12)):
            try:
                id1 = datestr.index(date12[i].split('_')[0])
                id2 = datestr.index(date12[i].split('_')[1])
            except:
                continue
            ax.plot([dates[id1], dates[id2]],[bperp_est[id1], bperp_est[id2]], '-', lw=2., c='gray', alpha=0.4)
            pp.auto_adjust_xaxis_date(ax, dates, fontsize=font_size)
            plt.tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True)

        if SLC_path is None:
            cmap = 'plasma'
            im=ax.scatter(dates, bperp_est, marker='o', s=200, c=rms, cmap=cmap, edgecolor='k', linewidth=1., alpha=0.8)
            cbar=fig.colorbar(im, ax=ax, cmap=cmap)
            im.set_clim(1*self.rms_med, self.threshold)
            cbar.ax.set_ylabel(cbarstr, rotation=270, labelpad=20)

        elif SLC_path is not None:
            im=ax.scatter(dates, bperp_est, marker='o', s=200, c=bcolor, edgecolor='k', linewidth=1., alpha=0.8)
            handles = [Line2D([0],[0],marker='o',color=c,lw=1,markersize=10,markeredgecolor='k',alpha=0.8) for c in [colorA, colorB]]
            labels = ['Sen1-A','Sen1-B']
            ax.legend(handles, labels, frameon=True, loc='upper right')
        ax.set_xlabel('Time [year]')
        ax.set_ylabel('Perp Baseline [m]')
        ax.set_title(title)

        # output
        plt.tight_layout()
        out_file = f'{pic_dir}/{filename}.png'
        plt.savefig(out_file, bbox_inches='tight', transparent=True, dpi=fig_dpi)
        print('save to file: '+out_file)
        plt.show()

        if return_values == True:
            return dates, datestr

    # ----------------------------------------------------------------------------------
    def plot_sceneNoise(self, average=False, shift=-1, SLC_path=None, ylog=False, return_values=False, plotTEC=False, pic_dic='./pic_supp'):
        from matplotlib.patches import Rectangle

        if plotTEC is False:
            filename = 'rms_ts_ramp_hist'
            title = 'Temporal noise distribution'
            ystr = 'Residual phase RMS [mm]'
        elif plotTEC is True:
            filename = 'tec_ts_hist'
            title = 'TEC delay history'
            ystr = 'Estimated TEC delay [mm]'

        fig_dpi=150
        font_size=16
        plt.rcParams.update({'font.size': font_size})

        dates     = self.dates
        datestr   = self.datestr
        rms       = self.rms
        bperp_est = self.bperp_est
        rms_med   = self.rms_med
        MAD       = self.MAD
        cutoff    = self.cutoff

        # get some weighting
        w = 1/rms          # weights are the inverse of rms residues
        w = len(w)*(w/sum(w))

        if average is True:
            filename = f'{filename}Avg'
            win  = 9                # window size (must be odd); 9 scenes ~ across 3 months
            hwin = int(0.5*(win-1)) # half window size
            rmsAvg = []
            for i in range(len(dates)):
                i1 = np.clip(i-hwin, 0, len(dates))
                i2 = np.clip(i+hwin, 0, len(dates))
                rmsWin = np.mean(rms[i1:i2+1])
                rmsAvg.append(rmsWin)

        threshold = self.threshold       # 3*MAD (Yunjun et al., 2019) for RMS residual
        # threshold = 15.                # manually set RMS threshold = 15 mm (by eyeballing the bad seasons)
        # threshold = 2.5*self.rms_med   # 2.5*median (eyeballing from log-log space normal distribution)

        lowRMS_date  = np.where(rms==np.min(rms))[0][0]
        highRMS_date = np.where(np.round(rms,0)>=np.round(threshold,0))[0]

        # save exclude dates
        if plotTEC is False:
            outfile = open('my_RMSexclude_date.txt', 'w')
        elif plotTEC is True:
            outfile = open('my_TECexclude_date.txt', 'w')

        for i in range(len(highRMS_date)+1):
            if i == len(highRMS_date):
                for j in range(len(highRMS_date)):
                    outfile.write(f'{datestr[highRMS_date[j]]},')
            else:
                outfile.write(f'{datestr[highRMS_date[i]]}\n')
        outfile.close()

        label_thresh = 'Threshold = {:.1f} * MAD ({:.1f})'.format(cutoff, threshold)
        #label_thresh  = 'Threshold = {:.1f} mm (2.5*med)'.format(threshold)
        label_percent1 = '95th percentile'
        label_percent2 = '99.7th percentile'
        label_lowRMS  = 'Reference date'
        label_highRMS = f'Noisy date ({len(highRMS_date)}/{len(datestr)})'
        label_SA = 'Sen1-A'
        label_SB = 'Sen1-B'
        label_avg = '3-month running mean'


        fig, ax = plt.subplots(figsize=[12,6])
        bar_width = np.min(np.diff(dates).tolist())*3/4
        if plotTEC is False:
            if SLC_path is None:
                ax.bar(dates, rms, width=bar_width.days)
                #lowbar = ax.bar(dates[lowRMS_date], rms[lowRMS_date], width=bar_width.days, color='orange')
                line = ax.axhline(y=threshold              , linestyle='--', color='k'   , lw=3, label=label_thresh)
                #line2 = ax.axhline(y=np.percentile(rms, 95), linestyle='--', color='gray', lw=2, label=label_percent1)
                #line3 = ax.axhline(y=np.percentile(rms, 99.7), linestyle='-.', color='gray', lw=2, label=label_percent2)
                for i in range(len(highRMS_date)):
                    highbar = ax.bar(dates[highRMS_date[i]], rms[highRMS_date[i]], width=bar_width.days, color='lightgrey')
                if average is True:
                    avgline, = ax.plot(dates, rmsAvg, c='r', lw=2)
                    try:
                        handles = [line, line2, line3, lowbar, highbar, avgline]
                        labels= [label_thresh, label_percent1, label_percent2, label_lowRMS, label_highRMS, label_avg]
                    except:
                        handles = [line,highbar, avgline]
                        labels= [label_thresh, label_highRMS, label_avg]
                else:
                    try:
                        handles = [line, line2, line3, lowbar, highbar]
                        labels= [label_thresh, label_percent1, label_percent2, label_lowRMS, label_highRMS]
                    except:
                        handles = [line, line2, line3, lowbar]
                        labels= [label_thresh, label_percent1, label_percent2, label_lowRMS]
                ax.legend(handles, labels, frameon=True, loc='upper right')
                if average is True:
                    ax.plot(dates, rmsAvg, c='r', lw=2)
                #ax.plot(dates, w, c='k', lw=2)
            elif SLC_path is not None:
                filename=filename+'_platform'
                colorA = 'dodgerblue'
                colorB = 'tomato'
                bcolor, _= self.MI_info(SLC_path, colorA, colorB)
                ax.bar(dates, rms, width=bar_width.days, color=bcolor)
                line = ax.axhline(y=threshold              , linestyle='--', color='k'   , lw=3, label=label_thresh)
                line2 = ax.axhline(y=np.percentile(rms, 95), linestyle='--', color='gray', lw=2, label=label_percent1)
                line3 = ax.axhline(y=np.percentile(rms, 99.7), linestyle='-.', color='gray', lw=2, label=label_percent2)
                if average is True:
                    avgline, = ax.plot(dates, rmsAvg, c='r', lw=2)
                    handles = [line]+[line2]+[line3]+[avgline]+[Rectangle((0,0),1,1,color=c) for c in [colorA, colorB]]
                    labels= [label_thresh, label_percent1, label_percent2, label_avg, label_SA, label_SB]
                else:
                    handles = [line]+[line2]+[line3]+[Rectangle((0,0),1,1,color=c) for c in [colorA, colorB]]
                    labels= [label_thresh, label_percent1, label_percent2, label_SA, label_SB]
                ax.legend(handles, labels, frameon=True, loc='upper right')

        elif plotTEC is True:
            ax.bar(dates, rms, width=bar_width.days, color='grey')
            #line = ax.axhline(y=threshold              , linestyle='--', color='k'   , lw=3, label=label_thresh)
            #for i in range(len(highRMS_date)):
                #highbar = ax.bar(dates[highRMS_date[i]], rms[highRMS_date[i]], width=bar_width.days, color='lightgrey')
            if average is True:
                avgline, = ax.plot(dates, rmsAvg, c='r', lw=2, label=label_avg)
                handles = [avgline]
                labels  = [label_avg]
            else:
                handles = [line, highbar]
                labels  = [label_thresh, label_highRMS]
            ax.legend(handles, labels, frameon=True, loc='upper right')

        pp.auto_adjust_xaxis_date(ax, dates, fontsize=font_size)
        ax.set_xlabel('Time [year]')
        ax.set_ylabel(ystr)
        #ax.set_ylim(0,32)
        if ylog == True:
            ax.set_yscale('log')
        ax.set_title(title)

        # output
        plt.tight_layout()
        out_file = f'{pic_dir}/{filename}.pdf'
        plt.savefig(out_file, bbox_inches='tight', transparent=True, dpi=fig_dpi)
        print('save to file: '+out_file)
        plt.show()

        if return_values == True:
            return dates, datestr



# ---------------------------------------------------------
# Plot Sentinel SLC process software versions
# ---------------------------------------------------------
def plot_slcproc(s1ver_file):
    dates = []
    versions = []
    with open(s1ver_file, 'r') as f:
        for line in f:
            tmp = line.split()
            try:
                fnstr = tmp[0].split('_') # filename strings
                MI = tmp[0][:3]           # mission identifier, S1A or S1B.
            except:
                continue
            if MI[:2] == 'S1':
                date = fnstr[5]
                yy, mo, dd, hr, mm, ss = int(date[:4]), int(date[4:6]), int(date[6:8]), int(date[9:11]), int(date[11:13]), int(date[13:15])
                dates.append(datetime(yy,mo,dd,hr,mm,ss))

                ver = tmp[2]
                versions.append(ver)

    versions_set = sorted(list(set(versions)))


    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')

    plt.figure(figsize=[10,8])
    for i in range(len(dates)):
        plt.scatter(dates[i], versions[i], s=2, marker='s', fc='b')
        if (i>1) & (versions[i] != versions[i-1]):
            plt.axvline(dates[i], lw=0.6, color='grey', ls='dashed')
    plt.xlabel('Dates')
    plt.ylabel('Processing software version')
    ax = plt.gca()
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    plt.show()
    out_file = f'{pic_dir}/slc_version.png'
    plt.savefig(out_file, bbox_inches='tight', transparent=True, dpi=150)



# ---------------------------------------------------------
# Plot elevation vs phase correlation
# ---------------------------------------------------------
# 1. Plot the observed raw timeseries data (each slice), there should present a correlation between elevation and phase (displacement)
# 2. Plot the weather model (each slice), there should also be some correlation.
# 3. If the weather model correlation agrees with the observation?
# 4. Is there any spatial variance in the elevation-phase correlation (e.g., latitude dependent, coastal or inland features)?

def plot_ele_atm(ts_file, atm_file, mask_file=mask_file, fname='', plotdate='all', withshift=1, noshow=True):
    if plotdate == 'all':
        from mintpy import info
        datelist = info.print_date_list(ts_file)
    elif len([plotdate]) == 1:
        datelist = [plotdate]
    elif len([plotdate]) > 1:
        datelist = plotdate

    for date in datelist:
        datasetName=f'timeseries-{date}'
        data    = readfile.read(ts_file, datasetName=datasetName)
        era5    = readfile.read(atm_file, datasetName=datasetName)[0]*1000

        disp    = data[0]*1000                                         # Unit: mm
        refdate = data[1]['REF_DATE']

        # Read timeseries geo info
        lats, lons, length, width, ref_lat, ref_lon = read_geo(ts_file, verbose=False)
        lats_array = np.tile(lats[0:-1], (len(lons)-1, 1)).T

        # Read DEM
        demfile = '../DEM/SRTM_3arcsec.dem'
        dem = np.array(readfile.read(demfile)[0], dtype=np.float32)         # Unit: m

        # Read mask and mask the dataset
        mask_file = mask_file   # 'waterMask.h5' or 'maskTempCoh.h5'
        mask_data = readfile.read(mask_file)[0]
        disp[mask_data==0] = np.nan
        dem[mask_data==0]  = np.nan
        era5[mask_data==0]  = np.nan

        # Plot time-series disp from one slice
        fig = plt.figure(figsize=[10,14])
        n = 10
        nlat = int(length/n)
        space = 50
        for i in range(n):
            p0 = i*width*nlat
            p1 = p0+width*nlat
            if i == n-1:
                p1 = -1
            shift = ((n-1) - i*space) * withshift
            lat_arr = lats_array.flatten()[p0:p1]
            x = dem.flatten()[p0:p1]
            y = disp.flatten()[p0:p1]
            #idx = np.isfinite(x) & np.isfinite(y)
            #c = np.polyfit(x[idx], y[idx], 2)
            im1 = plt.scatter(x, y+shift, marker='o', ec='k', c=lat_arr, cmap='hsv', lw=0.5)
            #plt.plot(x, c[0]+c[1]*x+shift, color='k', linestyle='dashed', lw=0.5, label='slope = {:.2f}'.format(c[1]))
            plt.clim(np.min(lats), np.max(lats))
        for i in range(n):
            p0 = i*width*nlat
            p1 = p0+width*nlat
            if i == n-1:
                p1 = -1
            shift = ((n-1) - i*space) * withshift
            lat_arr = lats_array.flatten()[p0:p1]
            x = dem.flatten()[p0:p1]
            y_pred = era5.flatten()[p0:p1]
            plt.scatter(x, y_pred+shift, marker='o', ec='gray', fc='lightgray', alpha=0.5, lw=0.1)


        cbar = fig.colorbar(im1, aspect=50, shrink=0.8, pad=0.01)
        cbar.ax.set_ylabel('Latitude of pixels [deg]', rotation=270, labelpad=20)
        plt.plot(np.array([2000,2000]), np.array([-50,0]), 'k-', lw=6)
        plt.text(2050, -35, '5 cm', rotation=270)
        ax = plt.gca()
        ax.axes.get_yaxis().set_ticks([])
        plt.title(f'Elevation correlated signals\n({date}-{refdate})')
        plt.xlabel('Elevation [m]')
        plt.ylabel(f'Raw range change')

        out_file = f'{pic_dir}/ele-atm_{fname}{date}_{int(withshift)}.png'
        fig.savefig(out_file, bbox_inches='tight', transparent=True, dpi=150)
        print(f'Save fig to {out_file}')
        if noshow == True:
            plt.close(fig)
        else:
            plt.show(fig)
            plt.close(fig)