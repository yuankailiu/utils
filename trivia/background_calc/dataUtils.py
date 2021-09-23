#!/usr/bin/env python3

############################################################
# Utilities for seismicity analysis                        #
# Plotting & calculation of background activity            #
#                                                          #
# Y.K. Liu @ 2021 June                                     #
############################################################

## Note:
## This script cannot be runned directly, import it as functinos to use.

import time
import json
import vptree
from pyproj import Geod
import pandas as pd
from scipy.ndimage import gaussian_filter
import numpy as np
from matplotlib import pyplot as plt, patches
from matplotlib.offsetbox import AnchoredText
from obspy.core.utcdatetime import UTCDateTime as UTC
from obspy.geodetics import kilometers2degrees
from obspy.geodetics import locations2degrees
from datetime import timedelta
from geopy.distance import geodesic


## set matplotlib canvas fontsize
plt.rcParams.update({'font.size': 22})



###################################################################################
## Define functions
###################################################################################


def read_meta(json_file):
    """Read the metadata from a json file"""
    with open(json_file) as f:
        print('Read metadata from %s' % json_file)
        meta = json.load(f)
    return meta


def read_plate_bound(plateBound_file, plateBound_name, v=True):
    """Read plate boundary geometries from file"""
    if v:
        print('Read the plate boundary data from file: {}'.format(plateBound_file))
    lola = []
    with open(plateBound_file, 'r') as f:
        for line in f:
            if line[0] == '>':
                if any(name in line for name in plateBound_name):
                    if v:
                        print(' '+line[:-1])
                    include = True
                    continue
                else:
                    include = False
                    continue
            if include:
                lola.append([float(ll) for ll in line.split()])
    lalo = np.flip(np.array(lola), axis=1)
    lalo = lalo[np.argsort(lalo[:, 1])]
    return lalo


def read_cat(catname, fullFile=False):
    """Read catalog of regular CSV format catalog"""
    tmp = pd.read_csv(catname, dtype='str')
    c   = tmp.to_numpy()
    if fullFile:
        return c
    else:
        time   = c[:,0]
        lat    = c[:,1].astype('float')
        lon    = c[:,2].astype('float')
        dep    = c[:,3].astype('float')
        mag    = c[:,4].astype('float')
        dtime  = pd.to_datetime(time, format='%Y-%m-%d %H:%M:%S.%f').to_pydatetime()
        for i in range(len(dtime)):
            dtime[i] = dtime[i].replace(tzinfo=None)
        dtime_s = pd.Series(dtime-dtime[0]).dt.total_seconds().to_numpy()
        if c.shape[1] == 22:
            # orignal USGS format (22 columns)
            evid = c[:,11].astype('str')
            arr  = (evid, dtime, dtime_s, lat, lon, dep, mag)
        elif c.shape[1] == 8:
            evid = c[:,7].astype('str')
            arr  = (evid, dtime, dtime_s, lat, lon, dep, mag)
        elif c.shape[1] == 10:
            evid = c[:,7].astype('str')
            td   = c[:,8].astype('float')
            sd   = c[:,9].astype('float')
            arr  = (evid, dtime, dtime_s, lat, lon, dep, mag, td, sd)
        elif c.shape[1] == 11:
            evid = c[:,7].astype('str')
            td   = c[:,8].astype('float')
            sd   = c[:,9].astype('float')
            mc   = c[:,10].astype('float')
            arr  = (evid, dtime, dtime_s, lat, lon, dep, mag, td, sd, mc)
        print('Read catalog of CSV file: {}, {} events'.format(catname, len(evid)))
    return arr


def read_xyz(file):
    """Read xyz format file"""
    tmp = pd.read_csv(file, dtype=float)
    tmp = tmp.dropna()
    c   = tmp.to_numpy()
    return tmp, c


def read_cat_isc(catname):
    """Read ISC-GEM catalog"""
    pdcat    = pd.read_csv(catname, header=97)
    c        = pdcat.to_numpy()
    time     = c[:,0]
    lat      = c[:,1].astype('float')
    lon      = c[:,2].astype('float')
    dep      = c[:,7].astype('float')
    mag      = c[:,10].astype('float')
    evid     = c[:,-1]
    dtime    = pd.to_datetime(time, format='%Y-%m-%d %H:%M:%S.%f').to_pydatetime()
    for i in range(len(dtime)):
        dtime[i] = dtime[i].replace(tzinfo=None)
    dtime_s = pd.Series(dtime-dtime[0]).dt.total_seconds().to_numpy()
    catalog = (c, evid, dtime, dtime_s, lat, lon, dep, mag)
    print('Read ISC-GEM catalog: {} events'.format(len(evid)))
    return catalog


def read_cat_ssn(catname): # need to fix
    """Read Servicio Sismologico Nacional (SSN) catalog"""
    pdcat    = pd.read_csv(catname, dtype='str', header=12)
    pdcat    = pdcat[pdcat["Profundidad"].str.contains("en")==False]
    pdcat    = pdcat[pdcat["Magnitud"].str.contains("no")==False]
    c        = pdcat.to_numpy()
    lat      = c[:,3].astype('float')
    lon      = c[:,4].astype('float')
    dep      = c[:,5].astype('float')
    mag      = c[:,2].astype('float')
    evid     = c[:,11].astype('str')
    dtime    = pd.to_datetime(time, format='%Y-%m-%d %H:%M:%S.%f').to_pydatetime()
    for i in range(len(dtime)):
        dtime[i] = dtime[i].replace(tzinfo=None)
    dtime_s = pd.Series(dtime-dtime[0]).dt.total_seconds().to_numpy()
    catalog = (c, evid, dtime, dtime_s, lat, lon, dep, mag)
    print('Read Servicio Sismologico Nacional (SSN) catalog: {} events'.format(len(evid)))
    return catalog


def cat_selection(cat, select):
    """Refine catalog event selection"""
    if len(cat) == 7:
        evid, dtime, dtime_s, lat, lon, dep, mag = np.array(cat).T[select].T
    elif len(cat) == 9:
        evid, dtime, dtime_s, lat, lon, dep, mag, td, sd = np.array(cat).T[select].T
    elif len(cat) == 10:
        evid, dtime, dtime_s, lat, lon, dep, mag, td, sd, mc = np.array(cat).T[select].T

    dtime_s = dtime_s.astype('float')
    lat  = lat.astype('float')
    lon  = lon.astype('float')
    dep  = dep.astype('float')
    mag  = mag.astype('float')

    if len(cat) == 7:
        newcat = (evid, dtime, dtime_s, lat, lon, dep, mag)
    elif len(cat) == 9:
        newcat = (evid, dtime, dtime_s, lat, lon, dep, mag, td, sd)
    elif len(cat) == 10:
        newcat = (evid, dtime, dtime_s, lat, lon, dep, mag, td, sd, mc)

    print('Refine event selection: {} events'.format(len(evid)))
    return newcat


def textonly(ax, txt, fontsize=14, loc=2, *args, **kwargs):
    """ pyplot write text in the plot just like legend style """
    at = AnchoredText(txt, prop=dict(size=fontsize), frameon=True, loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    return


def maxc_Mc(mag_arr, plot='no', save='./', title='', Mc=None, range=[0,4]):
    """
    calculate magnitude of completeness
    REFERENCE: Woessner, J., & Wiemer, S. (2005), BSSA
    """
    bin_size = 0.1
    fc = 'lightskyblue'
    ec = 'k'
    k, bin_edges = np.histogram(mag_arr,np.arange(-1,10,bin_size))
    centers      = bin_edges[:-1] + np.diff(bin_edges)[0]/2
    correction   = 0.20
    if Mc == None:
        Mc = centers[np.argmax(k)] + correction
    else:
        Mc = Mc
    msg  =        r'$M_{min} = %.3f$'         % (Mc)
    msg += '\n' + r'N($M \geq M_{min}$) = %d' % (len(mag_arr[mag_arr>=Mc]))
    msg += '\n' + r'N($M < M_{min}$) = %d'    % (len(mag_arr[mag_arr<Mc]))

    if plot == 'yes':
        fig ,axs = plt.subplots(figsize=[18,5], ncols=2)
        axs[0].bar(bin_edges[:-1], k, width=bin_size, fc=fc, ec=ec)
        axs[1].bar(bin_edges[:-1], np.cumsum(k)[-1]-np.cumsum(k), width=bin_size, fc=fc, ec=ec)
        axs[1].set_yscale('log')
        fig.suptitle(title, size=22)
        #axs[1].bar(bin_edges[:-1], 100*(1-np.cumsum(k)/np.cumsum(k)[-1]), width=bin_size, fc=fc, ec=ec)
        for ax in axs:
            ax.axvline(x=Mc, c='k', ls='--', lw=3)
            ax.set_xlim(range[0],range[1])
            textonly(ax, msg, loc='upper right')
            ax.set_xlabel('Magnitude')
            ax.set_ylabel('# events')
            # ax.text(0.5, 1.03, title, ha='center', fontsize=22, transform=ax.transAxes)
        plt.show()
        if save != 'no':
            fig.savefig('{}/magfreq_dist_{}.png'.format(save,title), dpi=300, bbox_inches='tight')
    elif plot == 'no':
        pass
    else:
        print('Plot keyword is either "yes" or "no"')
    return Mc


def epoch_Mc(mag, dtime, n_bin=10, plot='no', title=''):
    """calculate Mc for each epoch of the long-history catalog"""
    bin_sec = (dtime[-1]-dtime[0]).total_seconds()/n_bin
    epochs = []
    for i in range(int(n_bin)):
        epochs.append(np.where(dtime >= dtime[0]+timedelta(seconds=bin_sec*i))[0][0])
    Mcs = []
    for i in range(n_bin):
        if i==n_bin-1:
            sub_mag = mag[epochs[i]:-1]
        else:
            sub_mag = mag[epochs[i]:epochs[i+1]]
        Mcs.append(maxc_Mc(sub_mag, plot=plot, title=title, range=[2,8]))
    return epochs, bin_sec, Mcs


def pts_projline(lalo, start_lalo, end_lalo):
    """project one/multiple point(s) onto a line"""
    start_lalo = np.array(start_lalo)   # start_lalo = np.array([lat, lon])
    end_lalo   = np.array(end_lalo)     # end_lalo   = np.array([lat, lon])
    u    = (end_lalo-start_lalo)        # u = np.array([lat, lon])
    v    = (lalo-start_lalo)            # v = np.array([lat, lon])
    un   = np.linalg.norm(u)
    vn   = np.linalg.norm(v, axis=len(lalo.shape)-1)
    cos  = u.dot(v.T) / (un*vn).flatten()
    dpar = vn*cos                                    # distance parallel to line
    new_lalo = start_lalo.reshape(-1,2) + (dpar*((u/un).reshape(2,-1))).T
    new_lalo = new_lalo.reshape(lalo.shape)
    dper = np.linalg.norm(new_lalo-lalo, axis=len(lalo.shape)-1) # distance perpendicular to line
    if dpar.size == 1:
        dpar = float(dpar)
    return new_lalo, dpar, dper


def epi2projDist(lons, lats, start_lalo, end_lalo, dperMax=None, newlalo=False):
    """project points (lat, lon) onto a line and calc the distance along the line"""
    # Convert seismicity epicenter to projected distance on the approx fault trace
    # lons      :  longitude of the points to be projected
    # lats      :  latitude of the points to be projected
    # start_lalo:  proj line start
    # end_lalo  :  proj line end
    # dperMax   :  maximum perpendicular distance (deg) allowed (do not proj points that are too far)
    new_lalo, dpar, dper = pts_projline(np.array([lats,lons]).T, start_lalo, end_lalo)
    if dperMax is not None:
        idx = dper<=dperMax
        new_lalo = new_lalo[idx]
        dpar     = dpar[idx]
        dper     = dper[idx]
    tmp = new_lalo.reshape(-1,2)
    ev_dist  = np.zeros(len(tmp))
    for i in range(len(tmp)):
        ev_dist[i] = geodesic((start_lalo[0], start_lalo[1]), (tmp[i,0], tmp[i,1])).km
    if ev_dist.size == 1:
        ev_dist = float(ev_dist)
    if dperMax is not None:
        return ev_dist, idx
    if newlalo:
        return ev_dist, new_lalo
    return ev_dist


def init_metric_dict(vars):
    """initialize the metric dict for saving purposes"""
    metric = dict()
    for var in vars:
        metric[var] = []
    return metric


def run_fix_bin(m, x, cat, meta, bin_size='default', xlim='default'):
    """calc seismicity metrics along user-defined fixed bins"""
    if bin_size == 'default':
        bin_size = 100  # default = 100 km

    if xlim == 'default':
        xlim = [np.min(x), np.max(x)]

    b_edge  = np.arange(xlim[0], xlim[1], bin_size)
    b_mid   = b_edge+np.diff(b_edge)[0]/2
    N        = len(b_mid)

    # Get events within each bin
    inds = np.digitize(x, b_edge)

    # Loop bins and calc seismicity metrics
    for i in range(N):
        # get array_id of seismicity in each fault bin
        bid = inds==i+1

        # print progress
        print(i, b_mid[i], np.sum(bid))

        # Run calculation
        res = seis_metric(cat, bid, meta, nout=False)

        # append info seis metrcs for this fault bin
        if len(m) != len(res):
            raise ValueError('Input result m mismatch!')
        i = 0
        for key in m:
            m[key].append(res[i])
            i += 1
    m['BIN_MID_LOC'] = b_mid
    print('Fixed binning calculation is completed:\nbin_size = {}'.format(bin_size))
    return m


def run_moving_bin(m, x, cat, meta, bin_size='default', bin_step='default', xlim='default'):
    """calc seismicity metrics along user-defined moving window bins"""
    if bin_size == 'default':
        bin_size = 100  # default = 100 km

    if bin_step == 'default':
        bin_step = 10  # default = 10 km

    if xlim == 'default':
        xlim = [np.min(x), np.max(x)]

    N        = int(np.ceil((2 + (xlim[1]-bin_size)/bin_step)))
    b_mid   = np.zeros(N)

    # Loop bins and calc seismicity metrics
    for i in range(N):
        b_edge = np.array([0+bin_step*i, 0+bin_step*i+bin_size])
        b_mid[i] = np.mean(b_edge)

        # Get events within each bin
        inds = np.digitize(x, b_edge)

        # get id of array of seismicity in this fault bin
        bid = inds==1
        print(i, b_edge[0], b_mid[i], b_edge[1], np.sum(bid))

        # Run calculation
        res = seis_metric(cat, bid, meta, nout=False)

        # append info seis metrcs for this fault bin
        if len(m) != len(res):
            raise ValueError('Input result m mismatch!')
        i = 0
        for key in m:
            m[key].append(res[i])
            i += 1
    m['BIN_MID_LOC'] = b_mid
    print('Moving window calculation is completed:\nbin_size={}\nbin_step={}'.format(bin_size,bin_step))
    return m


def make_segment(start_lalo, end_lalo, seg_length, seg_width):
    """make segments of fault bins based on some geometry info (obsolete)"""
    lon12 = [end_lalo[1], start_lalo[1]]
    lat12 = [end_lalo[0], start_lalo[0]]

    dcos = np.diff(lon12) / (np.diff(lon12)**2 + np.diff(lon12)**2)**0.5
    dsin = np.diff(lat12) / (np.diff(lon12)**2 + np.diff(lon12)**2)**0.5
    seg_deg   = kilometers2degrees(seg_length)
    total_deg = locations2degrees(lat12[1], lon12[1], lat12[0], lon12[0])
    nbin      = int(np.round(total_deg/seg_deg))
    print('  number of segments made: ',nbin-1)

    lon_seg = np.linspace(lon12[1], lon12[0], nbin)
    lat_seg = np.linspace(lat12[1], lat12[0], nbin)

    width = kilometers2degrees(seg_width)
    lon_west = lon_seg - width*dsin
    lat_west = lat_seg + width*dcos
    lon_east = lon_seg + width*dsin
    lat_east = lat_seg - width*dcos
    return lon_seg, lat_seg, lon_west, lat_west, lon_east, lat_east, dcos, dsin


def seis_metric(catalog, bin_id, meta, nout=True):
    """ WORKFLOW:
    Calculate seismicity metrics within a certain fault bin
    input:  1) catalog array
            2) array id of events within bin
            3) array id of events > Mc
    """

    # selections
    if nout is True:
        print('Selected seismicity number = {}'.format(np.sum(bin_id)))

    # order events by origin times for interevent time calc
    if catalog.shape[0] == 7:
        evid, dtime, dtime_s, lat, lon, dep, mag = np.array(catalog).T[bin_id].T
        evid, dtime, dtime_s, lat, lon, dep, mag = np.array(catalog).T[bin_id][np.argsort(dtime_s)].T
    elif catalog.shape[0] == 10:
        evid, dtime, dtime_s, lat, lon, dep, mag, td, sd, mc = np.array(catalog).T[bin_id].T
        evid, dtime, dtime_s, lat, lon, dep, mag, td, sd, mc = np.array(catalog).T[bin_id][np.argsort(dtime_s)].T

    # calculate interevent times & mean seismic rate [num/sec]
    int_t = dtime_s[1:]-dtime_s[:-1]  # interevent times in each fault bin [sec]

    # Return NaN if only has one or zero event in the bin
    if (len(dtime_s)<2) or (len(int_t)==1) or  (dtime_s[-1]-dtime_s[0]<1e-6):
        print(' --> only has one/zero event, save as nans')
        return (evid, mag, dtime_s, int_t, lat, lon, dep, [np.nan]*3, [np.nan]*3, [np.nan]*3, [np.nan]*3, np.nan)

    ## seismic moment (Sum within 15-km window; N*m)
    moment = np.sum(10**(1.5*(mag.astype(float))+9.05))

    ## b-value:
    meanMag = np.mean(mag)
    b = (np.log10(np.exp(1)))/(meanMag - meta['Mc'])
    # Bootstrapping b for NN=10000 sets
    NN = 10000
    b_boot = []
    for k in range(NN):
        new_bmag = np.random.choice(mag, replace=True, size=len(mag))
        meanMag  = np.mean(new_bmag)
        b_boot.append((np.log10(np.exp(1)))/(meanMag - meta['Mc']))
    lb = np.percentile(b_boot,  2.5)
    ub = np.percentile(b_boot, 97.5)

    ## Event rate:
    meanRate = len(dtime_s)/(dtime_s[-1]-dtime_s[0])   # mean seismicity rate in each fault bin [num/sec]

    # calculate metrics of interevent times
    int_avg = np.mean(int_t)    # sec
    int_std = np.std(int_t)     # sec
    int_var = np.var(int_t)     # sec^2

    # calculate COV, background rate (BR), mainshock fraction (BF)
    cov = int_std / int_avg    # non-dimension
    BR  = (int_avg/int_var)    # num/sec
    BF  = BR / meanRate        # non-dimension
    BR  = BR * 365.25 * 86400  # num/year

    # Bootstrapping COV, BR, BF for NN sets
    Cov_boot = []
    BR_boot  = []
    BF_boot  = []
    for k in range(NN):
        new_dt = np.array(sorted(np.random.choice(int_t, replace=True, size=len(int_t))))
        Cov_boot.append(np.std(new_dt)/np.mean(new_dt))
        BR_boot.append(365.25*86400*np.mean(new_dt)/np.var(new_dt))
        BF_boot.append(BR_boot[k]/meanRate/(365.25 * 86400))
    lc = np.percentile(Cov_boot,  2.5)
    uc = np.percentile(Cov_boot, 97.5)
    lr = np.percentile(BR_boot,  2.5)
    ur = np.percentile(BR_boot, 97.5)
    lf = np.percentile(BF_boot,  2.5)
    uf = np.percentile(BF_boot, 97.5)

    # values and the corresponding uncertainties
    cov = [cov, lc, uc]
    BR  = [ BR, lr, ur]
    BF  = [ BF, lf, uf]
    b   = [  b, lb, ub]
    return (evid, mag, dtime_s, int_t, lat, lon, dep, cov, BR, BF, b, moment)


def plot_result(b_mid, num_arr, result, ystr, dist, meta, cat, x=0, y=0, ylim=[None,None], curve=False, lc='r', fc='lightpink', m_circ=5, log='no', cmap_max=50, titstr='metric', show=False):
    """
    The main script to plot the results.
    ## Plot Background seismicity versus creep rate.
    To-do:
        1. make the function inputs simpler
        2. ...
    """
    plt.rcParams.update({'font.size': 24})
    data, L, U = np.array(result).T
    if cat.shape[0] == 7:
        evid, dtime, dtime_s, lat, lon, dep, mag = cat
    elif cat.shape[0] == 10:
        evid, dtime, dtime_s, lat, lon, dep, mag, td, sd, mc = cat

    fig, ax = plt.subplots(nrows=3, gridspec_kw={'height_ratios':[1,0.4,1.8], 'hspace': 0.05}, sharex=True, figsize=[18,18])
    print(' 1. plot seismicity metrics: {}'.format(ystr))
    ax1 = ax[0]
    ax2 = ax1.twinx()
    if curve is False:
        ln1, = ax1.plot(b_mid, data, '-o', color=lc, lw=3, mfc=lc, mec='k', mew=2, markersize=15)
    elif curve is True:
        ln1, = ax1.plot(b_mid, data, c=lc, lw=5)
    ax1.fill_between(b_mid, L, U, fc=fc, ec='grey', ls='--', lw=2, alpha=0.5)
    ln2, = ax2.plot(x, np.abs(y), c='k', lw=5, zorder=0)
    ax1.set_ylabel(ystr, color=lc)
    ax2.set_ylabel('InSAR line-of-sight\ncreep rate [mm/year]', color='k', rotation=270, labelpad=50)

    trench = read_plate_bound(meta['PLATE_BOUND_FILE'], meta['PLATE_BOUND_NAME'], v=False)

    lalo = locs = []
    for key in meta:
        if key.startswith('LOC_'):
            locs.append(key.split('_')[1])
            lalo.append(np.array(meta[key]))
    lalo = np.array(lalo)
    loc_dist = calc_trench_project(lalo, trench)[-1] - meta['REF_LOC_DIST']

    for i in range(len(locs)):
        ax1.axvline(x=loc_dist[i], lw=1, c='k', alpha=0.2)
        ax1.text(loc_dist[i], ylim[1]*0.96, locs[i], rotation=270, ha='left', va='top', fontsize=18)
    ax1.set_ylim(ylim[0], ylim[1])
    ax2.set_ylim(0,1)
    ax1.xaxis.set_tick_params(which='both', labelbottom=False)
    plt.legend([ln1, ln2], [ystr, 'Creep rate'], loc='upper left', fontsize=18)

    print(' 2. plot event num histograms')
    ax3 = ax[1]
    ax3.bar(b_mid, num_arr, color='grey', width=len(num_arr)/15)
    ax3.set_ylabel('Num events')
    labels = 'Bin width {} km'.format(meta['BIN_WIDTH_MVW']), 'Bin step {} km'.format(meta['BIN_STEP_MVW'])
    handles = [patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * len(labels)
    ax3.legend(handles, labels, loc='best', fontsize=18, handlelength=0, handletextpad=0, borderaxespad=1.0)


    ax4 = ax[2]
    xedges = np.linspace(0, np.max(dist), int(np.max(dist)/10))
    yedges = np.linspace(0, (UTC(meta['ENDTIME'])-UTC(meta['STARTTIME']))/86400, int((UTC(meta['ENDTIME'])-UTC(meta['STARTTIME']))/86400/60))
    Dt = np.array(dtime_s)
    Dt = Dt - Dt[0]
    H, xedges, yedges = np.histogram2d(dist, np.array(Dt)/86400, bins=(xedges, yedges))
    H = H.T
    msg = ' 3. plot seismicity heatmap: '
    msg += 'time_bin {:.2f} days; space_bin {:.2f} km\n'.format(yedges[1]-yedges[0], xedges[1]-xedges[0])
    print(msg)
    scale = 30/(yedges[1]-yedges[0]) * 1/(xedges[1]-xedges[0])
    H = H * scale
    #H0 = np.mean(H, axis=0)
    #H  = (H-H0)/H0
    Hf = gaussian_filter(H, sigma=[1.5, 1.5])
    yedgdt = []
    for i in range(len(yedges)):
        yedgdt.append((UTC(meta['STARTTIME']) + yedges[i]*86400).datetime)
    X, Y = np.meshgrid(xedges, yedgdt)
    if log == 'no':
        im = ax4.pcolormesh(X, Y, 10*Hf, cmap='Reds', vmax=cmap_max)
        #print('\t\tplot normal scale intensity, cmap_max={}'.format(cmap_max))
    elif log == 'yes':
        #print('\t\tplot logscale intensity, ignore cmap_max')
        im = ax4.pcolormesh(X, Y, np.log10(10*Hf), cmap='Reds', vmin=-1, vmax=1.5)

    m_id = mag>=m_circ
    select = m_id
    sx = dist[select].astype('float')
    sy = np.array(dtime[select])
    ss = 2.9**mag[select].astype('float')
    ax4.scatter(sx, sy, s=ss, ec='k', fc='none', lw=1.5)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.20, 0.015, 0.2])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'Seismicity intensity [$\frac{1}{month \times 10km}$]', rotation=270, labelpad=40)
    ax4.set_xlabel('Along-strike distance [km]')
    ax4.set_ylabel('Time [calender year]')
    #ax4.set_xlim(0, xlim[1]-xlim[0])
    ax4.set_ylim(UTC(meta['STARTTIME']).datetime, UTC(meta['ENDTIME']).datetime)

    #ax1.set_title('{:} ({:})'.format(faultName, titstr), pad=50)
    if meta['SELECT_DEPTH']=='yes':
        figname = '{}/res_{}-{}_{}.png'.format(meta['PIC_DIR'], meta['DEP_MIN'], meta['DEP_MAX'], titstr)
    else:
        figname = '{}/res_{}.png'.format(meta['PIC_DIR'], titstr)
    plt.savefig(figname, bbox_inches='tight', dpi=300)
    plt.rcParams.update({'font.size': 22})
    if show:
        plt.show()


def plot_2D_distr(dist, dep, mag, meta):
    """plot 2D distribution of the hypocenters (distance vs depth)"""
    num, bin_edges = np.histogram(dep, np.arange(0,30,1))
    centers = bin_edges[:-1] + np.diff(bin_edges)[0]/2
    fig = plt.figure(figsize=[20,5])
    gs  = fig.add_gridspec(nrows=1, ncols=8)
    ax1 = fig.add_subplot(gs[:,:7])
    ax2 = fig.add_subplot(gs[:, 7])
    sc = ax1.scatter(dist, dep, s=10, c=mag, cmap='plasma_r', vmin=2, vmax=6.5)
    ax1.set_xlabel('Along-strike distance [km]')
    ax1.set_ylim(30,0)
    ax1.set_ylabel('Depth [km]')
    ax2.barh(bin_edges[:-1], num, fc='lightgray', ec='k')
    ax2.set_ylim(30,0)
    ax2.set_yticks([])
    ax2.set_xlabel('# events')
    plt.colorbar(sc, label='Magnitude')
    plt.savefig('{}/dist_depth.png'.format(meta['PIC_DIR']), bbox_inches='tight', dpi=300)
    plt.show()


def plot_3D_distr(lon, lat, dep, mag):
    """plot 3D distribution of the hypocenters with diff azimuth angles; output to png for making a movie"""

    X =  lon
    Y =  lat
    Z = -dep
    V =  mag

    fig, ax = plt.subplots(figsize=[10,10], subplot_kw={'projection':'3d'})
    plt.subplots_adjust(wspace=None, hspace=None)
    ax.set_box_aspect((1,1,0.2))
    ax.scatter(X, Y, Z, c=-Z, cmap='jet_r', marker='o', alpha=0.6, s=0.003*V**3)
    ax.set_xlabel('Lon')
    ax.set_ylabel('Lat')
    ax.set_zlabel('Depth')

    deg=2
    for ii in range(0,360,deg):
        ax.view_init(elev=22, azim=ii)
        fig.savefig("movie%03d.png" % int(ii//deg))


def find_indeces(pool, keys):
    idxs = []
    for key in keys:
        idxs.append(np.nonzero(pool==key)[0][0])
    return np.array(idxs)


def euclidean(p1, p2):
    """k-dim distance in euclidean space"""
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))


def geodesic2d(p1, p2):
    """lat lon geodesic distance"""
    return geodesic(p1, p2).km


def geodesic3d(p1, p2):
    """lat lon geodesic distance, considering depth as well"""
    hd = geodesic2d(p1[:2], p2[:2])
    vd = p1[2] - p2[2]
    return np.sqrt(hd**2 + vd**2)


def knn(query, pool, n=1, coord='geo'):
    """Geo/Euclidean space k-dim n_nearest-neighbor distance(s)"""
    N = pool.shape[0]
    k = pool.shape[1]
    print('Build k-dim VPtree in O(N log N) time complexity; k={} N={}'.format(k, N))

    if coord=='geo':
        if k==2:
            method = geodesic2d
        elif k==3:
            method = geodesic3d
        else:
            msg = 'Num of dimension {} is wrong with geo coordinates; should be either 2 or 3'
            raise ValueError(msg.format(k))
    elif coord=='euclidean':
        method = euclidean
    else:
        msg = '{} coordinate system not found; Either geo or euclidean'
        raise ValueError(msg.format(coord))

    time_go = time.time()
    tree = vptree.VPTree(pool, method)
    time_lapse = time.time() - time_go
    print('Tree developed, took {} min {:.2f} sec'.format(time_lapse//60, time_lapse%60))

    # Convert querying points to pandas dataframe
    query_df           = pd.DataFrame()
    query_df['coords'] = list(query)

    # Conduct the VPtree searching queries:
    print('Total {} queries to search'.format(len(query)))
    nn_df = query_df['coords'].apply(tree.get_n_nearest_neighbors, args=(n,))

    # get ndarray from DataFrame
    dists = np.array([*nn_df.values], dtype='object')[:,:,0]
    nns   = np.array([*nn_df.values], dtype='object')[:,:,1]

    # reshape arrays to output
    if n == 1:
        dists = dists.reshape(-1,).astype('float')
        nns   = np.concatenate(np.concatenate(nns)).reshape(-1,k)
    else:
        dists = dists.reshape(-1,n).astype('float')
        nns   = np.concatenate(np.concatenate(nns)).reshape(-1,n,k)
    return dists, nns


def calc_slab_distance(points, slab):
    """ Calculate distance from points to the slab model """
    dists, _ = knn(points, slab, n=1)
    return dists


def point2greatCircle(P, A, B):
    '''
    Calculates the shortest distance between a "single" point P
    and a great circle passing through A and B.
    Input:
    P       [lat, lon] of query point.
    A       [lat, lon] of first point defining the great circle.
    B       [lat, lon] of second point defining the great circle.

    Output:
    H       [lat, lon] on great circle closest to query point.
    dper    Perpendicular distance between great circle and point (km).
    dpar    Parallel distance from point A to the point H on the great circle (km).
    '''
    # Define an ellipsoid.
    ell = Geod(ellps='WGS84') # Use GRS80/WGS84 ellipsoid.

    # Great circle containing A and B
    az_ab, _,       _ = ell.inv(A[1], A[0], B[1], B[0])

    # Great circle containing A and P
    az_ap, _, dist_ap = ell.inv(A[1], A[0], P[1], P[0])

    # Find azimuth A to M (M is the mirrow image of P on the other side of line AB)
    az_am = az_ab + (az_ab - az_ap)

    # Find the mirror point M
    M = np.zeros(2)
    M[1], M[0],     _ = ell.fwd(A[1], A[0], az_am, dist_ap)

    # Point H is the half-way between P and M along a great circle.
    az_pm, _, dist_pm = ell.inv(P[1], P[0], M[1], M[0])
    dper = dist_pm/2
    H = np.zeros(2)
    H[1], H[0],     _ = ell.fwd(P[1], P[0], az_pm, dper)

    # Parellel distance along the great circle between A and H
    _, _, dpar = ell.inv(A[1], A[0], H[1], H[0])

    return H, M, dper/1e3, dpar/1e3


def calc_trench_project(points, trench):
    """ Calculate points projected on the trench trace """
    segs_length = []
    for i in range(len(trench)-1):
        segs_length.append(geodesic(trench[i], trench[i+1]).km)
    segs_length = np.array(segs_length)

    # find nearest 2 trench grids for each point
    _, nns = knn(points, trench, n=2)
    same_nn = (nns[:,0]==nns[:,1])[:,0] * (nns[:,0]==nns[:,1])[:,1]

    idxs = []
    for i in range(len(nns)):
        nns[i] = nns[i][np.argsort(nns[i][:,1])]
        idxs.append(np.where(trench == nns[i,0])[0][0])

    # point to greatCircle projection
    result = list(map(point2greatCircle, points, nns[:,0], nns[:,1]))
    result = np.array(result, dtype='object')

    H    = np.concatenate(result[:,0]).reshape(-1,2)
    M    = np.concatenate(result[:,1]).reshape(-1,2)
    dper = result[:,2].astype('float')
    dpar = result[:,3].astype('float')

    # Integrate segments to get a total trench-parallel distance
    N = len(points)
    Dpar = np.zeros(N)
    dpar[same_nn] = 0.0
    for i in range(N):
        Dpar[i] = np.sum(segs_length[:idxs[i]]) + dpar[i]

    if N == 1:
        return H[0], M[0], dper[0], Dpar[0]
    else:
        return H, M, dper, Dpar





##################################################################################
##                  Obsolete functions
##################################################################################

def find_nearest(array, value):
    ## temporarily deprecated
    array = np.asarray(array)
    value = np.asarray(value)
    idx = np.linalg.norm(array - value, axis=1).argmin()
    distance = np.sqrt(geodesic(array[idx][:2], value[:2]).km**2 + np.abs(array[idx][2]-value[2])**2)
    return idx, distance


def find_nearest_k(array, value, k, tl=0.2):
    ## temporarily deprecated
    array = np.asarray(array)
    value = np.asarray(value)

    lat0 = value[0]
    lon0 = value[1]
    sub  = (array[:,0]>lat0-tl)*(array[:,0]<lat0+tl)*(array[:,1]>lon0-tl)*(array[:,1]<lon0+tl)

    if np.sum(sub) < k:
        print(' less than {} candidates in nearest_{}th searching, report nan'.format(k,k))
        return [np.nan], [np.nan]

    sub_arr = array[sub]
    dist = []

    if len(value) == 3:
        # 3D distance
        for i in range(len(sub_arr)):
            dist.append(np.sqrt(geodesic(sub_arr[i,:2], value[:2]).km**2 + np.abs(sub_arr[i,2]-value[2])**2))
    elif len(value) == 2:
        # 2D distance
        for i in range(len(sub_arr)):
            dist.append(geodesic(sub_arr[i], value).km)
    dist = np.array(dist)

    dupl_n = 0
    while True:
        idxs = np.argsort(dist)[:k+dupl_n]
        dupl_n = k-len(set(dist[idxs]))
        if dupl_n <= 0:
            idxs = find_indeces(dist, sorted(list(set(dist[idxs]))))
            break

    nk_idxs = find_indeces(array, sub_arr[idxs])
    return nk_idxs, dist[idxs]


def dist_3Dplane(tri, p):
    ## temporarily deprecated
    p0 = tri[0]
    p1 = tri[1]
    p2 = tri[2]
    normal = np.cross( p1-p0, p2-p0)
    normal = normal / np.linalg.norm(normal)
    dist = np.abs( np.dot(p-p0, normal) )
    return dist, normal


def calc_trench_project_ob(lat, lon, trench, tl=10, newlalo=False, endpt=False, simple=False, v=False):
    ## temporarily deprecated
    """ Calculate trench projection location """
    segs_length = []
    for i in range(len(trench)-1):
        segs_length.append(geodesic(trench[i], trench[i+1]).km)
    segs_length = np.array(segs_length)
    td = []
    if (np.array(lat).size==1) and (np.array(lon).size==1):
        lat = np.array([lat])
        lon = np.array([lon])
    for i in range(len(lat)):
        if i%5000 == 0:
            if v:
                print('at {} event'.format(i))
        epi = np.array([lat[i], lon[i]])
        if simple:
            td.append(find_nearest_k(trench, epi, k=1, tl=tl)[1])
            idx = find_nearest_k(trench, epi, k=1, tl=tl)[0][0]
            new_lalo = trench[idx]
        else:
            idxs = find_nearest_k(trench, epi, k=2, tl=tl)[0]
            id1, id2 = sorted(idxs)
            lalo_s = trench[id1]
            lalo_e = trench[id2]
            if newlalo:
                di, new_lalo = epi2projDist(lon[i], lat[i], lalo_s, lalo_e, newlalo=newlalo)
            else:
                di = epi2projDist(lon[i], lat[i], lalo_s, lalo_e)
            if di < 0:
                print('{} {} got negative distance along trench'.format(lon[i], lat[i]))
            td.append(np.sum(segs_length[:id1]) + di)
    td = np.array(td)
    if td.size == 1:
        td = float(td)
    results = [td]
    if newlalo:
        results.append(new_lalo)
    if endpt:
        results.append(lalo_s)
        results.append(lalo_e)
    return results

def calc_slab_distance_ob(lat, lon, dep, slab, v=False):
    ## temporarily deprecated
    """ Calculate hypos from slab model distance """
    sd = []
    for i in range(len(lat)):
        if i%5000 == 0:
            if v:
                print('at {} event'.format(i))
        hypo = np.array([lat[i], lon[i], -dep[i]])
        sd.append(find_nearest_k(slab, hypo, k=1)[1][0])
    sd = np.array(sd)
    return sd
