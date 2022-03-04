#!/usr/bin/env python3
############################################################
# This code it meant to examine the products from MintPy
# YKL @ 2021-05-19
############################################################


import os
import sys
import pyproj
import argparse
import numpy as np
import matplotlib.pyplot as plt

from mintpy.view import viewer

from sarut.tools.geod import (
    transec_pick,
    make_transec_swath
    )

# Inscrese matplotlib font size when plotting
plt.rcParams.update({'font.size': 16})


#################################################################

def cmdLineParse():
    description = ' Plot profiles of data (e.g., velocity) from mintpy hdf5 file '

    ## basic input/output files and paths
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', dest='infile', type=str, required=True,
            help = 'Input file for plotting, e.g., velocity.h5')
    parser.add_argument('-d', dest='geom_file', type=str, required=True,
            help = 'DEM file, e.g., geometryGeo.h5')
    parser.add_argument('-m', dest='mask_file', type=str, required=True,
            help = 'Mask file, e.g., maskTempCoh.h5, waterMask.h5')
    parser.add_argument('-p', dest='pic_dir', type=str, default='./pic_supp',
            help = 'Output picture directory. (default: %(default)s)')
    parser.add_argument('-o', dest='outfile', type=str, default='rms_ts2velo.png',
            help = 'output figure file name. (default: %(default)s)')

    ## define how the profile swath is spanning
    parser.add_argument('-l', '--prof0', dest='prof0', type=float, nargs=4, required=True,
            help = 'Define an initial profile; [lat0, lon0, lat1, lon1] (degree)\
                        e.g., [31.4046, 35.0121, 30.84, 36.89]')
    parser.add_argument('-f', '--fault', dest='fault', type=float, nargs=4, default=None,
            help = 'Define a fault line for profiles to span; [lat0, lon0, lat1, lon1] (degree)\
                        e.g., [30.9055, 35.3844, 28.3033, 34.5927]    # DST fault line')
    parser.add_argument('-s', '--strike', dest='strike', type=float, nargs=4, default=None,
            help = 'Define a strike for profiles to span (degree). The strike argument will overwrites fault option.\
                        e.g., 196       # fault strike of DST\
                              78        # across-track direction for sentinel descending track')

    ## define how wide is the swath, how many profiles in it will be generated
    parser.add_argument('-w', '--width', dest='width', type=float, default=50.0,
            help = 'Profile swath width. (default: %(default)s km)')
    parser.add_argument('--nprof', dest='nprof', type=int, default=60,
            help = 'Number of profiles in the swath. (default: n=%(default)s)')

    ## parameters for presenting stacks of profiles
    parser.add_argument('--option', dest='option', type=str, default='L1',
            help = 'Option for averaging the stacks of profiles. (default: n=%(default)s)')
    parser.add_argument('--nstack', dest='nstack', type=int, default=20,
            help = 'Number of profiles to stack and average for presenting. (default: n=%(default)s)')
    parser.add_argument('--binsize', dest='bin_size', type=float, default=2.0,
            help = 'Averaged fault-perpendicular distance bin (km). (default: %(default)s km)')
    parser.add_argument('--shift', dest='shift', type=float, default=50.0,
            help = 'Approx. center of the fault, set as the distance origin (km). (default: %(default)s km)')
    parser.add_argument('-c', '--binsize', dest='cmap', type=str, default='RdYlBu_r',
            help = 'output figure colormap. (default: %(default)s)')
    parser.add_argument('--sub-lat', dest='sub_lat', type=float, nargs=2, default=[None, None],
            help = 'Subset of latitudes; [lat_min, lat_max] (degree). (default: %(default)s)')
    parser.add_argument('--sub-lon', dest='sub_lon', type=float, nargs=2, default=[None, None],
            help = 'Subset of longitudes; [lon_min, lon_max] (degree). (default: %(default)s)')
    parser.add_argument('-v', '--vlim', dest='vlim', type=float, nargs=2, default=[None, None],
            help = 'Value limit of displaying the map and profiles; [vmin, vmax] (degree). (default: %(default)s)')


    inps = parser.parse_args()
    if len(sys.argv)<1:
        print('')
        parser.print_help()
        sys.exit(1)
    elif (inps.fault is None) and (inps.strike is None):
        print('Define at least either fault or strike!!')
        parser.print_help()
        sys.exit(1)
    else:
        return inps


def generate_profiles_swath(prof0, fault=None, strike=None, width=10, nprof=20):
    """
    prof0       Initial profile; [lat0, lon0, lat1, lon1] (degree)
                e.g., [31.4046, 35.0121, 30.84, 36.89]        # along DST fault (to see tectonic signal)
                      [31.5581, 36.0175, 27.6944, 36.8021]    # along track (to check along-track ramp)
    fault       Define a fault line for profiles to span; [lat0, lon0, lat1, lon1] (degree)
                e.g., [30.9055, 35.3844, 28.3033, 34.5927]    # DST fault line
    strike      Define a strike for profiles to span (degree). The strike argument will overwrites fault option.
                e.g., 196       # fault strike of DST
                      78        # across-track direction for sentinel descending track
    width       Width of the profile swath (kilometers)
    nprof       Number of profiles in swath, evenly spanned
    """

    if fault and strike is None:
        print('Error: both fault and strike are not given. Please specify at least one of them. \
                      strike will overwrite fault argument if both are given')
        sys.exit(1)

    width *= 1e3   # convert kilometer to meter

    ## (1) spanning a given fault:
    if fault:
        print('Estimate the parallel profile based on the given fault geometry')
        geod = pyproj.Geod(ellps='WGS84')
        az12, az21, dist = geod.inv(fault[1], fault[0], fault[3], fault[2])
        print('Fault line geometry of {}:'.format(fault))
        print('az12 = {:.2f} deg'.format(az12))
        print('az21 = {:.2f} deg'.format(az21))
        print('fault_length = {:.2f} km'.format(dist/1e3))
        strike    = az12
        prof1     = list(geod.fwd(prof0[1], prof0[0], strike, width)[:2])[::-1]
        prof1[2:] = list(geod.fwd(prof0[3], prof0[2], strike, width)[:2])[::-1]

    ## (2) spanning a given strike (degree):
    if strike:
        print('Estimate the parallel profile based on the given strike')
        if fault:
            print('strike argument overwrites fault option!!')
        print('The specified strike: {:.2f} deg'.format(strike))
        prof1     = list(geod.fwd(prof0[1], prof0[0], strike, width)[:2])[::-1]
        prof1[2:] = list(geod.fwd(prof0[3], prof0[2], strike, width)[:2])[::-1]

    print('Est parallel profile', prof1)
    print('Now generate a swath of profiles between the initial and the parallel profiles')
    profs = make_transec_swath(prof0[0:2], prof0[2:4], prof1[0:2], prof1[2:4], nprof)
    return profs


def plot_prof_swath_loc(inps, cmap='RdYlBu_r', prof_txt='prof_tmp_pts.txt', sub_lon=[None, None], sub_lat=[None, None], vlim=[None, None]):
    tmp_str   = inps.velo_file.split('.')[-1].split('/')[-1]
    cmd =  f'view.py {inps.velo_file} velocity --noverbose '
    cmd += f'--pts-file {prof_txt} --pts-marker wo --pts-ms 5 '
    cmd += f'-m {inps.mask_file} -d {inps.geom_file} -c {cmap} '
    cmd += f'--alpha 0.7 --dem-nocontour --shade-exag 0.05 --figtitle {tmp_str} '
    cmd += f'--sub-lon {sub_lon[0]} {sub_lon[1]} --sub-lat {sub_lat[0]} {sub_lat[1]} '
    cmd += f'--vlim {vlim[0]} {vlim[1]} -u mm '
    cmd += f'--outfile {inps.pic_dir}/profiles_{tmp_str}.png'
    obj = viewer(cmd)
    obj.configure()
    obj.plot()


def plot_stack_profiles(res, inps, n=20, shift=50):
    """
    res     the result of all the profiles
    inps    other arguments inputs (for output dir)
    n       the number of profiles to stack (stack every n profiles)
    shift   approx center of the fault, set as the distance origin. E.g., 50
    """
    nplot = int(len(res)/n)
    fig, axs = plt.subplots(figsize=[8, 3*nplot], nrows=int(len(res)/n), sharex=True,  gridspec_kw = {'wspace':0, 'hspace':0.05})
    for i in range(nplot):
        ax = axs[i]
        xs = []
        zs = []
        zes= []
        for j in np.arange(i*n, (i+1)*n):
            xs = xs + list( res[j]['distance']/ 1e3)
            zs = zs + list( res[j]['value']   * 1e3)
            zes= zes+ list(rese[j]['value']   * 1e3)
        xs  = np.array(xs) - shift
        zs  = np.array(zs) - np.mean(zs)
        zes = np.array(zes)

        ax.scatter(xs, zs, fc='whitesmoke', ec='lightgrey', alpha=0.2)
        markers, caps, bars = ax.errorbar(xs, zs, yerr=1*zes, mfc='cornflowerblue', mec='k',
                                            fmt='o', errorevery=10, elinewidth=3, capsize=4, capthick=3)
        [bar.set_alpha(0.2) for bar in bars]
        [cap.set_alpha(0.2) for cap in caps]
        ax.set_xlim(0-shift, 180-shift)
        ax.set_ylim(-2.5, 2.5)
        ax.set_ylabel('LOS velo\n[mm/yr]')
        if ax == axs[-1]:
            ax.set_xlabel('Across-fault distance [km]')
    axs[0].set_title('All pixels in each profile, demean (vel & 1*ts2vel_std)')
    filename = '{}/transects_{}_err.png'.format(inps.pic_dir, n)
    plt.savefig(filename, dpi=150, bbox_inches='tight')


def plot_avg_stack_profiles(res, inps, bin_size=2, n=20, shift=50, option='L1'):
    """
    res         the result of all the profiles
    inps        other arguments inputs (for output dir)
    bin_size    averaged fault-perpendicular distance bin (km). E.g., 2
    n           the number of profiles to stack (stack every n profiles). E.g., 20
    shift       approx center of the fault, set as the distance origin. E.g., 50
    option      way to represent the average stack of profiles
                'L1':   using median and MAD
                'L2':   using mean and STD
    """
    nplot    = int(len(res)/n)
    n_bins   = int(np.abs(max(xs)-min(xs)) / bin_size)

    fig, axs = plt.subplots(figsize=[8, 3*nplot], nrows=int(len(res)/n), sharex=True,  gridspec_kw = {'wspace':0, 'hspace':0.05})
    for i in range(nplot):
        ax = axs[i]     # plotting axis

        # get values from input result
        xs, zs, zes, loc = [], [], [], []
        for j in np.arange(i*n, (i+1)*n):
            xs = xs + list( res[j]['distance']/ 1e3)
            zs = zs + list( res[j]['value']   * 1e3)
            zes= zes+ list(rese[j]['value']   * 1e3)
        xs  = np.array(xs) - shift
        zs  = np.array(zs) - np.mean(zs)
        zes = np.array(zes)
        xs_sort = xs[np.argsort(xs)]
        zs_sort = zs[np.argsort(xs)]

        # L1-norm
        med, mad = [], []
        # L2-norm
        avg, std = [], []

        for n_bin in range(n_bins):
            start =  n_bin    * bin_size - shift
            end   = (n_bin+1) * bin_size - shift
            if max(xs_sort) < start:
                continue
            start_id = np.where(xs_sort >= start)[0][0]
            end_id   = np.where(xs_sort < end)[0][-1]
            #print(' bin {}, {} pixels'.format(n_bin+1, end_id-start_id))
            ## check the bin range is positive
            if end_id-start_id > 0:
                med.append(np.median(zs_sort[start_id:end_id]))
                mad.append(np.median(np.abs(zs_sort[start_id:end_id]-np.median(zs_sort[start_id:end_id]))))
                avg.append(np.mean(zs_sort[start_id:end_id]))
                std.append(np.std(zs_sort[start_id:end_id]))
                loc.append((start+end)/2)
        med = np.array(med)
        mad = np.array(mad)
        avg = np.array(avg)
        std = np.array(std)
        loc = np.array(loc)

        if any(option in s for s in ['Med', 'med', 'median', 'L1']):
            t_str = ['Med', '(vel med & 1*mad)']
            plot_y = med
            plot_yerr = 1 * mad
            option = 'L1'
        elif any(option in s for s in ['Avg', 'avg', 'Mean', 'mean', 'L2']):
            t_str = ['Avg', '(vel avg & 1*std)']
            plot_y = avg
            plot_yerr = 1 * std
            option = 'L2'

        ax.scatter(xs, zs, fc='grey', ec='lightgrey', alpha=0.2)
        markers, caps, bars = ax.errorbar(loc, plot_y, yerr=plot_yerr, mfc='coral', mec='k',
                    fmt='o', errorevery=1, elinewidth=1, capsize=4, capthick=1, ecolor='r')
        [bar.set_alpha(0.6) for bar in bars]
        [cap.set_alpha(0.6) for cap in caps]
        ax.set_xlim(0-shift, 180-shift)
        ax.set_ylim(-4, 4)
        ax.set_ylabel('LOS velo\n[mm/yr]')
        if ax == axs[-1]:
            ax.set_xlabel('Across-fault distance [km]')

    filename = '{}/transects{}_{}_{}_{}km_err.pdf'.format(inps.pic_dir, option, t_str[0], n, bin_size)
    axs[0].set_title('Binned {}, demean {}'.format(t_str[0], t_str[1]))
    plt.savefig(filename, dpi=300, bbox_inches='tight')


#################################################################


if __name__ == '__main__':

    inps = cmdLineParse()

    inps.proj_dir  = os.path.expanduser(os.getcwd())
    inps.pic_dir   = os.path.expanduser(f'{inps.pic_dir}')
    inps.geom_file = os.path.expanduser(f'{inps.geom_file}')
    inps.velo_file = os.path.expanduser(f'{inps.velo_file}')
    inps.mask_file = os.path.expanduser(f'{inps.mask_file}')

    os.chdir(inps.proj_dir)
    if not os.path.exists(inps.pic_dir):
        os.makedirs(inps.pic_dir)

    print('MintPy project directory: {}'.format(inps.proj_dir))
    print('Pictures will be saved to: {}'.format(inps.pic_dir))


    ## generate swath and the location txt files
    prof0  = inps.prof0
    fault  = inps.fault
    strike = inps.strike
    width  = inps.width
    nprof  = inps.nprof
    profs  = generate_profiles_swath(prof0, fault=fault, strike=None, width=width, nprof=nprof)


    ## plot the profiles locations on the map
    sub_lon = inps.sub_lon      # [34,   37.5]    (subset for Gulf of Aqaba)
    sub_lat = inps.sub_lat      # [27.5, 31.7]    (subset for Gulf of Aqaba)
    vlim    = inps.vlim         # [-4,   4]       (subset for Gulf of Aqaba)
    plot_prof_swath_loc(inps, cmap=inps.cmap, sub_lon=sub_lon, sub_lat=sub_lat, vlim=vlim)


    ## Just test the plotting with a file
    if False:
        test_file = inps.infile
        x,  z,  res  = transec_pick(test_file, None, 'prof_tmp.txt', fmt='lalo', mask_file=None)
        idx = np.argsort(x)
        x = x[idx]
        z = z[idx]
        plt.figure(figsize=[10,10])
        plt.plot(x, z/1e3, '-ok', lw=0.5, ms=1)
        #plt.scatter(x, z/1e3, s=1)
        plt.xlabel('Southing distance along track [km]')
        plt.ylabel('Radian')
        plt.show()


    ## get values along the velocity profiles
    x,  z,  res  = transec_pick(inps.velo_file, 'velocity',    'prof_tmp.txt', fmt='lalo', mask_file=inps.mask_file)
    xe, ze, rese = transec_pick(inps.velo_file, 'velocityStd', 'prof_tmp.txt', fmt='lalo', mask_file=inps.mask_file)


    ##  Stack scatter profiles
    plot_stack_profiles(res, inps, n=inps.nstack, shift=inps.shift)


    ## Stack averaged profiles
    plot_avg_stack_profiles(res, inps, bin_size=inps.bin_size, n=inps.nstack, shift=inps.shift, option=inps.option)


    print('**********\nNormal end of the script\n**********')
