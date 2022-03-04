#!/usr/bin/env python3
############################################################
# This code it meant to examine the products from MintPy
# YKL @ 2021-05-19
############################################################

#%%
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


#%%
#proj_dir  = os.path.expanduser(os.getcwd())
proj_dir  = os.path.expanduser('/net/kraken/nobak/ykliu/aqaba/a087/isce/mintpy_ionsbas')
geom_file = 'geometryGeo.h5'
mask_file = 'maskTempCoh095.h5'  # or 'maskTempCoh.h5', 'waterMask.h5'
ts_file   = 'timeseries.h5'
#velo_file = 'velocity_P.h5'
velo_file = 'velocity02.h5'
pic_dir   = os.path.expanduser(f'{proj_dir}/pic_supp')
geom_file = os.path.expanduser(f'{proj_dir}/inputs/{geom_file}')
ts_file   = os.path.expanduser(f'{proj_dir}/{ts_file}')
velo_file = os.path.expanduser(f'{proj_dir}/velocity_out/{velo_file}')
mask_file = os.path.expanduser(f'{proj_dir}/{mask_file}')

os.chdir(proj_dir)
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

print(f'MintPy project directory:\t{proj_dir}')
print(f'Pictures will be saved to:\t{pic_dir}')

print('Read lat lon coordinate attributes from {}'.format(ts_file))
atr   = readfile.read(ts_file, datasetName='timeseries')[1]  # get metadata
coord = coordinate(atr)

#%% generate a transect swath

# define an approximate fault line in lat lon
fault_line = [[30.9055, 35.3844], [28.3033, 34.5927]]
strike     = line_azimuth(lalo2yx(fault_line[0]), lalo2yx(fault_line[1]))   # fault strike [deg]

step_deg        = np.hypot(float(atr['X_STEP']), float(atr['Y_STEP']))/np.sqrt(2)     # deg/pixel
earth_radius    = 6.3781e6    # in meter
transec_swath_w = 300 * 1e3 # meter
transec_swath_w = 200.0 * transec_swath_w / earth_radius / np.pi / step_deg

# define an initial profile line in lat lon
#prof0 = [[30.7629, 34.8204], [30.3963, 35.9621]]
#prof0 = [[31.4046, 35.0121], [31.0379, 36.1537]]
prof0 = [[31.4046, 35.0121], [30.84, 36.89]]

prof1 = parallel_line(lalo2yx(prof0[0]), lalo2yx(prof0[1]), strike, transec_swath_w)
prof1 = [yx2lalo(prof1[0]), yx2lalo(prof1[1])]
nprof = 60
profs = make_transec_swath(prof0[0], prof0[1], prof1[0], prof1[1], nprof)

#%%

tmp_str   = velo_file.split('.')[0].split('/')[-1]
colormap = 'RdYlBu_r'

cmd =  f'view.py {velo_file} velocity --noverbose '
cmd += f'--pts-file prof_tmp_pts.txt --pts-marker wo --pts-ms 5 '
cmd += f'-m {mask_file} -d {geom_file} -c {colormap} '
cmd += f'--alpha 0.7 --dem-nocontour --shade-exag 0.05 --figtitle {tmp_str} '
cmd += f'--sub-lon 34 37.5 --sub-lat 27.5 31.7  --vlim -4 4 -u mm '
cmd += f'--outfile {pic_dir}/profiles_{tmp_str}.png'
obj = viewer(cmd)
obj.configure()
obj.plot()

#%% get values along the velocity profiles

x,  z,  res  = transec_pick(velo_file, 'velocity',    'prof_tmp.txt', fmt='lalo', mask_file=mask_file)
xe, ze, rese = transec_pick(velo_file, 'velocityStd', 'prof_tmp.txt', fmt='lalo', mask_file=mask_file)

#%%  Stack scatter profiles
n     = 20
nplot = int(len(res)/n)
center_shift = 50  # approx center of the fault, set as the distance origin
fig, axs = plt.subplots(figsize=[8, 3*nplot], nrows=int(len(res)/n), sharex=True,  gridspec_kw = {'wspace':0, 'hspace':0.05})
for i in range(nplot):
    ax = axs[i]
    xs = []
    zs = []
    zes= []
    for j in np.arange(i*n, (i+1)*n):
        xs = xs + list(res[j]['distance']/ 1000)
        zs = zs + list(res[j]['value']   * 1000)
        zes= zes+ list(rese[j]['value']  * 1000)
    xs  = np.array(xs) - center_shift
    zs  = np.array(zs) - np.mean(zs)
    zes = np.array(zes)

    ax.scatter(xs, zs, fc='whitesmoke', ec='lightgrey', alpha=0.2)
    markers, caps, bars = ax.errorbar(xs, zs, yerr=1*zes, mfc='cornflowerblue', mec='k',
                                        fmt='o', errorevery=10, elinewidth=3, capsize=4, capthick=3)
    [bar.set_alpha(0.2) for bar in bars]
    [cap.set_alpha(0.2) for cap in caps]
    ax.set_xlim(0-center_shift, 180-center_shift)
    ax.set_ylim(-2.5, 2.5)
    ax.set_ylabel('LOS velo\n[mm/yr]')
    if ax == axs[-1]:
        ax.set_xlabel('Across-fault distance [km]')
axs[0].set_title('All pixels in each profile, demean (vel & 1*ts2vel_std)')
filename = '{}/transects_{}_err.png'.format(pic_dir, n)
plt.savefig(filename, dpi=150, bbox_inches='tight')




#%% Stack averaged profiles
option = 'L1'

n        = 20         # stack how many profiles
nplot    = int(len(res)/n)
center_shift = 50  # approx center of the fault, set as the distance origin

bin_size = 2      # km; averaged fault-perpendicular distance bin
n_bins   = int(np.abs(max(xs)-min(xs)) / bin_size)


fig, axs = plt.subplots(figsize=[8, 3*nplot], nrows=int(len(res)/n), sharex=True,  gridspec_kw = {'wspace':0, 'hspace':0.05})
for i in range(nplot):
    ax = axs[i]
    xs = []
    zs = []
    zes= []
    for j in np.arange(i*n, (i+1)*n):
        xs = xs + list(res[j]['distance']/ 1000)
        zs = zs + list(res[j]['value']   * 1000)
        zes= zes+ list(rese[j]['value']  * 1000)
    xs  = np.array(xs) - center_shift
    zs  = np.array(zs) - np.mean(zs)
    zes = np.array(zes)

    xs_sort = xs[np.argsort(xs)]
    zs_sort = zs[np.argsort(xs)]

    # L1-norm
    med = []
    mad = []

    # L2-norm
    avg = []
    std = []

    loc = []
    for n_bin in range(n_bins):
        start    =  n_bin    * bin_size - center_shift
        end      = (n_bin+1) * bin_size - center_shift
        if max(xs_sort) < start:
            continue
        start_id = np.where(xs_sort >= start)[0][0]
        end_id   = np.where(xs_sort < end)[0][-1]
        print(' bin {}, {} pixels'.format(n_bin+1, end_id-start_id))
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

    if any(option in s for s in ['Avg', 'avg', 'Mean', 'mean', 'L2']):
        t_str = ['Avg', '(vel avg & 1*std)']
        plot_y = avg
        plot_yerr = 1 * std
    elif any(option in s for s in ['Med', 'med', 'median', 'L1']):
        t_str = ['Med', '(vel med & 1*mad)']
        plot_y = med
        plot_yerr = 1 * mad

    ax.scatter(xs, zs, fc='whitesmoke', ec='lightgrey', alpha=0.2)
    markers, caps, bars = ax.errorbar(loc, plot_y, yerr=plot_yerr, mfc='coral', mec='k',
                                        fmt='o', errorevery=1, elinewidth=3, capsize=4, capthick=3)
    [bar.set_alpha(0.2) for bar in bars]
    [cap.set_alpha(0.2) for cap in caps]
    ax.set_xlim(0-center_shift, 180-center_shift)
    ax.set_ylim(-4, 4)
    ax.set_ylabel('LOS velo\n[mm/yr]')
    if ax == axs[-1]:
        ax.set_xlabel('Across-fault distance [km]')

pic_dir = f'{proj_dir}/pic_supp'
filename = '{}/transects{}_P_{}_{}km_err.png'.format(pic_dir, t_str[0], n, bin_size)
axs[0].set_title('Binned {}, demean {}'.format(t_str[0], t_str[1]))
plt.savefig(filename, dpi=150, bbox_inches='tight')






#%% Get a sense of how near-range and far-range will affect my LOS observation sensitivity

if False:
    geom_file = '{}/inputs/geometryGeo.h5'.format(proj_dir)
    mask_file = '{}/maskTempCoh.h5'.format(proj_dir)   # 'waterMask.h5' or 'maskTempCoh.h5'
    mask_data = readfile.read(mask_file)[0]

    # prepare LOS geometry: geocoding if in radar-coordinates
    inc_angle, head_angle, atr = solid_earth_tides.prepare_los_geometry(geom_file)
    length     = int(atr['LENGTH'])
    width      = int(atr['WIDTH'])
    x_min      = float(atr['X_FIRST'])
    x_step     = float(atr['X_STEP'])
    y_min      = float(atr['Y_FIRST'])
    y_step     = float(atr['Y_STEP'])
    lats       = np.arange(y_min,length*y_step+y_min, y_step)
    lons       = np.arange(x_min, width*x_step+x_min, x_step)
    wavelength = float(atr['WAVELENGTH'])
    Lons, Lats = np.meshgrid(lons, lats)

    # get LOS unit vector
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        unit_vec = [
            np.sin(inc_angle) * np.cos(head_angle) * -1,
            np.sin(inc_angle) * np.sin(head_angle),
            np.cos(inc_angle),
        ]

    ## convert ENU slip to LOS direction
    # sign convention: positive for motion towards satellite
    print('Calc ENU slip in LOS direction...')
    de_los = (1 * unit_vec[0] + 0 * unit_vec[1] + 0 * unit_vec[2])
    dn_los = (0 * unit_vec[0] + 1 * unit_vec[1] + 0 * unit_vec[2])
    du_los = (0 * unit_vec[0] + 0 * unit_vec[1] + 1 * unit_vec[2])
    unit='mm/yr'
    cbarstr1 = f'LOS [{unit}]'
    cbarstr2 = 'angle [deg]'

    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=[11,12], sharey=True, sharex=True, gridspec_kw = {'wspace':0, 'hspace':0.1})
    for ax, data, fig_title, cbarstr in zip(axs.flatten(),
                                [de_los, dn_los, du_los, np.rad2deg(inc_angle), np.rad2deg(head_angle)],
                                [f'1{unit} East', f'1{unit} North', f'1{unit} Up', 'Incidence angle', 'Head angle'],
                                [cbarstr1, cbarstr1, cbarstr1, cbarstr2, cbarstr2]):
        data[mask_data==0] = np.nan
        cntr = ax.contour(Lons, Lats, data, 5, colors='black', linewidths=2)
        ax.clabel(cntr, inline_spacing=1, fmt='%.2f', fontsize=10)
        im   = ax.imshow(data, extent=[lons[0],lons[-1],lats[-1],lats[0]])
        cbax = inset_axes(ax, width="40%", height="4%", loc='upper left',
                        bbox_to_anchor=(+0.1,0,1,1), bbox_transform=ax.transAxes)
        cbar = plt.colorbar(im, cax=cbax, orientation='horizontal', label=cbarstr)
        if ax in axs.flatten()[3:-1]:
            ax.set_xlabel('Longitude')
        if ax == axs.flatten()[0]:
            ax.set_ylabel('Latitude')
        ax.set_title(fig_title)
    axs[-1, -1].axis('off')
    plt.savefig(f'{pic_dir}/slip_enu.png', dpi=150, bbox_inches='tight')


    ## Strike-slip motion of the fault
    print('Calc assumed strike-slip motion in LOS direction...')
    vmin, vmax = 1.2, 1.6

    # Fault slip orientation
    s      = 5     # mm
    strike = line_azimuth(lalo2yx(fault_p1), lalo2yx(fault_p2))   # deg
    dip    = 0     # deg

    # fault slip into enu components
    s_u    = s * -np.sin(np.deg2rad(dip))
    s_n    = s *  np.cos(np.deg2rad(dip)) * np.cos(np.deg2rad(strike))
    s_e    = s *  np.cos(np.deg2rad(dip)) * np.sin(np.deg2rad(strike))
    data   = s_e*de_los + s_n*dn_los + s_u*du_los
    data[mask_data==0] = np.nan

    print(' North \t East \t Up')
    print(' {:.2f} \t {:.2f} \t {:.2f}'.format(s_n, s_e, s_u))
    fig_title = r'Strike-slip ({:.0f}$^{{\circ}}$) motion {:.1f} mm/yr'.format(strike, s)

    fig, ax = plt.subplots(figsize=[8,12])
    cntr    = ax.contour(Lons, Lats, data, 15, colors='black', linewidths=2)
    ax.clabel(cntr, inline_spacing=1, fmt='%.2f', fontsize=10)
    im      = ax.imshow(data, extent=[lons[0],lons[-1],lats[-1],lats[0]], vmin=vmin, vmax=vmax)
    cbax    = inset_axes(ax, width="30%", height="3%", loc='upper left',
                        bbox_to_anchor=(+0.05,0,1,1), bbox_transform=ax.transAxes)
    cbar    = plt.colorbar(im, cax=cbax, orientation='horizontal', label=f'LOS [{unit}]',
                            ticks=np.arange(vmin,vmax,0.2))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(fig_title)
    plt.savefig(f'{pic_dir}/slip_strike.png', dpi=150, bbox_inches='tight')


    # %%
