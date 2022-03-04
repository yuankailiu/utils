#!/usr/bin/env python3
############################################################
# This code it meant to examine the products from MintPy
# YKL @ 2021-05-19
############################################################

# This code is not complete, need to work on it...

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from mintpy.utils import readfile
from mintpy import solid_earth_tides
from sarut.tools.geod import line_azimuth, lalo2yx


plt.rcParams.update({'font.size': 16})


##################################################################

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


## define a fault with endpoints (todo...)
fault_p1 = []
fault_p2 = []


# Get a sense of how near-range and far-range will affect my LOS observation sensitivity

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
