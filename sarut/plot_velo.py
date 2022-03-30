#!/usr/bin/env python3

############################################################
# This code it meant to examine/plot the products from MintPy
# Yuan-Kai Liu @ 2022-03-28
############################################################

# Plot the velocity.h5 files as pngs. You can also geocode them into radar coord and plot them.
# Original terminal commands in bash:
#   view.py velocity.h5 blablabla...
#   geocode.py velocity.h5 -l ../inputs/radar/geometryRadar.h5 --outdir test/ --geo2radar

import os
import sys
import json
import glob
import argparse
from mintpy import view, geocode
from mintpy.utils import utils as ut
from mintpy.utils import readfile


def create_parser():
    description = ' Plot the velocity.h5 files as pngs. You can also geocode them into radar coord and plot them '

    ## basic input/output files and paths
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-j', '--json', dest='myjson', type=str, required=True,
            help = 'Input JSON file for parameters, e.g., params.json')
    parser.add_argument('-dir', '--dir', dest='dir', type=str, default='./',
            help = 'Velocity files directory. (default: %(default)s)')
    parser.add_argument('--outdir', dest='outdir', type=str, default='./pic_velo',
            help = 'Picture output directory. (default: %(default)s)')
    parser.add_argument('--dpi', dest='dpi', type=int, default=300,
            help = 'Picture DPI. (default: %(default)s)')
    parser.add_argument('-c', '--cmap', dest='cmap', type=str, default='RdYlBu_r',
            help = 'Colormap. (default: %(default)s)')
    parser.add_argument('-u', '--unit', dest='unit', type=str, default='mm',
            help = 'Unit. (default: %(default)s)')
    parser.add_argument('--radar', dest='radar', action='store_true',
            help = 'Turn on conversion to radar coordinate. Save to a radar/ folder. (default: turn off)')
    parser.add_argument('--rdrtable', dest='rdrtable', type=str, default='../inputs/radar/geometryRadar.h5',
            help = 'Radar coordinate geometry table. (default: %(default)s)')
    inps = parser.parse_args()
    if len(sys.argv)<1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return inps


#########################################
## Read parser arguments
inps = create_parser()

## Other parameters from the JSON file
with open(inps.myjson) as json_file:
    jdic = json.load(json_file)

proc_home  = os.path.expanduser(jdic['proc_home'])
veloDir    = os.path.expanduser('{}/{}'.format(proc_home, jdic['velo_dir']))
config     = '{}/{}'.format(proc_home, jdic['config'])
tmCoh_mask = '{}/{}'.format(proc_home, jdic['tcoh_mask'])
water_mask = '{}/{}'.format(proc_home, jdic['water_mask'])
refdate    = jdic['ref_date']
refla      = jdic['ref_lat']
reflo      = jdic['ref_lon']
xmin       = jdic['lon_min']
xmax       = jdic['lon_max']
ymin       = jdic['lat_min']
ymax       = jdic['lat_max']
dem_file   = jdic['dem_out']
alpha      = jdic['velo_alpha']
shade      = jdic['shade_exag']
v1         = jdic['vlim1']  # standard velocity field [mm/yr] (with ERA5, SET, demErr corrections)
v2         = jdic['vlim2']  # velocity after deramp/iono correction [mm/yr]
v3         = jdic['vlim3']  # velocity STD [mm/yr]
v4         = jdic['vlim4']  # for seasonal amplitudes [mm]
v5         = jdic['vlim5']  # SET field  [mm/yr]
v6         = jdic['vlim6']  # ERA5 or iono field  [mm/yr]


# Read basic attributes from .h5
head_title = 'vel'
atr_file = glob.glob('{}/{}*.h5'.format(inps.dir, head_title))[0]
_, atr   = readfile.read(atr_file, datasetName='velocity')


# Whether to convert the geocoded velocities to radar coord; save them under velocity_out/radar/
if inps.radar:
    rdrtable = inps.rdrtable
    rdrdir   = './radar/'
    if not os.path.exists(rdrdir):
        os.makedirs(rdrdir)

    opt = ['-l', rdrtable, '--outdir', rdrdir, '--geo2radar']
    geocode.main([tmCoh_mask]  + opt)
    geocode.main([dem_file]    + opt)
    geocode.main(['./vel*.h5'] + opt)
    geocode.main(['../avgSpatialCoh.h5']     + opt)
    geocode.main(['../temporalCoherence.h5'] + opt)

    inps.outdir = os.path.abspath(rdrdir+inps.outdir)
    inps.dir    = os.path.abspath(rdrdir+inps.dir)
    head_title  = 'rdr_' + head_title


# Output directory
if not os.path.exists(inps.outdir):
    os.makedirs(inps.outdir)


# Glob all files from input directory
vfiles = glob.glob('{}/{}*.h5'.format(inps.dir, head_title))


# Plot specifications
if inps.radar:
    coord = ut.coordinate(atr)
    refy = coord.lalo2yx(refla, coord_type='lat')
    refx = coord.lalo2yx(reflo, coord_type='lon')
    tmCoh_mask = inps.dir + '/rdr_' + os.path.basename(tmCoh_mask)
    dem_file   = inps.dir + '/rdr_' + os.path.basename(dem_file)
    opt  = ['--ref-yx', str(refy), str(refx), '--mask', tmCoh_mask, '--dem', dem_file]
else:
    roi  = ['--sub-lon', str(xmin), str(xmax), '--sub-lat', str(ymin), str(ymax)]
    opt  = roi + ['--ref-lalo', str(refla), str(reflo), '--mask', tmCoh_mask, '--dem', dem_file]

opt += ['--unit', inps.unit, '--dem-nocontour', '--shade-exag', str(shade)]
opt += ['-c', inps.cmap, '--alpha', str(alpha), '--dpi', str(inps.dpi), '--nodisplay', '--update']

dsets  = ['velocity', 'velocityStd']

# Loop over all the files and plot
for vfile in vfiles:
    key = os.path.basename(vfile).split('.')[0].split('velocity')[1]
    key = head_title + key
    for dset in dsets:
        if any(k in key for k in ['2', 'lr', 'qr']):
            vlim = ['--vlim', *[str(v) for v in v2]]
        elif 'SET' in key:
            vlim = ['--vlim', *[str(v) for v in v5]]
        elif any(k in key for k in ['ERA', 'Ion']):
            vlim = ['--vlim', *[str(v) for v in v6]]
        else:
            vlim = ['--vlim', *[str(v) for v in v1]]
        if dset == 'velocityStd':
            vlim = ['--vlim', *[str(v) for v in v3]]
            title = key+'_Std'
        else:
            title = key

        outname = '{}/{}.png'.format(os.path.expanduser(inps.outdir), title)
        out = ['-o', outname, '--figtitle', title]
        iargs = [vfile] + [dset] + opt + vlim + out
        #print(' '.join(iargs))
        view.main(iargs)

print('Complete plotting!')