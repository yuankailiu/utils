#!/usr/bin/env python3

############################################################
# This code it meant to examine/plot the products from MintPy
# Yuan-Kai Liu @ 2022.04.18
############################################################

## Use GDAL to resample the orignal DEM to match the extent, dimension, and resolution of
## MintPy geocoded .h5 products.
## [This is optional], just to cover the full extent when using topsStack radar coord datsets
##  (when geocode geometryRadar.h5 to geometryGeo.h5, the height will have large gaps; not pretty)
## Should be run after having the geometryGeo.h5 file (must in geo-coord to allow reading lon lat)
## The output DEM is then saved separetly (defined in `params.json` as "dem_out")
## The output DEM is mainly for plotting purposes using view.py

import os
import sys
import json
import argparse
from mintpy.utils import readfile


def create_parser():
    description = ' Plot the velocity.h5 files as pngs. You can also geocode them into radar coord and plot them '

    ## basic input/output files and paths
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-j', '--json', dest='myjson', type=str, required=True,
            help = 'Input JSON file for parameters, e.g., params.json')
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

proc_home = os.path.expanduser(jdic['proc_home'])
dem_out   = '{}/{}'.format(proc_home, jdic['dem_out'])
geo_file  = '{}/{}'.format(proc_home, jdic['geom_file'])
dem_orig  =    '{}'.format(jdic['dem_orig'])

# Read basic attributes from .h5
hgt = readfile.read(geo_file, datasetName='height')[0]
atr = readfile.read(geo_file, datasetName='height')[1]

# compute latitude and longitude min max

lon_min = float(atr['X_FIRST'])
lon_max = float(atr['X_FIRST']) + float(atr['X_STEP']) * int(atr['WIDTH'])
lat_max = float(atr['Y_FIRST'])
lat_min = float(atr['Y_FIRST']) + float(atr['Y_STEP']) * int(atr['LENGTH'])


print('Dimension of the dataset (length, width): {}, {}'.format(atr['LENGTH'], atr['WIDTH']))
print('S N W E: {} {} {} {}'.format(lat_min, lat_max, lon_min, lon_max))


# Check the directory
outdir = os.path.dirname(dem_out)
if not os.path.exists(outdir):
    os.makedirs(outdir)


# do gdalwarp on teh orignal DEM and output it


cmd = 'gdalwarp {} {} -te {} {} {} {} -ts {} {} -of ISCE '.format(
        dem_orig, dem_out, lon_min, lat_min, lon_max, lat_max, atr['WIDTH'], atr['LENGTH'])
os.system(cmd)

cmd = 'fixImageXml.py -i {} -f '.format(dem_out)
os.system(cmd)

print('Normal finish.')
