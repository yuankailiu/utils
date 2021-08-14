#!/usr/bin/env python3
# ykliu @ Jul 07, 2021
#
# Get lat and lon files in geo-coordinates
# read a template geocoded files to get the lon/lat coordinates

import sys
import numpy as np
import isce
import argparse
import isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml
from mintpy.utils import readfile

def cmdLineParse():
    '''
    Command line parsers
    '''
    description = 'Generate geo-coord lat/lon files from a template geocoded file, or given lat/lon info'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-t', dest='templFile', type=str, default=None,
            help = 'geocoded template file, e.g., hgt.geo.vrt, hgt.geo.xml')
    parser.add_argument('--lalo-start', dest='ll_start', type=float, nargs='+', default=[],
            help = 'starting lat, lon (mandatory if templFile not given)')
    parser.add_argument('--lalo-step', dest='ll_step', type=float, nargs='+', default=[],
            help = 'step of lat, lon (mandatory if templFile not given)')
    parser.add_argument('--width', dest='width', type=int, default=None,
            help = 'width of the cooridnates (mandatory if templFile not given)')
    parser.add_argument('--length', dest='length', type=int, default=None,
            help = 'length of the cooridnates (mandatory if templFile not given)')
    parser.add_argument('--out-lat', dest='out_lat', type=str, default='./lat.geo',
            help = 'output filename for the latitude file (default: %(default)s).')
    parser.add_argument('--out-lon', dest='out_lon', type=str, default='./lon.geo',
            help = 'output filename for the longitude file (default: %(default)s).')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        print('')
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()

    if inps.templFile is None:
        if (inps.ll_start==[]) or (inps.ll_step==[]) or (inps.width is None) or (inps.length is None):
            raise ValueError('You must either specify 1) a template file or 2) lat/lon info including all of the following: \n \
            --lalo-start, --lalo-step, --width, --length\n')
        else:
            print('Use input lat/lon information to create lat lon files')
            width   = int(inps.width)
            length  = int(inps.length)
            y0      = float(inps.ll_start[0])
            x0      = float(inps.ll_start[1])
            dy      = float(inps.ll_step[0])
            dx      = float(inps.ll_step[1])
    else:
        print('Use input geocoded template file (.vrt or .xml) to create lat lon files')
        infile = inps.templFile
        print(' use the template file: {}'.format(infile))
        if infile.endswith('.vrt'):
            attr = readfile.read_gdal_vrt(infile)
        elif infile.endswith('.xml'):
            attr = readfile.read_isce_xml(infile)
        else:
            raise ValueError('template file must be either .xml or .vrt file')
        width   = int(attr['WIDTH'])
        length  = int(attr['LENGTH'])
        y0      = float(attr['Y_FIRST'])
        x0      = float(attr['X_FIRST'])
        dy      = float(attr['Y_STEP'])
        dx      = float(attr['X_STEP'])


    # stop of x/y
    x_stop = x0 + (dx * width)
    y_stop = y0 + (dy * length)

    # get the 2d array of Lat
    lat = np.arange(y0, y_stop, dy)
    Lat = np.tile(lat, (width, 1)).T

    # get the 2d array of Lon
    lon = np.arange(x0, x_stop, dx)
    Lon = np.tile(lon, (length, 1))

    # print shapes
    print('Latitude file shape:\t{}'.format(Lat.shape))
    print('Longitude file shape:\t{}'.format(Lon.shape))

    # save to binary files
    Lat.tofile(inps.out_lat)
    Lon.tofile(inps.out_lon)

    # create .xml and .vrt for the new lat/lon files
    create_xml(inps.out_lat, width, length, 'double')
    create_xml(inps.out_lon, width, length, 'double')

    print('Normal ending of the code')
