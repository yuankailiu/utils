#!/usr/bin/env python3
# ykliu @ Jul 07, 2021
#
# Generate the water body file based on an originally downloaded large-scale water body file

import sys
import isce
import argparse
import isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import waterBodyRadar


def cmdLineParse():
    '''
    Example datasets:
    wbdFile     = '/net/kraken/nobak/ykliu/aqaba/d021/process/s1a/merged/waterBody/swbdLat_N25_N35_Lon_E032_E039.wbd'
    wbdOutFile  = './waterBody.rdr'
    '''
    description = 'Generate a new waterBody file given lat/lon grids and an input waterBody file'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--lat', dest='latFile', type=str, required=True,
            help = 'latitude grids binary file')
    parser.add_argument('--lon', dest='lonFile', type=str, required=True,
            help = 'longitude grids binary file')
    parser.add_argument('-i', dest='infile', type=str, required=True,
            help = 'input waterBody file')
    parser.add_argument('-o', dest='outfile', type=str, default='./waterBody.geo',
            help = 'output waterBody file (default: %(default)s).')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        print('')
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()

    latFile     = inps.latFile
    lonFile     = inps.lonFile
    wbdFile     = inps.infile
    wbdOutFile  = inps.outfile

    waterBodyRadar(latFile, lonFile, wbdFile, wbdOutFile)

    print('Normal ending of the code')
