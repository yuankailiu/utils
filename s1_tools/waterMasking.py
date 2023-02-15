#!/usr/bin/env python3
# ykliu @ Feb 13, 2023
#
# Resample original waterBody.wbd to a waterMask matching the geo/rdr files
# Can also mask the ISCE products based on the waterMask file
#
# Integrate the following two scripts:
#   getlalo.py
#   getwbd.py

import argparse
import glob
import os
import shutil
import sys

import isce
import numpy as np
#import isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import create_xml, waterBodyRadar
from mintpy.mask import mask_file
from mintpy.utils import readfile


def cmdLineParse():
    '''
    Command line parsers
    '''
    description = 'Generate geo-coord lat/lon files from a template geocoded file\n'+\
                  'Then resample a waterBody/waterMask file from an input waterBody file'

    EXAMPLE = """Examples:
        ## Generate waterMask and mask several files
        python waterMasking.py -t merged/los.rdr.geo.xml -i swbdLat_N33_N41_Lon_E032_E042.wbd -f filt_topophase.unw.geo phsig.cor.geo

        ## Generate waterMask and mask a default list of files
        python waterMasking.py -t merged/los.rdr.geo.vrt -i swbdLat_N33_N41_Lon_E032_E042.wbd -f default

        ## Generate waterMask and do not apply masking
        python waterMasking.py -t merged/los.rdr.geo.vrt -i swbdLat_N33_N41_Lon_E032_E042.wbd
    """
    epilog = EXAMPLE
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter,  epilog=epilog)

    parser.add_argument('-t', dest='templFile', type=str, required=True,
            help = 'geocoded template xml or vrt file, e.g., *.geo.vrt, *.geo.xml in the merged/ folder')
    parser.add_argument('-i', dest='inwater', type=str, required=True,
            help = 'input waterBody file. E.g., wbd_1_arcsec/swbdLat_N33_N41_Lon_E032_E042.wbd')
    parser.add_argument('-o', dest='outwater', type=str, default='waterMask.geo',
            help = 'output waterMask file (default: %(default)s).')
    parser.add_argument('-f', dest='file_list', nargs='+',
            help = 'a list of files to be masked (default: %(default)s; only generate waterMask file)')
    parser.add_argument('-e', dest='fext', type=str, default='wbdmsk',
            help = 'masked file name extension (default: %(default)s).')
    parser.add_argument('-r', dest='remove', action='store_true',
            help = 'Remove intermediate lat/lon files (default: %(default)s).')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        print('')
        return parser.parse_args()


if __name__ == '__main__':

    ## 1. get user inputs
    inps = cmdLineParse()
    outdir  = os.path.dirname(inps.templFile)
    latFile = os.path.join(outdir, 'lat.geo')
    lonFile = os.path.join(outdir, 'lon.geo')

    if inps.file_list is None:
        inps.file_list = []
    elif inps.file_list == ['default']:
        inps.file_list = [
            os.path.join(outdir, 'filt_topophase.unw.geo'),
            os.path.join(outdir, 'filt_topophase.flat.geo'),
            os.path.join(outdir, 'filt_dense_offsets.bil.geo'),
            os.path.join(outdir, 'dense_offsets_snr.bil.geo'),
            os.path.join(outdir, 'topophase.ion.geo'),
            os.path.join(outdir, 'topophase.cor.geo'),
            os.path.join(outdir, 'phsig.cor.geo'),
            os.path.join(outdir, 'los.rdr.geo'),
        ]
    print(f'Files to be masked ({len(inps.file_list)}): {inps.file_list}')


    ## 2. Read the geocoded lat lon coordinates
    print('Use input geocoded template file (.vrt or .xml) to create lat lon files')
    infile = inps.templFile
    print(f' use the template file: {infile}')
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
    print(f'Latitude file shape:\t{Lat.shape}')
    print(f'Longitude file shape:\t{Lon.shape}')

    # save to binary files
    Lat.tofile(latFile)
    Lon.tofile(lonFile)

    # create .xml and .vrt for the new lat/lon files
    create_xml(latFile, width, length, 'double')
    create_xml(lonFile, width, length, 'double')


    ## 3. Resample waterBody/waterMask file
    # although in the same coord as input template file, no geo metadata written in this file
    # water body. (0) --- land; (-1;255) --- water; (-2) --- no data.
    waterBody = inps.inwater
    waterMask = os.path.join(outdir, inps.outwater)
    waterBodyRadar(latFile, lonFile, waterBody, waterMask)

    # flip the body to mask:
    mask = readfile.read(waterMask)[0]
    mask[mask==0] = 1
    mask[mask==255] = 0
    mask.tofile(waterMask)


    ## 4. Apply the masking and make ISCE metadata files
    if len(inps.file_list) != 0:
        print('Apply masking to files: ', inps.file_list)
        for in_file in inps.file_list:
            if not os.path.isfile(in_file):
                continue
            dir_name  = os.path.dirname(in_file)
            base, ext = os.path.basename(in_file).split('.', maxsplit=1)
            out_file  = os.path.join(dir_name, base+'_'+inps.fext+'.'+ext)
            mask_file(fname=in_file, mask_file=waterMask, out_file=out_file, fill_value=0)

            # prepare ISCE metadata file by
            # 1. copy and rename metadata files
            # 2. update file path inside files
            for mext in ['xml', 'vrt']:
                # copy
                in_meta_file = f'{in_file}.{mext}'
                out_meta_file = f'{out_file}.{mext}'
                shutil.copy2(in_meta_file, out_meta_file)
                print(f'copy {in_meta_file} to {out_meta_file}')
                print('   and update the corresponding filename')
                # update file path
                with open(out_meta_file) as f:
                    s = f.read()
                s = s.replace(os.path.basename(in_file),
                            os.path.basename(out_file))
                with open(out_meta_file, 'w') as f:
                    f.write(s)


    ## 5. Remove useless lat lon files
    if inps.remove:
        for f in glob.glob(latFile+'*'):
            print('delete intermediate file: ', f)
            os.remove(f)
        for f in glob.glob(lonFile+'*'):
            print('delete intermediate file: ', f)
            os.remove(f)

    print('\nNormal ending of the code.')
