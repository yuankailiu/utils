#!/usr/bin/env python3
# Originally written by Cunren Liang
# Modified by Ollie Stephenson

import argparse
import glob
import ntpath
import os
import sys

import isce
import isceobj
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd


def cmdLineParse():
    '''
    Command line parser.
    '''
    parser = argparse.ArgumentParser( description='take looks')
    parser.add_argument('-dir', dest='dir', type=str, required=True,
            help = 'data directory')
    parser.add_argument('-svg', dest='svg', type=str, required=True,
            help = 'output svg filename')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()

    hdr =   '''<?xml version="1.0" standalone="no"?>
            <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
            "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
            <svg width="30cm" height="120cm" version="1.1"
                xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
            '''
    tlr =   '''
            </svg>
            '''
    svg = hdr


    #images per line
    ipl = 20
    imgdir = 'img'
    os.mkdir(imgdir)
    files = sorted(glob.glob( os.path.join(inps.dir, '*-*', 'ion/ion_cal/filt.ion') ))
    nfiles = len(files)
    for i in range(nfiles):
        pair = files[i].split('/')[-4]
        mdate = pair.split('-')[0]
        sdate = pair.split('-')[1]

        img = isceobj.createImage()
        img.load(files[i])
        width = img.width
        length = img.length

        cmd = 'mdx {} -s {} -ch2 -r4 -rhdr {} -wrap 6.28 -addr -3.14 -cmap CMY -P -workdir {}'.format(
            files[i],
            width,
            width*4,
            imgdir)
        runCmd(cmd)
        # Can change the compression here if we want
        cmd = 'convert {} -resize 12% {}'.format(
            os.path.join(imgdir, 'out.ppm'),
            os.path.join(imgdir, pair + '.tiff'))
        runCmd(cmd)
        os.remove(os.path.join(imgdir, 'out.ppm'))

        #line and column indexes, indexes start from 1
        ii = int((i + 1 - 0.1) / ipl) + 1
        jj = i + 1 - (ii - 1) * ipl

        first_row_gap = 0.6
        first_col_gap = 0.3

        img = '''    <image xlink:href="{}" x="{}cm" y="{}cm"/>
    <text x="{}cm" y="{}cm" style="font-family:'Times New Roman';font-weight:normal;font-style:normal;font-stretch:normal;font-variant:normal;font-size:11px">{}</text>
    <text x="{}cm" y="{}cm" style="font-family:'Times New Roman';font-weight:normal;font-style:normal;font-stretch:normal;font-variant:normal;font-size:11px">{}</text>
'''.format(os.path.join(imgdir, pair + '.tiff'),
           first_col_gap + (jj-1)*1.2,
           first_row_gap + (ii-1)*3.8,
           first_col_gap + (jj-1)*1.2,
           first_row_gap + (ii-1)*3.8-0.26,
           mdate,
           first_col_gap + (jj-1)*1.2,
           first_row_gap + (ii-1)*3.8-0.03,
           sdate
        )

        svg += img


    svg += tlr


    with open(inps.svg, 'w') as f:
        f.write(svg)