#!/usr/bin/env python3
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Author: Cunren Liang
# Co-authors: Ollie Stephenson, Yuan-Kai Liu
# Last update: May, 2023
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# refer to: https://github.com/isce-framework/isce2/blob/main/contrib/stack/topsStack/plotIonPairs.py

import argparse
import glob
import os
import sys
from argparse import RawTextHelpFormatter

import isce
import isceobj
import numpy as np
from isceobj.Alos2Proc.Alos2ProcPublic import runCmd


def cmdLineParse():
    '''
    Command line parser.
    '''
    EXAMPLE = """
    plot_imgs.py -i 'ion/*_*/ion_cal/filt.ion' --redo --loc -3 --chan 2 --out pic/img_ion --mark pairs_diff_starting_ranges.txt
    plot_imgs.py -i 'ion_azshift_dates/*.ion' --redo --loc 1 --chan 1 --out pic/img_azshiftDate --wrap 0.00628
    plot_imgs.py -i 'ion_burst_ramp_merged_dates/*.float' --redo --loc -1 --chan 1 --out pic/img_ionRampDate --wrap 0.0628
    plot_imgs.py -i 'merged/interferograms/*_*/filt_fine.unw' --redo --loc -2 --chan 2 --out pic/img_unw
    """

    parser = argparse.ArgumentParser(description='mdx plot a bunch of images (.int, .unw, .ion, .float, etc.).',
                                     formatter_class=RawTextHelpFormatter,
                                     epilog=EXAMPLE)
    parser.add_argument('--in', '-i', dest='input', type=str, required=True,
            help = 'data path and filename patterns for glob')
    parser.add_argument('--loc', '-l', dest='loc', type=int, default=-3,
            help = 'pair/date pattern location in the path. E.g., -3 for ./ion/*_*/ion_cal/filt.ion. (default: %(default)s)')
    parser.add_argument('--chan', '-b', dest='chan', type=int, default=2,
            help = 'usually, 1 for amplitude, 2 for phase. (default: %(default)s)')
    parser.add_argument('--wrap', '-w', dest='wrap', type=float, default=6.28,
            help = 'Wrap range. (default: %(default)s)')
    parser.add_argument('--out', '-o', dest='outdir', type=str, default='./img',
            help = 'output image folder. (default: %(default)s)')
    parser.add_argument('--redo', '-r', dest='redo', action='store_true', default=False,
            help = 'Replot all the individual .tif plots. (default: %(default)s)')
    parser.add_argument('--collate', '-c', dest='collate', action='store_true', default=True,
            help = 'Collate all .tif to a single .svg. (default: %(default)s)')
    parser.add_argument('--amp', '-a', dest='overamp', action='store_true', default=False,
            help = 'turn on overlaying the amplitude. (default: %(default)s)')
    parser.add_argument('--mark', '-m', dest='mark_txt', type=str, default=None,
            help = 'Mark certain dates/pairs from a text file (default: %(default)s)')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()

    # output folder
    odir = inps.outdir
    if inps.overamp:
        odir += '_amp'
    if not os.path.exists(odir):
        os.makedirs(odir)

    # glob the files to be plotted
    files   = sorted(glob.glob(os.path.join(inps.input)))

    # read the dates/pairs to mark
    marks = []
    if inps.mark_txt:
        with open(inps.mark_txt) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if len(line)>0 and line[0].isdigit():
                    marks.append(line)
        marks = list(set(marks))


    ##########################################
    # 0. preparation
    ##########################################
    # combine plot dimension
    ipl    = 20  # images per row
    ppc    = 30  # pixels per cm (the conversion that I guess)
    WIDTH  = 30  # artborad width [cm]
    n_rows = np.ceil(len(files) / ipl)  # number of img rows
    print(f'Total number of files : {len(files)}')
    print(f'Collate {ipl} images per line')
    print(f'Expect {n_rows} rows in the collate plot')

    # gauge the image dimension by the first file
    img = isceobj.createImage()
    img.load(files[0]+'.xml')
    width  = img.width
    length = img.length
    print('Image dimension [px]: WIDTH, LENGTH = ', width, length)

    widthMax = WIDTH*ppc / ipl
    if width >= widthMax:
        ratio = widthMax / width
        resize = f' -resize {100.0*ratio}%'
        pz = f' -pz -{1/ratio}'
    else:
        ratio = 1.0
        resize = ''
        pz = ''

    LENGTH = (length * ratio * n_rows) / ppc # artboard length [cm]
    print(resize)
    print('Collate image dimension [px]: WIDTH, LENGTH = ', width*ratio, length*ratio)
    print('Collate artboard dimension [cm]: WIDTH, LENGTH = ', WIDTH, LENGTH)

    # set up combine plot
    svg =   '''<?xml version="1.0" standalone="no"?>
            <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
            "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
            <svg width="{}cm" height="{}cm" version="1.1"
                xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
            '''.format(WIDTH, LENGTH)
    tlr =   '''
            </svg>
            '''
    rW  = width  * ratio / ppc  # cm width of each panel
    rL  = length * ratio / ppc  # cm length of each panel


    ##########################################
    # 1. plot each file as tif
    ##########################################
    for i, file in enumerate(files):
        pair  = file.split('/')[inps.loc]
        if '.' in pair:
            pair = pair.split('.')[0]
        if '_' in pair:
            mdate = pair.split('_')[0]
            sdate = pair.split('_')[1]
        else:
            date = str(pair)

        # generate each .tif plot
        if inps.redo:
            img = isceobj.createImage()
            img.load(file+'.xml')
            width  = img.width
            length = img.length

            if inps.chan == 1:
                cmd = 'mdx {} -s {} -ch1 -r4 -wrap {} -addr -{} -cmap CMY -P -workdir {} {}'.format(
                    file,
                    width,
                    inps.wrap,
                    inps.wrap/2,
                    odir, pz)
            elif inps.chan == 2:
                if not inps.overamp:
                    cmd = 'mdx {} -s {} -ch2 -r4 -rhdr {} -wrap {} -addr -{} -cmap CMY -P -workdir {} {}'.format(
                        file,
                        width,
                        width*4,
                        inps.wrap,
                        inps.wrap/2,
                        odir, pz)
                elif inps.overamp:
                    cmd = 'mdx {} -s {} -amp -r4 -rtlr {} -CW -unw -r4 -rhdr {} -wrap {} -addr -{} -cmap CMY -P -workdir {} {}'.format(
                        file,
                        width,
                        width*4,
                        width*4,
                        inps.wrap,
                        inps.wrap/2,
                        odir, pz)

            runCmd(cmd)

            # Can change the compression here if we want
            #cmd = 'convert {} {} {}.tif'.format(os.path.join(odir, 'out.ppm'), resize, os.path.join(odir, pair))
            cmd = 'convert {} {}.tif'.format(os.path.join(odir, 'out.ppm'), os.path.join(odir, pair))
            runCmd(cmd)
            os.remove(os.path.join(odir, 'out.ppm'))


        ##########################################
        # 2. generate the collate plot as svg
        ##########################################
        if inps.collate:
            #line and column indexes, indexes start from 1
            ii = int((i + 1 - 0.1) / ipl) + 1
            jj = i + 1 - (ii - 1) * ipl

            first_row_gap = rL / 8
            first_col_gap = rW / 5

            # plot the interferograms
            if '_' in pair or '-' in pair:
                if any(x in marks for x in [f'{mdate}_{sdate}', f'{sdate}_{mdate}', f'{mdate}-{sdate}', f'{sdate}-{mdate}']):
                    print(f' > mark the pair {pair}')
                    font_color = ';fill:red'
                    add_box    = '''<rect fill="none" stroke="red" stroke-width="2" x="{}cm" y="{}cm" width="{}cm" height="{}cm"/>
                                 '''.format(first_col_gap + (jj-1)*rW*0.85,
                                            first_row_gap + (ii-1)*rL*0.82,
                                            rW-first_col_gap, rL-first_row_gap)
                    print(add_box)
                else:
                    font_color = ''
                    add_box    = ''

                # write the svg
                img =   '''<image xlink:href="{}" x="{}cm" y="{}cm"/>
                            {}
                            <text x="{}cm" y="{}cm" style="font-family:'Times New Roman';font-weight:normal;font-style:normal;font-stretch:normal;font-variant:normal;font-size:8px{}">{}</text>
                            <text x="{}cm" y="{}cm" style="font-family:'Times New Roman';font-weight:normal;font-style:normal;font-stretch:normal;font-variant:normal;font-size:8px{}">{}</text>
                        '''.format(os.path.join(pair + '.tif'),
                                   first_col_gap + (jj-1)*rW*0.85,
                                   first_row_gap + (ii-1)*rL*0.82,
                                   add_box,
                                   first_col_gap + (jj-1)*rW*0.85,
                                   first_row_gap + (ii-1)*rL*0.82+rW*0.1,
                                   font_color, mdate,
                                   first_col_gap + (jj-1)*rW*0.85,
                                   first_row_gap + (ii-1)*rL*0.82+rW*0.1*2.5,
                                   font_color, sdate)

            # plot the acquisitions
            else:
                if date in marks:
                    print(f' > mark the date {date}')
                    font_color = ';fill:red'
                else:
                    font_color = ''

                # write the svg
                img =   '''    <image xlink:href="{}" x="{}cm" y="{}cm"/>
                        <text x="{}cm" y="{}cm" style="font-family:'Times New Roman';font-weight:normal;font-style:normal;font-stretch:normal;font-variant:normal;font-size:8px{}">{}</text>
                        '''.format(os.path.join(pair + '.tif'),
                                    first_col_gap + (jj-1)*rW*0.85,
                                    first_row_gap + (ii-1)*rL*0.82,
                                    first_col_gap + (jj-1)*rW*0.85,
                                    first_row_gap + (ii-1)*rL*0.82+rW*0.1,
                                    font_color,
                                    date)

            svg += img

    svg += tlr

    with open(os.path.join(odir, 'collate.svg'), 'w') as f:
        f.write(svg)


    ##########################################
    # 3. create colorbar
    ##########################################
    width_colorbar = 100
    length_colorbar = 20
    colorbar = np.ones((length_colorbar, width_colorbar), dtype=np.float32) * \
               (np.linspace(-inps.wrap/2, inps.wrap/2, num=width_colorbar, endpoint=True, dtype=np.float32))[None,:]
    colorbar.astype(np.float32).tofile(os.path.join(odir, 'colorbar'))
    runCmd('mdx {} -s {} -cmap cmy -wrap {} -addr -{} -P -workdir {}'.format(os.path.join(odir, 'colorbar'), width_colorbar, inps.wrap, inps.wrap/2, odir))
    runCmd('convert {} -compress LZW -resize 100% {}'.format(os.path.join(odir, 'out.ppm'), os.path.join(odir, f'colorbar_-{inps.wrap/2}_{inps.wrap/2}.tiff')))
    runCmd('rm {} {}'.format(os.path.join(odir, 'colorbar'), os.path.join(odir, 'out.ppm')))

    print('Normal complete.')
