#!/usr/bin/env python3

import argparse
import glob
import ntpath
import os
import sys


def runCmd(cmd, silent=0):
    import os

    if silent == 0:
        print(f"{cmd}")
    status = os.system(cmd)
    if status != 0:
        raise Exception(f'error when running:\n{cmd}\n')


def getWidth(xmlfile):
    from xml.etree.ElementTree import ElementTree
    xmlfp = None
    try:
        xmlfp = open(xmlfile)
        print(f'reading file width from: {xmlfile}')
        xmlx = ElementTree(file=xmlfp).getroot()
        #width = int(xmlx.find("component[@name='coordinate1']/property[@name='size']/value").text)
        tmp = xmlx.find("component[@name='coordinate1']/property[@name='size']/value")
        if tmp == None:
            tmp = xmlx.find("component[@name='Coordinate1']/property[@name='size']/value")
        width = int(tmp.text)
        print(f"file width: {width}")
    except OSError as strerr:
        print("IOError: %s" % strerr)
        return []
    finally:
        if xmlfp is not None:
            xmlfp.close()
    return width



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

    hdr = '''<?xml version="1.0" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="30cm" height="120cm" version="1.1"
     xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
'''
    tlr = '''
</svg>'''
    svg = hdr


    #images per line
    ipl = 20
    imgdir = 'img'
    os.mkdir(imgdir)
    files = sorted(glob.glob( os.path.join(inps.dir, '*-*', 'ion/lower/merged/filt_topophase.unw') )) # (pair is -5)
    # files = sorted(glob.glob( os.path.join(inps.dir, '*-*', 'merged/filt_topophase.unw.geo') )) # (pair is -3)
    nfiles = len(files)
    for i in range(nfiles):
        pair = files[i].split('/')[-5] # Adjust based on file path
        mdate = pair.split('-')[0]
        sdate = pair.split('-')[1]
        width = getWidth(files[i] + '.xml')

        cmd = 'mdx {} -s {} -ch2 -r4 -rhdr {} -wrap 20 -addr -10 -cmap CMY -P -workdir {}'.format(
            files[i],
            width,
            width*4,
            imgdir)
        runCmd(cmd)
        # Originally 12%
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
