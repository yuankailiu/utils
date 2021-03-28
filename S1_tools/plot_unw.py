#!/usr/bin/env python3

import os
import sys
import glob
import ntpath
import argparse


def runCmd(cmd, silent=0):
    import os

    if silent == 0:
        print("{}".format(cmd))
    status = os.system(cmd)
    if status != 0:
        raise Exception('error when running:\n{}\n'.format(cmd))


def getWidth(xmlfile):
    from xml.etree.ElementTree import ElementTree
    xmlfp = None
    try:
        xmlfp = open(xmlfile,'r')
        print('reading file width from: {0}'.format(xmlfile))
        xmlx = ElementTree(file=xmlfp).getroot()
        #width = int(xmlx.find("component[@name='coordinate1']/property[@name='size']/value").text)
        tmp = xmlx.find("component[@name='coordinate1']/property[@name='size']/value")
        if tmp == None:
            tmp = xmlx.find("component[@name='Coordinate1']/property[@name='size']/value")
        width = int(tmp.text)
        print("file width: {0}".format(width))
    except (IOError, OSError) as strerr:
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
<svg width="20cm" height="20cm" version="1.1"
     xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
'''
    tlr = '''
</svg>'''
    svg = hdr


    #images per line
    ipl = 15
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

        img = '''    <image xlink:href="{}" x="{}cm" y="{}cm"/>
    <text x="{}cm" y="{}cm" style="font-family:'Times New Roman';font-weight:normal;font-style:normal;font-stretch:normal;font-variant:normal;font-size:11px">{}</text>
    <text x="{}cm" y="{}cm" style="font-family:'Times New Roman';font-weight:normal;font-style:normal;font-stretch:normal;font-variant:normal;font-size:11px">{}</text>
'''.format(os.path.join(imgdir, pair + '.tiff'),
           0+(jj-1)*1.2,
           0+(ii-1)*3.7,
           0+(jj-1)*1.2,
           0+(ii-1)*3.7-0.26,
           mdate,
           0+(jj-1)*1.2,
           0+(ii-1)*3.7-0.03,
           sdate
        )

        svg += img


    svg += tlr


    with open(inps.svg, 'w') as f:
        f.write(svg)

