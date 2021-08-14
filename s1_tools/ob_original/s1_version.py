#!/usr/bin/env python3

#Cunren Liang, 22-MAR-2018

import os
import sys
import glob
import zipfile
import argparse
import datetime
import numpy as np
import xml.etree.ElementTree as ET


def get_group(dir0):
    '''
    this routine group the slices
    each group is an acquisition
    the returned result is a list containing a number of lists (groups/acquistions)
    this routine has no problem with same slice having different versions
    '''
    #sort by starting time
    zips = sorted(glob.glob(os.path.join(dir0, 'S1*_IW_SLC_*.zip')), key=lambda x: x.split('_')[-5], reverse=False)
    nzips = len(zips)

    group = []
    for i in range(nzips):
        datefmt = "%Y%m%dT%H%M%S"
        fields = zips[i].split('_')
        tbef = datetime.datetime.strptime(fields[-5], datefmt)
        taft = datetime.datetime.strptime(fields[-4], datefmt)
        
        if i == 0:
            #create new group
            tbef0 = tbef
            group0 = []

        #S-1A is capable of operating up to 25 min per orbit [21]
        #Yague-Martinez et al., "Interferometric Processing of Sentinel-1 TOPS Data,"
        #S1A/B revisit time is 6 days, here we use 1 day to check if from the same orbit
        if np.absolute((tbef - tbef0).total_seconds()) < 24 * 3600:
            group0.append(zips[i])
        else:
            group.append(group0)
            #create new group
            tbef0 = tbef
            group0 = []
            group0.append(zips[i])

        if i == nzips - 1:
            group.append(group0)

    return group


def check_verion(group):
    '''
    this routine check the slice versions in each acquistion
    this routine checks the swath start ranges in each slice of each acquistion
    '''
    #speed of light (m/s)
    c = 299792458.0

    ngroup = len(group)
    nslice = 0
    for x in group:
       nslice += len(x)

    print('versions and start ranges')
    #print('slice                                                                    acq. no.  version    site')
    #print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('slice                                                                    no   ver         IW1 (m)           IW2 (m)           IW3 (m)')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    info = ''
    for i in range(ngroup):
        ngroup0 = len(group[i])
        same = True
        for j in range(ngroup0):
            #1. processing software version
            zf = zipfile.ZipFile(group[i][j], 'r')
            manifest = [item for item in zf.namelist() if '.SAFE/manifest.safe' in item][0]
            xmlstr = zf.read(manifest)
            root = ET.fromstring(xmlstr)
            elem = root.find('.//metadataObject[@ID="processing"]')

            ####Setup namespace
            nsp = "{http://www.esa.int/safe/sentinel-1.0}"
            rdict = elem.find('.//xmlData/' + nsp + 'processing/' + nsp + 'facility').attrib
            site = rdict['site'] +', '+ rdict['country']

            rdict = elem.find('.//xmlData/' + nsp + 'processing/' + nsp + 'facility/' + nsp + 'software').attrib
            #ver = rdict['name'] + ' ' + rdict['version']
            ver = rdict['version']

            #print_stuff = '%s    %3d     %s     %s'%(group[i][j].split('/')[-1], i+1, ver, site)
            print_stuff = '%s %3d  %s  '%(os.path.basename(group[i][j]), i+1, ver)


            #2. start ranges
            anna = sorted([item for item in zf.namelist() if '.SAFE/annotation/s1' in item])
            #dual polarization. for the same swath, the slant ranges of two polarizations should be the same.
            if len(anna) == 6:
                anna = anna[1:6:2]

            startingRange = []
            for k in range(3):
                xmlstr = zf.read(anna[k])
                root = ET.fromstring(xmlstr)
                startingRange.append(
                    float(root.find('imageAnnotation/imageInformation/slantRangeTime').text)*c/2.0
                    )

            print_stuff += "{} {} {}".format(startingRange[0], startingRange[1], startingRange[2])
            print(print_stuff)


            if j == 0:
                ver0 = ver
            else:
                if ver0 != ver:
                    same = False
        print()
        if same == False:
            info += 'in acquistion {}, versions are different\n'.format(i+1)

    print('number of slices: {}'.format(nslice))
    print('number of acquisitions: {}'.format(ngroup))

    if info == '':
        print('great! no acquisition has slices with different versions')
    else:
        print(info)


def check_gap(group):
    '''
    this routine checks if there are gaps in each acquistion
    this routine has no problem with same slice having different versions
    '''
    nogap = True
    ngroup = len(group)
    for i in range(ngroup):
        ngroup0 = len(group[i])
        if ngroup0 == 1:
            continue
        else:
            for j in range(1, ngroup0):
                datefmt = "%Y%m%dT%H%M%S"
                fields = group[i][j-1].split('_')
                tbef0 = datetime.datetime.strptime(fields[-5], datefmt)
                taft0 = datetime.datetime.strptime(fields[-4], datefmt)

                fields = group[i][j].split('_')
                tbef = datetime.datetime.strptime(fields[-5], datefmt)
                taft = datetime.datetime.strptime(fields[-4], datefmt)

                if (tbef - taft0).total_seconds() > 0:
                    nogap = False
                    print('in acquistion {}, there is a gap between the following two slices:'.format(i+1))
                    print(group[i][j-1].split('/')[-1])
                    print(group[i][j].split('/')[-1])

    if nogap == True:
        print('great! no acquistion has gaps between slices')


def check_redundancy(group, threshold=1):
    '''
    threshold: time difference threshold between two slices in second.
    this routine checks, for a slice, if there are multiple versions.
    '''

    multiple_version = False
    ngroup = len(group)
    for i in range(ngroup):
        ngroup0 = len(group[i])
        if ngroup0 == 1:
            continue
        else:
            for j in range(ngroup0-1):
                for k in range(j+1, ngroup0):
                    datefmt = "%Y%m%dT%H%M%S"
                    fields = group[i][j].split('_')
                    tbef0 = datetime.datetime.strptime(fields[-5], datefmt)
                    taft0 = datetime.datetime.strptime(fields[-4], datefmt)

                    fields = group[i][k].split('_')
                    tbef = datetime.datetime.strptime(fields[-5], datefmt)
                    taft = datetime.datetime.strptime(fields[-4], datefmt)

                    if np.absolute((tbef - tbef0).total_seconds()) < threshold and \
                       np.absolute((taft - taft0).total_seconds()) < threshold:
                        multiple_version = True
                        print('in acquistion {}, the following two slices are the same, but with different versions:'.format(i+1))
                        print(group[i][j-1].split('/')[-1])
                        print(group[i][j].split('/')[-1])

    if multiple_version == False:
        print('great! no slice has multiple versions')


def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser( description='report sentinel-1 product processing software info')
    parser.add_argument('-dir', dest='dir', type=str, required=True,
            help = 'directory containing the "S1*_IW_SLC_*.zip" files')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()

    group = get_group(inps.dir)
    check_verion(group)
    check_gap(group)
    check_redundancy(group, threshold=1)











