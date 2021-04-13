#!/usr/bin/env python3

#Cunren Liang, 26-MAR-2018
# Modified by Ollie Stephenson, 14-DEC-2020
#   Master/slave to reference/secondary 
# Still using sdate and mdate variables
# Modified by YKL, 12-Apr-2021
#   add optional function: user defined pairing list from a text file

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


def findslice(date, group):
    '''
    this routine finds the index of the group that contains a certain date string
    '''
    if len(date) != 8:
        print('Input date of findslice() should be YYYYMMDD format!')
        return np.nan
    datestr = '_'+date+'T'
    for i in range(len(group)):
        if any(datestr in slc for slc in group[i]):
            #print(group[i])
            return i
    print('No such date {}'.format(date))
    return np.nan    


def cmdLineParse():
    '''
    Command line parser.
    '''
    description = 'Modified version from YKL (Apr 2021):\n \
        Report sentinel-1 product processing software info\n \
        Adding function to pair user defined pairs from a txt file'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-dir', dest='dir', type=str, required=True,
            help = 'directory containing the "S1*_IW_SLC_*.zip" files')
    parser.add_argument('-xml', dest='xml', type=str, required=True,
            help = 'example reference.xml/secondary.xml file')
    parser.add_argument('-num', dest='num', type=int, default=2,
            help = 'number of pairs for each acquistion. default: 2')
    parser.add_argument('-yr', dest='yr', type=float, default=1.0,
            help = 'time span threshhold. default: 1.0 year')
    parser.add_argument('-pairtxt', dest='pairtxt', type=str, default=None,
            help = 'Pairs list text file (disregard -num and -yr). default: None')


    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()

    print('Pairing Sentinel-1 SLC files under {}'.format(os.path.abspath(inps.dir)))

    group = get_group(inps.dir)
    #read standard configurations
    tree = ET.parse(inps.xml)
    root = tree.getroot()

    datefmt = "%Y%m%dT%H%M%S"
    ngroup = len(group)
    pairs = []
    pairs2 = []

    if inps.pairtxt is not None:
        print('Pairing based on user defined -pairtxt: {}'.format(os.path.abspath(inps.pairtxt)))
        print('Disregard -num and -yr criteria')
        with open(inps.pairtxt) as f:
            date12 = f.read().splitlines()
        for i in range(len(date12)):
            udr  = '20'+date12[i].split('-')[0]
            uds  = '20'+date12[i].split('-')[1]
            udri = findslice(udr, group)
            udsi = findslice(uds, group)

            fields = group[udri][0].split('_')
            mdate = fields[-5][2:8] # YYMMDD format
            # mdate = fields[-5][0:8] # YYYYMMDD format
            mtime = datetime.datetime.strptime(fields[-5], datefmt)

            fields = group[udsi][0].split('_')
            sdate = fields[-5][2:8] # YYMMDD format
            #sdate = fields[-5][0:8] # YYYYMMDD format
            stime = datetime.datetime.strptime(fields[-5], datefmt)
            ms = mdate + '-' + sdate

            if not os.path.exists(ms):
                os.mkdir(ms)
                pairs.append(ms)
            else:
                print('Skip creating existing pair directory {}'.format(ms))

            #reference xml
            root.set('name', 'reference')
            safe = root.find("property[@name='safe']")
            safe.text = '{}'.format(group[udri])
            safe = root.find("property[@name='output directory']")
            safe.text = 'referencedir'
            tree.write(os.path.join(ms, 'reference.xml'))

            #secondary xml
            root.set('name', 'secondary')
            safe = root.find("property[@name='safe']")
            safe.text = '{}'.format(group[udsi])
            safe = root.find("property[@name='output directory']")
            safe.text = 'secondarydir'
            tree.write(os.path.join(ms, 'secondary.xml'))

    else:
        print('Pairing based on -num and -yr: ')
        for i in range(ngroup):
            fields = group[i][0].split('_')
            mdate = fields[-5][2:8] # YYMMDD format
            # mdate = fields[-5][0:8] # YYYYMMDD format
            mtime = datetime.datetime.strptime(fields[-5], datefmt)
            for j in range(i+1, i+inps.num+1):
                if j > ngroup - 1:
                    continue

                fields = group[j][0].split('_')
                sdate = fields[-5][2:8] # YYMMDD format
                #sdate = fields[-5][0:8] # YYYYMMDD format
                stime = datetime.datetime.strptime(fields[-5], datefmt)
                ms = mdate + '-' + sdate
                
                if np.absolute((stime - mtime).total_seconds()) > inps.yr * 365.0 * 24.0 * 3600:
                    pairs2.append(ms)
                    #print('WARNING: time span of pair {} larger than threshhold, skip this pair...')
                    continue

                os.mkdir(ms)
            
                #reference xml
                root.set('name', 'reference')
                safe = root.find("property[@name='safe']")
                safe.text = '{}'.format(group[i])
                safe = root.find("property[@name='output directory']")
                safe.text = 'referencedir'
                tree.write(os.path.join(ms, 'reference.xml'))

                #secondary xml
                root.set('name', 'secondary')
                safe = root.find("property[@name='safe']")
                safe.text = '{}'.format(group[j])
                safe = root.find("property[@name='output directory']")
                safe.text = 'secondarydir'
                tree.write(os.path.join(ms, 'secondary.xml'))

                pairs.append(ms)

    print('created the following pairs:')
    for x in pairs:
        print('{}'.format(x))
    #print('***************************************')
    print('total number of pairs created: {}'.format(len(pairs)))


    if len(pairs2) != 0:
        print('time spans of the following pairs are larger than threshold')
        print('they are not created')
        for x in pairs2:
            print('{}'.format(x))
        print('total number of pairs not created: {}'.format(len(pairs2)))
    else:
        print('all possible pairs created')






