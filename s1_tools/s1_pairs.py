#!/usr/bin/env python3

#Cunren Liang, 26-MAR-2018
# Modified by Ollie Stephenson, 14-DEC-2020
#   Master/slave to reference/secondary
# Still using sdate and mdate variables
# Modified by YKL, 12-Apr-2021
#   add optional function: user defined pairing list from a text file
# Modified by OLS, 10-May-2021
#   add optional pairing above minimum temporal threshold (e.g. form pairs with a baseline above half a year)

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
    this routine groups the SLC slices
    each group is an acquisition
    the returned result is a list containing a number of lists (groups/acquistions)
    the list is orderd by acquisition date
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
            group0= []
            group0.append(zips[i])

        if i == nzips - 1:
            group.append(group0)

    return group


def findslice(date, group):
    '''
    this routine finds the index of the group that contains a certain date string
    '''
    if len(date) != 8:
        raise Exception('Input date of findslice() should be YYYYMMDD format! ({})'.format(date))
    datestr = '_'+date+'T'
    for i in range(len(group)):
        if any(datestr in slc for slc in group[i]):
            #print(group[i])
            return i
    raise Exception('No such date {}'.format(date))

def make_pair(ms,group1,group2,tree,root):
        ''' Make xml files for a given pair of dates '''

        #reference xml
        root.set('name', 'reference')
        safe = root.find("property[@name='safe']")
        safe.text = '{}'.format(group1)
        safe = root.find("property[@name='output directory']")
        safe.text = 'referencedir'
        tree.write(os.path.join(ms, 'reference.xml'))

        #secondary xml
        root.set('name', 'secondary')
        safe = root.find("property[@name='safe']")
        safe.text = '{}'.format(group2)
        safe = root.find("property[@name='output directory']")
        safe.text = 'secondarydir'
        tree.write(os.path.join(ms, 'secondary.xml'))

        return

def cmdLineParse():
    '''
    Command line parser.
    '''
    description = 'Modified version from YKL (Apr 2021):\n \
        Report sentinel-1 product processing software info\n \
        Adding function to pair user defined pairs from a txt file'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', '--dir', dest='dir', type=str, required=True,
            help = 'directory containing the "S1*_IW_SLC_*.zip" files')
    parser.add_argument('-x', '--xml', dest='xml', type=str, required=True,
            help = 'example reference.xml/secondary.xml file')
    parser.add_argument('-n', '--num', dest='num', type=int, default=2,
            help = 'number of pairs for each acquistion (default: %(default)s).')
    parser.add_argument('--skip_aqns', dest='skip_aqns', type=int, default=0,
            help = 'skip-pairing scheme; skip num of nearest acquistions (disregard -n) (default: %(default)s).')
    parser.add_argument('--yrmin', dest='yr_min', type=float, default=0,
            help = 'minimum temporal baseline (years; disregard --skip_aqns) (default: %(default)s).')
    parser.add_argument('--yrmax', dest='yr_max', type=float, default=1.0,
            help = 'maximum temporal baseline (years) (default: %(default)s).')
    parser.add_argument('--txt', dest='pairtxt', type=str, default=None,
            help = 'text file contains the list of pairs (disregard -n, --yr, --skip variables) (default: %(default)s).')


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
    ngroup = len(group) # number of SLC dates
    pairs = []
    pairs2 = []

    # If we have a file with a list of pairs, use that
    if inps.pairtxt is not None:
        print('Pairing based on user defined -pairtxt: {}'.format(os.path.abspath(inps.pairtxt)))
        print('Disregard -num, -yr and -skip criteria')
        with open(inps.pairtxt) as f:
            date12 = f.read().splitlines()
        for i in range(len(date12)):
            # udr  = '20'+date12[i].split('-')[0]
            # uds  = '20'+date12[i].split('-')[1]
            udr  = date12[i].split('-')[0]
            uds  = date12[i].split('-')[1]

            udri = findslice(udr, group)
            udsi = findslice(uds, group)

            fields = group[udri][0].split('_')
            # mdate = fields[-5][2:8] # YYMMDD format
            mdate = fields[-5][0:8] # YYYYMMDD format
            mtime = datetime.datetime.strptime(fields[-5], datefmt)

            fields = group[udsi][0].split('_')
            # sdate = fields[-5][2:8] # YYMMDD format
            sdate = fields[-5][0:8] # YYYYMMDD format
            stime = datetime.datetime.strptime(fields[-5], datefmt)
            ms = mdate + '-' + sdate

            if not os.path.exists(ms):
                os.mkdir(ms)
                pairs.append(ms)
                make_pair(ms,group[udri],group[udsi],tree,root)
            else:
                print('Skip creating existing pair directory {}'.format(ms))

    # If we're doing yr_min
    elif inps.yr_min > 0:

        print('Pairing based on -yr_min and -yr_max, skip {} yr pairing scheme'.format(inps.yr_min))

        if inps.skip_aqns != 0:
            print('Ignoring skip_aqns')
            inps.skip_aqns=0 # Ignore the skip_aqns variable

        if inps.yr_min > inps.yr_max:
            raise Exception("yr_min is larger than yr_max, can't make any pairs")

        for i in range(ngroup):
            fields = group[i][0].split('_')
            # mdate = fields[-5][2:8] # YYMMDD format
            mdate = fields[-5][0:8] # YYYYMMDD format
            mtime = datetime.datetime.strptime(fields[-5], datefmt)
            counter = 0 # count how many pairs we've made

            # Loop over each acqusition and figure out the timespan
            for j in range(i+1,ngroup):
                if j > ngroup - 1: # Skip if we're greater than the no. of SLCs
                    continue
                fields = group[j][0].split('_')
                # sdate = fields[-5][2:8] # YYMMDD format
                sdate = fields[-5][0:8] # YYYYMMDD format
                stime = datetime.datetime.strptime(fields[-5], datefmt)
                # If temporal baseline is over the yr_min threshold, make a pair
                if np.absolute((stime - mtime).total_seconds()) > inps.yr_min * 365.0 * 24.0 * 3600:
                    # Unless it's also over the max temporal threshold, when we make stop making pairs
                    if np.absolute((stime - mtime).total_seconds()) > inps.yr_max * 365.0 * 24.0 * 3600:
                        pairs2.append(ms)
                        print('WARNING: time span of pair {} larger than threshhold, skip this pair...')
                        continue
                    ms = mdate + '-' + sdate
                    if not os.path.exists(ms):
                        pairs.append(ms)
                        os.mkdir(ms)
                        make_pair(ms,group[i],group[j],tree,root)
                    else:
                        print('Skip creating existing pair directory {}'.format(ms))

                    # When the counter reaches num, stop
                    # Only want to make 'num' pairs
                    counter += 1
                    if counter == inps.num:
                        break
            # When we find one over the required timespan, increment the counter

    else:
        if inps.skip_aqns == 0 and inps.yr_min == 0:
            print('Pairing based on -num and -yr')
        elif inps.yr_min > 0:
            print('Pairing based on -yr_min and -yr_max, skip {} yr pairing scheme'.format(inps.yr_min))
            inps.skip_aqns=0
        elif inps.skip_aqns > 0:
            inps.num = int(1)
            print('Pairing based on -skip_aqns and -yr_max, skip-{} pairing scheme'.format(inps.skip_aqns))
        for i in range(ngroup):
            fields = group[i][0].split('_')
            # mdate = fields[-5][2:8] # YYMMDD format
            mdate = fields[-5][0:8] # YYYYMMDD format
            mtime = datetime.datetime.strptime(fields[-5], datefmt)

            for j in range(i+1+inps.skip_aqns, i+1+inps.skip_aqns+inps.num):
                if j > ngroup - 1: # Skip if we're greater than the no. of SLCs
                    continue

                fields = group[j][0].split('_')
                # sdate = fields[-5][2:8] # YYMMDD format
                sdate = fields[-5][0:8] # YYYYMMDD format
                stime = datetime.datetime.strptime(fields[-5], datefmt)
                ms = mdate + '-' + sdate

                if np.absolute((stime - mtime).total_seconds()) > inps.yr_max * 365.0 * 24.0 * 3600:
                    pairs2.append(ms)
                    #print('WARNING: time span of pair {} larger than threshhold, skip this pair...')
                    continue


                if not os.path.exists(ms):
                    pairs.append(ms)
                    os.mkdir(ms)
                    make_pair(ms,group[i],group[j],tree,root)
                else:
                    print('Skip creating existing pair directory {}'.format(ms))

    if len(pairs) == 0:
        print('Created no pairs!')
    else:
        print('created the following pairs:')
        for x in pairs:
            print('{}'.format(x))
        #print('***************************************')
        print('total number of pairs created: {}'.format(len(pairs)))

    # Write to logfile
    filename1 = 'pair_log_{}.txt'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    f = open(filename1,"w")
    if len(pairs) == 0:
        f.write('Created no pairs!')
    else:
        f.write('Pairs created:\n')
        for x in pairs:
            f.write('{}\n'.format(x))
    f.close()


    if len(pairs2) != 0:
        print('time spans of the following pairs are larger than threshold')
        print('they are not created')
        for x in pairs2:
            print('{}'.format(x))
        print('total number of pairs not created: {}'.format(len(pairs2)))
    else:
        if len(pairs) == 0:
            pass
        else:
            print('all possible pairs created')






