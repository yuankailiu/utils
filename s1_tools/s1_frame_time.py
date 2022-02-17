#!/usr/bin/env python3
##
## Reading the burst files in each subswath (IW*.xml) to get the frame time information
## This is for checking potential gaps between frames if weird stripes of decorrelation occurs
## No need to use it in a standard processing workflow
##
##  ykliu @ Jul 20, 2021


import sys
import glob
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime as dt
import argparse


## Set the path and dates to read the xml file
def cmdLineParse():
    '''
    Command line parser.
    '''
    description = 'Get the frame startign and ending sensing times \n'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dir', dest='basedir', type=str, default='./',
            help = 'Base directory for reading the pair folders. (default: %(default)s)')
    parser.add_argument('--dates', dest='dates', type=str, required=True, nargs='+',
            help = 'Date of SLCs you want to check')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()



if __name__ == '__main__':

    inps = cmdLineParse()
    basedir = inps.basedir
    dates   = inps.dates


    ## Reading the xml file
    SWATHS     = ['IW1', 'IW2', 'IW3']
    name_frame = ['a', 'b', 'c', 'd', 'e'] # the pairs need to be processed frame-by-frame. Pair folder named as YYYYMMDD-YYYYMMDD_a is the first frame.
    A=dict()

    for date in dates:
        print('Working on date {}'.format(date))
        A[date] = dict()

        tmp = glob.glob(basedir+'/*{}*/'.format(date))
        tmp.sort()

        for path in tmp:
            pair = path.split('/')[-2]
            frame_num = 'frame_'+str(name_frame.index(pair.split('_')[-1])+1)
            A[date][frame_num] = dict()

            if date in pair.split('-')[0]:
                xml = path+'reference.xml'
                dir = '/referencedir'
            elif date in pair.split('-')[1]:
                xml = path+'secondary.xml'
                dir = '/secondarydir'

            # read the SLC filename from major xml
            with open(xml, 'r') as f:
                root = ET.fromstring(f.read())
            for prop in root.iter('property'):
                if prop.attrib['name'] == 'safe':
                    A[date][frame_num]['SLC_NAME'] = prop.text[2:-2].split('/')[-1].split('.')[0]


            # read sensing times from swath xml
            for i in SWATHS:
                A[date][frame_num][i] = dict()

                swath_xml = path+dir+'/'+i+'.xml'

                with open(swath_xml, 'r') as f:
                    root = ET.fromstring(f.read())

                tmp = []
                for prop in root.iter('property'):
                    if prop.attrib['name'] == 'azimuthtimeinterval':
                        for val in prop.iter('value'):
                            tmp.append(val.text)
                if len(list(set(tmp))) == 1:
                    ati = float(list(set(tmp))[0])
                else:
                    ati = float(list(set(tmp))[0])
                    print('Warning: have varying azimuth time interval, take {} s'.format(ati))

                s_start = []
                for prop in root.iter('property'):
                    if prop.attrib['name'] == 'sensingstart':
                        for val in prop.iter('value'):
                            s_start.append(val.text)

                s_stop = []
                for prop in root.iter('property'):
                    if prop.attrib['name'] == 'sensingstop':
                        for val in prop.iter('value'):
                            s_stop.append(val.text)

                res       = [None]*(len(s_start)+len(s_stop))
                res[::2]  = s_start
                res[1::2] = s_stop

                res_obj = [dt.strptime(x,'%Y-%m-%d %H:%M:%S.%f') for x in res]

                A[date][frame_num][i]['AzimuthTimeInt'] = ati
                A[date][frame_num][i]['SensingTime']    = [start+'\t'+stop for start, stop in zip(s_start, s_stop)]
                A[date][frame_num][i]['SensingStart']   = res_obj[0]
                A[date][frame_num][i]['SensingStop']    = res_obj[-1]


    ## Save the results
    for date in dates:
        fname = 'frameInfo_{}.txt'.format(date)
        f = open(fname, 'w')
        print('Save info to text file {}'.format(fname))

        head =  '# Date of acquisition: {}\n'.format(date)
        head += '# Frame start/stop sensing times\n'
        head += '# Identify any anomalous overlapping frames or gaps\n\n'
        f.write(head)

        for frame in A[date]:
            f.write('> {} SLC\t{}\n'.format(frame, A[date][frame]['SLC_NAME']))

        for swath in SWATHS:
            ts = []
            f.write('\n'+'+'*60+'\n')
            f.write('#### Subswath:  {}\n'.format(swath))

            for frame in A[date]:
                i = 1
                for res_i in A[date][frame][swath]['SensingTime']:
                    f.write('{}_burst_{:02d}\t{}\n'.format(frame, int(i), res_i))
                    i += 1
                ts.append(A[date][frame][swath]['SensingStart'])
                ts.append(A[date][frame][swath]['SensingStop'] )

            int_time = np.array([x.total_seconds() for x in np.diff(ts)][::2])
            ovp_time = np.array([x.total_seconds() for x in np.diff(ts)][1::2])

            f.write('\n## Frame azimuth interval times [seconds]\n')
            for i in range(len(int_time)):
                f.write('frame_{}\t\t{:.6f}\n'.format(i+1, int_time[i]))

            f.write('\n## Frame num of lines (ati={} s)\n'.format(ati))
            for i in range(len(int_time)):
                f.write('frame_{}\t\t{:.6f}\n'.format(i+1, int_time[i]/ati))

            f.write('\n## Frame overlapped times [seconds] (- for overlap; + for gap)\n')
            for i in range(len(ovp_time)):
                f.write('frame_{}-{}\t{:.6f}\n'.format(i+1, i+2, ovp_time[i]))

            f.write('\n## Frame overlapped num of lines (ati={} s)\n'.format(ati))
            for i in range(len(ovp_time)):
                f.write('frame_{}-{}\t{:.6f}\n'.format(i+1, i+2, ovp_time[i]/ati))

    print('Complete!')
