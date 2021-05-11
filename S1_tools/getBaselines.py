#!/usr/bin/env python3
# ykliu @ May 03, 2021
#
# Prepare baseline timeseries for a batch of interferograms
#   - baseline time-series pairs (as topsStack.py):
#       reference-secondary1, reference-secondary2, ...
#   - workflow:
#       1. get all dates in the working directory, e.g., process/
#       2. use `computeBaseline.py` to calculate baselines (isce2/contrib/stack/topsStack/computeBaseline.py)
#       3. saved baselines to txt files under a baselines/ directory as topsStack.py
# Required: 
#       1. load isce2 topsStack from path, do one of the the following:
#           1) load_tops_stack
#           2) export PATH=${PATH}:${ISCE_STACK}/topsStack
#       2. load mintpy from path (optional):
#           This is just to use ptime function to convert YYMMDD to YYYYMMDD format

import os
import sys
import glob
import argparse
import subprocess
import numpy as np
from datetime import datetime as dt
from mintpy.utils import ptime

def cmdLineParse():
    '''
    Command line parser.
    '''
    description = 'Prepare baseline timeseries for a batch of interferograms \n \
        - baseline time-series pairs (as topsStack.py):\n \
            reference-secondary1, reference-secondary2, ...\n \
        - workflow: \n \
            1. get all dates in the working directory, e.g., process/ \n \
            2. use `computeBaseline.py` to calculate baselines (isce2/contrib/stack/topsStack/computeBaseline.py) \n \
            3. saved baselines to txt files under a baselines/ directory as topsStack.py \n \
        Required: \n \
            1. load isce2 topsStack from path, do one of the the following: \n \
                1) load_tops_stack \n \
                2) export PATH=${PATH}:${ISCE_STACK}/topsStack \n \
            2. load mintpy from path: \n \
                This is just to use ptime function to convert YYMMDD to YYYYMMDD format \n'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', dest='dir', type=str, default='./baselines',
            help = 'output directory for the baselines. Default: `./baselines`')
    parser.add_argument('-r', dest='reference_dir', type=str, default='referencedir',
            help = 'default directory name for the reference acquisition. Default: `referencedir`')
    parser.add_argument('-s', dest='secondary_dir', type=str, default='secondarydir',
            help = 'default directory name for the secondary acquisition. Default: `secondarydir`')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()

    outdir = inps.dir
    current_path = os.path.abspath(os.getcwd())
    print('Read acquisition dates from pair directories in the working path: %s' % current_path)

    pairs = glob.glob('*-*/')
    dates = []
    for i in range(len(pairs)):
        reference = pairs[i].split('-')[0]
        secondary = pairs[i].split('-')[1][:-1]
        dates.append(reference)
        dates.append(secondary)
    dates = list(set(dates))
    dates.sort()

    reference = dates[0]

    print('Total number of dates: %d' % len(dates))
    print('Set the first date as the common reference for baseline computation: %s' % reference)
    print('Start making baseline folders, compute baselines...')
    print('Output files will be saved under {}/*_*'.format(outdir))

    n = 0
    for i in range(1,len(dates[:])):
        start_dt = dt.now()

        secondary = dates[i]
        baseline_pair = '{}_{}'.format(ptime.yyyymmdd(reference), ptime.yyyymmdd(secondary))
        baseline_path = '{}/{}'.format(outdir, baseline_pair)

        # Get paths of ref and sec acquisitions for this baseline pair
        p_ref = [p for p in pairs if reference in p][0][:-1]
        p_sec = [p for p in pairs if secondary in p][0][:-1]
        if p_ref.split('-').index(reference) == 0:
            pos_ref=inps.reference_dir
        elif p_ref.split('-').index(reference) == 1:
            pos_ref=inps.secondary_dir
        if p_sec.split('-').index(secondary) == 0:
            pos_sec=inps.reference_dir
        elif p_sec.split('-').index(secondary) == 1:
            pos_sec=inps.secondary_dir
        refpath = './{}/{}'.format(p_ref, pos_ref)
        secpath = './{}/{}'.format(p_sec, pos_sec)
        
        # Make baseline directory and have a baseline filename
        print('  ', baseline_pair)
        if not os.path.exists(baseline_path):
            os.makedirs(baseline_path)
            
        # Write a logfile
        f = open('{}/baselineCompute.log'.format(baseline_path), 'w+')
        f.write('### [ Run computeBaseline.py ]\n')    
        f.write('#   >> Time   now    : {}\n'.format(start_dt))
        f.write('#   >> Baseline pair : {}\n'.format(baseline_pair))
        f.write('#   >> Reference path: {}\n'.format(refpath))
        f.write('#   >> Secondary path: {}\n\n'.format(secpath))

        # Run `computeBaseline.py` and keep the output to the logfile
        baseline_file = '{}/{}.txt'.format(baseline_path, baseline_pair)

        # method 1:
        bashCmd = 'computeBaseline.py -m {} -s {} -b {}'.format(refpath, secpath, baseline_file)
        process = subprocess.run(bashCmd, stdout=f, shell=True)

        # method 2:
        #bashCmd = ['computeBaseline.py', '-m', refpath, '-s', secpath, '-b', baseline_file]
        #process = subprocess.Popen(bashCmd, stdout=f)

        n += 1

    print('Computed %d pairs' % int(n))
    print('Normal complete')