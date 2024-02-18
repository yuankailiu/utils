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

import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime as dt

import numpy as np
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
    parser.add_argument('-w', dest='workdir', type=str, default='.',
            help = 'current working directory. (default: %(default)s)')
    parser.add_argument('-d', dest='dir', type=str, default='./baselines',
            help = 'output directory for the baselines. (default: %(default)s)')
    parser.add_argument('-r', dest='reference_dir', type=str, default='referencedir',
            help = 'default directory name for the reference acquisition. (default: %(default)s)')
    parser.add_argument('-s', dest='secondary_dir', type=str, default='secondarydir',
            help = 'default directory name for the secondary acquisition. (default: %(default)s)')
    parser.add_argument('-m', dest='method', type=str, default='single',
            help = 'reference_secondary pairing. (default: %(default)s)')
    parser.add_argument('-p', dest='subproc_opt', type=str, default='popen',
            help = 'choose either `run` or `popen`. (default: %(default)s)')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()

    outdir = inps.dir
    method = inps.method
    workdir = inps.workdir
    current_path = os.path.abspath(os.getcwd())
    print('Read acquisition dates from pair directories in the working path: %s' % current_path)

    pairs = glob.glob(f'{workdir}/20*-20*')
    refs = []
    secs = []
    for i in range(len(pairs)):
        ref = pairs[i].split('/')[-1].split('-')[0]
        sec = pairs[i].split('/')[-1].split('-')[1]
        refs.append(ref)
        secs.append(sec)

    if method == 'single':
        secs = list(set(refs+secs))
        secs.sort()
        refs = list([secs[0]] * (len(secs)-1))
        secs = list(np.array(secs)[1:])
        print('Total number of dates: %d' % int(len(secs)+1))
        print('Set the first date as the common reference for baseline computation: %s' % refs[0])
        res = "\n".join(f"{x} {y}" for x, y in zip(refs, secs))
        print(res)

    elif method == 'pairwise':
        refs = list(refs)
        secs = list(secs)
        print('Total number of pairs: %d' % len(secs))
        print('Pairwise reference_secondary for baseline computation')
        res = "\n".join(f"{x} {y}" for x, y in zip(refs, secs))
        print(res)


    print('Start making baseline folders, compute baselines...')
    print(f'Output files will be saved under {outdir}/*_*')
    print(f'subprocess option: {inps.subproc_opt}')

    n = 0
    for i in range(len(secs)):
        start_dt = dt.now()

        reference = refs[i]
        secondary = secs[i]
        baseline_pair = f'{ptime.yyyymmdd(reference)}_{ptime.yyyymmdd(secondary)}'
        baseline_path = f'{outdir}/{baseline_pair}'

        # Get paths of ref and sec acquisitions for this baseline pair
        p_ref = [p for p in pairs if reference in p][0]
        p_sec = [p for p in pairs if secondary in p][0]

        if p_ref.split('/')[-1].split('-').index(reference) == 0:
            pos_ref=inps.reference_dir
        elif p_ref.split('/')[-1].split('-').index(reference) == 1:
            pos_ref=inps.secondary_dir
        if p_sec.split('/')[-1].split('-').index(secondary) == 0:
            pos_sec=inps.reference_dir
        elif p_sec.split('/')[-1].split('-').index(secondary) == 1:
            pos_sec=inps.secondary_dir
        refpath = f'{p_ref}/{pos_ref}'
        secpath = f'{p_sec}/{pos_sec}'

        # Make baseline directory and have a baseline filename
        print('  ', baseline_pair)
        if not os.path.exists(baseline_path):
            os.makedirs(baseline_path)

        # Write a logfile
        f = open(f'{baseline_path}/baselineCompute.log', 'w+')
        f.write('### [ Run computeBaseline.py ]\n')
        f.write(f'#   >> Time   now    : {start_dt}\n')
        f.write(f'#   >> Baseline pair : {baseline_pair}\n')
        f.write(f'#   >> Reference path: {refpath}\n')
        f.write(f'#   >> Secondary path: {secpath}\n\n')

        # Run `computeBaseline.py` and keep the output to the logfile
        baseline_file = f'{baseline_path}/{baseline_pair}.txt'

        if inps.subproc_opt == 'run':
            # subprocess option 1: sequential computing
            bashCmd = f'computeBaseline.py -m {refpath} -s {secpath} -b {baseline_file}'
            process = subprocess.run(bashCmd, stdout=f, shell=True)

        elif inps.subproc_opt == 'popen':
            # subprocess option 2: parallel in background; faster
            bashCmd = ['computeBaseline.py', '-m', refpath, '-s', secpath, '-b', baseline_file]
            process = subprocess.Popen(bashCmd, stdout=f)

        n += 1

    print('Computed %d pairs' % int(n))
    print('Normal complete')
