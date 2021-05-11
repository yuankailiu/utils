#!/usr/bin/env python3
# ykliu @ May 02, 2021
#
# Multilook a batch of pairs of interferogram
#   - workflow:
#       1. go into each pair directory
#       2. use `looks.py` to multilook -range -azimuth
#       3. save new files in merged/YYYYMMDD-YYYYMMDD/ under the working dir as topsStack.py
#  E.g. looks.py -i filt_topophase.unw.geo -o mul10_filt_topophase.unw.geo -r 10 -a 10

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
    description = 'Multilook a batch of pairs of interferogram \n \
        - workflow:\n \
            1. go into each pair directory \n \
            2. use `looks.py` to multilook -range -azimuth \n \
            3. save new files in merged/YYYYMMDD-YYYYMMDD/ under the working dir as topsStack.py \n'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', dest='dir', type=str, default='./merged/interferograms',
            help = 'output directory. Default: `./merged/interferograms`')    
    parser.add_argument('-r', dest='rlooks', type=int, required=True,
            help = 'range looks (int), e.g., 10')
    parser.add_argument('-a', dest='alooks', type=int, required=True,
            help = 'azimuth looks (int), e.g., 10')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps = cmdLineParse()

    # Define the multilooking
    rlooks = inps.rlooks
    alooks = inps.alooks

    current_path = os.path.abspath(os.getcwd())
    print('Read acquisitions from the current working path: %s' % current_path)

    pairs = glob.glob('*-*/')
    print('Start multilooking are acquisitions...')

    n = 0
    for i in range(len(pairs)):
        start_dt = dt.now()
        pair = pairs[i].split('/')[0].split('-')
        new_pair = '{}_{}'.format(ptime.yyyymmdd(pair[0]), ptime.yyyymmdd(pair[1]))
        output_path = '{}/{}'.format(inps.dir, new_pair)

        # Get original path for this pair
        input_path = './{}merged'.format(pairs[i])
        
        # Make baseline directory and have a baseline filename
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Read from these input files
        i_unw = '{}/filt_topophase.unw.geo'.format(input_path)
        i_cor = '{}/phsig.cor.geo'.format(input_path)
        i_cmp = '{}/filt_topophase.unw.conncomp.geo'.format(input_path)
        i_ion = '{}/topophase.ion.geo'.format(input_path)

        # Save these output files
        o_unw = '{}/{}'.format(output_path, i_unw.split('/')[-1])
        o_cor = '{}/{}'.format(output_path, i_cor.split('/')[-1])
        o_cmp = '{}/{}'.format(output_path, i_cmp.split('/')[-1])
        o_ion = '{}/{}'.format(output_path, i_ion.split('/')[-1])

        bashCmd = 'looks.py -i {} -o {} -r {} -a {}'.format(i_unw, o_unw, rlooks, alooks)
        print('\n >> ', bashCmd)
        process = subprocess.run(bashCmd, shell=True)

        bashCmd = 'looks.py -i {} -o {} -r {} -a {}'.format(i_cor, o_cor, rlooks, alooks)
        print('\n >> ', bashCmd)
        process = subprocess.run(bashCmd, shell=True)

        bashCmd = 'looks.py -i {} -o {} -r {} -a {}'.format(i_cmp, o_cmp, rlooks, alooks)
        print('\n >> ', bashCmd)
        process = subprocess.run(bashCmd, shell=True)

        bashCmd = 'looks.py -i {} -o {} -r {} -a {}'.format(i_ion, o_ion, rlooks, alooks)
        print('\n >> ', bashCmd)
        process = subprocess.run(bashCmd, shell=True)

        n += 1
    print('Multilooked %d pairs' % int(n))
    print('Normal complete')