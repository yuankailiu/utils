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
            help = 'output directory. (default: %(default)s)')
    parser.add_argument('-r', dest='rlooks', type=int, required=True,
            help = 'range looks (int), e.g., 10')
    parser.add_argument('-a', dest='alooks', type=int, required=True,
            help = 'azimuth looks (int), e.g., 10')
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

    # Define the multilooking
    rlooks = inps.rlooks
    alooks = inps.alooks

    current_path = os.path.abspath(os.getcwd())
    print('Read acquisitions from the current working path: %s' % current_path)

    pairs = glob.glob('*-*/')
    print('Start multilooking all acquisitions...')
    print('Subprocess option: {}'.format(inps.subproc_opt))

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

        # Write a logfile
        f = open('{}/looksCompute.log'.format(output_path), 'w+')
        f.write('### [ Run looks.py ]\n')
        f.write('#   >> Time   now     : {}\n'.format(start_dt))
        f.write('#   >> subprocess opt : {}\n'.format(inps.subproc_opt))
        f.write('#   >> range looks    : {}\n'.format(rlooks))
        f.write('#   >> azimuth looks  : {}\n\n'.format(alooks))

        if (rlooks!=1) and (alooks!=1):
            print('Multilook the following files to {}'.format(output_path))
            ## run looks.py on datasets:
            if inps.subproc_opt == 'run':
                for infile, outfile in list(zip([i_unw, i_cor, i_cmp, i_ion], [o_unw, o_cor, o_cmp, o_ion])):
                    bashCmd = 'looks.py -i {} -o {} -r {} -a {}'.format(infile, outfile, rlooks, alooks)
                    print('\n >> ', bashCmd)
                    process = subprocess.run(bashCmd, stdout=f, shell=True)

            # still problematic, may overwrite input files (not recommended)
            elif inps.subproc_opt == 'popen':
                for infile, outfile in list(zip([i_unw, i_cor, i_cmp, i_ion], [o_unw, o_cor, o_cmp, o_ion])):
                    bashCmd = ['looks.py', '-i', infile, '-o', outfile, '-r', str(rlooks), '-a', str(alooks)]
                    process = subprocess.Popen(bashCmd, stdout=f)

        else:
            # if multilook 1, just copy files over there
            print('Without multilook, copy the following files to {}'.format(output_path))
            for infile in [i_unw, i_cor, i_cmp, i_ion]:
                bashCmd = 'rsync {}* {}'.format(infile, output_path)
                print('\n >> ', bashCmd)
                process = subprocess.run(bashCmd, stdout=f, shell=True)

        n += 1
    print('Multilooked %d pairs' % int(n))
    print('Normal complete')