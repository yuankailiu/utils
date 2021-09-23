#!/usr/bin/env python3
# ykliu @ Sep 17, 2021
#
# Add the iono phase back to the corrected unwrapPhase to get the original phase
#   - workflow:
#       1. go into each merged pair directory
#       2. use `imageMath.py` to add two files and export a new file


import os
import sys
import glob
import argparse
import subprocess


def cmdLineParse():
    '''
    Command line parser.
    '''
    description = ' Add the iono phase back to the corrected unwrapPhase to get the original phase \n \
        - workflow:\n \
            1. go into each merged pair directory \n \
            2. use `imageMath.py` to add two files and export a new file \n \
        Example: loopimgMath.py -d . -a filt_topophase.unw.geo -b topophase.ion.geo -o filt_topophase_ori.unw.geo -m + -p run '
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', dest='dir', type=str, default='cwd',
            help = 'working directory that contains lots of pair folders. (default: %(default)s)')
    parser.add_argument('-a', dest='file_a', type=str, required=True,
            help = 'file a')
    parser.add_argument('-b', dest='file_b', type=str, required=True,
            help = 'file b')
    parser.add_argument('-o', '--out', '--output', dest='outfile', type=str, required=True,
            help = 'output file')
    parser.add_argument('-m', dest='method', type=str, default='+',
            help = 'imageMath.py operation method. (default: %(default)s)')
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

    # Operation string
    if inps.method == '+':
        op_method = "'a_0;a_1+b'"
    elif inps.method == '-':
        op_method = "'a_0;a_1-b'"
    else:
        op_method = inps.method


    # Working directory
    if inps.dir == 'cwd':
        inps.dir = os.path.abspath(os.getcwd())
        print('Pair folders under the path: %s' % inps.dir)


    # Fetch the pair directories
    pairs = glob.glob('*_*/')
    print('Start the math operation on all pairs...')
    print('Subprocess option: {}'.format(inps.subproc_opt))


    # Loop over pairs
    for i in range(len(pairs)):
        pair_path   = os.path.join(inps.dir,  pairs[i])
        outfile     = os.path.join(pair_path, inps.outfile)
        file_a       = os.path.join(pair_path, inps.file_a)
        file_b       = os.path.join(pair_path, inps.file_b)

        if inps.subproc_opt == 'run':
            bashCmd = 'imageMath.py -e {} -o {} --a {} --b {}'.format(op_method, outfile, file_a, file_b)
            print('\n >> ', bashCmd)
            process = subprocess.run(bashCmd, shell=True)

        elif inps.subproc_opt == 'popen':
            bashCmd = ['imageMath.py', '-e', op_method, '-o', outfile, '--a', file_a, '--b', file_b]
            process = subprocess.Popen(bashCmd)

        elif inps.subproc_opt == 'dry':
            bashCmd = 'imageMath.py -e {} -o {} --a {} --b {}'.format(op_method, outfile, file_a, file_b)
            print('>> ', bashCmd)


    # Done
    print('Operated %d pairs' % int(i+1))
    print('Normal complete')
