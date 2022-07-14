#!/usr/bin/env python3
############################################################
# Program is for Sentinel-1 post-analysis                  #
# Author: Yuan-Kai Liu, 2022                               #
############################################################

import os
import sys
import glob
import argparse
import shutil


def cmdLineParse():
    '''
    Command line parser.
    '''
    description = 'Operate (cp mv rsync rm) files with keywords '
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', '--cmd', dest='cmd', type=str, required=True,
            help = 'bash commands operationg on the files (e.g., mv cp rsync rm).')
    parser.add_argument('-k', '--key', dest='keywords', type=str, nargs='+',
            help = 'filename keywords in the current path (e.g., apple F14 20160101)')
    parser.add_argument('-t', '--txt', dest='txt', type=str,
            help = 'a text file contains all the keywords and/or filenames (overwrites --key)')
    parser.add_argument('-d', '--dest', dest='dest', type=str, default=None,
            help = 'output path of the files, needed for cp mv rsync (default: %(default)s).')
    parser.add_argument('--print', dest='print', action='store_true', default=False, help = 'only print results to screen (no execution)')

    inps = parser.parse_args()

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    elif (inps.keywords is None) and (inps.txt is None):
        print('Both --key and --txt are none. Please use either of them.')
        sys.exit(1)
    else:
        return inps


def main(inps):
    if inps.keywords:
        src_list = []
        for k in inps.keywords:
	        src_list += glob.glob('*{}*'.format(k))

    if inps.txt:
        txt_file = open(inps.txt, 'r')
        src_list = txt_file.read()

    dst = inps.dest

    if inps.print:
        print('Number of files matching the patterns: {}'.format(len(src_list)))
        print('-'*80)
        for src in src_list:
            print(src)
        print('-'*80)
        print('command operation: {}'.format(inps.cmd))
        if dst:
            print('destination: {}'.format(dst))
    else:
        if not os.path.exists(dst):
            os.makedirs(dst)
        for src in src_list:
            if inps.cmd == 'cp':
                shutil.copy(src, dst)
            elif 'cp -r' in inps.cmd:
                shutil.copytree(src, dst)
            elif inps.cmd == 'mv':
                shutil.move(src, dst)
            elif inps.cmd == 'rm':
                os.remove(src)
            elif 'rm -r' in inps.cmd:
                os.rmdir(src)


if __name__ == '__main__':

    inps = cmdLineParse()

    main(inps)
