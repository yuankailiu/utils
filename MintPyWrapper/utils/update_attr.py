#!/usr/bin/env python3
############################################################
# Author: Yuan-Kai Liu June 2023                           #
############################################################

import glob
import os
import sys

import h5py
from mintpy.utils.arg_utils import create_argument_parser

#####################################################################################
EXAMPLE = """example:
  update_attr.py -r ./inputs/radar/geometryRadar.h5 -f ./inputs/geometryGeo.h5 ./inputs/ifgramStack.h5
  update_attr.py -r ./inputs/radar/geometryRadar.h5 -d .
"""

def create_parser(subparsers=None):
    synopsis = 'Update the attributes of the hdf5 files.'
    epilog = EXAMPLE
    name = __name__.split('.')[-1]
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=epilog, subparsers=subparsers)

    parser.add_argument('-r', '--file-ref', required=True, help='file to be the reference.')
    parser.add_argument('-f', '--file', nargs='+', help='files to be updated')
    parser.add_argument('-d', '--dir', help='main dir name where the path (& subdirs) contains some *.h5')
    parser.add_argument('--action', default='fillin', help='fillin or overwrite')
    return parser


def update_attr(file, file_ref, action='fillin'):
    file = h5py.File(file, 'a')
    file_ref = h5py.File(file_ref)

    keys_ref = list(file_ref.attrs.keys())
    keys = list(file.attrs.keys())

    # what to update
    if action == 'fillin':
        update_keys = list(set(keys_ref) - set(keys))
    elif action == 'overwrite':
        update_keys = keys_ref

    if len(update_keys) > 0:
        print(f'update the attributes: {update_keys}')
    else:
        print('nothing to update')

    # update keys in attr
    for key in update_keys:
        if key not in file.attrs.keys():
            print(f'update {key} {file_ref.attrs[key]}')
            file.attrs[key] = file_ref.attrs[key]
        else:
            print(f'{key} already exists')

    file.close()
    file_ref.close()
    return


def main(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    if (not inps.file) and (not inps.dir):
        print('please use -f or -d options')

    if inps.dir:
        files  = glob.glob(os.path.join(inps.dir+'/*.h5'))
        files += glob.glob(os.path.join(inps.dir+'/**/*.h5'))
        files += glob.glob(os.path.join(inps.dir+'/**/**/*.h5'))
        files = sorted(list(set(files)))
    else:
        files = inps.file

    print(files)

    for file in files:
        if file == inps.file_ref:
            print(f'skip self file: {file}')
        else:
            print(file)
            update_attr(file, inps.file_ref, action=inps.action)

#####################################################################################

if __name__ == '__main__':
    main(sys.argv[1:])
