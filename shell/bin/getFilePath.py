#!/usr/bin/env python3

import sys
import os

usage = """
Usage: getFilePath.py file
"""

def main(filename):

    cdir = os.getcwd()
    print(os.path.join(cdir, filename))


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        print(usage)
        sys.exit()
    filename = args[0]
    main(filename)
