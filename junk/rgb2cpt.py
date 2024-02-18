#!/usr/bin/env python3

## ------------------------------------------ ##
#  Convert RGB table to a 8-column CPT file
#  with overrule background, foreground, and NaN colors
#  Y.K. Liu @ 2021 June
## ------------------------------------------ ##

# 1. Customize your rgb table here: http://jdherman.github.io/colormap/
# 2. Save it as rgb plaintext file: 'your_custom_rgb_table.rgb'
# 3. Use this script to convert it to a 8-column CPT file

import sys
import argparse
import numpy as np


def cmdLineParse():
    description = ' Convert RGB table to a 8-column CPT file:\n \
            1. Customize your rgb table here: http://jdherman.github.io/colormap/ \n \
            2. Save it as rgb plaintext file: "your_custom_rgb_table.rgb" \n \
            3. Use this script to convert it to a 8-column CPT file \n \
                e.g., rgb2cpt.py -i your_custom_rgb_table.rgb -o my_newcpt.cpt'
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', dest='infile', type=str, required=True,
            help = 'input RGB filename')
    parser.add_argument('-o', '--output', dest='outfile', type=str, default='my_newcpt.cpt',
            help = 'output CPT filename')
    parser.add_argument('--nc', dest='nc', type=int, nargs=3, default=[255, 255, 255],
            help = 'NaN color. (default: %(default)s)')
    parser.add_argument('--bc', dest='bc', type=int, nargs=3, default=None,
            help = 'Background color. (default: as lower bound)')
    parser.add_argument('--fc', dest='fc', type=int, nargs=3, default=None,
            help = 'Foreground color. (default: as upper bound)')
    parser.add_argument('--head', dest='add_head', type=str, default=None,
            help = 'additional headings. (default: %(default)s)')

    # print argparse help if less than 2 argv
    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


if __name__ == '__main__':

    inps    = cmdLineParse()
    inname  = inps.infile
    outname = inps.outfile

    # Add headings
    heading =  '# \n'
    heading += '#         {}\n'.format(inname.split('.')[0])
    heading += '#                   self-defined cpt\n'
    if inps.add_head:
        heading += '# '+inps.add_head+'\n'

    with open(outname, 'w+') as outfile:
        outfile.write(heading)
        # Read the RGB table file
        with open(inname) as f:
            lines = f.readlines()
            pos = np.linspace(0,1,len(lines))
            for i in range(len(lines)-1):
                r1, g1, b1 = lines[i].split()
                r2, g2, b2 = lines[i+1].split()
                if i == 0:
                    l_bound = r1, g1, b1
                elif i == len(lines)-2:
                    u_bound = r2, g2, b2
                pos1 = pos[i]
                pos2 = pos[i+1]
                line = '{:.6f} {:s} {:s} {:s} {:.6f} {:s} {:s} {:s}'
                #print(line.format(pos1, r1, g1, b1, pos2, r2, g2, b2))
                outfile.write(line.format(pos1, r1, g1, b1, pos2, r2, g2, b2)+'\n')

        nc = inps.nc
        if not inps.bc:
            bc = l_bound
        else:
            bc = inps.bc
        if not inps.fc:
            fc = u_bound
        else:
            fc = inps.fc

        # append NaN color
        outfile.write('N {} {} {}\n'.format(*nc))
        # append background color
        outfile.write('B {} {} {}\n'.format(*bc))
        # append foreground color
        outfile.write('F {} {} {}'.format(*fc))

    print('Complete converting {} to {}'.format(inname, outname))
