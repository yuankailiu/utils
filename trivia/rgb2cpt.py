#!/usr/bin/env python3

## ------------------------------------------ ##
#  Convert RGB table to a 8-column CPT file
#  with overrule background, foreground, and NaN colors
#  Y.K. Liu @ 2021 June
## ------------------------------------------ ##

# 1. Customize your rgb table here: http://jdherman.github.io/colormap/
# 2. Save it as rgb plaintext file: 'your_custom_rgb_table.rgb'
# 3. Use this script to convert it to a 8-column CPT file

import numpy as np

inname = 'your_custom_rgb_table.rgb'
outname = '{}.cpt'.format(inname.split('.')[0])

with open(outname, 'w+') as outfile:
    # Add headings
    outfile.write('# \n')
    outfile.write('#         {}\n'.format(inname.split('.')[0]))
    outfile.write('#                   self-defined cpt\n')

    # Read the RGB table file
    with open(inname) as f:
        lines = f.readlines()
        pos = np.linspace(0,1,len(lines))
        for i in range(len(lines)-1):
            r1, g1, b1 = lines[i].split()
            r2, g2, b2 = lines[i+1].split()
            if i == 0:
                bc = r1, g1, b1
            elif i == len(lines)-2:
                fc = r2, g2, b2
            pos1 = pos[i]
            pos2 = pos[i+1]
            #print('{:.6f} {:s} {:s} {:s} {:.6f} {:s} {:s} {:s}'.format(pos1, r1, g1, b1, pos2, r2, g2, b2))
            outfile.write('{:.6f} {:s} {:s} {:s} {:.6f} {:s} {:s} {:s}\n'.format(pos1, r1, g1, b1, pos2, r2, g2, b2))

    # append NaN color
    outfile.write('N 255 255 255\n')

    # append background color
    outfile.write('B {} {} {}\n'.format(bc[0], bc[1], bc[2]))

    # append foreground color
    outfile.write('F {} {} {}'.format(fc[0], fc[1], fc[2]))

print('Complete converting {} to {}'.format(inname, outname))
