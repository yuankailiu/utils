#!/usr/bin/env python3

## Read and plot a batch of single look complex images
##
##  Author: ykliu 2022

import os
import sys
import glob
import argparse
from functools import partial
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from osgeo import gdal
from multiprocessing import Pool
matplotlib.rcParams.update({'font.size': 14})


#####################################################################

def create_parser():
    parser = argparse.ArgumentParser(description='Read and plot the amplitude of sigle look complex',
                                    formatter_class=argparse.RawTextHelpFormatter,
                                    epilog = '''
Example:

plot_slc.py -d ./SLC -o ./pic -p 8 -b 7000 19000 20000 60000 -v 0 250 -a 4 --dpi 300 --oform png

''')
    parser.add_argument('-d','--dir', type=str, required=True,
            help='Directory contains pair/date folders.', dest='dpath')
    parser.add_argument('-o','--outdir',type=str, default='./pic/',
            help='Output figure directory. Default: %(default)s', dest='outdir')
    parser.add_argument('--date', type=str, default=None,
            help='Only plot this date. Default: %(default)s', dest='date')
    parser.add_argument('--ext', type=str, default='.slc.full',
            help='SLC extension. Default: %(default)s', dest='ext')
    parser.add_argument('-p','--numproc',type=int, default=1,
            help='Num of multi processes. Default: %(default)s', dest='numproc')
    parser.add_argument('-b','--bbox',type=int, nargs=4, default=[0,-1,0,-1],
            help='Bounding box; first_line, end_line, first_samp, end_samp. Default: %(default)s', dest='bbox')
    parser.add_argument('-v','--vlim', nargs=2, metavar=('VMIN', 'VMAX'), type=float, default=[None, None],
            help='Display limits for matrix plotting.', dest='vlim')
    parser.add_argument('-a','--aspect',type=float, default=1.0,
            help='Aspect ratio (>1 means Y is longer). Default: %(default)s', dest='aspect')
    parser.add_argument('--north', action='store_true',
            help='Rotate the plot to shown North being roughly up. Default: %(default)s')
    parser.add_argument('--dpi',type=float, default=300,
            help='Dpi of output images. Default: %(default)s', dest='dpi')
    parser.add_argument('--oform', type=str, default='png',
            help='Output image format (png, tif, jpeg). Default: %(default)s', dest='outform')

    values = parser.parse_args()
    return values


## ==================================================================
## Define functions (from Radar Imaging class)
## ==================================================================

def readbinary(file, nsamp, nline, dtype):
    with open(file, 'rb') as fn:
        print('doing np.frombuffer...')
        load_arr = np.frombuffer(fn.read(), dtype=dtype)
        print('done with np.frombuffer')
        load_arr = load_arr.reshape((nline, nsamp))
    return np.array(load_arr)


def plot_img(data, nhdr=0, title='Data', scale=1, cmap='gray', vlim=None,
            orglim=None, origin='upper', aspect="equal", xlabel='Range [bins]', ylabel='Azimuth [lines]',
            clabel='Value [-]', interpolation='none', savefig=None, figsize=[6,6],
            lim=[None,None,None,None], ex=None, yticks=None, dpi=300):
    if scale > 1:
        clabel = '{} * {}'.format(clabel, scale)
    else:
        clabel = '{}'.format(clabel)

    # Adjust the data header part for better visualization
    val = np.array(data)
    val[:,nhdr:] = scale * val[:,nhdr:]

    if vlim is None:
        vlim = [None, None]

    # original value limit (for overlap image coloring output which varies from 0~1)
    if orglim is not None:
        if vlim[0] is not None:
            vlim[0] = (vlim[0]-orglim[0])/np.diff(orglim)[0]
        else:
            vlim[0] = 0
        if vlim[1] is not None:
            vlim[1] = (vlim[1]-orglim[0])/np.diff(orglim)[0]
        else:
            vlim[1] = 1
        cticks      = np.linspace(  vlim[0],   vlim[1], 4)
        cticklabels = np.linspace(orglim[0], orglim[1], 4)

    # plot the 2D image
    plt.figure(figsize=figsize)
    im   = plt.imshow(val, cmap=cmap, interpolation=interpolation, vmin=vlim[0], vmax=vlim[1],
                        origin=origin, aspect=aspect, extent=ex)
    cbar = plt.colorbar(im, shrink=0.4, pad=0.06)
    cbar.set_label(clabel, rotation=270, labelpad=30)
    if orglim is not None:
        cbar.set_ticks(cticks)
        cbar.ax.set_yticklabels('{:.2f}'.format(x) for x in cticklabels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(lim[0], lim[1])
    plt.ylim(lim[2], lim[3])
    if (yticks is not None) and (len(yticks)==3):
        plt.yticks(np.linspace(lim[2], lim[3], yticks[2]), np.linspace(yticks[0], yticks[1], yticks[2]))
    if savefig is not None:
        form = savefig.split('.')[-1]
        if form.startswith('tif'):
            plt.savefig('{}'.format(savefig), dpi=dpi, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
        else:
            plt.savefig('{}'.format(savefig), dpi=dpi, format=form)
    print('Saved {}'.format(savefig))
    plt.close()
    return

## ==================================================================
## ##################################################################
## ==================================================================


def mk_outdir(inps):
    if not os.path.exists(inps.outdir):
        os.makedirs(inps.outdir)
        print('Make the output directory ', inps.outdir)
    return


def get_flist(inps):
    tmp = inps.dpath + '/*/*' + inps.ext
    fnames = glob.glob(tmp)
    for f in fnames:
        print(f)
    print('Total {} files \n'.format(len(fnames)))
    return fnames


def plot_slc(fname, inps):
    if True:
        date = fname.split('/')[-1].split('.')[0]
        if len(date) != 8:
            print('Weird date identified: ', date)
            print('Nothing to do. Exiting ...')
            sys.exit(0)
        outname = os.path.expanduser(inps.outdir + '/amp_' + date + '.' + inps.outform)

        if not os.path.exists(outname):
            # subset region
            l1 = inps.bbox[0]
            l2 = inps.bbox[1]
            s1 = inps.bbox[2]
            s2 = inps.bbox[3]

            # read slc: ReadAsArray(start_col, start_row, num_col, num_row)
            data = gdal.Open(fname, gdal.GA_ReadOnly)
            slc  = data.GetRasterBand(1).ReadAsArray(s1, l1, s2-s1, l2-l1)

            # compute magnitude
            #amp = abs(slc[l1:l2, s1:s2])
            amp = abs(slc)

            # take care the roatation to make North being roughly up
            # (hacky way.... need to be careful!)
            if inps.north:
                print(' reading... ; north up', fname)
                origin = 'lower'
                amp = np.flip(amp.T, axis=1)
                inps.aspect = 1/inps.aspect
                ex = [amp.shape[1], 0, 0, amp.shape[0]]
                xlabel = 'Azimuth [lines]'
                ylabel = 'Range [samples]'
            else:
                print(' reading... ', fname)
                origin = 'upper'
                ex = None
                xlabel = 'Range [samples]'
                ylabel = 'Azimuth [lines]'

            # plot amp (without multilook; with antialiasing)
            titstr  = date[:4]+'-'+date[4:6]+'-'+date[6:]
            plot_img(amp, title=titstr, aspect=inps.aspect, vlim=[inps.vlim[0], inps.vlim[1]], clabel='Amp [-]',
                    figsize=[16,16], origin=origin, ex=ex, xlabel=xlabel, ylabel=ylabel,
                    interpolation='antialiased', savefig=outname, dpi=inps.dpi)



#####################################################################


if __name__ == '__main__':

    inps = create_parser()

    mk_outdir(inps)

    fnames = get_flist(inps)

    if inps.north:
        print('Rotate the plot to show north as roughly up (hacky... watch out!)')

    if inps.date:
        print('Only plot one SLC: ', inps.date)
        fname = inps.dpath+'/'+inps.date+'/'+inps.date+inps.ext
        plot_slc(fname=fname, inps=inps)
    else:
        print('Multi-processing, num of processes: ', inps.numproc)
        pool = Pool(inps.numproc)                           # create a multiprocessing Pool
        pool.map(partial(plot_slc, inps=inps), fnames)      # process data_inputs iterable with pool
        pool.close()
        pool.join()

    print('Done!!')
