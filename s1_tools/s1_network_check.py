#!/usr/bin/env python3
############################################################
# Check the network connectivity before running topsStack
# YKL @ 2022-03-15
############################################################

import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mintpy.utils import ptime
from mintpy.utils import plot as pp
from mintpy.objects import ifgramStack

import sarut.tools.plot as sarplt
import sarut.tools.math as sarmath


# Change font
if False:
    from matplotlib import font_manager
    font_dirs = ['/net/kraken.gps.caltech.edu/export/bak/ykliu/fonts']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
    plt.rcParams['font.family'] = 'Helvetica'

plt.rcParams.update({'font.size': 16})



######################################################################

def cmdLineParse():
    description = ' Check the network connectivity before running topsStack\n'+\
                  '  Must specify one from [-i IFGFILE], [-l LISTFILE], [-d DIR]'

    ## basic input/output files and paths
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-v', '--ver', dest='s1_version', type=str, required=True,
            help = 'A text file listing acquisition versions, starting ranges, e.g., s1_version.txt')
    parser.add_argument('-i', '--ifg', dest='ifgfile', type=str,
            help = 'Input ifg stack HDF5 file, e.g., path/to/ifgramStack.h5')
    parser.add_argument('-l', '--list', dest='listfile', type=str,
            help = 'Custom list file containing the pairs, e.g., run_files/run_16_unwrap, run_files/run_20_unwrap_ion')
    parser.add_argument('-d', '--dir', dest='dir', type=str,
            help = 'Directory containing the pair folders, e.g., merged/interferograms/')
    parser.add_argument('-n', '--name', dest='name', type=str, default='unw',
            help = 'Name of the ifg stack. (E.g. unw, ion, ... default: %(default)s)')
    parser.add_argument('--spread', dest='spread', type=float, default=0.,
            help = 'Random spread of the starting ranges (meter in y-axis) for visualization. (default: %(default)s)')

    inps = parser.parse_args()
    if len(sys.argv)<1:
        print('')
        parser.print_help()
        sys.exit(1)
    elif (inps.ifgfile is None) and (inps.listfile is None) and (inps.dir is None):
        print('Need to give --ifg or --list or --dir. Stop!')
        parser.print_help()
        sys.exit(1)
    else:
        source = ['from pair directories', 'from pairs list file', 'from ifgramStack.h5']
        if inps.dir:
            inps.src=1
        elif inps.listfile:
            inps.src=2
            if 'ion' in inps.listfile:
                inps.name = 'ion'
            else:
                inps.name = 'unw'
        elif inps.ifgfile:
            inps.src=3
        print('Reading ifgram pairs: {}'.format(source[inps.src-1]))
        return inps


def dates_with_Npairs(dateList, numPair, N):
    dates = list(np.array(dateList)[numPair==N])
    nears = nearest_dates(dateList, dates, 3)
    return dates


def nearest_dates(dateList, dates, n):
    idxs = [i for i, x in enumerate(dateList) if x in dates]
    if len(idxs) > 0:
        for idx in idxs:
            print('Dates:', np.array(dateList[idx]))
            nears = dateList[idx-n:idx] + dateList[idx+1:idx+n+1]
            nearsStr = ' '.join(nears)
            print('Nearest-{}: {}'.format(n, nearsStr))
    else:
        nears = None
        print('no such thing, all good!')
    return nears


def check_rank(A):
    rk = np.linalg.matrix_rank(A)
    if rk < A.shape[1]-1:
        print('\nRank deficient! The network is disconnected!')
        print('Num of cols of design matrix A (numDate-1) = ', A.shape[1]-1)
        print('Rank = ', rk, '\n')
    else:
        print('The network is good. The design matrix has full rank.')


def find_npair(A):
    """
    Find the gap from a design matrix A
    Input:
        A:          Design matrix (num_ifgs, num_dates)
    Return:
        num_pair:   num of pairs for each column (date)
    """
    num_pair = []
    for j in range(A.shape[1]):
        num_r = sum(A[:,j]==1)
        num_s = sum(A[:,j]==-1)
        num_pair.append(num_r + num_s)
    num_pair = np.array(num_pair)
    return num_pair


def find_networks(A, date_list, date12_list, s1_dict):
    pairs = []
    for row in A:
        date12_idx = list(np.nonzero(row)[0])
        pairs.append(date12_idx)

    nets = sarmath.networking(pairs)
    print('\nNumber of networks found: {}\n'.format(len(nets)))

    date_groups = []
    date12_groups = []
    for i, date_id in enumerate(nets):
        dates = list(np.array(date_list)[date_id])
        print("Connected network no.{} ({} dates):".format(i+1, len(date_id)))
        print('Acquisition\tIW1 starting range (m)')
        for date in dates:
            print('{}\t{}'.format(date, s1_dict[date][2]))
        date_groups.append(dates)
        date12_groups.append([])
        for date12 in date12_list:
            if bool(set(date12.split('_')) & set(dates)):
                date12_groups[i].append(date12)
    return nets, date_groups, date12_groups


def read_s1_version(file):
    ddict = dict()
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('S1'):
                zipf, no, ver, iw1, iw2, iw3 = line.split()
                date = zipf.split('T')[0].split('_')[-1]
                if date not in ddict:
                    ddict[date] = [int(no), ver, float(iw1), float(iw2), float(iw3)]
                else:
                    if int(no) != ddict[date][0]:
                        print('{}, the no. is not consistent!'.format(date))
                    if ver != ddict[date][1]:
                        print('{}, the version is not consistent!'.format(date))
                    if float(iw1) != ddict[date][2]:
                        print('{}, the starting range of IW1 is not consistent!'.format(date))
                    if float(iw2) != ddict[date][3]:
                        print('{}, the starting range of IW2 is not consistent!'.format(date))
                    if float(iw3) != ddict[date][4]:
                        print('{}, the starting range of IW3 is not consistent!'.format(date))
    return ddict


def readfromDir(dir_list, sep):
    date12_list = []
    mDates      = []
    sDates      = []
    for dir in dir_list:
        date12 = dir.split('/')[-1]
        date12 = '_'.join(date12.split(sep))
        mdate = date12.split('_')[0]
        sdate = date12.split('_')[1]
        date12_list.append(date12)
        mDates.append(mdate)
        sDates.append(sdate)
    date_list = sorted(list(set(mDates + sDates)))
    A = ifgramStack.get_design_matrix4timeseries(date12_list, refDate='no')[0]
    return A, date12_list ,date_list


def readListFile(listfile, prestr, sep):
    date12_list = []
    mDates      = []
    sDates      = []
    with open(listfile) as f:
        lines = f.readlines()
        for line in lines:
            date12 = line.split(prestr)[-1].split('\n')[0]
            date12 = '_'.join(date12.split(sep))
            mdate = date12.split('_')[0]
            sdate = date12.split('_')[1]
            date12_list.append(date12)
            mDates.append(mdate)
            sDates.append(sdate)
        date_list = sorted(list(set(mDates + sDates)))
    A = ifgramStack.get_design_matrix4timeseries(date12_list, refDate='no')[0]
    return A, date12_list ,date_list


def readStackHDF5(ifgfile):
    obj         = ifgramStack(ifgfile)
    date12_list = obj.get_date12_list(dropIfgram=False)
    date_list   = obj.get_date_list(dropIfgram=False)
    # get the design matrix of the network
    A = obj.get_design_matrix4timeseries(date12_list, refDate='no')[0]    # , refDate='no'
    return A, date12_list, date_list



######################################################################

def main(inps, show=False):

    ## Get the version and starting ranges
    s1_dict = read_s1_version(inps.s1_version)


    ## Get the date12 and date lists:
    if inps.src == 1:
        sep = '_'
        print('Reading design matrix from: {}'.format(inps.dir))
        dir_list = glob.glob(inps.dir, '*{}*'.format(sep))
        A, date12_list ,date_list = readListFile(dir_list, sep)
    elif inps.src == 2:
        if inps.name == 'unw':
            prestr = 'config_igram_unw_'
            sep = '_'
        elif inps.name == 'ion':
            prestr = 'config_unwrap_ion_'
            sep = '-'
        print('Reading design matrix from: {}'.format(inps.listfile))
        A, date12_list ,date_list = readListFile(inps.listfile, prestr, sep)
    elif inps.src == 3:
        print('Reading design matrix from: {}'.format(inps.ifgfile))
        A, date12_list, date_list = readStackHDF5(inps.ifgfile)
    print('')
    print('number of pairs:', len(date12_list))
    print('number of dates:', len(date_list))


    ## check rank deficiency. It will be rank deficient if num of networks > 1
    check_rank(A)

    ## get the number of pairs for each acquisition; find the probable gap(s)
    npairs = find_npair(A)

    ## Find and group the network(s). The best case is to have only one fully connected network!
    nets, date_groups, date12_groups = find_networks(A, date_list, date12_list, s1_dict)


    ## Plot the num of pairs
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(figsize=[7,3.5])
    ax = plt.subplot(1,1,1)
    for i, net in enumerate(nets):
        x_range = net
        ax.vlines(x=x_range, ymin=0, ymax=npairs[net], alpha=0.6, linewidth=1, color=colors[i])
        ax.plot(x_range, npairs[net], "o", markersize=8, alpha=0.6, label='network_{}'.format(i+1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0,None])
    ax.set_xlabel('Dates of SLC')
    ax.set_ylabel('Num of pairs')
    plt.legend(loc='lower right')
    plt.savefig('numPairs_{}.pdf'.format(inps.name), bbox_inches='tight')
    if show:
        plt.show()


    ## Plot the network
    plt.figure(figsize=[10,4.5])
    ax = plt.subplot(1, 1, 1)
    srange_list_all = []
    for i, net in enumerate(nets):
        # the y-axis is starting range (although the variable in `plot_network()` corresponds to `pbase`)
        np.random.seed(10)  # fix a random seed
        spread=inps.spread	# random spread for starting range in meter (just for visualization purpose)
        srange_list = []
        for dd in date_groups[i]:
            range0_iw1 = s1_dict[dd][2]
            srange_list.append(range0_iw1 + spread*2*(np.random.rand()-0.5))
        srange_list_all.extend(srange_list)
        p_dict={'fontsize'      :   16,
                'linewidth'     :   2,
                'linewidths'    :   1.2,
                'markersize'    :   10,
                'transparency'  :   0.4,
                'markercolor'   :   npairs[net], # np.delete(npairs, gaps); TO-DO
                'edgecolor'     :   colors[i],
                'linecolor'     :   colors[i],
                'cbar_label'    :   '',
                'colormap'      :   'summer_r',
                'vlim'          :   [1,max(npairs)],
                'ylabel'        :   'IW1 starting range (m)\n(±{}m random spread for visual.)'.format(spread),
                'disp_legend'   :  False}

        # TO-DO: how can the date12List_drop be useful??
        date12List_drop=[]   # 20150807_20160215

        # plot it
        ax, cbar = sarplt.plot_network(ax, date12_groups[i], date_groups[i], srange_list, p_dict, date12List_drop)

    cbar.set_ticks(np.arange(1,max(npairs)))
    cbar.ax.set_yticklabels(np.arange(1,max(npairs)).astype(str))
    cbar.set_label('Number of pairs', fontsize=p_dict['fontsize'], rotation=270, labelpad=35)
    #ax.axes.yaxis.set_visible(False)
    dates, datevector = ptime.date_list2vector(date_list)

    ax = pp.auto_adjust_xaxis_date(ax, datevector, fontsize=p_dict['fontsize'], every_year=p_dict['every_year'])[0]
    ax = pp.auto_adjust_yaxis(ax, srange_list_all, fontsize=p_dict['fontsize'])
    ax.set_title('Show the network gap(s)')
    plt.savefig('networkCheck_{}.pdf'.format(inps.name), bbox_inches='tight')
    if show:
        plt.show()


    ## Print other acquisitions with low number of pairs
    print(' ======= Acquisitions with only 1 pairs =======')
    tmp_dates = dates_with_Npairs(date_list, npairs, 1)

    print(' ======= Acquisitions with only 2 pairs =======')
    tmp_dates = dates_with_Npairs(date_list, npairs, 2)

    print(' ======= Acquisitions with only 3 pairs =======')
    tmp_dates = dates_with_Npairs(date_list, npairs, 3)



######################################################################
######################################################################

if __name__ == '__main__':

    inps = cmdLineParse()

    main(inps)
