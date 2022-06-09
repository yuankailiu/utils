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

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mintpy.objects.colors import ColormapExt

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
    ## Print other acquisitions with low number of pairs (Not used, this is not helpful)
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
        print('\nRank deficient. The network is disconnected.')
        print('Rank = {}; Num of coloumns (numDate-1) = {}'.format(rk, A.shape[1]-1))
    else:
        print('Full rank. The network is fully connected')
    lack_ref = []
    lack_sec = []
    for j in range(A.shape[1]):
        if (-1 not in A[:,j]) and (j != A.shape[1]-1):
            lack_ref.append(j)
        if ( 1 not in A[:,j]) and (j != 0):
            lack_sec.append(j)
    return lack_ref, lack_sec

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


def networking(friends, sortbylen=True):
    # person's friendship circle (network) is a person themselves
    # plus friendship circles of all their direct friends
    # minus already seen people
    # https://stackoverflow.com/questions/15331877/creating-dictionaries-of-friends-that-know-other-friends-in-python
    def friends_graph(people_all, friends):
        # graph of friends (adjacency lists representation)
        G = {p: [] for p in people_all} # person -> direct friends list
        for friend in friends:
            for p in friend:
                f_list = list(friend)
                f_list.remove(p)
                G[p].extend(f_list)
        return G

    def friendship_circle(person): # a.k.a. connected component
        seen.add(person)
        yield person
        for friend in direct_friends[person]:
            if friend not in seen:
                yield from friendship_circle(friend)

    people_all = list(set(sum(friends, [])))
    direct_friends = friends_graph(people_all, friends)
    seen = set() # already seen people

    # group people into friendship circles
    circs = (friendship_circle(p) for p in people_all if p not in seen)

    # convert generator to a list with sublist(s)
    circ_lists = []
    for circ in circs:
        member = sorted(list(circ))
        circ_lists.append(member)

    if sortbylen:
        circ_lists.sort(key=len, reverse=True)

    return circ_lists


def find_networks(A, date_list, date12_list, s1_dict=None):
    pairs = []
    for row in A:
        date12_idx = list(np.nonzero(row)[0])
        pairs.append(date12_idx)

    nets = networking(pairs)
    print('\nNumber of networks found: {}\n'.format(len(nets)))

    date_groups = []
    date12_groups = []
    for i, date_id in enumerate(nets):
        dates = list(np.array(date_list)[date_id])
        print("Connected network no.{} ({} dates):".format(i+1, len(date_id)))
        if s1_dict:
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


def print_gaps(date_list, lack_ref, lack_sec):
    if len(lack_ref)>0:
        print('Dates not as reference: {}'.format(np.array(date_list)[lack_ref]))
    if len(lack_sec)>0:
        print('Dates not as secondary: {}'.format(np.array(date_list)[lack_sec]))
    return


def plot_num_pairs(nets, npairs, name, show=False):
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
    plt.savefig('numPairs_{}.pdf'.format(name), bbox_inches='tight')
    if show:
        plt.show()


## modified from mintpy plot_network function
## now can plot network and mark the gaps. see `find_network_gap.py`
## to-do: combine plotting funcs down below to this func (2022-3-3 ykl)
def plot_network(ax, date12List, dateList, pbaseList, p_dict={}, date12List_drop=[], print_msg=True):
    """Plot Temporal-Perp baseline Network
    Inputs
        ax : matplotlib axes object
        date12List : list of string for date12 in YYYYMMDD_YYYYMMDD format
        dateList   : list of string, for date in YYYYMMDD format
        pbaseList  : list of float, perp baseline, len=number of acquisition
        p_dict   : dictionary with the following items:
                      fontsize
                      linewidth
                      markercolor
                      markersize

                      cohList : list of float, coherence value of each interferogram, len = number of ifgrams
                      colormap : string, colormap name
                      disp_title : bool, show figure title or not, default: True
                      disp_drop: bool, show dropped interferograms or not, default: True
    Output
        ax : matplotlib axes object
    """

    # Figure Setting
    if 'fontsize'    not in p_dict.keys():  p_dict['fontsize']    = 12
    if 'linewidth'   not in p_dict.keys():  p_dict['linewidth']   = 2
    if 'linewidths'  not in p_dict.keys():  p_dict['linewidths']  = 2
    if 'linecolor'   not in p_dict.keys():  p_dict['linecolor']   = False
    if 'markercolor' not in p_dict.keys():  p_dict['markercolor'] = 'orange'
    if 'markersize'  not in p_dict.keys():  p_dict['markersize']  = 12
    if 'edgecolor'   not in p_dict.keys():  p_dict['edgecolor']   = 'k'
    if 'transparency' not in p_dict.keys():  p_dict['transparency']   = 0.7

    # For colorful display of coherence
    if 'cohList'     not in p_dict.keys():  p_dict['cohList']     = None
    if 'xlabel'      not in p_dict.keys():  p_dict['xlabel']      = None #'Time [years]'
    if 'ylabel'      not in p_dict.keys():  p_dict['ylabel']      = 'Perp Baseline [m]'
    if 'cbar_label'  not in p_dict.keys():  p_dict['cbar_label']  = 'Average Spatial Coherence'
    if 'cbar_size'   not in p_dict.keys():  p_dict['cbar_size']   = '3%'
    if 'disp_cbar'   not in p_dict.keys():  p_dict['disp_cbar']   = True
    if 'colormap'    not in p_dict.keys():  p_dict['colormap']    = 'RdBu'
    if 'vlim'        not in p_dict.keys():  p_dict['vlim']        = [0.2, 1.0]
    if 'disp_title'  not in p_dict.keys():  p_dict['disp_title']  = True
    if 'disp_drop'   not in p_dict.keys():  p_dict['disp_drop']   = True
    if 'disp_legend' not in p_dict.keys():  p_dict['disp_legend'] = True
    if 'every_year'  not in p_dict.keys():  p_dict['every_year']  = 1
    if 'number'      not in p_dict.keys():  p_dict['number']      = None

    # support input colormap: string for colormap name, or colormap object directly
    if isinstance(p_dict['colormap'], str):
        cmap = ColormapExt(p_dict['colormap']).colormap
    elif isinstance(p_dict['colormap'], mpl.colors.LinearSegmentedColormap):
        cmap = p_dict['colormap']
    else:
        raise ValueError('unrecognized colormap input: {}'.format(p_dict['colormap']))

    cohList = p_dict['cohList']
    transparency = p_dict['transparency']

    # Date Convert
    dateList = ptime.yyyymmdd(sorted(dateList))
    dates, datevector = ptime.date_list2vector(dateList)
    tbaseList = ptime.date_list2tbase(dateList)[0]

    ## maxBperp and maxBtemp
    date12List = ptime.yyyymmdd_date12(date12List)
    ifgram_num = len(date12List)
    pbase12 = np.zeros(ifgram_num)
    tbase12 = np.zeros(ifgram_num)
    for i in range(ifgram_num):
        m_date, s_date = date12List[i].split('_')
        m_idx = dateList.index(m_date)
        s_idx = dateList.index(s_date)
        pbase12[i] = pbaseList[s_idx] - pbaseList[m_idx]
        tbase12[i] = tbaseList[s_idx] - tbaseList[m_idx]
    if print_msg:
        print('max perpendicular baseline: {:.2f} m'.format(np.max(np.abs(pbase12))))
        print('max temporal      baseline: {} days'.format(np.max(tbase12)))

    ## Keep/Drop - date12
    date12List_keep = sorted(list(set(date12List) - set(date12List_drop)))
    if not date12List_drop:
        p_dict['disp_drop'] = False

    ## Keep/Drop - date
    m_dates = [i.split('_')[0] for i in date12List_keep]
    s_dates = [i.split('_')[1] for i in date12List_keep]
    dateList_keep = ptime.yyyymmdd(sorted(list(set(m_dates + s_dates))))
    dateList_drop = sorted(list(set(dateList) - set(dateList_keep)))
    idx_date_keep = [dateList.index(i) for i in dateList_keep]
    idx_date_drop = [dateList.index(i) for i in dateList_drop]

    # Ploting
    disp_min = p_dict['vlim'][0]
    disp_max = p_dict['vlim'][1]
    if cohList is not None:
        data_min = min(cohList)
        data_max = max(cohList)
        if print_msg:
            print('showing coherence')
            print('data range: {}'.format([data_min, data_max]))
            print('display range: {}'.format(p_dict['vlim']))

        # plot low coherent ifgram first and high coherence ifgram later
        cohList_keep = [cohList[date12List.index(i)] for i in date12List_keep]
        date12List_keep = [x for _, x in sorted(zip(cohList_keep, date12List_keep))]

    if p_dict['disp_cbar']:
        cax = make_axes_locatable(ax).append_axes("right", p_dict['cbar_size'], pad=p_dict['cbar_size'])
        norm = mpl.colors.Normalize(vmin=disp_min, vmax=disp_max)
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cbar.ax.tick_params(labelsize=p_dict['fontsize'])
        cbar.set_label(p_dict['cbar_label'], fontsize=p_dict['fontsize'], rotation=270, labelpad=25)

    # Dot - SAR Acquisition
    if idx_date_keep:
        x_list = [dates[i] for i in idx_date_keep]
        y_list = [pbaseList[i] for i in idx_date_keep]
        if isinstance(p_dict['markercolor'], str):
            ax.plot(x_list, y_list, 'ko', alpha=0.7, ms=p_dict['markersize'], mfc=p_dict['markercolor'])
        else:
            ax.scatter(x_list, y_list, s=p_dict['markersize']**2, marker='o', c=p_dict['markercolor'], cmap=p_dict['colormap'],
                        alpha=0.7, edgecolors=p_dict['edgecolor'], linewidths=p_dict['linewidths'])
    if idx_date_drop:
        x_list = [dates[i] for i in idx_date_drop]
        y_list = [pbaseList[i] for i in idx_date_drop]
        ax.plot(x_list, y_list, 'ko', alpha=0.7, ms=p_dict['markersize'], mfc='r')

    ## Line - Pair/Interferogram
    # interferograms dropped
    if p_dict['disp_drop']:
        for date12 in date12List_drop:
            date1, date2 = date12.split('_')
            idx1 = dateList.index(date1)
            idx2 = dateList.index(date2)
            x = np.array([dates[idx1], dates[idx2]])
            y = np.array([pbaseList[idx1], pbaseList[idx2]])
            if cohList is not None:
                val = cohList[date12List.index(date12)]
                val_norm = (val - disp_min) / (disp_max - disp_min)
                ax.plot(x, y, '--', lw=p_dict['linewidth'], alpha=transparency, c=cmap(val_norm))
            else:
                ax.plot(x, y, '--', lw=p_dict['linewidth'], alpha=transparency, c='r')

    # interferograms kept
    for date12 in date12List_keep:
        date1, date2 = date12.split('_')
        idx1 = dateList.index(date1)
        idx2 = dateList.index(date2)
        x = np.array([dates[idx1], dates[idx2]])
        y = np.array([pbaseList[idx1], pbaseList[idx2]])
        if cohList is not None:
            val = cohList[date12List.index(date12)]
            val_norm = (val - disp_min) / (disp_max - disp_min)
            ax.plot(x, y, '-', lw=p_dict['linewidth'], alpha=transparency, c=cmap(val_norm))
        elif p_dict['linecolor'] is not False:
            ax.plot(x, y, '-', lw=p_dict['linewidth'], alpha=transparency, c=p_dict['linecolor'])
        else:
            ax.plot(x, y, '-', lw=p_dict['linewidth'], alpha=transparency, c='k')

    if p_dict['disp_title']:
        ax.set_title('Interferogram Network', fontsize=p_dict['fontsize'])

    # axis format
    ax = pp.auto_adjust_xaxis_date(ax, datevector, fontsize=p_dict['fontsize'],
                                every_year=p_dict['every_year'])[0]
    ax = pp.auto_adjust_yaxis(ax, pbaseList, fontsize=p_dict['fontsize'])
    ax.set_xlabel(p_dict['xlabel'], fontsize=p_dict['fontsize'])
    ax.set_ylabel(p_dict['ylabel'], fontsize=p_dict['fontsize'])
    ax.tick_params(which='both', direction='in', labelsize=p_dict['fontsize'],
                   bottom=True, top=True, left=True, right=True)

    if p_dict['number'] is not None:
        ax.annotate(p_dict['number'], xy=(0.03, 0.92), color='k',
                    xycoords='axes fraction', fontsize=p_dict['fontsize'])

    # Legend
    if p_dict['disp_drop'] and p_dict['disp_legend']:
        solid_line = mpl.lines.Line2D([], [], color='k', ls='solid',  label='Ifgram used')
        dash_line  = mpl.lines.Line2D([], [], color='k', ls='dashed', label='Ifgram dropped')
        ax.legend(handles=[solid_line, dash_line])

    return ax, cbar


def call_plot_networks(nets, npairs, date_list, date_groups, date12_groups, s1_dict, spread, name, show=False):
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(figsize=[10,4.5])
    ax = plt.subplot(1, 1, 1)
    srange_list_all = []
    for i, net in enumerate(nets):
        # the y-axis is starting range (although the variable in `plot_network()` corresponds to `pbase`)
        np.random.seed(10)  # fix a random seed
        spread=float(spread)	# random spread for starting range in meter (just for visualization purpose)
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
        ax, cbar = plot_network(ax, date12_groups[i], date_groups[i], srange_list, p_dict, date12List_drop)

    cbar.set_ticks(np.arange(1,max(npairs)))
    cbar.ax.set_yticklabels(np.arange(1,max(npairs)).astype(str))
    cbar.set_label('Number of pairs', fontsize=p_dict['fontsize'], rotation=270, labelpad=35)
    #ax.axes.yaxis.set_visible(False)
    dates, datevector = ptime.date_list2vector(date_list)

    ax = pp.auto_adjust_xaxis_date(ax, datevector, fontsize=p_dict['fontsize'], every_year=p_dict['every_year'])[0]
    ax = pp.auto_adjust_yaxis(ax, srange_list_all, fontsize=p_dict['fontsize'])
    ax.set_title('Show the network gap(s)')
    plt.savefig('networkCheck_{}.pdf'.format(name), bbox_inches='tight')
    if show:
        plt.show()


######################################################################

def main(inps):

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
    lack_ref, lack_sec = check_rank(A)
    print_gaps(date_list, lack_ref, lack_sec)


    ## get the number of pairs for each acquisition; find the probable gap(s)
    npairs = find_npair(A)

    ## Find and group the network(s). The best case is to have only one fully connected network!
    nets, date_groups, date12_groups = find_networks(A, date_list, date12_list, s1_dict=None)


    ## Plot number of pairs for each date
    plot_num_pairs(nets, npairs, inps.name, show=False)


    ## Plot the network
    call_plot_networks(nets, npairs, date_list, date_groups, date12_groups, s1_dict, inps.spread, inps.name, show=False)


######################################################################
######################################################################

if __name__ == '__main__':

    inps = cmdLineParse()

    main(inps)
