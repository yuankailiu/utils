#!/usr/bin/env python3

import glob
import numpy as np
import matplotlib.pyplot as plt
from mintpy.objects import ifgramStack

import sarut.tools.plot as sarplt

# fonts
from matplotlib import font_manager
font_dirs = ['/net/kraken/bak/ykliu/fonts']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

# set font
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams.update({'font.size': 16})


def dates_with_Npairs(dateList, numPair, N):
    dates = list(np.array(dateList)[numPair==N])
    nears = nearest_dates(dateList, dates, 3)
    return dates


def nearest_dates(dateList, dates, n):
    idxs = [i for i, x in enumerate(dateList) if x in dates]
    for idx in idxs:
        print('Dates:', np.array(dateList[idx]))
        nears = dateList[idx-n:idx] + dateList[idx+1:idx+n+1]
        nearsStr = ' '.join(nears)
        print('Nearest-{}: {}'.format(n, nearsStr))
    return nears


def find_gap(A):
    """
    Find the gap from a design matrix A
    Input:
        A:          Design matrix (num_ifgs, num_dates)
    Return:
        num_pair:   num of pairs for each column (date)
        gap_j:      the index of gap(s) in the network
    """
    num_pair = []
    gap_j    = []
    for j in range(A.shape[1]):
        num_r = sum(A[:,j]==1)
        num_s = sum(A[:,j]==-1)
        num_pair.append(num_r + num_s)
        if num_pair[j] < 2:
            gap_j.append(j)
            nears = nearest_dates(date_list, date_list[j], 3)
            print('gap at {} (col={}), as_ref={}, as_sec={}, current pairs as below'.format(date_list[j], j, sum(A[:,j]==1), sum(A[:,j]==-1)))
            for pair in date12_list:
                if date_list[j] in pair:
                    print(pair)
            print('')
    num_pair = np.array(num_pair)
    return num_pair, gap_j


## First get the date12 and date lists:

# get lists from ifgramStack.h5
ifgfile = '/marmot-nobak/ykliu/aqaba/topsStack/a087/mintpy_ion/inputs/ifgramStack.h5'
#ifgfile = '/marmot-nobak/ykliu/aqaba/topsStack/a087/mintpy/inputs/ifgramStack.h5'

# get from directory names


obj = ifgramStack(ifgfile)
date12_list = obj.get_date12_list(dropIfgram=False)
date_list   = obj.get_date_list(dropIfgram=False)

# get the design matrix of the network
A = obj.get_design_matrix4timeseries(date12_list, refDate='no')[0]    # , refDate='no'

# check rank deficiency
rk = np.linalg.matrix_rank(A)
if rk < A.shape[1]-1:
    print('Rank deficient! The network is disconnected!')
    print('Num of cols of design matrix A (numDate-1) = ', A.shape[1]-1)
    print('Rank = ', rk, '\n')

else:
    print('The network is good.')

# find the gap
npairs, gaps = find_gap(A)


## Plot the num of pairs
x_range = np.arange(len(date_list))
plt.figure(figsize=[7,3.5])
ax = plt.subplot(1,1,1)
ax.vlines(x=x_range, ymin=0, ymax=npairs, color='gray', alpha=0.2, linewidth=1)
ax.plot(x_range, npairs, "o", markersize=8, color='gray', alpha=0.6)
ax.vlines(x=x_range[gaps], ymin=0, ymax=npairs[gaps], color='r', alpha=0.2, linewidth=2)
ax.plot(x_range[gaps], npairs[gaps], "o", markersize=8, color='r', alpha=0.6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Dates of SLC')
ax.set_ylabel('Num of pairs')
plt.show()


## Plot the network
pbase_list = np.random.rand(len(date_list))
p_dict={'fontsize'      :   16,
        'linewidth'     :   0.6,
        'markersize'    :   10,
        'markercolor'   :   np.delete(npairs, gaps),
        'cbar_label'    :   'Num of pairs',
        'colormap'      :   'summer_r',
        'vlim'          :   [2,max(npairs)],
        'ylabel'        :   '',
        'disp_legend'   :  False}
date12List_drop=['20150807_20160215']

plt.figure(figsize=[7,3.5])
ax = plt.subplot(1, 1, 1)
ax, cbar = sarplt.plot_network(ax, date12_list, date_list, pbase_list, p_dict, date12List_drop)
cbar.set_ticks([2,3,4])
ax.axes.yaxis.set_visible(False)
ax.set_title('Show the network gap(s)')
plt.show()

print(' ======= Acquisitions with only 2 pairs =======')
_ = dates_with_Npairs(date_list, npairs, 2)
print(' ======= Acquisitions with only 3 pairs =======')
_ = dates_with_Npairs(date_list, npairs, 3)
# %%
