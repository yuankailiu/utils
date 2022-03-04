#!/usr/bin/env python3
# --------------------------------------------------------------
# Plotting ts residual RMS
#
# Yuan-Kai Liu, 2021-09-23
# --------------------------------------------------------------


import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
from mintpy.utils import utils as ut, plot as pp


def cmdLineParse():
    description = ' Read and plot residual RMS of velocity fitting '
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', dest='dir', type=str, default='.',
            help = 'figure output directory. (default: %(default)s)')
    parser.add_argument('-i', dest='infile', type=str, required=True,
            help = 'input RMS txt file')
    parser.add_argument('-o', dest='outfile', type=str, default='rms_ts2velo.png',
            help = 'output figure file name. (default: %(default)s)')
    parser.add_argument('-c', '--cutoff', dest='cutoff', type=float, nargs=1, default=3.0,
            help = 'cutoff factor of MAD. (default: %(default)s)')
    parser.add_argument('-w', '--window', dest='window', type=float, nargs=1, default=90.0,
            help = 'Running window legnth. (default: %(default)s days)')
    parser.add_argument('--wt', '--window_type', dest='window_type', type=str, default='mean',
            help = 'Running window type, ["mean", "median"]. (default: %(default)s)')
    parser.add_argument('-t', '--fig_title', '--title', dest='fig_title', type=str, default='Residual RMS',
            help = 'output figure title')
    parser.add_argument('-y', '--ylim', dest='ylim', type=float, nargs=2, default=[0, 65],
            help = 'plotting y-axis limit')
    parser.add_argument('-x', '--xlim', dest='xlim', type=str, nargs=2, default=['20140701', '20210301'],
            help = 'plotting x-axis limit (date_str YYYYMMDD)')
    parser.add_argument('--ylabel', dest='ylabel', type=str, default='Residual RMS [mm]',
            help = 'figure y-label')
    parser.add_argument('--hide', '--hideDates', dest='hideDates', type=str, default=None,
            help = 'Ignore certain dates when plotting RMS. (default: %(default)s)')

    if len(sys.argv) < 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        return parser.parse_args()


def read_snd_txt(txt_file):
    date_str = []
    date_num = []
    with open(txt_file) as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            else:
                tmp = line.split()
                string = '{:04d}{:02d}{:02d}'.format(int(tmp[0]),int(tmp[1]),int(tmp[2]))
                date_str.append(string)
                date_num.append(float(tmp[4]))
    date_num = np.array(date_num)
    return date_str, date_num


def read_rms_txt(txt_file):
    date_str = []
    date_rms = []
    with open(txt_file) as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            else:
                tmp = line.split()
                date_str.append(tmp[0])
                date_rms.append(float(tmp[1]))
    date_rms = np.array(date_rms)

    # meter to mm
    date_rms *= 1e3
    return date_str, date_rms


def read_hide_txt(txt_file):
    date_str = []
    with open(txt_file) as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            else:
                tmp = line.split()
                date_str.append(tmp[0])
    return date_str


def get_show_data(dates, rms, dates_hide):
    show_index = []
    for i in np.arange(len(dates)):
        if dates[i] in dates_hide:
            continue
        else:
            show_index.append(i)
    new_dates = list(np.array(dates)[show_index])
    new_rms   = rms[show_index]
    return new_dates, new_rms


def YYYYMMDD2datetime(YYYYMMDD):
    date = datetime.strptime(YYYYMMDD, '%Y%m%d')
    return date


def list2datetimes(YYYYMMDD_list):
    dates = []
    for YYYYMMDD in YYYYMMDD_list:
        dates.append(YYYYMMDD2datetime(YYYYMMDD))
    return dates


def plot_rmsdates(inps, date_str, date_rms, SLC_path=None, ylog=False, return_values=False, plotTEC=False, save_excl=False):

    filename = inps.outfile
    title    = inps.fig_title
    ylim     = inps.ylim
    xlim     = inps.xlim
    xlim     = list2datetimes(xlim)
    window   = inps.window
    window_type = inps.window_type

    dates = list2datetimes(date_str)

    try:
        threshold = ut.median_abs_deviation_threshold(date_rms, center=0., cutoff=inps.cutoff)
    except:
        # equivalent calculation using numpy assuming Gaussian distribution
        threshold = np.median(date_rms) / .6745 * inps.cutoff

    low_idx   = np.where(date_rms==np.min(date_rms))[0][0]
    high_bool = date_rms >= threshold
    high_idx  = np.where(high_bool)[0]

    date_rms_i = date_rms[~high_bool]

    try:
        threshold_i = ut.median_abs_deviation_threshold(date_rms_i, center=0., cutoff=inps.cutoff)
    except:
        # equivalent calculation using numpy assuming Gaussian distribution
        threshold_i = np.median(date_rms_i) / .6745 * inps.cutoff

    if plotTEC or 'TEC' in title:
        title    = 'TEC delay history'
        ystr     = 'Estimated TEC delay [mm]'
    else:
        ystr     = inps.ylabel

    if window is not None:
        rmsAvg = []
        rmsMed = []
        for i in np.arange(len(dates)):
            t1 = dates[i] - timedelta(days=window/2)
            t2 = dates[i] + timedelta(days=window/2)
            for j in np.arange(0,i+1):
                if dates[j]>t1:
                    t1_id = int(j)
                    break
            for j in np.arange(i,len(dates)):
                if dates[j]>t2:
                    t2_id = int(j-1)
                    break
                else:
                    t2_id = int(j)
            # print(dates[t1_id], dates[i], dates[t2_id])
            rmsAvg.append(np.mean(date_rms[t1_id:t2_id+1]))
            rmsMed.append(np.median(date_rms[t1_id:t2_id+1]))
            if window_type.lower().startswith(('avg','mean')):
                rms_win = rmsAvg
            elif window_type.lower().startswith(('med')):
                rms_win = rmsMed


    # save exclude dates
    if save_excl:
        if plotTEC is False:
            exclfile = open('my_RMSexclude_date.txt', 'w')
        elif plotTEC is True:
            exclfile = open('my_TECexclude_date.txt', 'w')

        for i in range(len(high_idx)+1):
            if i == len(high_idx):
                for j in range(len(high_idx)):
                    exclfile.write(f'{date_str[high_idx[j]]},')
            else:
                exclfile.write(f'{date_str[high_idx[i]]}\n')
        exclfile.close()


    label_thresh   = 'Threshold (MAD * {:.1f}) = {:.1f}'.format(inps.cutoff, threshold)
    label_thresh_i = 'New MAD * {:.1f} = {:.1f}'.format(inps.cutoff, threshold_i)
    label_percent1 = '95th percentile'
    label_percent2 = '99.7th percentile'
    label_lowRMS   = 'Reference date'
    label_highRMS  = 'Noisy dates ({:d}/{:d})'.format(len(high_idx), len(date_str))
    label_SA       = 'Sen1-A'
    label_SB       = 'Sen1-B'
    label_avg      = '3-month running {}'.format(window_type)


    fig, ax = plt.subplots(figsize=[14,6])
    bar_width = np.min(np.diff(dates).tolist())*3/4
    if plotTEC is False:
        if SLC_path is None:
            ax.plot(dates, date_rms)
            #ax.bar(dates, date_rms, width=bar_width.days)
            #lowbar = ax.bar(dates[low_idx], date_rms[low_idx], width=bar_width.days, color='orange')
            #line = ax.axhline(y=threshold              , linestyle='--', color='k'   , lw=3, label=label_thresh)
            #line2 = ax.axhline(y=threshold_i              , linestyle='--', color='lightgrey'   , lw=3, label=label_thresh)
            #line3 = ax.axhline(y=np.percentile(date_rms, 95), linestyle='--', color='lightgrey', lw=2, label=label_percent1)
            #line4 = ax.axhline(y=np.percentile(date_rms, 99.7), linestyle='-.', color='lightgrey', lw=2, label=label_percent2)
            print('High rms dates: ',len(high_idx))
            for i in range(len(high_idx)):
                highbar = ax.bar(dates[high_idx[i]], date_rms[high_idx[i]], width=bar_width.days, color='lightgrey')
            if window is not None:
                winline, = ax.plot(dates, rms_win, c='r', lw=2)
                try:
                    handles = [line, line2, line3, line4, lowbar, highbar, winline]
                    labels  = [label_thresh, label_thresh_i, label_percent1, label_percent2, label_lowRMS, label_highRMS, label_avg]
                except:
                    try:
                        handles = [line, line2, highbar, winline]
                        labels  = [label_thresh, label_thresh_i, label_highRMS, label_avg]
                    except:
                        try:
                            handles = [line, line2, winline]
                            labels  = [label_thresh, label_thresh_i, label_avg]
                        except:
                            handles = [winline]
                            labels  = [label_avg]
            else:
                try:
                    handles = [line, line2, line3, line4, lowbar, highbar]
                    labels  = [label_thresh, label_thresh_i, label_percent1, label_percent2, label_lowRMS, label_highRMS]
                except:
                    try:
                        handles = [line, line2, highbar, lowbar]
                        labels  = [label_thresh, label_thresh_i, label_highRMS, label_lowRMS]
                    except:
                        try:
                            handles = [line, line2, lowbar]
                            labels  = [label_thresh, label_thresh_i, label_lowRMS]
                        except:
                            handles = [lowbar]
                            labels  = [label_lowRMS]
            ax.legend(handles, labels, frameon=True, loc='upper right')
            if window is not None:
                ax.plot(dates, rms_win, c='r', lw=2)
            #ax.plot(dates, w, c='k', lw=2)
        elif SLC_path is not None:
            filename += '_platform'
            colorA = 'dodgerblue'
            colorB = 'tomato'
            bcolor, _= MI_info(SLC_path, colorA, colorB)
            ax.bar(dates, date_rms, width=bar_width.days, color=bcolor)
            line = ax.axhline(y=threshold              , linestyle='--', color='k'   , lw=3, label=label_thresh)
            line2 = ax.axhline(y=threshold_i              , linestyle='--', color='lightgrey'   , lw=3, label=label_thresh)
            #line3 = ax.axhline(y=np.percentile(date_rms, 95), linestyle='--', color='lightgrey', lw=2, label=label_percent1)
            #line4 = ax.axhline(y=np.percentile(date_rms, 99.7), linestyle='-.', color='lightgrey', lw=2, label=label_percent2)
            if window is not None:
                winline, = ax.plot(dates, rms_win, c='r', lw=2)
                handles = [line]+[line2]+[winline]+[Rectangle((0,0),1,1,color=c) for c in [colorA, colorB]]
                labels= [label_thresh, label_thresh_i, label_avg, label_SA, label_SB]
            else:
                handles = [line]+[line2]+[Rectangle((0,0),1,1,color=c) for c in [colorA, colorB]]
                labels= [label_thresh, label_thresh_i, label_SA, label_SB]
            ax.legend(handles, labels, frameon=True, loc='upper right')

    elif plotTEC is True:
        ax.bar(dates, date_rms, width=bar_width.days, color='coral')
        #line = ax.axhline(y=threshold              , linestyle='--', color='k'   , lw=3, label=label_thresh)
        #for i in range(len(high_idx)):
            #highbar = ax.bar(dates[high_idx[i]], rms[high_idx[i]], width=bar_width.days, color='lightgrey')
        if window is not None:
            winline, = ax.plot(dates, rms_win, c='r', lw=2, label=label_avg)
            handles = [winline]
            labels  = [label_avg]
        else:
            handles = [line, highbar]
            labels  = [label_thresh, label_highRMS]
        ax.legend(handles, labels, frameon=True, loc='upper right')

    pp.auto_adjust_xaxis_date(ax, dates, fontsize=font_size)
    ax.set_xlabel('Time [year]')
    ax.set_ylabel(ystr)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    if ylog == True:
        ax.set_yscale('log')
    ax.set_title(title)
    plt.tight_layout()

    # output
    if not filename.endswith('.png'):
        filename += '.png'
    out_file = f'{pic_dir}/{filename}'
    plt.savefig(out_file, bbox_inches='tight', transparent=True, dpi=fig_dpi)
    print('save to file: '+out_file)

    if return_values == True:
        return dates, date_str



if __name__ == '__main__':

    inps        = cmdLineParse()

    # Read inputs / set plotting stuff
    pic_dir     = inps.dir
    txt_file    = inps.infile
    hide_file   = inps.hideDates
    fig_dpi     = 150
    font_size   = 16
    plt.rcParams.update({'font.size': font_size})
    print('Reading file: {}'.format(txt_file))

    # mkdir
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)

    # Read rms file
    #date_str, date_rms = read_rms_txt(txt_file)
    date_str, date_rms = read_snd_txt(txt_file)

    # Hide specific dates (exclude_dates)
    if hide_file is not None:
        date_str_hide      = read_hide_txt(hide_file)
        date_str, date_rms = get_show_data(date_str, date_rms, date_str_hide)

    # Plot
    print('Plotting the residual RMS')
    plot_rmsdates(inps, date_str, date_rms, SLC_path=None, ylog=False, return_values=False, plotTEC=False, save_excl=False)