#!/usr/bin/env python3

#########################################################
# 1. calling tsview.py to plot/save ts
# 2. then plot all the output files to a combined figure
#########################################################

import numpy as np
from datetime import datetime as dt
from matplotlib import pyplot as plt
from mintpy.tsview import timeseriesViewer
from mintpy.utils import plot as pp


def tsview(fname, yx=None, lalo=None, model=None, oname=None, v=False, nodisplay=True):
    """Visualize input file by calling tsview.py"""
    cmd = 'tsview.py {} --ms 6 --ylim -10 10 --multilook-num 10 --save'.format(fname)
    if yx is not None:
        cmd += ' --yx {} {}'.format(yx[0], yx[1])
    if lalo is not None:
        cmd += ' --lalo {} {}'.format(lalo[0], lalo[1])
    if model is not None:
        cmd += ' {}'.format(model)
    if oname is not None:
        cmd += ' --outfile {}'.format(oname)
    if nodisplay:
        cmd += ' --nodisplay'
    if v is False:
        cmd += ' --noverbose'
    obj = timeseriesViewer(cmd)
    obj.configure()
    obj.figsize_img = [5, 4]
    obj.figsize_pts = [5, 2]
    obj.plot()


def read_ptstxt(fname):
    """Read the pt locations, get coords and names"""
    yx   = []
    locs = []
    with open(fname) as f:
        for line in f:
            yx.append(np.array(line.split()[:2]).astype(int))
            locs.append(line.split()[2])
    yx = np.array(yx)
    return yx, locs


def init_data_dict(vars):
    """initialize the metric dict for saving purposes"""
    data = dict()
    for var in vars:
        data[var] = dict()
        data[var]['Date_obs'] = []
        data[var]['Disp_obs'] = []
        data[var]['Date_fit'] = []
        data[var]['Disp_fit'] = []
    return data


def read_tstxt(ts_txt):
    """ Read the ts data and header into dict
    # time-series file = timeseries_ERA5_demErr.h5
    # Y/X = 521, 462, lat/lon = 35.7663, -117.6154
    # reference pixel: y=1175, x=800
    # estimated time function parameters:
    #     velocity          : -30.63 +/-  12.42 cm/year
    #     exp20190705Tau30.0:  -7.91 +/-   6.85 cm
    #     log20190705Tau30.0:  15.95 +/-   6.36 cm
    # residual root-mean-square: 1.0931304693222046 cm
    # unit: cm
    """
    atr = dict()
    with open(ts_txt) as f:
        print('read ts file  {}'.format(ts_txt))
        c = 'head'
        for line in f:
            if line.startswith('# time-series file'):
                atr['ts_file'] = line.split(' = ')[1]
            elif line.startswith('# Y/X'):
                atr['y'] = int(line.split(',')[0].split(' = ')[1])
                atr['x'] = int(line.split(',')[1])
                atr['lat'] = float(line.split(',')[2].split(' = ')[1])
                atr['lon'] = float(line.split(',')[3])
            elif line.startswith('# reference'):
                atr['refy'] = int(line.split(',')[0].split('y=')[1])
                atr['refx'] = int(line.split(',')[1].split('x=')[1])
            elif line.startswith('# estimated time function'):
                c = 'head_model'
                atr['model'] = dict()
            elif line.startswith('# residual'):
                c = 'residual'
                atr['rmsr'] = float(line.split()[3])
            elif line.startswith('# unit'):
                atr['unit'] = line.split()[2]
                c = 'obs'
                atr['Date_obs'] = []
                atr['Disp_obs'] = []
            elif line.startswith('# fit'):
                c = 'fit'
                atr['Date_fit'] = []
                atr['Disp_fit'] = []

            if c == 'head_model' and not line.startswith('# estimated'):
                val_str = line.split('+/-')[0].split()
                std_str = line.split('+/-')[1].split()
                atr['model'][line.split()[1]] = float(val_str[-1])
                if len(std_str) == 1:
                    atr['model'][line.split()[1]+'_Std'] = np.nan
                else:
                    atr['model'][line.split()[1]+'_Std'] = float(std_str[0])

            if c == 'obs':
                if line[0] != '#':
                    atr['Date_obs'].append(dt.strptime(line.split()[0], '%Y%m%d'))
                    atr['Disp_obs'].append(float(line.split()[1]))

            if c == 'fit':
                if line[0] != '#':
                    atr['Date_fit'].append(dt.strptime(line.split()[0], '%Y%m%d'))
                    atr['Disp_fit'].append(float(line.split()[1]))

        atr['Date_obs'] = np.array(atr['Date_obs'])
        atr['Disp_obs'] = np.array(atr['Disp_obs'])
        atr['Date_fit'] = np.array(atr['Date_fit'])
        atr['Disp_fit'] = np.array(atr['Disp_fit'])
        return atr


def plot_timeseries(TS, demean=True, vlim=[None, None], rvlim=[None, None], ll='best', fig_title='custom time series', fig_dpi=300, fig_ratio=4):
    """ USAGE:
    Plot a pile of time series stored in a dictionary
    Input
        TS:  A dictionary of ts data from several locations
    Explanations of the required variables in the TS dict()
        TS[loc]      -    a dictionary of ts data from a location
        TS[loc][var] -    var can be the following:
                            'Date_obs'   dates list of the observation ts data
                            'Date_fit'   dates list of the modeled ts data
                            'Disp_obs'   displacement array of the observation ts data
                            'Disp_fit'   displacement array of the model ts data
                            'model'      model dict() of the fitted time function model:
                                > 'velocity'     secular velocity
                                > 'velocity_Std' standard deviation of the secular velocity
                            'rmsr'       residual root-mean-square between the model and the obs
                            'unit'       unit of the ts file
    """
    # setting
    font_size = 18
    font_size_lb = 20
    num_pts   = len(TS)
    ncols     = 2
    nrows     = num_pts//ncols + num_pts%ncols
    W         = 8            # width of each subplot
    H         = W/fig_ratio  # height of each subplot
    fig_space = {'hspace':0.1, 'wspace':0.02}
    fig_size  = [ncols*W,nrows*H]
    ha='center'
    va='center'

    # print info
    print('figure 1: plot the time-series obs/pred')
    print('figure 2: plot the time-series residues')
    if demean:
        print('Mean of the time-series data is removed')

    # innitialize fig1 for ts data; fig2 for residuals
    fig1, axs1 = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, gridspec_kw=fig_space, figsize=fig_size)
    fig2, axs2 = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, gridspec_kw=fig_space, figsize=fig_size)

    # loop over files
    for ax1, ax2, loc in list(zip(axs1.flatten(), axs2.flatten(), locs)):
        m     = TS[loc]
        x     = m['Date_obs']
        y     = m['Disp_obs']
        xf    = m['Date_fit']
        yf    = m['Disp_fit']
        v     = m['model']['velocity']
        v_std = m['model']['velocity_Std']
        rmsr  = m['rmsr']
        unit  = m['unit']

        # remove mean value when plotted
        if demean:
            y  -= np.nanmean(y)
            yf -= np.nanmean(yf)

        # plot ts data
        ax1.scatter(x, y, marker='o', s=40, facecolors='lightgrey', edgecolors='k')
        ax1.plot(xf, yf, '-', lw=2, c='tomato', label='{} ({:.2f}±{:.2f} {}/y, RMSR={:.2f} {})'.format(loc, v, v_std, unit, rmsr, unit))

        # plot the residues
        yf_sub = yf[np.in1d(xf, x).nonzero()[0]]
        r = y-yf_sub
        r -= np.nanmean(r)
        ax2.scatter(x, r, marker='s', s=40, facecolors='lightgrey', edgecolors='b', label='{} (RMSR={:.2f} {})'.format(loc, rmsr, unit))
        ax2.plot(x, r, '-', c='k')

        # axis format
        for ax, ylim in list(zip([ax1, ax2], [vlim, rvlim])):
            ax.set_ylim(ylim[0], ylim[1])
            #pp.auto_adjust_xaxis_date(ax, x, fontsize=font_size)
            ax.tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True)
            ax.legend(loc=ll, fontsize=font_size/1.5)
            ax.tick_params(axis='x', labelrotation = 45)

    # add axis labels, title
    xstr  = 'Time [year]'
    ystrs = ['LOS displacement [{}]'.format(unit), 'LOS residues [{}]'.format(unit)]
    fig_titles = [fig_title, fig_title+' (residues)']
    for fig, ystr, titl in list(zip([fig1,fig2], ystrs, fig_titles)):
        fig.text(0.50, 0.02, xstr, ha=ha, va=va, fontsize=font_size_lb)
        fig.text(0.08, 0.50, ystr, ha=ha, va=va, fontsize=font_size_lb, rotation='vertical')
        fig.text(0.50, 0.90, titl, ha=ha, va=va, fontsize=font_size_lb)

    # output
    out1 = ('_'.join(fig_title.split()))+'.png'
    out2 = ('_'.join(fig_title.split()))+'_resid.png'
    for fig, out in list(zip([fig1, fig2],[out1, out2])):
        fig.savefig(out, bbox_inches='tight', transparent=True, dpi=fig_dpi)
        print('save to file: '+out)
        fig.show()



###############################################################################
if __name__ == '__main__':

    ## read the user-specified locations in a TXT file
    yx, locs = read_ptstxt('locs.txt')

    if False:
        ## Define time function model
        model = '--exp 20190705 30 --log 20190705 30'


        ## Run tsview.py for the given locations
        for i in range(len(locs)):
            tsview('timeseries_ERA5_demErr.h5', yx=yx[i], model=model, oname=locs[i])


    ## Read the TS TXT files
    TS_Dict = init_data_dict(locs)
    for loc in locs:
        ts_file = '{}_ts.txt'.format(loc)
        TS_Dict[loc] = read_tstxt(ts_file)


    ## make a pile of ts plots
    plot_timeseries(TS_Dict, fig_title='a064_tspile')