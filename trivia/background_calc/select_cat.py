#!/usr/bin/env python3

############################################################
# Doing seismicicty event selection                        #
#                                                          #
# Y.K. Liu @ 2021 June                                     #
############################################################


import os
import json
import pandas as pd
import matplotlib.dates as mdates
from shapely.geometry import Point, Polygon
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import timedelta
plt.rcParams.update({'font.size': 22})

import dataUtils as du




def run_select(my_json):
    ## Plot/select events by polygons

    def plot_select_events(sel, back_sel=False, draw_poly=False, savename=False):
        # Define cursor clicking

        def onclick(event):
            click = event.xdata, event.ydata
            if None not in click: # clicking outside the plot area produces a coordinate of None, so we filter those out.
                print('x = {}, y = {}'.format(*click))
                coords.append(click)

        if draw_poly:
            print('Please specify a {} polygon'.format(draw_poly))
            coords = []
        plt.figure(figsize=[14,10])
        if back_sel is not False:
            plt.scatter(lon[back_sel], lat[back_sel], s=(2.2**mag[back_sel]), fc='lightgrey', ec='grey', marker='o', alpha=0.3, linewidths=0.2)
        sc = plt.scatter(lon[sel], lat[sel], s=(2.2**mag[sel]), c=dep[sel], cmap='jet_r', ec='k', marker='o', linewidths=0.4)
        cbar = plt.colorbar(sc)
        cbar.set_label('Depth [km]')
        plt.xlim(meta['EXTENT'][0], meta['EXTENT'][1])
        plt.ylim(meta['EXTENT'][2], meta['EXTENT'][3])
        plt.plot(plate_xy[:,0], plate_xy[:,1], '--', color='k', lw=2, clip_on=True)
        plt.title('{} events (draw a polygon for {})'.format(np.sum(sel), draw_poly))
        if draw_poly:
            plt.gca().figure.canvas.mpl_connect('button_press_event', onclick)
            plt.savefig('{}/{}.png'.format(meta['PIC_DIR'], draw_poly), bbox_inches='tight', dpi=300)
        if savename:
            plt.savefig('{}/{}.png'.format(meta['PIC_DIR'], savename), bbox_inches='tight', dpi=300)
        plt.show()
        if draw_poly:
            coords.append(coords[0])
            if len(coords) != 0:
                np.savetxt(meta[draw_poly], coords, delimiter=", ")
                print('Polygon saved to file: {}'.format(meta[draw_poly]))
            return coords


    ## Opening JSON file for metadata
    meta = du.read_meta(my_json)


    ## Create output pics folder
    if not os.path.exists(meta['PIC_DIR']):
        print('Make pic output dir %s' % meta['PIC_DIR'])
        os.makedirs(meta['PIC_DIR'])


    ## Read the plate boundary data from file
    if meta['PLOT_PLATE_BOUND'] == 'yes':
        plate_xy = du.read_plate_bound(meta['PLATE_BOUND_FILE'], meta['PLATE_BOUND_NAME'])


    ## Read the slab model data from file
    if (meta['SLAB_DEPTH'] is not None) and (meta['SLAB_THICK'] is not None):
        _, tmp     = du.read_xyz(meta['SLAB_DEPTH'])
        _, tmp_thk = du.read_xyz(meta['SLAB_THICK'])
        slab = np.vstack((tmp[:,1], tmp[:,0]-360, tmp[:,2], tmp_thk[:,2])).T


    ## Read the earthquake catalog
    print('Reading the seismicity catalog %s' % meta['CATALOG'])
    cat = du.read_cat(meta['CATALOG'])
    evid, dtime, dtime_s, lat, lon, dep, mag = cat


    ## plot Mc history
    # first check the mc for every epoch (each year)
    print('Checking the history of completeness magnitude')
    starttime = dt.datetime.strptime(meta['STARTTIME'],'%Y%m%d')
    endtime   = dt.datetime.strptime(meta['ENDTIME'] ,'%Y%m%d')
    sel_time  = (dtime >= starttime) * (dtime < endtime)
    print('There are {} events between {} and {}'.format(np.sum(sel_time), starttime.strftime("%Y%m%d"), endtime.strftime("%Y%m%d")))
    n_yr      = int((endtime-starttime).days/365.25)
    epochs, bin_sec, Mcs = du.epoch_Mc(mag[sel_time], dtime[sel_time], n_yr, plot='no')
    bin_day   = bin_sec/86400.0

    # choose some colors for plotting
    fc1 = 'r'
    fc2 = 'lightskyblue'

    # now make the plot
    fig, ax1 = plt.subplots(figsize=[14,14])
    ax2 = ax1.twinx()
    ax1.scatter(dtime, mag+np.random.uniform(-0.05, 0.05, len(mag)), marker='o', fc='grey', s=10, zorder=0)
    ax1.scatter(dtime[sel_time][epochs]+timedelta(days=bin_day/2), np.array(Mcs), s=100, c=fc1, ec='k')
    ax1.plot(dtime[sel_time][epochs]+timedelta(days=bin_day/2), np.array(Mcs), lw=2, c=fc1 ,zorder=0)
    ax2.hist(dtime[sel_time], bins=n_yr, fc=fc2, ec='k', alpha=0.6)
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.text(0.1, 0.8, 'Binning ~{:.0f} days'.format(bin_day), c=fc1, transform=ax2.transAxes)
    ax1.text(0.1, 0.9, 'Binning ~{:.0f} days'.format(bin_day), c=fc2, transform=ax2.transAxes)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('# events', rotation=270)
    ax1.set_ylabel('Magnitude of \ncompleteness', color=fc1, labelpad=50)
    ax1.set_yticks(np.arange(1,8,0.5))
    ax1.grid(True, alpha=0.8)
    fig.savefig('{}/McHistory.png'.format(meta["PIC_DIR"]), dpi=300, bbox_inches='tight')


    ## Look at overall magnitude-frequency distribution
    # Change the starttime to a more recent and reliable period
    meta["STARTTIME"] = '19900101'
    starttime = dt.datetime.strptime(meta['STARTTIME'],'%Y%m%d')
    sel_time  = (dtime >= starttime) * (dtime < endtime)

    # plot mag-freq dist.
    guess_Mc = du.maxc_Mc(mag[sel_time], plot='yes', save=meta["PIC_DIR"], title='ori', range=[0,8.5])
    print('Estimated overall Mc = {:.2f}'.format(guess_Mc))

    Mc = du.maxc_Mc(mag[sel_time], plot='yes', save=meta["PIC_DIR"], title='final', Mc=meta["Mc"], range=[0,8.5])
    print('Decided Mc = {:.2f}'.format(Mc))


    ## Update the event selection based on Mc
    sel_mag = (mag>=Mc)
    sel_1 = sel_time * sel_mag
    print('There are {} events between {} and {}, Mag >= {}'.format(np.sum(sel_1), starttime.strftime("%Y%m%d"), endtime.strftime("%Y%m%d"), Mc))


    ## Draw a polygon for further event subset (e.g., around the fault, subduction zone)
    if meta['SELECT_ROI'] == 'yes':
        polytype='POLYGON_ROI'
        if (meta["UPDATE_ROI"] == 'no') and (os.path.exists(meta[polytype])):
            print('Read the existing polygon file: {}'.format(meta[polytype]))
            coords =  pd.read_csv(meta[polytype], dtype='float', header=None).to_numpy()
        else:
            coords = plot_select_events(sel_1, draw_poly=polytype)
        poly = Polygon(coords)
        sel_roi = []
        print('Classifying points in/out of the {}'.format(polytype))
        for i in range(len(lat)):
            pp = Point(lon[i], lat[i])
            sel_roi.append(pp.within(poly))
        sel_roi = np.array(sel_roi)


    ## Try to exclude the shallow volcanic arc events
    if meta['SELECT_ARC'] == 'yes':
        polytype='POLYGON_ARC'
        if (meta["UPDATE_ARC"] == 'no') and (os.path.exists(meta[polytype])):
            print('Read the existing polygon file: {}'.format(meta[polytype]))
            coords =  pd.read_csv(meta[polytype], dtype='float', header=None).to_numpy()
        else:
            coords = plot_select_events(sel_1, draw_poly=polytype)
        poly = Polygon(coords)
        sel_arc = []
        print('Classifying points in/out of the {}'.format(polytype))
        for i in range(len(lat)):
            pp = Point(lon[i], lat[i])
            sel_arc.append(pp.within(poly))
        sel_arc = ~(np.array(sel_arc)*(dep<=meta['ARC_EV_DEPTH']))


    ## Make a final plot showing the selected seismicity
    if meta['SELECT_ROI'] or meta['SELECT_ARC']:
        msg  = ' >> Time {} to {}\n'.format(starttime.strftime("%Y%m%d"), endtime.strftime("%Y%m%d"))
        msg += ' >> Mag >= {}\n'.format(Mc)
        sel_2 = sel_time * sel_mag
        if meta['SELECT_ROI']:
            sel_2 *= sel_roi
            msg   += ' >> Within the ROI polygon\n'
        if meta['SELECT_ARC']:
            sel_2 *= sel_arc
            msg   += ' >> Excluded the ARC events <= {} km depth\n'.format(meta['ARC_EV_DEPTH'])
        msg += ' >> Total {} events'.format(np.sum(sel_2))
        print(msg)

        # Plot final selection
        print('Plot the final events further selected by the polygon')
        plot_select_events(sel_2, back_sel=sel_1, savename='polygon_final')

        # Save the new selected catalog to file
        cat_full = du.read_cat(meta['CATALOG'], fullFile=True)
        np.savetxt(meta['OUT_CATALOG'], cat_full[sel_2,:13], fmt='%s', delimiter=",")
        print('Final selected events saved to file: {}'.format(meta['OUT_CATALOG']))


###########################################
    print('Normal complete of the code.')


############################################################################
if __name__ == '__main__':
    my_json = './init.json'
    run_select(my_json)