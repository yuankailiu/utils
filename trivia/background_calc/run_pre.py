#!/usr/bin/env python3

############################################################
# Doing seismicicty event selection                        #
#                                                          #
# Y.K. Liu @ 2021 June                                     #
############################################################
#%%

import os
import pandas as pd
import matplotlib.dates as mdates
from shapely.geometry import Point, Polygon
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import timedelta
from geopy.distance import geodesic
import dataUtils as du

plt.rcParams.update({'font.size': 20})


class pre_analysis():
    """
    pre-analysis of the seismicity
    """

    def __init__(self, in_json=None, catalog='default'):
        ## Opening JSON file for metadata
        self.meta = du.read_meta(in_json)
        self.catalog = catalog
        ## Create output pics folder
        if not os.path.exists(self.meta['PIC_DIR']):
            print('Make pic output dir %s' % self.meta['PIC_DIR'])
            os.makedirs(self.meta['PIC_DIR'])
        self.load_data(self.catalog)
        return


    def load_data(self, catalog):
        ## Read the plate boundary data from file; format=(latitude, longitude)
        if self.meta['PLOT_PLATE_BOUND'] == 'yes':
            self.trench = du.read_plate_bound(self.meta['PLATE_BOUND_FILE'], self.meta['PLATE_BOUND_NAME'])

        ## Read the slab model data from file; format=(latitude, longitude, depth, thickness)
        if (self.meta['SLAB_DEPTH'] is not None) and (self.meta['SLAB_THICK'] is not None):
            _, tmp     = du.read_xyz(self.meta['SLAB_DEPTH'])
            _, tmp_thk = du.read_xyz(self.meta['SLAB_THICK'])
            self.slab = np.vstack((tmp[:,1], tmp[:,0]-360, tmp[:,2], tmp_thk[:,2])).T

        ## Read the earthquake catalog
        if catalog == 'default':
            catalog = self.meta['CATALOG']
        print('Reading the seismicity catalog %s' % catalog)
        cat = du.read_cat(catalog)

        if len(cat) == 7:
            self.evid, self.dtime, self.dtime_s, self.lat, self.lon, self.dep, self.mag = cat
        elif len(cat) == 9:
            self.evid, self.dtime, self.dtime_s, self.lat, self.lon, self.dep, self.mag, self.td, self.sd = cat
        elif len(cat) == 10:
            self.evid, self.dtime, self.dtime_s, self.lat, self.lon, self.dep, self.mag, self.td, self.sd, self.mc = cat
        return


    def plot_select_events(self, o, o_back=False, draw_poly=False, savename=False):
        # Define cursor clicking

        def onclick(event):
            click = event.xdata, event.ydata
            if None not in click: # clicking outside the plot area produces a coordinate of None, so we filter those out.
                print('x = {}, y = {}'.format(*click))
                coords.append(click)

        if draw_poly:
            print(' > Please specify a {} polygon'.format(draw_poly))
            coords = []

        plt.figure(figsize=[14,10])
        if o_back is not False:
            plt.scatter(self.lon[o_back], self.lat[o_back], s=(2.2**self.mag[o_back]), ec='grey', marker='o', linewidths=0.2, fc='lightgrey', alpha=0.3)
        sc =plt.scatter(self.lon[o],      self.lat[o],      s=(2.2**self.mag[o]),      ec='k',    marker='o', linewidths=0.4, c=self.dep[o], cmap='jet_r')
        cbar = plt.colorbar(sc)
        cbar.set_label('Depth [km]')
        plt.xlim(self.meta['EXTENT'][0], self.meta['EXTENT'][1])
        plt.ylim(self.meta['EXTENT'][2], self.meta['EXTENT'][3])
        plt.plot(self.trench[:-1,1], self.trench[:-1,0], '--', color='k', lw=2, clip_on=True)
        plt.plot(self.trench[1:,1],  self.trench[1:,0],  '--', color='k', lw=2, clip_on=True)
        plt.title('{} events (draw a polygon for {})'.format(np.sum(o), draw_poly))
        if draw_poly:
            plt.gca().figure.canvas.mpl_connect('button_press_event', onclick)
            plt.savefig('{}/{}.png'.format(self.meta['PIC_DIR'], draw_poly), bbox_inches='tight', dpi=300)
        if savename:
            plt.savefig('{}/{}.png'.format(self.meta['PIC_DIR'], savename), bbox_inches='tight', dpi=300)
        plt.show()
        if draw_poly:
            coords.append(coords[0])
            if len(coords) != 0:
                np.savetxt(self.meta[draw_poly], coords, delimiter=", ")
                print('Polygon saved to file: {}'.format(self.meta[draw_poly]))
            return coords
        else:
            return


    def Mc_history(self):
        """Plot the magnitude completeness history"""
        # first check the mc for every epoch (each year)
        print('Checking the history of completeness magnitude')
        self.starttime = dt.datetime.strptime(self.meta['STARTTIME'],'%Y%m%d')
        self.endtime   = dt.datetime.strptime(self.meta['ENDTIME']  ,'%Y%m%d')
        o_time  = (self.dtime >= self.starttime) * (self.dtime < self.endtime)
        n_yr      = int((self.endtime - self.starttime).days/365.25)

        print('There are {} events between {} and {}'.format(np.sum(o_time), self.starttime.strftime("%Y%m%d"), self.endtime.strftime("%Y%m%d")))
        epochs, bin_sec, Mcs = du.epoch_Mc(self.mag[o_time], self.dtime[o_time], n_yr, plot='no')
        bin_day   = bin_sec/86400.0

        # choose some colors for plotting
        fc1 = 'r'
        fc2 = 'lightskyblue'

        # now make the plot
        fig, ax1 = plt.subplots(figsize=[14,14])
        ax2 = ax1.twinx()
        ax1.scatter(self.dtime, self.mag+np.random.uniform(-0.05, 0.05, len(self.mag)), marker='o', fc='grey', s=10, zorder=0)
        ax1.scatter(self.dtime[o_time][epochs]+timedelta(days=bin_day/2), np.array(Mcs), s=100, c=fc1, ec='k')
        ax1.plot(self.dtime[o_time][epochs]+timedelta(days=bin_day/2), np.array(Mcs), lw=2, c=fc1 ,zorder=0)
        ax2.hist(self.dtime[o_time], bins=n_yr, fc=fc2, ec='k', alpha=0.6)
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.text(0.1, 0.8, 'Binning ~{:.0f} days'.format(bin_day), c=fc1, transform=ax2.transAxes)
        ax1.text(0.1, 0.9, 'Binning ~{:.0f} days'.format(bin_day), c=fc2, transform=ax2.transAxes)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('# events', rotation=270)
        ax1.set_ylabel('Magnitude of \ncompleteness', color=fc1, labelpad=50)
        ax1.set_yticks(np.arange(1,8,0.5))
        ax1.grid(True, alpha=0.8)
        fig.savefig('{}/McHistory.png'.format(self.meta["PIC_DIR"]), dpi=300, bbox_inches='tight')
        return


    def update_time(self, yyyymmdd):
        """ Change the starttime to a more recent and reliable period """
        self.meta["STARTTIME"] = yyyymmdd
        self.starttime = dt.datetime.strptime(self.meta['STARTTIME'],'%Y%m%d')
        self.o_time  = (self.dtime >= self.starttime) * (self.dtime < self.endtime)
        return


    def magfreq_dist(self):
        """Look at overall magnitude-frequency distribution"""
        guess_Mc = du.maxc_Mc(self.mag[self.o_time], plot='yes', save=self.meta["PIC_DIR"], title='ori', range=[0,8.5])
        print('Estimated overall Mc = {:.2f}'.format(guess_Mc))

        self.Mc = du.maxc_Mc(self.mag[self.o_time], plot='yes', save=self.meta["PIC_DIR"], title='final', Mc=self.meta["Mc"], range=[0,8.5])
        print('Decided Mc = {:.2f}'.format(self.Mc))
        return


    def update_Mc(self, Mc=None):
        """Update the event selection based on Mc"""
        if Mc is None:
            Mc = self.Mc
        self.o_mag = (self.mag>=Mc)
        o_out = self.o_time * self.o_mag
        print('There are {} events between {} and {}, Mag >= {}'.format(np.sum(o_out), self.starttime.strftime("%Y%m%d"), self.endtime.strftime("%Y%m%d"), Mc))
        return o_out


    def manual_select(self, o_in, key):
        """Draw a polygon for further event subset
           Example:
            around the faultl;
            within a region near the subduction zone;
        """
        if self.meta['SELECT_{}'.format(key)] == 'yes':
            polytype='POLYGON_{}'.format(key)
            if (self.meta['UPDATE_{}'.format(key)] == 'no') and (os.path.exists(self.meta[polytype])):
                print('Read the existing polygon file: {}'.format(self.meta[polytype]))
                coords =  pd.read_csv(self.meta[polytype], dtype='float', header=None).to_numpy()
            else:
                coords = self.plot_select_events(o_in, draw_poly=polytype)
            poly = Polygon(coords)
            o_man = []
            print('Classifying points in/out of the {}'.format(polytype))
            for i in range(len(self.lat)):
                pp = Point(self.lon[i], self.lat[i])
                o_man.append(pp.within(poly))
            o_man = np.array(o_man)
        return o_man


    def include_roi(self, o_in, key='ROI'):
        print('Make a polygon to include events within the region of interest')
        self.o_roi = self.manual_select(o_in, key)
        return


    def exclude_arc(self, o_in, key='ARC'):
        min_depth = self.meta['ARC_EV_DEPTH']
        print('Make a polygon to exclude arc events; will exclude events shallower than {} km'.format(min_depth))
        o_arc = self.manual_select(o_in, key)
        self.o_arc = ~(np.array(o_arc)*(self.dep<=min_depth))
        return


    def get_trench_proj_distance(self, o_in):
        """ Calculate trench projection location """
        print('Calculate trench projection location')
        lalo    = np.vstack([self.lat[o_in], self.lon[o_in]]).T
        self.td = du.calc_trench_project(lalo, self.trench)[-1]
        return self.td


    def get_slab_shortest_distance(self, o_in):
        """ Calculate hypos from slab model distance """
        print('Calculate hypos from slab model distance')
        lalod   = np.vstack([self.lat[o_in], self.lon[o_in], self.dep[o_in]]).T
        self.sd = du.calc_slab_distance(lalod, self.slab)
        return self.sd


    def update_catalog(self, infile, outfile='default', ext='', o_in=None, append_info=[]):
        """ Generate a new catalog file """
        if outfile == 'default':
            outname = self.meta['OUT_CATALOG']+ext+'.csv'
        else:
            outname = outfile+ext+'.csv'
        cat = du.read_cat(infile, fullFile=True)
        if o_in is None:
            idx = np.ones(len(cat), dtype=bool)
        else:
            idx = o_in

        # get columns to save to a new catalog
        n    = np.sum(idx)
        if cat.shape[1] == 22:
            # USGS original format (22 columns)
            data = np.concatenate([cat[idx,:6], cat[idx,10].reshape(n,1), cat[idx,11].reshape(n,1)], axis=1)
            head =  'time, latitude, longitude, depth, mag, magType, net, id'
        elif cat.shape[1] == 10:
            # format appended with trench dist & slab dist
            data = np.array(cat[idx])
            head =  'time, latitude, longitude, depth, mag, magType, net, id, td, sd'

        # append additional info (e.g., trench proj distance, distance to slab model, segment Mc)
        if len(append_info) != 0:
            for key, val in append_info.items():
                head += ', {}'.format(key)
                val = np.round(val, 4)
                data = np.concatenate([data, val.reshape(n,1)], axis=1)
        np.savetxt(outname, data, fmt='%s', header=head, delimiter=", ")
        print('Final selected events saved to file: {}'.format(outname))
        return


    def final_plot(self, o_in):
        ## Make a final plot showing the selected seismicity
        if self.meta['SELECT_ROI'] or self.meta['SELECT_ARC']:
            msg  = ' >> Time {} to {}\n'.format(self.starttime.strftime("%Y%m%d"), self.endtime.strftime("%Y%m%d"))
            msg += ' >> Mag >= {}\n'.format(self.Mc)
            o_out = self.o_time * self.o_mag
            if self.meta['SELECT_ROI']:
                o_out *= self.o_roi
                msg   += ' >> Within the ROI polygon\n'
            if self.meta['SELECT_ARC']:
                o_out *= self.o_arc
                msg   += ' >> Excluded the ARC events <= {} km depth\n'.format(self.meta['ARC_EV_DEPTH'])
            msg += ' >> Total {} events'.format(np.sum(o_out))
            print(msg)

            # Plot final selection
            print('Plot the final events further selected by the polygon')
            self.plot_select_events(o_out, o_back=o_in, savename='polygon_final')

            # Save the new selected catalog to file
            self.update_catalog(infile=self.meta['CATALOG'], ext='', o_in=o_out)
        return o_out


    def spatial_chunck_mc(self):
        td = self.td - min(self.td)
        chunk_size = 100    # analyze Mc in each chunck (km)
        chunks = np.arange(min(td), max(td), chunk_size)
        chunks[-1] += chunk_size

        depth_groups = dict()
        depth_groups['shallow']   = [0,    30]
        depth_groups['interm']    = [30,   70]

        Mc_arr = np.zeros(len(self.mag))
        for key, val in depth_groups.items():
            idx1 = (self.dep>=val[0]) * (self.dep<val[1])
            for i in range(len(chunks)-1):
                idx2  = (td>=chunks[i]) * (td<chunks[i+1])
                idx   = idx1 * idx2
                title = '{}; along trench {:.0f}-{:.0f}km'.format(key, chunks[i], chunks[i+1])
                mc    = du.maxc_Mc(self.mag[idx], plot='no', save='./pic', title=title, range=[0,8.5])
                mc    = round(mc+0.05,1) # round up to the 1st decimal
                Mc_arr += (mc * idx)
        return Mc_arr




############################################################################
#%%
if __name__ == '__main__':
# 0. Prepare data
    my_json = './init.json'
    meta = du.read_meta(my_json)

    obj = pre_analysis(in_json = './init.json')

#%% 1. Look at a overall Mc
    obj.Mc_history()
    obj.update_time('20000101')
    obj.magfreq_dist()
    opt = obj.update_Mc()

#%% 2. Polygon selections
    obj.include_roi(opt, key='ROI')
    obj.exclude_arc(opt, key='ARC')
    opt_final = obj.final_plot(opt)

#%% 3. Calculate trench and slab distance
    td = obj.get_trench_proj_distance(opt_final)
    sd = obj.get_slab_shortest_distance(opt_final)

#%% 4. Save distances to a new catalog file
    add_info = dict()
    add_info['trenchDist'] = td
    add_info['slabDist']   = sd
    obj.update_catalog(meta['CATALOG'], ext='_add', o_in=opt_final, append_info=add_info)

#%% 5. Analyze Mc for fault chuncks, save each chunck_Mc into a new catalog file
    obj.load_data('outcat_add.csv')
    Mc_arr = obj.spatial_chunck_mc()
    add_info = dict()
    add_info['Mc'] = Mc_arr
    obj.update_catalog(meta['OUT_CATALOG']+'_add.csv', ext='_add_mc', append_info=add_info)

    print('Normal complete of the code.')