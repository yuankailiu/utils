#!/usr/bin/env python3
# Calculate distance between two list of lat/lon


#%%
import os
import vptree
import warnings
import numpy as np
import pandas as pd
import datetime as dt
import dataUtils as du
from datetime import timedelta
import matplotlib.pyplot as plt
from geopy.distance import geodesic

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({'font.size': 22})

## Set input files
in_json = 'init.json'
catalog = 'outcat_add_mc.csv'

## Opening JSON file for metadata
meta = du.read_meta(in_json)
catalog = catalog
## Create output pics folder
if not os.path.exists(meta['PIC_DIR']):
    print('Make pic output dir %s' % meta['PIC_DIR'])
    os.makedirs(meta['PIC_DIR'])

## Read the plate boundary data from file; format=(latitude, longitude)
if meta['PLOT_PLATE_BOUND'] == 'yes':
    trench = du.read_plate_bound(meta['PLATE_BOUND_FILE'], meta['PLATE_BOUND_NAME'])

## Read the slab model data from file; format=(latitude, longitude, depth, thickness)
if (meta['SLAB_DEPTH'] is not None) and (meta['SLAB_THICK'] is not None):
    _, tmp     = du.read_xyz(meta['SLAB_DEPTH'])
    _, tmp_thk = du.read_xyz(meta['SLAB_THICK'])
    slab = np.vstack((tmp[:,1], tmp[:,0]-360, tmp[:,2], tmp_thk[:,2])).T

## Read the earthquake catalog
if catalog == 'default':
    catalog = meta['CATALOG']
print('Reading the seismicity catalog %s' % catalog)
cat = du.read_cat(catalog)

if len(cat) == 7:
    evid, dtime, dtime_s, lat, lon, dep, mag = cat
elif len(cat) == 9:
    evid, dtime, dtime_s, lat, lon, dep, mag, td, sd = cat
elif len(cat) == 10:
    evid, dtime, dtime_s, lat, lon, dep, mag, td, sd, mc = cat


select = (dep>=0) * (dep<70) * (mag>=mc)
print(np.sum(select))
cat = du.cat_selection(cat, select)

if len(cat) == 7:
    evid, dtime, dtime_s, lat, lon, dep, mag = cat
elif len(cat) == 9:
    evid, dtime, dtime_s, lat, lon, dep, mag, td, sd = cat
elif len(cat) == 10:
    evid, dtime, dtime_s, lat, lon, dep, mag, td, sd, mc = cat


# %%
num = 10

query = np.array(list(zip(lat[:num], lon[:num])))
dists, nns = du.knn(query, trench, n=2)

#query = np.array(list(zip(lat[:num], lon[:num], dep[:num])))
#dists, nns = du.knn(query, slab[:,:3], n=2)

print(dists.shape)
print(nns.shape)


# %%
def test():

    import  matplotlib.pyplot       as      plt
    from    mpl_toolkits.basemap    import  Basemap

    a   = np.array([ 30.0,  80.0])
    b   = np.array([ 50.0, 100.0])

    p   = np.array([ 45.0,  70.0])

    h, m, dper, dpar = du.point2greatCircle(p, a, b)

    fig = plt.figure()
    ax  = plt.gca()
    bm  = Basemap(  projection  = 'ortho',
                    lat_0       = 35.0,
                    lon_0       = 85.0)
    bm.drawcoastlines(linewidth=0.25)

    points          = [a, b, p, m, h]
    point_names     = ['A', 'B', 'P', 'M', 'H']
    point_colors    = ['k', 'k', 'k', 'r', 'r']
    point_label_xy  = [(-10, -10), (10, 10), (-10, 5), (10, -5), (10, -3)]

    lines           = [[a, b], [p, m]]
    line_colors     = ['k', 'r']

    for point, color, name, label_xy in zip(
            points, point_colors, point_names, point_label_xy):
        bm.scatter( point[1], point[0],
                    color       = color,
                    marker      = '.',
                    latlon      = True)

        plt.annotate(   name,
                        xy          = bm(point[1], point[0]),
                        xycoords    = 'data',
                        xytext      = label_xy,
                        textcoords  = 'offset points',
                        color       = color)

    for line, color in zip(lines, line_colors):
        print(line)
        bm.drawgreatcircle(
                    line[0][1], line[0][0],
                    line[1][1], line[1][0],
                    color = color)

    plt.show()

test()
# %%
num = 1
query = np.array(list(zip(lat[:num], lon[:num])))

H, M, dper, Dpar = du.calc_trench_project(query, trench)

# %%
loc_dist = np.array([10,20,30])
loc = ['A', 'B', 'C']

plt.figure()
plt.axvline(x=10, lw=1, c='k', alpha=0.2)
plt.text(loc_dist, 20, loc, rotation=270, ha='left', va='top', fontsize=18)
plt.show()
# %%
