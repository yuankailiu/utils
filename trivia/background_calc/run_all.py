#!/usr/bin/env python3

# Quick calculation of seismicity metrics
# ykliu @ Mar31 2021
#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
import dataUtils as du
from select_cat import run_select
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams.update({'font.size': 22})


""" WORKFLOW:
## 1. catalog script
#       - read the original raw catalog
#       - estimate the Mc history for every year (save pic)
#       - set a universal Mc (Mmin) and check the mag-freq distribution (save pic)
#       - restrain the catalog, >Mc and within a time period
#       - plot GMT with the restrained catalog
#       - select a polygon to subset the catalog (output a polygon file)
#       - plot the subset catalog (output a subset catalog)

## 2. calc script
#       - pick fault line end points (convert lalo to along-strike location)
#       - binning & calc (location, mag, time, depth) => (location, metric, uncertainty, time) putput a file

## 3. plotting script
#       - plot final seismicity metrics figures
"""

# convert the base filename to a filename with depths info
def make_depth_filename(basename, meta):
    delim = '.'
    tmp = basename.split(delim)
    outname = tmp[0]+'_d'+str(meta['DEP_MIN'])+'-'+str(meta['DEP_MAX'])+delim+tmp[1]
    return outname

#%%
## =====================  0. Read metadata  ============================ ##
my_json = './init.json'
meta = du.read_meta(my_json)



## =====================  1. Data selection  ============================ ##
# >> Skip, if set "UPDATE_ROI" and "UPDATE_ARC" to "no"

if (meta['UPDATE_ROI']=='yes') or (meta['UPDATE_ARC']=='yes'):
    run_select(my_json)


## =====================  2. Binning & calc  ============================ ##
# >> Skip, if set "MODE" != "calc"

infile = meta['OUT_CATALOG']+'_add_mc.csv'
cat = du.read_cat(infile)
evid, dtime, dtime_s, lat, lon, dep, mag, td, sd, mc = cat
#dist = du.epi2projDist(lon, lat, meta['FAULT_START'], meta['FAULT_END'])
dist = td - meta['REF_LOC_DIST']

# Get output filenames
out_fix = meta['OUT_RESULT_FILE1']
out_mvw = meta['OUT_RESULT_FILE2']

# further set depth constraints
if meta['SELECT_DEPTH'] == 'yes':
    out_fix = make_depth_filename(meta['OUT_RESULT_FILE1'], meta)
    out_mvw = make_depth_filename(meta['OUT_RESULT_FILE2'], meta)
    select = (dep>=meta['DEP_MIN']) * (dep<meta['DEP_MAX']) * (mag>=mc)
    msg  = 'Depth ranges: {} to {} km\n'.format(meta['DEP_MIN'], meta['DEP_MAX'])
    msg += 'Considering spatially varying Mc\n'
    msg += 'Total events: {}'.format(np.sum(select))
    print(msg)
    #cat = du.cat_selection(cat, select_depth)
    #dist = du.epi2projDist(lon, lat, meta['FAULT_START'], meta['FAULT_END'])
    cat = np.array(cat).T[select].T
    evid, dtime, dtime_s, lat, lon, dep, mag, td, sd, mc = cat
    dist = td - meta['REF_LOC_DIST']

    # initialize the key for dict saving results
    result_names = ['EVID','MAG','DTIME','INTT','LAT','LON','DEPTH','COV','B_RATE','B_FRAC','b-value','M0']

if meta['MODE'][:4] == 'calc':
    # 1) run calc for fixed binning (skip once run)
    print('Calculation using fixed bins...')
    res = du.init_metric_dict(result_names)
    res = du.run_fix_bin(res, dist, cat, meta, bin_size=meta["BIN_WIDTH_FIX"])
    with open(out_fix, "wb") as f:
        pickle.dump(res, f)

    # 2) run calc for moving binning (skip once run)
    print('Calculation using moving window bins...')
    res = du.init_metric_dict(result_names)
    res = du.run_moving_bin(res, dist, cat, meta, bin_size=meta["BIN_WIDTH_MVW"], bin_step=meta["BIN_STEP_MVW"])
    with open(out_mvw, "wb") as f:
        pickle.dump(res, f)


#%%
## =====================  3. Plot the result  ============================ ##
# read the saved/calculated results
with open(out_mvw, "rb") as f:
    res = pickle.load(f)

# get num of events along fault
num_arr = []
for evid in res['EVID']:
    num_arr.append(len(evid))

# get bin center locations
bx = res['BIN_MID_LOC']
x = np.arange(min(bx),max(bx))      # x of aseismic slip (N/A)
y = np.ones_like(x)                 # y of aseismic slip (N/A)

# plot different metrics
du.plot_result(bx, num_arr, res['COV'],     'COV',                 dist, meta, cat, ylim=[0,4],     lc='g',          fc='lightgreen', m_circ=6, cmap_max=20, titstr='cov',    log='yes', curve=True)
du.plot_result(bx, num_arr, res['B_FRAC'],  'Background fraction', dist, meta, cat, ylim=[0,1],     lc='r',          fc='lightpink',  m_circ=6, cmap_max=20, titstr='backFr', log='yes', curve=True)
du.plot_result(bx, num_arr, res['B_RATE'],  'Background rate',     dist, meta, cat, ylim=[0,50],    lc='b',          fc='lightblue',  m_circ=6, cmap_max=20, titstr='backRt', log='yes', curve=True)
du.plot_result(bx, num_arr, res['b-value'], 'b-value',             dist, meta, cat, ylim=[0.4,1.2], lc='darkorange', fc='bisque',     m_circ=6, cmap_max=20, titstr='bv',     log='yes', curve=True)
# %%

import map as mmap
trench = du.read_plate_bound(meta['PLATE_BOUND_FILE'], meta['PLATE_BOUND_NAME'], v=False)



lalo = []
locs = []
for key in meta:
    if key.startswith('LOC_'):
        locs.append(key.split('_')[1])
        lalo.append(np.array(meta[key]))
lalo = np.array(lalo)

new_lalo, _, _, td = du.calc_trench_project(lalo, trench)

pts = np.hstack([lalo, new_lalo])


#%%
slab_model = './Slab2_AComprehe/alu_slab2_dep_02.23.18.grd'
mmap.gmt_map(lon.astype('float'), lat.astype('float'), mag.astype('float'), dep.astype('float'), meta, pts=pts, title='test_proj', slab_model=slab_model)
#mmap.gmt_map(lon.astype('float'), lat.astype('float'), mag.astype('float'), dep.astype('float'), meta, title='zmap_70km', slab_model=slab_model)

# %%
