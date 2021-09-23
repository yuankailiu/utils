#!/usr/bin/env python3
# ykliu @ 14-Jun-2021

#%% Read a MintPy time-series file and get iono TEC model

import os
import numpy as np
import matplotlib.pyplot as plt
import insar.ionotec as tec


# The path to save tec data downloaded from IGS
tec_dir     = '/home/ykliu/kamb-nobak/z_common_data/atmosphere/tec_a087'

# The geometry file for generating TEC model in your region of interest
geom_file   = '/home/ykliu/kamb-nobak/aqaba/a087/isce/mintpy/inputs/geometryGeo.h5'

# A template MintPy timeseries for generating TEC model timeseries
ref_ts_file = '/home/ykliu/kamb-nobak/aqaba/a087/isce/mintpy/timeseries.h5'

# Output file of TEC model tiemseries
out_file    = '/home/ykliu/kamb-nobak/aqaba/a087/isce/mintpy/inputs/TEC.h5'


if not os.path.exists(tec_dir):
    os.makedirs(tec_dir)


#%% Downlaod / Make TEC timeseries
tec.igs_iono_ramp_timeseries(tec_dir, out_file, geom_file, ref_ts_file)


# Make an global animation of TEC model in 24 hr at a specific date
#tec_file = '{}/jplg0100.16i'.format(tec_dir)
#tec.plot_tec_animation(tec_file, save=True)

# %%
