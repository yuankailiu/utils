#! /bin/bash
# --------------------------------------------------------------
# Easily run MintPy routines step by step:
#  - prep_aria.py : prepare files from ARIA-tools output
#  - smallbaselineApp.py : run different steps
# 
# Make sure change the paths and config file
#
# Example: (run this script, time it, write to a logfile) 
#       time bash 1_run_SBApp.sh 2>&1 | tee -a 1_run_SBApp.log
#
# Yuan-Kai Liu, 2020-08-07
# --------------------------------------------------------------

printf "\n\n\n"
printf "########################################################\n"
printf "# >>> Start standard processing routines for MintPy >>> \n"
printf "########################################################\n\n"



# =============== Run the MintPy routine step by step ==================
MINTPY_path=.      # Path for MintPy processing. Default: current directory
config=${MINTPY_path}/AqabaSenDT021.cfg

smallbaselineApp.py ${config} --dostep load_data
#smallbaselineApp.py ${config} --dostep modify_network
#plot_network.py inputs/ifgramStack.h5 --nodisplay --vlim 0.2 1.0 --cmap-vlist 0.2 0.7 1.0 --show-kept --lw 2 --ms 10
#smallbaselineApp.py ${config} --dostep reference_point
#smallbaselineApp.py ${config} --dostep quick_overview

#smallbaselineApp.py ${config} --dostep correct_unwrap_error
#smallbaselineApp.py ${config} --dostep invert_network

#smallbaselineApp.py ${config} --dostep correct_LOD
#smallbaselineApp.py ${config} --dostep correct_SET
#smallbaselineApp.py ${config} --dostep correct_troposphere
#smallbaselineApp.py ${config} --dostep deramp

#solid_earth_tide.py timeseries_ERA5.h5 -g inputs/geometryGeo.h5

#smallbaselineApp.py ${config} --dostep correct_topography
#dem_error.py timeseries_ERA5_SET.h5 -t smallbaselineApp.cfg -o timeseries_ERA5_SET_demErr.h5 -g inputs/geometryGeo.h5

#smallbaselineApp.py ${config} --dostep residual_RMS
#smallbaselineApp.py ${config} --dostep reference_date

#smallbaselineApp.py ${config} --dostep velocity
#bash ./3_fit_velocity.sh

# smallbaselineApp.py ${config} --dostep geocode
# smallbaselineApp.py ${config} --dostep google_earth
# smallbaselineApp.py ${config} --dostep hdfeos5
