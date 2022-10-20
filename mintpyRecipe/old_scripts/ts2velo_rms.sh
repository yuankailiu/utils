#! /bin/bash
# --------------------------------------------------------------
# Run timeseries_rms.py using:
#
# YKL @ 2021-09-27
# updated @ 2021-09-27
# --------------------------------------------------------------

config="smallbaselineApp.cfg"
mask_file="../maskTempCoh095.h5"


mkdir -p resid_01 resid_02


opt=" -m ${mask_file} --cutoff 3 --figsize 10 5 "

timeseries_rms.py velocity01_Residual.h5 $opt -r no && mv *_date.txt *rms_velocity01_* resid_01
timeseries_rms.py velocity02_Residual.h5 $opt -r no && mv *_date.txt *rms_velocity02_* resid_02
