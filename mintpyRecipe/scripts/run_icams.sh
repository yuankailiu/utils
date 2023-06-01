#! /bin/sh

main_dir=$(pwd)

#conda activate icams

### @ mintpy/

## delete orbit files and re-run orbit and los.npy
#rm -rf *orbit
#rm -rf ./icams/ERA5/*.npy ./icams/ERA5/sar
tropo_icams.py timeseries_SET.h5 inputs/geometryGeo.h5 --sar-par IW1.xml --ref-file timeseries_SET.h5 --project los --nproc 8
cp timeseries_icams*.h5 ./icams                                 # backup icams results

## prepare ERA5_stochastic + ion corrected time series
cp timeseries_icamsCor_los_sklm.h5 timeseries_SET_ERA5S.h5      # the icams corrected file is timeseries_SET_ERA5S
diff.py timeseries_SET_ERA5S.h5      inputs/ion.h5          -o timeseries_SET_ERA5S_Ion0.h5 --force
diff.py timeseries_SET_ERA5S_Ion0.h5 inputs/ionBurstRamp.h5 -o timeseries_SET_ERA5S_Ion.h5  --force

# backup topo residuals and final residuals
mkdir -p ./topoResidual/afterERA5
cp demErr.h5 reference_date.txt rms_timeseriesResidual_ramp* timeseriesResidual*.h5 ./topoResidual/afterERA5


## @ mintpy/topoResidual/afterERA5_ICAMS

# go to this path and run mintpy (NB: edit .cfg name)
mkdir -p ./topoResidual/afterERA5_ICAMS
cd ./topoResidual/afterERA5_ICAMS
dem_error.py $main_dir/timeseries_SET_ERA5S_Ion.h5 --poly-order 1 --periodic 1.0 0.5 -g $main_dir/inputs/geometryGeo.h5 -o timeseries_SET_ERA5S_Ion_demErr.h5 --cluster local --num-worker 8 --ram 16 --update
smallbaselineApp.py $main_dir/AqabaSenAT087.cfg --dostep residual_RMS
smallbaselineApp.py $main_dir/AqabaSenAT087.cfg --dostep deramp
rm -rf inputs pic

# velocity and reference frame (NB: edit the reference point/date, euler pole)
timeseries2velocity.py timeseries_SET_ERA5S_Ion_demErr.h5 --poly-order 1 --periodic 1.0 0.5 -o velocity.h5 --ref-lalo 29.5463 36.0810 --ref-date 20190118 --update
plate_motion.py --geom $main_dir/inputs/geometryGeo.h5 --plate Arabia --velo velocity.h5
diff.py velocity_ITRF14.h5 $main_dir/velocity_out/velocity_ITRF14.h5 -o ICAMS_ornot_diff.h5

# plot velocity
opt=" --unit mm -c RdYlBu_r --dpi 300 --dem  $main_dir/inputs/srtm.dem  --dem-nocontour --shade-exag 0.02 --shade-min -4000 --shade-max 4000 --coastline 10m --coastline-lw 0.5  --mask $main_dir/closurePhase/maskTempCohClosurePhaseNumTriNonzero.h5 --nodisplay "
view.py $main_dir/velocity_out/velocity_ITRF14.h5   velocity $opt -v -4   4   -o velocity_ITRF14_orig.png
view.py velocity_ITRF14.h5                          velocity $opt -v -4   4   -o velocity_ITRF14_icams.png
view.py ICAMS_ornot_diff.h5                         velocity $opt -v -0.5 0.5 -o ICAMS_ornot_diff.png



## @ mintpy/

# go back
cd $main_dir
echo "completed!"
