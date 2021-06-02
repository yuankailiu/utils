#! /bin/sh

# --------------------------------------------------------------
# A modification of Yunjun's plotting script
#
# Yuan-Kai Liu, 2020-08-07
# --------------------------------------------------------------

###############################################################
# Plot Results from Routine Workflow with smallbaselineApp.py
# Author: Zhang Yunjun, 2017-07-23
# Latest update: 2020-03-20
###############################################################


## Change to 0 if you do not want to re-plot loaded dataset again
plot_network=1
plot_key_files=1
plot_loaded_data=1
plot_loaded_data_aux=1
plot_timeseries=1
plot_timeseries_residual=1
plot_TEC_ts=1
plot_geocoded_data=1
plot_the_rest=1


# Default file name
tmCoh_mask='maskTempCoh.h5'
water_mask='waterMask.h5'
dem_file='./inputs/geometryRadar.h5'
if [ ! -f $dem_file ]; then
    dem_file='./inputs/geometryGeo.h5'
fi

## Log File
log_file='2_plot_SBApp.log'
touch $log_file
printf "\n\n\n\n\n" >> $log_file
echo "########################  ./plot_smallbaselineApp.sh  ########################" >> $log_file
date >> $log_file
echo "##############################################################################" >> $log_file
#use "echo 'yoyoyo' | tee -a log" to output message to both screen and file.
#use "echo 'yoyoyo' >> log" to output message to file only.

## Create pic folder
if [ ! -d "pic" ]; then
    echo 'Create ./pic folder'
    mkdir pic
fi


## common view.py option for all files
view='view.py --nodisplay --dpi 150 --nrows 3 --ncols 8 --update '


## plot the ifgram network 
plotnet='plot_network.py --nodisplay'
if [ $plot_network -eq 1 ]; then
    file=inputs/ifgramStack.h5;    test -f $file && $plotnet $file >> $log_file
fi


## Plot Key files
opt=' --dem '$dem_file' --mask '$tmCoh_mask' -u cm '
#opt=' --dem '$dem_file' --mask '$tmCoh_mask' -u cm --vlim -2 2'
if [ $plot_key_files -eq 1 ]; then
    file=velocity.h5;              test -f $file && $view $file $opt               >> $log_file
    file=temporalCoherence.h5;     test -f $file && $view $file -c gray --vlim 0 1 >> $log_file
    file=maskTempCoh.h5;           test -f $file && $view $file -c gray --vlim 0 1 >> $log_file
    file=inputs/geometryRadar.h5;  test -f $file && $view $file                    >> $log_file
    file=inputs/geometryGeo.h5;    test -f $file && $view $file                    >> $log_file
fi


## Loaded Dataset
if [ $plot_loaded_data -eq 1 ]; then
    file=inputs/ifgramStack.h5
    opt='--noaxis --fontsize 8'
    test -f $file && h5ls $file/unwrapPhase      && $view $file unwrapPhase-      $opt --zero-mask --wrap        >> $log_file
    test -f $file && h5ls $file/unwrapPhase      && $view $file unwrapPhase-      $opt --zero-mask --vlim -15 15 >> $log_file
    test -f $file && h5ls $file/coherence        && $view $file coherence-        $opt --mask no                 >> $log_file
    test -f $file && h5ls $file/connectComponent && $view $file connectComponent- $opt --zero-mask               >> $log_file

    # phase-unwrapping error correction
    for dset in unwrapPhase_bridging unwrapPhase_phaseClosure unwrapPhase_bridging_phaseClosure; do
        test -f $file && h5ls $file/$dset            && $view $file $dset-             --zero-mask --vlim -15 15 >> $log_file
    done
fi


## Auxliary Files from loaded dataset
if [ $plot_loaded_data_aux -eq 1 ]; then
    file=avgPhaseVelocity.h5;   test -f $file && $view $file --mask $water_mask   >> $log_file
    file=avgSpatialCoh.h5;      test -f $file && $view $file -c gray --vlim 0 1   >> $log_file
    file=maskConnComp.h5;       test -f $file && $view $file -c gray --vlim 0 1   >> $log_file
fi


## Time-series files
opt='--mask '$tmCoh_mask' --noaxis -u cm --wrap --wrap-range -10 10 '
if [ $plot_timeseries -eq 1 ]; then
    file=timeseries.h5;                             test -f $file && $view $file $opt >> $log_file

    #LOD for Envisat
    file=timeseries_LODcor.h5;                      test -f $file && $view $file $opt >> $log_file
    file=timeseries_LODcor_ECMWF.h5;                test -f $file && $view $file $opt >> $log_file
    file=timeseries_LODcor_ECMWF_demErr.h5;         test -f $file && $view $file $opt >> $log_file
    file=timeseries_LODcor_ECMWF_ramp.h5;           test -f $file && $view $file $opt >> $log_file
    file=timeseries_LODcor_ECMWF_ramp_demErr.h5;    test -f $file && $view $file $opt >> $log_file

    #w tropo delay corrections
    for tropo in ERA5 ECMWF MERRA NARR tropHgt; do
        file=timeseries_${tropo}.h5;                test -f $file && $view $file $opt >> $log_file
        file=timeseries_${tropo}_demErr.h5;         test -f $file && $view $file $opt >> $log_file
        file=timeseries_${tropo}_ramp.h5;           test -f $file && $view $file $opt >> $log_file
        file=timeseries_${tropo}_ramp_demErr.h7;    test -f $file && $view $file $opt >> $log_file
    done

    #w/o trop delay correction
    file=timeseries_ramp.h5;                        test -f $file && $view $file $opt >> $log_file
    file=timeseries_demErr_ramp.h5;                 test -f $file && $view $file $opt >> $log_file
fi



## Time-series Residual files
opt='--mask '$tmCoh_mask' --noaxis -u cm --wrap --wrap-range -10 10'
if [ $plot_timeseries_residual -eq 1 ]; then
    # residual time-series
    file=timeseriesResidual.h5;                     test -f $file && $view $file $opt >> $log_file

    # long-wavelength ramp removed
    file=timeseriesResidual_ramp.h5;                test -f $file && $view $file $opt >> $log_file
fi


## Plot Ionosphere TEC time series
opt='--mask '$tmCoh_mask' --noaxis -u cm --vlim 0 10'
opt2='--mask '$tmCoh_mask' --noaxis -u cm'
if [ $plot_TEC_ts -eq 1 ]; then
    # TEC time-series
    file=inputs/IGS_TEC_ref.h5;                     test -f $file && $view $file $opt >> $log_file
    file=inputs/IGS_TEC.h5;                         test -f $file && $view $file $opt2 >> $log_file
fi


## Geo coordinates for UNAVCO Time-series InSAR Archive Product
if [ $plot_geocoded_data -eq 1 ]; then
    file=./geo/geo_maskTempCoh.h5;                  test -f $file && $view $file -c gray  >> $log_file
    file=./geo/geo_temporalCoherence.h5;            test -f $file && $view $file -c gray  >> $log_file
    file=./geo/geo_velocity.h5;                     test -f $file && $view $file velocity >> $log_file
    file=./geo/geo_timeseries_ECMWF_demErr_ramp.h5; test -f $file && $view $file --noaxis >> $log_file
    file=./geo/geo_timeseries_ECMWF_demErr.h5;      test -f $file && $view $file --noaxis >> $log_file
    file=./geo/geo_timeseries_demErr_ramp.h5;       test -f $file && $view $file --noaxis >> $log_file
    file=./geo/geo_timeseries_demErr.h5;            test -f $file && $view $file --noaxis >> $log_file
fi


if [ $plot_the_rest -eq 1 ]; then
    for tropo in ERA5 ECMWF MERRA NARR; do
        file=velocity${tropo}.h5;   test -f $file && $view $file --mask no >> $log_file
    done
    file=numInvIfgram.h5;           test -f $file && $view $file --mask no >> $log_file
fi


## Move/copy picture files to pic folder
echo "Copy *.txt files into ./pic folder."
cp *.txt pic/
echo "Move *.png/pdf/kmz files into ./pic folder."
mv *.png *.pdf *.kmz ./geo/*.kmz pic/

