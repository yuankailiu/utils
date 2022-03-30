#! /bin/sh
###############################################################
# Plot Results from Routine Workflow with smallbaselineApp.py
# Author: Zhang Yunjun, 2017-07-23
# Latest update: 2021-03-08 > 2021-07-08
###############################################################
# Update the date above to enable auto copyover/overwrite


## Change to 0 if you do not want to re-plot loaded dataset again
plot_network=1
plot_key_files=1
plot_loaded_data=1
plot_loaded_data_aux=1
plot_timeseries=1
plot_TEC_ts=0
plot_geocoded_data=1
plot_the_rest=1
move_and_copy=1


# =============== Read defined variables from json file ==================
my_json="./params.json"
declare -A dic
while IFS="=" read -r key value
do
    dic[$key]="$value"
done < <(jq -r 'to_entries|map("\(.key)=\(.value)")|.[]' $my_json)
# =============== ===================================== ==================
# Get parameters
proc_home="${dic['proc_home']}"
log_file="${proc_home}/${dic['plotnetwork_log']}"
water_mask="${proc_home}/${dic['water_mask']}"
tmCoh_mask="${proc_home}/${dic['tcoh_mask']}"
dem_file='./inputs/geometryRadar.h5'
if [ ! -f $dem_file ]; then
    dem_file="${proc_home}/${dic['geom_file']}"
fi
vel_model="${proc_home}/${dic['velo_model']}"
n_worker=${dic['num_worker']}
rms_cutoff=${dic['rms_cutoff']}
picdir="${proc_home}/${dic['mintpy_pic']}"



## Log File
touch $log_file
printf "\n\n\n\n\n" >> $log_file
echo "########################  ./plot_smallbaselineApp.sh  ########################" >> $log_file
date >> $log_file
echo "##############################################################################" >> $log_file
#use "echo 'yoyoyo' | tee -a log" to output message to both screen and file.
#use "echo 'yoyoyo' >> log" to output message to file only.

## Create pic folder
if [ ! -d $picdir ]; then
    echo 'Create ./pic folder'
    mkdir $picdir
fi


## common view.py option for all files
view='view.py --nodisplay --dpi 300 --nrows 3 --ncols 8 --update --ram 8.0 '


## plot the ifgram network
plotnet='plot_network.py --nodisplay --figsize 10 6 --dpi 300 '
opt=" --vlim 0.2 1.0  --lw 2 --ms 10 --show-kept --nosplit-cmap --mc lightgrey -c romanian_r "
#opt=" --vlim 12 365  --lw 2 --ms 10 --show-kept --nosplit-cmap --mc lightgrey -c romanian "
if [ $plot_network -eq 1 ]; then
    file=inputs/ifgramStack.h5;    test -f $file && $plotnet $file $opt >> $log_file
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
    test -f $file && h5ls $file/ionoPhase        && $view $file ionoPhase-        $opt --zero-mask --wrap        >> $log_file
    test -f $file && h5ls $file/ionoPhase        && $view $file ionoPhase-        $opt --zero-mask --vlim   -5 5 >> $log_file

    # phase-unwrapping error correction
    for dset in 'unwrapPhase_bridging' 'unwrapPhase_phaseClosure' 'unwrapPhase_bridging_phaseClosure'; do
        test -f $file && h5ls $file/$dset        && $view $file $dset-            $opt --zero-mask --vlim -15 15 >> $log_file
    done
fi


## Auxliary Files from loaded dataset
if [ $plot_loaded_data_aux -eq 1 ]; then
    file='avgPhaseVelocity.h5';   test -f $file && $view $file --mask $water_mask   >> $log_file
    file='avgSpatialCoh.h5';      test -f $file && $view $file -c gray --vlim 0 1   >> $log_file
    file='maskConnComp.h5';       test -f $file && $view $file -c gray --vlim 0 1   >> $log_file
fi


## Time-series files
opt='--mask '$tmCoh_mask' --noaxis -u cm --wrap --wrap-range -10 10 '
if [ $plot_timeseries -eq 1 ]; then
    file='timeseries.h5'; test -f $file && $view $file $opt >> $log_file
    find . -name 'timeseries_*.h5' -exec   $view {}    $opt >> $log_file \;
fi


## Plot ionoPhase and TEC time series
opt1='--mask '$tmCoh_mask' --noaxis -u cm --vlim 0 10'
opt2='--mask '$tmCoh_mask' --noaxis -u cm'
if [ $plot_TEC_ts -eq 1 ]; then
    # TEC time-series
    file=inputs/timeseriesIon.h5;               test -f $file && $view $file $opt1 >> $log_file
    file=inputs/TEC_ref.h5;                     test -f $file && $view $file $opt1 >> $log_file
    file=inputs/TEC.h5;                         test -f $file && $view $file $opt2 >> $log_file
fi


## Geo coordinates for UNAVCO Time-series InSAR Archive Product
if [ $plot_geocoded_data -eq 1 ]; then
    file='./geo/geo_maskTempCoh.h5';          test -f $file && $view $file -c gray  >> $log_file
    file='./geo/geo_temporalCoherence.h5';    test -f $file && $view $file -c gray  >> $log_file
    file='./geo/geo_velocity.h5';             test -f $file && $view $file velocity >> $log_file
    find . -name './geo/geo_timeseries_*.h5' -exec             $view {}    $opt     >> $log_file \;
fi


if [ $plot_the_rest -eq 1 ]; then
    for tropo in 'ERA5' 'ERAI' 'ECMWF' 'MERRA' 'NARR'; do
        file='velocity'${tropo}'.h5';  test -f $file && $view $file --mask no >> $log_file
    done
    file='numInvIfgram.h5';            test -f $file && $view $file --mask no >> $log_file
fi


## Move/copy picture files to pic folder
if [ $move_and_copy -eq 1 ]; then
    echo "Copy *.txt files into ./pic folder."
    cp *.txt $picdir
    echo "Move *.png/pdf/kmz files into ./pic folder."
    mv *.png *.pdf *.kmz ./geo/*.kmz $picdir
fi
