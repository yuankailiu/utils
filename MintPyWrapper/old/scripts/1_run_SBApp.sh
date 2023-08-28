#! /bin/bash
# --------------------------------------------------------------
# Run MintPy routines with self-defined steps:
#  - smallbaselineApp.py : run different steps
#
# Make sure to have the params.json file
#
# Example: (run this script, time it, write to a logfile)
#       time bash 1_run_SBApp.sh 1234 2>&1 | tee -a 1_run_SBApp.log
#
# Yuan-Kai Liu, 2022-02-07
# --------------------------------------------------------------


display_usage() {
	echo "This script must be run with steps specified (e.g., 1, 2, 1234)"
	echo -e "\nUsage: $0 [steps] \n"
	}
# if less than two arguments supplied, display usage
if [  $# -ne 1 ]
then
    display_usage
    exit 1
fi
# check whether user had supplied -h or --help . If yes display usage
if [[ ( $1 == "--help") ||  $1 == "-h" ]]
then
    display_usage
    exit 0
fi

steps=$1

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
config="${proc_home}/${dic['config']}"
ifgStack="${proc_home}/${dic['ifgs_file']}"
mask_file="${proc_home}/${dic['tcoh_mask']}"
tCoh_file="${proc_home}/${dic['tcoh_file']}"
geom_file="${proc_home}/${dic['geom_file']}"
rms1="${proc_home}/${dic['rms1_dir']}"
rms2="${proc_home}/${dic['rms2_dir']}"
threshold="${dic['tcoh_threshold']}"
vel_model="${dic['velo_model']}"
n_worker=${dic['num_worker']}
rms_cutoff=${dic['rms_cutoff']}
processor=${dic['isce_proc']}

# custom file names for output
ts1=timeseries_SET_ERA5.h5
ts2=timeseries_SET_ERA5_Ion.h5
ts1d=timeseries_SET_ERA5_demErr
ts2d=timeseries_SET_ERA5_Ion_demErr
tsRe=timeseriesResidual


printf "\n\n\n"
printf "########################################################\n"
printf "# >>> Start standard processing routines for MintPy >>> \n"
printf "########################################################\n\n"


# =============== Run the MintPy routine step by step ==================

## 1: Prepare data network
if [[ $steps == *"1"* ]]; then
    smallbaselineApp.py ${config} --dostep load_data
    if [[ $processor == "topsStack" ]]; then
        bash run_geoInputs.sh
    fi
    smallbaselineApp.py ${config} --dostep modify_network
    smallbaselineApp.py ${config} --dostep reference_point
    smallbaselineApp.py ${config} --dostep quick_overview
    bash 2_plot_SBApp.sh
fi


## 2: Network inversion
if [[ $steps == *"2"* ]]; then
    smallbaselineApp.py ${config} --dostep correct_unwrap_error
    smallbaselineApp.py ${config} --dostep invert_network
    ifgram_inversion.py ${ifgStack} -t ${config} -i ionoPhase --update && mv *Ion.h5 inputs/
    generate_mask.py ${tCoh_file} -m ${threshold} -o ${mask_file}
fi


## 3: Obs noise correction
if [[ $steps == *"3"* ]]; then
    smallbaselineApp.py ${config} --dostep correct_LOD
    smallbaselineApp.py ${config} --dostep correct_SET
    smallbaselineApp.py ${config} --dostep correct_troposphere
    bash 4_change_refpoint.sh
    diff.py timeseries_SET_ERA5.h5 inputs/timeseriesIon.h5 -o timeseries_SET_ERA5_Ion.h5 --force
fi


# 4: Est topographic error
if [[ $steps == *"4"* ]]; then
    mkdir -p ${rms1}/ex ${rms2}/ex
    opt=" ${vel_model} -g ${geom_file} --num-worker ${n_worker} "
    #smallbaselineApp.py ${config} --dostep correct_topography
    dem_error.py $ts1 $opt -o ${ts1d}.h5 && mv ${tsRe}.h5 ${tsRe}_1.h5 && mv demErr.h5 demErr_1.h5
    dem_error.py $ts2 $opt -o ${ts2d}.h5 && mv ${tsRe}.h5 ${tsRe}_2.h5 && mv demErr.h5 demErr_2.h5
    #dem_error.py $ts1 $opt --ex ${rms1}/exclude_date.txt -o ${ts1d}_1_exc.h5 && mv ${tsRe}.h5 ${tsRe}_1_exc.h5 && mv demErr.h5 demErr_1_exc.h5
    #dem_error.py $ts2 $opt --ex ${rms2}/exclude_date.txt -o ${ts2d}_2_exc.h5 && mv ${tsRe}.h5 ${tsRe}_2_exc.h5 && mv demErr.h5 demErr_2_exc.h5
fi


## 5: Est RMS from other noise sources
if [[ $steps == *"5"* ]]; then
    opt=" -m ${mask_file} --cutoff $rms_cutoff --figsize 10 5 "
    #smallbaselineApp.py ${config} --dostep residual_RMS
    timeseries_rms.py ${tsRe}_1.h5 ${opt} -r no && mv *_date.txt rms*_1.* ${rms1}
    timeseries_rms.py ${tsRe}_2.h5 ${opt} -r no && mv *_date.txt rms*_2.* ${rms2}
    #timeseries_rms.py ${tsRe}_1_exc.h5 $opt -r no && mv *_date.txt *_1_exc.* ${rms1}/exc
    #timeseries_rms.py ${tsRe}_2_exc.h5 $opt -r no && mv *_date.txt *_2_exc.* ${rms2}/exc
    #timeseries_rms.py ${tsRe}_2.h5 $opt -r quadratic
fi


## 6: Deramp
if [[ $steps == *"6"* ]]; then
    #smallbaselineApp.py ${config} --dostep deramp
    remove_ramp.py ${ts1d}.h5 -m ${mask_file} -s linear    -o ${ts1d}_rampl.h5 --save-ramp-coeff
    remove_ramp.py ${ts1d}.h5 -m ${mask_file} -s quadratic -o ${ts1d}_rampq.h5 --save-ramp-coeff
    remove_ramp.py ${ts2d}.h5 -m ${mask_file} -s linear    -o ${ts2d}_rampl.h5 --save-ramp-coeff
    remove_ramp.py ${ts2d}.h5 -m ${mask_file} -s quadratic -o ${ts2d}_rampq.h5 --save-ramp-coeff
    bash 4_change_refpoint.sh
fi


## 7: Velocity models
if [[ $steps == *"7"* ]]; then
    #smallbaselineApp.py ${config} --dostep velocity
    bash ./3_fit_velocity.sh
fi


## 8: Post-result
if [[ $steps == *"8"* ]]; then
    smallbaselineApp.py   ${config} --dostep geocode
    fismallbaselineApp.py ${config} --dostep google_earth
    fismallbaselineApp.py ${config} --dostep hdfeos5
fi

echo "Complete customized MintPy routines."
