#! /bin/bash
# --------------------------------------------------------------
# Run timeseries2velocity.py using:
#       1. whether or not periodic models
#       2. whether or not start_date
#
# Save output velocity files into different directories
#
# + Need to reference your ERA5.h5 and SET.h5
# YKL @ 2020-08-17
# updated @ 2021-09-19
# --------------------------------------------------------------

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
config=smallbaselineApp.cfg
veloDir="${proc_home}/${dic['velo_dir']}"
refdate=${dic['ref_date']}
refla=${dic['ref_lat']}
reflo=${dic['ref_lon']}
velo_model=${dic['velo_model']}


# print information to terminal
printf "\n"
printf "Velocity estimation using temporal functions \n"
printf "Reference date: $refdate"
printf "Ref_lat, Ref_lon: $refla $reflo \n"
printf "Velocity models: $velo_model \n"
printf "velocity output path: $veloDir \n"
mkdir -p ${veloDir}

# Define file names to be operated
ts1=timeseries_SET_ERA5_demErr.h5
ts2=timeseries_SET_ERA5_Ion_demErr.h5
ts1lr=timeseries_SET_ERA5_demErr_rampl.h5
ts1qr=timeseries_SET_ERA5_demErr_rampq.h5
ts2lr=timeseries_SET_ERA5_Ion_demErr_rampl.h5
ts2qr=timeseries_SET_ERA5_Ion_demErr_rampq.h5
TRO=inputs/ERA5.h5
SET=inputs/SET.h5
ION=inputs/timeseriesIon.h5
TEC=inputs/TEC.h5

# Run timeseries2velocity.py
# Output files naming:
#       1  = velocity w/o iono correction
#       2  = velocity w/  iono correction
#       lr = apply linear    deramp
#       qr = apply quadratic deramp
#       vl = with only linear velo fit
ts2velo=" timeseries2velocity.py -t ${config} --ref-lalo ${refla} ${reflo} --ref-date ${refdate} "

$ts2velo  ${ts1}    ${velo_model}  -o ${veloDir}/velocity1.h5
$ts2velo  ${ts2}    ${velo_model}  -o ${veloDir}/velocity2.h5

$ts2velo  ${ts1lr}  ${velo_model}  -o ${veloDir}/velocity1lr.h5
$ts2velo  ${ts1qr}  ${velo_model}  -o ${veloDir}/velocity1qr.h5

$ts2velo  ${ts2lr}  ${velo_model}  -o ${veloDir}/velocity2lr.h5
$ts2velo  ${ts2qr}  ${velo_model}  -o ${veloDir}/velocity2qr.h5

$ts2velo  ${TRO}                   -o ${veloDir}/velocityERA5_vl.h5
$ts2velo  ${TRO}    ${velo_model}  -o ${veloDir}/velocityERA5.h5

$ts2velo  ${SET}                   -o ${veloDir}/velocitySET_vl.h5
$ts2velo  ${SET}    ${velo_model}  -o ${veloDir}/velocitySET.h5

$ts2velo  ${ION}                   -o ${veloDir}/velocityIon_vl.h5
$ts2velo  ${ION}    ${velo_model}  -o ${veloDir}/velocityIon.h5
