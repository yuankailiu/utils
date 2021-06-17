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
# updated @ 2021-05-19
# --------------------------------------------------------------

refdate=20160203
refx=470
refy=735

tsfile=timeseries_SET_ERA5_demErr.h5
tsfile_rl=timeseries_SET_ERA5_demErr_rampl.h5
tsfile_rq=timeseries_SET_ERA5_demErr_rampq.h5
weather=inputs/ERA5.h5
setmodel=inputs/SET.h5
ionotec=inputs/TEC.h5
templ=smallbaselineApp.cfg
#startdate="20160101"
periods="1.0 0.5"
steps=""
WLS=""


# common arguments (template + ref-yx + ref-date)
arg_common="-t ${templ} --ref-yx ${refy} ${refx} --ref-date ${refdate}"
explog="--exp 20160910 90 --exp 20171014 60.4 300.4 --log 20171026 200.7 --log 20191010 60 300"

# output path
veloDir=velocity_out

printf "\n"
printf "Velocity estimation using temporal functions \n"
printf "Reference date: $refdate    Ref_X: $refx    Ref_Y: $refy \n"
printf "Periods: $periods   Steps: $steps   WLS: $WLS \n"
printf "velocity output path: $veloDir \n"


# Run timeseries2velocity.py
# Output files naming: Rl=linear deramp; Rq=quadratic deramp;  P=periodic; S=step; W=weighted
if true; then
    mkdir -p ${veloDir}

    #timeseries2velocity.py ${tsfile}   ${arg_common}                        -o ${veloDir}/velocity.h5
    #timeseries2velocity.py ${tsfile}   ${arg_common} --periodic ${periods}  -o ${veloDir}/velocity_P.h5

    #timeseries2velocity.py ${tsfile_rl} ${arg_common}                       -o ${veloDir}/velocity_Rl.h5
    #timeseries2velocity.py ${tsfile_rl} ${arg_common} --periodic ${periods} -o ${veloDir}/velocity_RlP.h5

    timeseries2velocity.py ${tsfile_rl} ${arg_common} --periodic ${periods} ${explog} -o ${veloDir}/velocity_RlP_explog.h5


    #timeseries2velocity.py ${tsfile_rq} ${arg_common}                       -o ${veloDir}/velocity_Rq.h5
    #timeseries2velocity.py ${tsfile_rq} ${arg_common} --periodic ${periods} -o ${veloDir}/velocity_RqP.h5

    #timeseries2velocity.py ${weather}  ${arg_common}                        -o ${veloDir}/velocityERA5.h5
    #timeseries2velocity.py ${weather}  ${arg_common} --periodic ${periods}  -o ${veloDir}/velocityERA5_P.h5

    #timeseries2velocity.py ${setmodel} ${arg_common}                        -o ${veloDir}/velocitySET.h5
    #timeseries2velocity.py ${setmodel} ${arg_common} --periodic ${periods}  -o ${veloDir}/velocitySET_P.h5

    ##timeseries2velocity.py ${ionotec}  ${arg_common}                        -o ${veloDir}/velocityTEC.h5
    ##timeseries2velocity.py ${ionotec}  ${arg_common} --periodic ${periods}  -o ${veloDir}/velocityTEC_p.h5
fi
