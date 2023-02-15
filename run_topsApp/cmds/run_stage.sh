#!/bin/bash
# Run topsApp steps
# ykliu @ Jul21 2021

if [[ $2 == runall ]]; then
    logfile=cmd_runall.log
    use_steps=(
        'startup'
        'preprocess'
        'computeBaselines'
        'verifyDEM'
        'topo'
        'subsetoverlaps'
        'coarseoffsets'
        'coarseresamp'
        'overlapifg'
        'prepesd'
        'esd'
        'rangecoreg'
        'fineoffsets'
        'fineresamp'
        'ion'
        'burstifg'
        'mergebursts'
        'filter'
        'unwrap'
        'unwrap2stage'
        'geocode'
        'denseoffsets'
        'filteroffsets'
        'geocodeoffsets'
    )
elif [[ $2 == run2geo ]]; then
    logfile=cmd_run2geo.log
    use_steps=(
        'startup'
        'preprocess'
        'computeBaselines'
        'verifyDEM'
        'topo'
        'subsetoverlaps'
        'coarseoffsets'
        'coarseresamp'
        'overlapifg'
        'prepesd'
        'esd'
        'rangecoreg'
        'fineoffsets'
        'fineresamp'
        'ion'
        'burstifg'
        'mergebursts'
        'filter'
        'unwrap'
        'unwrap2stage'
        'geocode'
    )
elif [[ $2 == mlk_filt ]]; then
    logfile=cmd_mlkfilt.log
    use_steps=(
        'mergebursts'
        'filter'
        'unwrap'
        'unwrap2stage'
        'geocode'
    )
elif [[ $2 == geos ]]; then
    logfile=cmd_rungeos.log
    use_steps=(
        'geocode'
        'geocodeoffsets'
    )
elif [[ $2 == stage_0 ]]; then
    logfile=cmd_0_preproc.log
    use_steps=(
        'startup'
        'preprocess'
    )
elif [[ $2 == stage_1 ]]; then # this stage can turn on GPU modules
    logfile=cmd_1_fineoffset.log
    use_steps=(
        'computeBaselines'
        'verifyDEM'
        'topo'              # can use GPU
        'subsetoverlaps'
        'coarseoffsets'
        'coarseresamp'
        'overlapifg'
        'prepesd'
        'esd'
        'rangecoreg'
        'fineoffsets'       # can use GPU
    )
elif [[ $2 == stage_2 ]]; then
    logfile=cmd_2_merge.log
    use_steps=(
        'fineresamp'
        'ion'               # can use GPU
        'burstifg'
        'mergebursts'
    )
elif [[ $2 == stage_3 || $2 == filt ]]; then
    logfile=cmd_3_unwrap.log
    use_steps=(
        'filter'
        'unwrap'
        'unwrap2stage'
        'geocode'
    )
elif [[ $2 == stage_4 || $2 == geo ]]; then
    logfile=cmd_4_geocode.log
    use_steps=(
        'geocode'
    )
elif [[ $2 == stage_5 || $2 == offsets ]]; then
    logfile=cmd_5_denseoffset.log
    use_steps=(
        'denseoffsets'
        'filteroffsets'
        'geocodeoffsets'
    )
elif [[ $2 == custom ]]; then
    logfile=cmd_custom.log
    use_steps=(
        'put_your_custom_steps'
    )
else
    printf "No steps found to match the user input ${2}, exit...\n\n"
    exit
fi



################ Set time and chdir ####################
start=`date +%s`

IFS='-' read -ra array <<< "$1"
MASTER=${array[0]}
SLAVE=${array[1]}

cd ${MASTER}-${SLAVE}


############### Create the log file ####################

rm -rf ${logfile}

printf "####=========================================####\n" >> ${logfile}
printf "####=========================================####\n" >> ${logfile}
printf "    Starttime: `date` \n" >> ${logfile}
printf "####=========================================####\n" >> ${logfile}
printf "####=========================================####\n\n\n\n" >> ${logfile}


for ((i=0;i<${#use_steps[@]};i++)); do
    step=${use_steps[i]}
    if [[ $step == geocode ]]; then
	xmlfile='../example/topsApp_geocode.xml'
    elif [[ $step == geocodeoffsets ]]; then
	xmlfile='../example/topsApp_geocodeDense.xml'
    else
	xmlfile='../example/topsApp.xml'
    fi

    printf "##########################################################################################\n" >> ${logfile}
    printf "####     RUNSTEP ${i}: ${step}        \n" >> ${logfile}
    printf "##########################################################################################\n" >> ${logfile}
    printf "  --> using XML file: ${xmlfile} \n" >> ${logfile}
    (time topsApp.py ${xmlfile} --dostep=${step}) 2>&1 | tee -a ${logfile}
    printf "\n\n\n" >> ${logfile}
done


## Record total elasped time in the logfile
end=`date +%s`
runtime=$( echo "$end - $start" | bc -l )
Elaspsed="Total elapsed: $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
printf "\n\n\n####=========================================####\n" >> ${logfile}
printf "####=========================================####\n" >> ${logfile}
printf "    ${Elaspsed} \n" >> ${logfile}
printf "    Endtime: `date` \n" >> ${logfile}
printf "####=========================================####\n" >> ${logfile}
printf "####=========================================####\n" >> ${logfile}

cd ../

################ The End ####################
