#!/bin/bash
# Run topsApp processing: 'filt' to 'unwrap2stage'
# This stage does not use GPU modules, run with parallelization

use_steps=(
'filter'
'unwrap'
'unwrap2stage'
)


###############################################
start=`date +%s`
logfile=cmd_2_filt2unw.log

IFS='-' read -r -a array <<< "$1"
MASTER=${array[0]}
SLAVE=${array[1]}

cd ${MASTER}-${SLAVE}

rm -rf ${logfile}

printf "####=========================================####\n" >> ${logfile}
printf "####=========================================####\n" >> ${logfile}
printf "    Starttime: `date` \n" >> ${logfile}
printf "####=========================================####\n" >> ${logfile}
printf "####=========================================####\n\n\n\n" >> ${logfile}


for ((i=0;i<${#use_steps[@]};i++)); do
    step=${use_steps[i]}
    echo $stepstr
    printf "##########################################################################################\n" >> ${logfile}
    printf "####     RUNSTEP ${i}: ${step}        \n" >> ${logfile}
    printf "##########################################################################################\n" >> ${logfile}
    (time topsApp.py ../example/topsApp.xml --dostep=${step}) 2>&1 | tee -a ${logfile}
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
