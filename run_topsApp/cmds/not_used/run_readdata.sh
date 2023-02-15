#!/bin/bash
# Run topsApp processing Stage 1, 2, and 3: 'startup' to 'geocode'
# Requires GPU modules for steps 'topo' and 'fineoffset'
# ykliu @ Apr22 2021

use_steps=(
'startup'
'preprocess'
)


###############################################
start=`date +%s`
logfile=cmd_readdata.log

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
    if [ "$step" = "geocode" ]; then
	xmlfile='../example/topsApp_geocode.xml'
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
