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
)


###############################################
logfile=cmd_offset.log

IFS='-' read -r -a array <<< "$1"
MASTER=${array[0]}
SLAVE=${array[1]}

cd ${MASTER}-${SLAVE}

rm -rf ${logfile}

for ((i=0;i<${#use_steps[@]};i++)); do
    step=${use_steps[i]}
    printf "##########################################################################################\n" >> ${logfile}
    printf "####################          STEP ${i}: ${step}          ####################\n" >> ${logfile}
    printf "##########################################################################################\n" >> ${logfile}
    (time topsApp.py ../example/topsApp.xml --dostep=${step}) 2>&1 | tee -a ${logfile}
    printf "\n\n\n" >> ${logfile}
done

cd ../
