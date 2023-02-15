steps1=('startup'
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

steps2=(
'fineresamp'
'ion'
'burstifg'
'mergebursts'
'filter'
'unwrap'
'unwrap2stage'
)

steps3=(
'geocode'
'denseoffsets'
'filteroffsets'
'geocodeoffsets'
)

###############################################

for ((i=0;i<${#steps1[@]};i++)); do
    step=${steps1[i]}
    printf "#############################################\n" >> runstep.log
    printf "#################### STEP ${i}: ${step} \n" >> runstep.log
    printf "#############################################\n" >> runstep.log
    topsApp.py ../example/topsApp.xml --dostep=${step} 2>&1 | tee -a runstep.log
    printf "\n\n" >> runstep.log
done
