#! /bin/sh

CONFIG='AqabaSen*.cfg'
ORIG='ifgInv_orig'

thres=0.3
OUT=ifgInv_msk${thres}

mkdir $ORIG
mkdir $OUT

#--------------------------------
copylist=(
${CONFIG}
'smallbaselineApp.cfg'
'exclude_date.txt'
'reference_date.txt'
)

movelist=(
'avgPhaseVelocity.h'
'maskTempCoh*.h5'
'maskPoly*.h5'
'numInvIfgram.h5'
'temporalCoherence.h5'
'demErr.h5'
'rms_timeseriesResidual_ramp.pdf'
'rms_timeseriesResidual_ramp.txt'
'timeseriesDecorCov.h5'
'timeseries*.h5'
'velo*'
'cmd_ifgram_inversion.log'
'run_3_corrections.log'
'run_4_velocity.log'
'run_5_velocityPlot.log'
)
#--------------------------------


## 0. Run masking ifgramStack
mask.py inputs/ifgramStack.h5 -m waterMask.h5 --fill 0 -o inputs/ifgramStack_msk.h5

## 1. Copy the original timeseries and related files
for file in ${copylist[@]}; do
    cp $file $ORIG
done
for file in ${movelist[@]}; do
    mv $file $ORIG
done
mv $ORIG/timeseriesIon.h5 .


## 2. Re-do the inversion with coherence masking
time ifgram_inversion.py ./inputs/ifgramStack_msk.h5 -w var -m waterMask.h5 --mask-dset coherence --mask-thres $thres --cluster local --num-worker 8 --ram 24 2>&1 | tee cmd_ifgram_inversion.log

generate_mask.py temporalCoherence.h5 -m 0.9 -o maskTempCoh_0.9.h5 --update

time bash run_3_corrections 2>&1 | tee run_3_corrections.log

time bash run_4_velocity 2>&1 | tee run_4_velocity.log

time bash run_5_velocityPlot 2>&1 | tee run_5_velocityPlot.log


## 3. Store the output to a separate folder
for file in ${copylist[@]}; do
    cp $file $OUT
done
for file in ${movelist[@]}; do
    mv $file $OUT
done
mv $OUT/timeseriesIon.h5 .


## 4. Bring the original timeseries back
mv $ORIG/* .


echo "Normal finish."
