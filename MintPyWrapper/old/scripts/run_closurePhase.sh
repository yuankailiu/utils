#! /bin/sh

CONFIG='AqabaSen*.cfg'


## 0. Run masking ifgramStack and closure_phase_bias
mask.py inputs/ifgramStack.h5 -m waterMask.h5 --fill 0 -o inputs/ifgramStack_msk.h5

time closure_phase_bias.py -i inputs/ifgramStack_msk.h5 --nl 3 --bw 3 -a mask --wm waterMask.h5 --ram 24 --num-worker 8 2>&1 | tee -a closure_phase_bias.log
time closure_phase_bias.py -i inputs/ifgramStack_msk.h5 --nl 3 --bw 3 -a quick_estimate --wm waterMask.h5 --ram 24 --num-worker 8 2>&1 | tee -a closure_phase_bias.log
time closure_phase_bias.py -i inputs/ifgramStack_msk.h5 --nl 3 --bw 3 -a estimate --wm waterMask.h5 --ram 24 --num-worker 8 2>&1 | tee -a closure_phase_bias.log


echo "Normal finish."
