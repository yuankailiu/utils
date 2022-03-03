#! /bin/bash

mask.py velocity1.h5        -m ../maskTempCoh095.h5     -o velocity1_msk.h5
mask.py velocity2.h5        -m ../maskTempCoh095.h5     -o velocity2_msk.h5
mask.py velocity1lr.h5      -m ../maskTempCoh095.h5     -o velocity1lr_msk.h5
mask.py velocity1qr.h5      -m ../maskTempCoh095.h5     -o velocity1qr_msk.h5
mask.py velocity2lr.h5      -m ../maskTempCoh095.h5     -o velocity2lr_msk.h5
mask.py velocity2qr.h5      -m ../maskTempCoh095.h5     -o velocity2qr_msk.h5
mask.py velocitySET_vl.h5   -m ../maskTempCoh095.h5     -o velocitySET_vl_msk.h5
mask.py velocitySET.h5      -m ../maskTempCoh095.h5     -o velocitySET_msk.h5
mask.py velocityIon_vl.h5   -m ../maskTempCoh095.h5     -o velocityIon_vl_msk.h5
mask.py velocityIon.h5      -m ../maskTempCoh095.h5     -o velocityIon_msk.h5
