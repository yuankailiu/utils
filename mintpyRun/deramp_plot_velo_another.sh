#! /bin/bash

# Using `view.py` to plot all the velocity outputs
# YKL @ 2021-05-19

picdir=./pic_velo_ramp

xmin=34
xmax=37.5
ymin=27.7
ymax=31.7

refx=513
refy=998
refla=29.54435
reflo=36.08216

tmCoh_mask='../../maskTempCoh_high.h5'
water_mask='../../waterMask.h5'
dem_file='../../inputs/geometryGeo.h5'

view='view.py --nodisplay --dpi 300 -c RdYlBu_r --update'
roi=" --sub-lon $xmin $xmax --sub-lat $ymin $ymax "
#roi=' '
opt=" --dem $dem_file --alpha 0.6 --dem-nocontour --shade-exag 0.05 --mask $tmCoh_mask -u mm $roi --ref-lalo ${refla} ${reflo} "

v1=' --vlim -13  13  '   # velocity field [mm/yr]
v2=' --vlim -3   3   '   # velocity field [mm/yr]
v3=' --vlim -5   5   '   # velocity field [mm/yr]

## First do deramp and save the ramp files
ramp_type=linear
remove_ramp.py velocity1.h5   -s ${ramp_type} -m ${tmCoh_mask} --save-ramp-coeff -o velocity1_noramp.h5
remove_ramp.py velocity2.h5   -s ${ramp_type} -m ${tmCoh_mask} --save-ramp-coeff -o velocity2_noramp.h5
remove_ramp.py velocityIon.h5 -s ${ramp_type} -m ${tmCoh_mask} --save-ramp-coeff -o velocityIon_noramp.h5

diff.py velocity1.h5   velocity1_noramp.h5   -o velocity1_ramp.h5
diff.py velocity2.h5   velocity2_noramp.h5   -o velocity2_ramp.h5
diff.py velocityIon.h5 velocityIon_noramp.h5 -o velocityIon_ramp.h5


## Now plot the files
f='velocity1_ramp.h5     velocity';   $view $f $opt $v1 -o vel1_ramp.png          --figtitle vel1_ramp
f='velocity1_noramp.h5   velocity';   $view $f $opt $v2 -o vel1_noramp.png        --figtitle vel1_noramp
f='velocity2_ramp.h5     velocity';   $view $f $opt $v3 -o vel2_ramp.png          --figtitle vel2_ramp
f='velocity2_noramp.h5   velocity';   $view $f $opt $v2 -o vel2_noramp.png        --figtitle vel2_noramp
f='velocityIon_ramp.h5   velocity';   $view $f $opt $v1 -o velIon_ramp.png        --figtitle velIon_ramp
f='velocityIon_noramp.h5 velocity';   $view $f $opt $v2 -o velIon_noramp.png      --figtitle velIon_noramp

## these need to be changed to first use add.py diff.py, then rename it to *pred
## velocityIon_multiply1.383_plus0.001011.h5
slope=1.383
intcp=0.001011
image_math.py velocityIon.h5         '*'   $slope   -o  velocityIon_tmp.h5
image_math.py velocityIon_tmp.h5     '+'   $intcp   -o  velocityIon_scaled.h5
rm -rf velocityIon_tmp.h5
diff.py        velocity1.h5       velocityIon_scaled.h5     -o  velocity2_pred.h5
remove_ramp.py velocity2_pred.h5   -s ${ramp_type} -m ${tmCoh_mask} --save-ramp-coeff -o velocity2_pred_noramp.h5
diff.py        velocity2_pred.h5  velocity2_pred_noramp.h5  -o  velocity2_pred_ramp.h5

f='velocity2_pred.h5        velocity';   $view $f $opt $v3 -o vel2_pred.png          --figtitle vel2_pred
f='velocity2_pred_ramp.h5   velocity';   $view $f $opt $v2 -o vel2_pred_ramp.png     --figtitle vel2_pred_ramp
f='velocity2_pred_noramp.h5 velocity';   $view $f $opt $v2 -o vel2_pred_noramp.png   --figtitle vel2_pred_noramp


mkdir -p $picdir
mv *.png $picdir
