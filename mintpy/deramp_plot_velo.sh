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

tmCoh_mask='../../maskTempCoh_high2.h5'
water_mask='../../waterMask.h5'
dem_file='../../inputs/geometryGeo.h5'
dem_file='/home/ykliu/marmot-nobak/aqaba/broad_dem/dem_resamp_d021/d021.dem'

view='view.py --nodisplay --dpi 300 -c RdYlBu_r --update'
roi=" --sub-lon $xmin $xmax --sub-lat $ymin $ymax "
#roi=' '
opt=" --dem $dem_file --alpha 0.6 --dem-nocontour --shade-exag 0.05 --mask $tmCoh_mask -u mm $roi --ref-lalo ${refla} ${reflo} "

v1=' --vlim -4  4  '   # velocity field [mm/yr]
v2=' --vlim -2  2   '   # velocity field [mm/yr]
v3=' --vlim -4  4 '   # velocity field [mm/yr]

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


mkdir -p $picdir
mv *.png $picdir
