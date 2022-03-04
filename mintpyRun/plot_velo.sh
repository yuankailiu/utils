#! /bin/bash

# Using `view.py` to plot all the velocity outputs
# YKL @ 2021-05-19

picdir=./pic_velo

xmin=34
xmax=37.5
ymin=27.7
ymax=31.7

refx=513
refy=998
refla=29.54435
reflo=36.08216

tmCoh_mask='../maskTempCoh_high2.h5'
water_mask='../waterMask.h5'
dem_file='../inputs/geometryGeo.h5'
dem_file='/home/ykliu/marmot-nobak/aqaba/broad_dem/dem_resamp_d021/d021.dem'

view='view.py --nodisplay --dpi 300 -c RdYlBu_r --update'
roi=" --sub-lon $xmin $xmax --sub-lat $ymin $ymax "
#roi=' '
opt=" --dem $dem_file --alpha 0.6 --dem-nocontour --shade-exag 0.05 --mask $tmCoh_mask -u mm $roi --ref-lalo ${refla} ${reflo} "

v1=' --vlim -5  5 '     # velocity field [mm/yr]
v2=' --vlim -5  5   '   # velocity field after deramp [mm/yr]
v3=' --vlim  0  1.0 '   # velocity STD [mm/yr]
v4=' --vlim  0  16  '   # velocity periodic amplitude [mm/yr]
v5=' --vlim -0.2 0.2 '  # SET  field  [mm/yr]
v6=' --vlim -2  2   '   # iono field  [mm/yr]


# I will do something like this for all the velocity files and datasets:
# view.py velocity_P.h5 annualAmp -m ../maskTempCoh.h5 -d ../inputs/geometryGeo.h5 --alpha 0.5 --dem-nocontour --shade-exag 0.05 --figtitle velocty_P_annAmp --sub-lon 34 37.5 --sub-lat 27.5 31.7

f='velocity1.h5 velocity';              $view $f $opt $v1 -o vel1.png           --figtitle vel
f='velocity1.h5 velocityStd';           $view $f $opt $v3 -o vel1Std.png        --figtitle velStd
f='velocity2.h5 velocity';              $view $f $opt $v2 -o vel2.png           --figtitle vel2
f='velocity2.h5 velocityStd';           $view $f $opt $v3 -o vel2Std.png        --figtitle vel2Std

f='velocity1lr.h5 velocity';            $view $f $opt $v2 -o vel1_lr.png        --figtitle vel1_lr
f='velocity1lr.h5 velocityStd';         $view $f $opt $v3 -o vel1_lrStd.png     --figtitle vel1_lrStd
f='velocity2lr.h5 velocity';            $view $f $opt $v2 -o vel2_lr.png        --figtitle vel2_lr
f='velocity2lr.h5 velocityStd';         $view $f $opt $v3 -o vel2_lrStd.png     --figtitle vel2_lrStd

f='velocity1qr.h5 velocity';            $view $f $opt $v2 -o vel1_qr.png        --figtitle vel1_qr
f='velocity1qr.h5 velocityStd';         $view $f $opt $v3 -o vel1_qrStd.png     --figtitle vel1_qrStd
f='velocity2qr.h5 velocity';            $view $f $opt $v2 -o vel2_qr.png        --figtitle vel2_qr
f='velocity2qr.h5 velocityStd';         $view $f $opt $v3 -o vel2_qrStd.png     --figtitle vel2_qrStd

f='velocityERA5.h5 velocity';           $view $f $opt $v6 -o velERA5.png        --figtitle velERA5
f='velocityERA5.h5 velocityStd';        $view $f $opt $v3 -o velERA5Std.png     --figtitle velERA5Std
f='velocityERA5_vl.h5 velocity';        $view $f $opt $v6 -o velERA5_vl.png     --figtitle velERA5_vl
f='velocityERA5_vl.h5 velocityStd';     $view $f $opt $v3 -o velERA5_vlStd.png  --figtitle velERA5_vlStd

f='velocitySET.h5 velocity';            $view $f $opt $v5 -o velSET.png         --figtitle velSET
f='velocitySET.h5 velocityStd';         $view $f $opt $v3 -o velSETStd.png      --figtitle velSETStd
f='velocitySET_vl.h5 velocity';         $view $f $opt $v5 -o velSET_vl.png      --figtitle velSET_vl
f='velocitySET_vl.h5 velocityStd';      $view $f $opt $v3 -o velSET_vlStd.png   --figtitle velSET_vlStd

f='velocityIon.h5 velocity';            $view $f $opt $v6 -o velIon.png         --figtitle velIon
f='velocityIon.h5 velocityStd';         $view $f $opt $v3 -o velIonStd.png      --figtitle velIonStd
f='velocityIon_vl.h5 velocity';         $view $f $opt $v6 -o velIon_vl.png      --figtitle velIon_vl
f='velocityIon_vl.h5 velocityStd';      $view $f $opt $v3 -o velIon_vlStd.png   --figtitle velIon_vlStd



## Move/copy picture files to pic folder
mkdir -p ${picdir}
echo "Move *.png/pdf/kmz files into ${picdir} folder."
mv *.png *.pdf *.kmz ${picdir}
