#!/bin/bash

module load GMT

#demfile="Sinai_GMRT_61m.grd"
demfile="RedSeaSaudi_GMRT_490m.grd"
psfile="aqaba.ps"
acpt="aqaba.cpt"

gmt set MAP_FRAME_TYPE plain

gmt makecpt -Cmby.cpt -T-4167/2604/20 > $acpt 
#gmt makecpt -CGMT_globe.cpt -T-2000/2000/5 > $acpt 

#gmt grdgradient $demfile -Ghillshade-grad.nc -A345 -Ne0.6 -V
#gmt grdhisteq hillshade-grad.nc -Ghillshade-hist.nc -N -V
#gmt grdmath hillshade-hist.nc 5.5 DIV = hillshade-int.nc

#grdimage $demfile -R/32.6/27.5/35.8/31.2r -JM5i -Xc0 -Yc0 -Ba -BWSne+t"Gulf of Aqaba" -C$acpt -P -K > $psfile

grdimage $demfile -R/31/22/43/33r -JM -Xc0 -Yc0 -Ba -BWSne+t"Northern Red Sea" -C$acpt -Ihillshade-int.nc -P -K > $psfile

#pscoast -R -J -Ir -N1/0.5p -Df -W,black/0.1p -O -K >> $psfile

gmt pstext city_redsea.txt -R -J -F+f8p,Helvetica-Bold,black -O -K >> $psfile
gmt pstext region_redsea.txt -R -J -F+f+a -O -K >> $psfile

psscale -D5.4i/2.5i/3i/0.5c -C$acpt -Bxaf+l"Elevation" -By -O -K >> $psfile

psxy AR_rigid_obs.dat -R -J -Sc.08/0/0/0 -W0.1 -G255 -O -K >> $psfile

awk '{print $1, $2, $8 }' AR_rigid_obs.dat | gmt pstext -R -J -F+f7,Helvetica+jLB -O -K >> $psfile

#awk '{print $1, $2, $3, $4, $5, $6, $7, $8 }' AR_rigid_obs.dat | \
#    gmt psvelo -R -J -Se.15/0.95/0 -G0 -A0.03/0.16/0.07 -W -O -K >> $psfile

#gmt psvelo << END -R -J -Se.15/0.95/0 -G0 -A0.03/0.16/0.08 -W -O -K >> $psfile
#32.2 22.8 10 0 0.708 0.708 0 Example
#END

#gmt pstext -R -J -F+f8,Helvetica-Bold -O -K << END >> $psfile
#32.6    23.0    10 \261 1 mm/yr
#END

gmt psxy track087.txt -R -J -W1,white -O -K >> $psfile
gmt psxy track160.txt -R -J -W1,white -O -K >> $psfile

gmt pstext -R -J -F+f10,Helvetica-Bold,white+a11 -O << END >> $psfile
33.6    25.35   A.160
35.6    25.3    A.087
END

gmt psconvert $psfile -A1.0c+m1.0c+pthick -Tf

#gv $psfile
evince aqaba.pdf
