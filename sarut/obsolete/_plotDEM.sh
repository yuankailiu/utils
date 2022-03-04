#!/bin/bash
#               GMT EXAMPLE 42
#
# Purpose:      Illustrate Antarctica and stereographic projection
# GMT modules:  makecpt, grdimage, pscoast, pslegend, psscale, pstext, psxy
# Unix progs:   [curl grdconvert]
#
ps=david.ps


demfile="./dem_30_arcsec/dem_david.wgs84"

# gmt set FONT_ANNOT_PRIMARY 12p FONT_LABEL 12p PROJ_ELLIPSOID WGS-84 FORMAT_GEO_MAP dddF
# Data obtained via website and converted to netCDF thus:
# curl http://www.antarctica.ac.uk//bas_research/data/access/bedmap/download/bedelev.asc.gz
# gunzip bedelev.asc.gz
# grdconvert bedelev.asc BEDMAP_elevation.nc=ns -V
gmt makecpt -Cearth -T-2000/4000 > z.cpt
gmt grdimage -Cz.cpt $demfile -Jx1:60000000 -Q -P  > $ps
# gmt pscoast -R-180/180/-90/-60 -Js0/-90/-71/1:60000000 -Bafg -Di -W0.25p -O -K >> $ps
# gmt psscale -Cz.cpt -DJRM+w2.5i/0.2i+o0.5i/0+mc -R -J -O  -F+p+i -Bxa1000+lELEVATION -By+lm >> $ps

gmt psconvert $ps -A1.0c+m1.0c+pthick -Tf

#gv $psfile
#evince david.pdf
