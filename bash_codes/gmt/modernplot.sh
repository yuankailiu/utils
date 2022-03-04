#!/bin/bash
# Create a pseudo-color image plot of Antarctica using difference colormaps.
# Use Earth relief data provided by GMT (automatically downloaded)
# special files starting with @ are automatically downloaded from the server


if false; then
    gmt begin images_antarctica pdf
        gmt basemap -R0/360/-90/-60 -JG0/-90/20c -B
        # plot the oceans
        gmt makecpt -Cabyss -T-5000/0
        # clip the "wet areas" to remove the land
        gmt coast -Sc
            gmt grdimage @earth_relief_05m -C
        gmt coast -Q
        gmt colorbar -DJBC+o-6c/0.5c+w8c -B1500 -Bx+l"bathymetry [m]" -C
        # plot the land
        gmt makecpt -Cgray -T-6000/6000
        # clip the land to remove the "wet areas"
        gmt coast -Gc
            gmt grdimage @earth_relief_05m -C -I+d
        gmt coast -Q
        gmt colorbar -DJBC+o6c/0.5c+w8c -B1500 -Bx+l"topography [m]" -C -G0/6000
        # add a light contour to the land portion to make it look nicer
        gmt grdcontour @earth_relief_30m -Lp -C500 -Wcthinnest -A2000+f6p -Wathin
        gmt coast -Wthick,black
    gmt end
fi



if true; then
    #demfile="./dem_30_arcsec/dem_david.wgs84"
    #demfile="/marmot-nobak/ykliu/david/bedMachine_v2/dem/surface.grd"
    demfile="/marmot-nobak/ykliu/david/rema/dem_3_arcsec/dem_david.grd"

    gmt begin images_david_small pdf
        gmt basemap -R0/360/-90/-70 -JG0/-90/20c -B
        # plot the oceans
        gmt makecpt -Cabyss -T-5000/0
        # clip the "wet areas" to remove the land
        gmt coast -Sc
            gmt grdimage @earth_relief_05m -C
        gmt coast -Q
        gmt colorbar -DJBC+o-6c/0.5c+w8c -B1500 -Bx+l"bathymetry [m]" -C
        # plot the land
        gmt makecpt -Cdem1 -T-500/6000
        # clip the land to remove the "wet areas"
        echo "plotting the land DEM"
        gmt coast -Gc
            gmt grdimage ${demfile} -C -I+d -Q
        gmt coast -Q
        echo "DEM plotted"
        gmt colorbar -DJBC+o6c/0.5c+w8c -B1500 -Bx+l"topography [m]" -C -G0/6000
        # add a light contour to the land portion to make it look nicer
        #gmt grdcontour @earth_relief_30m -Lp -C500 -Wcthinnest -A2000+f6p -Wathin
        gmt coast -Wthinnest,black
    gmt end
fi