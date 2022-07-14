#!/bin/bash

### To extract REMA dem .tif files from a batch of *.gz.tar
# ------------------------------------------------------------------------------------------------------------------
### How to get REMA data
###	+ REMA main website: https://www.pgc.umn.edu/data/rema/
###	+ REMA explorer to see index of a certain DEM tile: https://livingatlas2.arcgis.com/antarcticdemexplorer/
### 	+ use wget to download *.gz.tar: https://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v1.1/
###		can copy-paste your urls to a text file and parallelize the download by running
###		`parallel -a rema_mosaic_tile_urls.txt --jobs 8 wget`
###
### The output durectory is the current path
# ------------------------------------------------------------------------------------------------------------------
###	ykliu @ 2022-02-08


outfile="dem_david.wgs84"


## Tiff files extraction
mkdir -p tiff
for pkg in *.tar.gz; do
	# Set comma as delimiter
	IFS='.'

	#Read the split words into an array based on comma delimiter
	read -a str <<< "$pkg"

	#Print the splitted words
	echo "Index name : ${str[0]}"
	#echo "gz ext: ${str[1]}"
	#echo "tar ext : ${str[2]}"

	# untar the *.dem.tif files to current path
	tar -xvf "${str[0]}.${str[1]}.${str[2]}" "${str[0]}*dem.tif"
done

mv *.dem.tif tiff



## The `merged.tif` is the stitched DEM formed by using GDAL:
#cd tiff
#gdal_merge.py -init -9999 -v -n -9999 -o merged.tif *.tif -of ISCE
#cd ..


## Transform coordinate system; Downsample the dem
mkdir -p dem_1_arcsec/ dem_3_arcsec/ dem_30_arcsec/
gdalwarp -t_srs EPSG:4326 tiff/merged.tif dem_1_arcsec/tmp.dem  -r cubicspline -tr 0.000277778 0.000277778 -of ISCE
gdalwarp -t_srs EPSG:4326 tiff/merged.tif dem_3_arcsec/tmp.dem  -r cubicspline -tr 0.000833333 0.000833333 -of ISCE
gdalwarp -t_srs EPSG:4326 tiff/merged.tif dem_30_arcsec/tmp.dem -r cubicspline -tr 0.008333333 0.008333333 -of ISCE


## Mask nan and zeros
gdal_calc.py -A dem_1_arcsec/tmp.dem  --outfile="dem_1_arcsec/tmp2.dem"  --calc="A*(A>-9000)" --NoDataValue=nan --format ISCE
gdal_calc.py -A dem_3_arcsec/tmp.dem  --outfile="dem_3_arcsec/tmp2.dem"  --calc="A*(A>-9000)" --NoDataValue=nan --format ISCE
gdal_calc.py -A dem_30_arcsec/tmp.dem --outfile="dem_30_arcsec/tmp2.dem" --calc="A*(A>-9000)" --NoDataValue=nan --format ISCE
gdalwarp -srcnodata 0 -dstnodata nan dem_1_arcsec/tmp2.dem  "dem_1_arcsec/${outfile}"  -of ISCE
gdalwarp -srcnodata 0 -dstnodata nan dem_3_arcsec/tmp2.dem  "dem_3_arcsec/${outfile}"  -of ISCE
gdalwarp -srcnodata 0 -dstnodata nan dem_30_arcsec/tmp2.dem "dem_30_arcsec/${outfile}" -of ISCE


## fix image xml file
fixImageXml.py -i "dem_1_arcsec/${outfile}" -f
fixImageXml.py -i "dem_3_arcsec/${outfile}" -f
fixImageXml.py -i "dem_30_arcsec/${outfile}" -f


## remove tmp files
rm -rf dem_*_arcsec/tmp*.dem*

echo "Complete the DEM outputs"
