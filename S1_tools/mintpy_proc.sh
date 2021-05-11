#!/bin/bash

# Ollie Stephenson 2021 April
# Do a few things to prepare an InSAR track for mintpy processing 
# Prep work before running
#   Load ISCE 
#   Make sure we have a topsApp_geo_los.xml file with the correct region, DEM, and geocode list
#   Choose example pair for the track (don't pick a short one)

# [ Modified by YKL @ 2021 May 9 ]
# Recommended steps (should be implemented in this script) to take for loading topsApp products to MintPy:
#   1. Baselines calculation for all dates
#   2. Reference folder with *.xml files
#   3. Put all interferograms (can do multilook) into a merged/ folder
#   4. Take a copy of geom_reference/ (with same crop region, geocoding, and multilook):
#       (1). hgt.geo, hgt.geo.vrt, hgt.geo.xml (take the dem.crop file, hardwire the lon lat info in the metadata .vrt and .xml)
#       (2). los.geo, los.geo.vrt, los.geo.xml (take los.rdr.geo file, multilook it)
#
#   Data directory structure:
#       process/ ______ merged/ ________ baselines/       _______ YYYYMMDD_YYYYMMDD, ...
#                                 |_____ referencedir/    _______ IW1/, IW2/, IW3/, IW1.xml, ...
#                                 |_____ interferograms/  _______ YYYYMMDD_YYYYMMDD, ...
#                                 |_____ geom_reference/  _______ hgt.geo, los.geo, ...


## Set default arguments
outdir=merged
rlooks=1
alooks=1

if [ $# -ne 4 ]; then
    echo ""
    echo "Prepare an topsApp track data for MintPy processing (Ollie -> YKL modified)"
    echo "usage: `basename $0` example_pair outdir rlooks alooks"
    echo "  example_pair:   a pair for copying geometry (have *.rdr, *.vrt, *.xml) and reference (IW*.xml) files"
    echo "  outdir:         output directory for saving files, fedualt=merged/"
    echo "  rlooks:         range multilook (using isce2 loosk.py), default=1"
    echo "  alooks:         azimuth multilook (using isce2 looks.py), default=1"
    echo ""
    echo "example of usage:"
    echo "  `basename $0` 201226-210107 merged 10 10"
    echo ""
    echo "Note:"
    echo "  load isce2 topsStack for computing baselines"
    echo "  load MintPy too"
    echo ""
    exit 1
fi


## Set these before running
process_dir=$(pwd)
printf "Current path (should be the process/ dir): $process_dir \n"
example_pair=$1
outdir=$2
rlooks=$3
alooks=$4
pair_dir=$process_dir/$example_pair
OUTDIR=$process_dir/$outdir


## Ask for confirmation
echo "Going to prepare topsApp products for MintPy..."
printf "  - topsApp process directory: $process_dir \n"
printf "  - Reference and geometry example pair: $pair_dir \n"
printf "  - Multilook using <isce2 looks.py>, (rlooks, alooks) = ($rlooks , $alooks) \n"
printf "  - Ouput compilation will be saved to: $OUTDIR \n"
read -p "Are you sure? " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi


## Make folder for integrated data
printf "\n>>> Create directory for saving \n"
mkdir -p $OUTDIR/interferograms/ 

# Copy the referencedir/ and geom_refenrece/
cd $pair_dir
#cp -r referencedir $OUTDIR/
#cp -r geom_reference $OUTDIR/
cd $OUTDIR/geom_reference && mkdir -p full && mv *.rdr *.vrt *.xml full/
cd $pair_dir


## Geocode LOS file
printf "\n>>> Geocode the LOS file \n"
#    topsApp_geocodeLOS.xml file with just the los file to geocode, the right bounding box and the right DEM 
#topsApp.py ../example/topsApp_geocodeLOS.xml --dostep=geocode
cd $pair_dir/merged


## Get the geotransform from a vrt file, add the geotransform to the dem.crop.vrt
geo_vrt_file='filt_topophase.unw.geo.vrt'
dem_old_file='dem.crop'
los_old_file='los.rdr.geo'
dem_vrt_file=$dem_old_file.vrt

printf "\n>>> Get geotransform from $geo_vrt_file, add to $dem_vrt_file \n"
l1=$(sed -n '2p' $geo_vrt_file)
l2=$(sed -n '3p' $geo_vrt_file)
if [ `cat $dem_vrt_file | wc -l` -le "9" ]
then
    sed -i "2 i $l2" $dem_vrt_file   # -i does additions directly to the file
    sed -i "2 i $l1" $dem_vrt_file 
fi

## GDAL translate the DEM file
printf "\n>>> GDAL translate for $dem_old_file \n"
gdal_translate $dem_vrt_file $dem_old_file -of ISCE
# Change variable labels for mintpy compatibility
sed -i 's/Coordinate1/coordinate1/'     $dem_old_file.xml
sed -i 's/Coordinate2/coordinate2/'     $dem_old_file.xml
sed -i 's/startingValue/startingvalue/' $dem_old_file.xml


## Multilook the geometry files if specified
dem_new_file='hgt.geo'
los_new_file='los.geo'
printf "\n>>> Multilook geometry files \n"
if [ $rlooks -gt 1 -o $alooks -gt 1 ]; then
    looks.py -i $dem_old_file -o $dem_new_file -r $rlooks -a $alooks
    looks.py -i $los_old_file -o $los_new_file -r $rlooks -a $alooks
fi


## Copy DEM and LOS files across
printf "\n>>> Copy DEM and LOS files to output directory \n"
cp $dem_new_file* $los_new_file* $OUTDIR/geom_reference/
rm -rf *aux*
cd $process_dir 

printf "\n>>> Do baselines and pairs multilook... \n"
## Lets do baselines or the topsApp products
#echo "Computing baselines"
#getBaselines.py -d ./baselines

## Lets do ISCE2 multilook for all pairs
#echo "Multilooking all pairs"
#if [ $rlooks -gt 1 -o $alooks -gt 1 ]; then
#    getLooks.py -d $OUTDIR/interferograms -r $rlooks -a $alooks
#fi

printf "\n>>> Preparations for loading topsApp products to MintPy is completed!\n"