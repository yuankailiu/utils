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
    echo "Prepare an topsApp track data for MintPy processing (Ollie's code, YKL modified)"
    echo "usage: `basename $0` example_pair outdir rlooks alooks"
    echo "  example_pair:   a pair for copying geometry (have *.rdr, *.vrt, *.xml) and reference (IW*.xml) files"
    echo "  outdir:         output directory for saving files, default=merged/"
    echo "  rlooks:         range multilook (using isce2 loosk.py), default=1"
    echo "  alooks:         azimuth multilook (using isce2 looks.py), default=1"
    echo ""
    echo "example of usage:"
    echo "  `basename $0` 201226-210107 merged 10 10"
    echo ""
    echo "Note:"
    echo "  load isce2 topsStack for computing baselines"
    echo "  load MintPy"
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

# input paths for copying files
refdir="referencedir"
secdir="secondarydir"
geodir="geom_reference"

# other files
dem_new_file='hgt.geo'
los_new_file='los.geo'
lat_geo='lat.geo'
lon_geo='lon.geo'
waterBody_ori='/marmot-nobak/ykliu/aqaba/broad_dem/wbd_1_arcsec/swbdLat_N25_N35_Lon_E032_E040.wbd'
waterBody_new='./waterBody.geo'

# output paths for saving files
basedir=$OUTDIR/baselines
ifgmdir=$OUTDIR/interferograms


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



# =================== Running starting from below ========================= ##

## Make folder for integrated data
printf "\n>>> Create directory for saving \n"
mkdir -p $ifgmdir/


## ========================= Folder: reference ============================ ##
## Copy the referencedir/ and geom_refenrece/
if true; then
    cd $OUTDIR
    if [ -d $refdir ]; then
        echo "$refdir exists."
    else
        cd $pair_dir
        msrsync -P -p 8 --stats $refdir $OUTDIR/
    fi

    cd $OUTDIR
    if [ -d $geodir ]; then
        echo "$geodir exists."
    else
        cd $pair_dir
        msrsync -P -p 8 --stats $geodir $OUTDIR/
    fi

    cd $OUTDIR/$geodir && mkdir -p full
    [ -f *.rdr ] && mv *.rdr full/
    [ -f *.vrt ] && mv *.vrt full/
    [ -f *.xml ] && mv *.xml full/
fi


## ======================== Folder: geom_reference ======================== ##
## Get the geotransform from a vrt file
if true; then
    cd $pair_dir/merged
    geo_vrt_file='filt_topophase.unw.geo.vrt'
    dem_old_file='dem.crop'
    los_old_file='los.rdr.geo'
    dem_vrt_file=$dem_old_file.vrt
fi

## Geocode LOS file
if true; then
    cd $pair_dir
    if [ -f "merged/$los_old_file" ]; then
        echo "merged/$los_old_file exists."
    else
        printf "\n>>> Geocode the LOS file (los.rdr to $los_old_file) \n"
        # Run geocode on ['merged/los.rdr'] using topsApp_geocodeLOS.xml with the same bounding box and DEM
        topsApp.py ../example/topsApp_geocodeLOS.xml --dostep=geocode
    fi
fi


## Append the geotransform to the dem.crop.vrt
if true; then
    cd $pair_dir/merged
    printf "\n>>> Get geotransform from $geo_vrt_file, add to $dem_vrt_file \n"
    l1=$(sed -n '2p' $geo_vrt_file)
    l2=$(sed -n '3p' $geo_vrt_file)
    if [ `cat $dem_vrt_file | wc -l` -le "9" ]
    then
        sed -i "2 i $l2" $dem_vrt_file   # -i does additions directly to the file
        sed -i "2 i $l1" $dem_vrt_file
    fi
fi


## GDAL translate the DEM file / change variable labels
if true; then
    cd $pair_dir/merged
    printf "\n>>> GDAL translate for $dem_old_file \n"
    gdal_translate $dem_vrt_file $dem_old_file -of ISCE
    # Change variable labels for mintpy compatibility
    sed -i 's/Coordinate1/coordinate1/'     $dem_old_file.xml
    sed -i 's/Coordinate2/coordinate2/'     $dem_old_file.xml
    sed -i 's/startingValue/startingvalue/' $dem_old_file.xml
fi


## Multilook the geometry files if specified
if true; then
    cd $pair_dir/merged
    if [ $rlooks -gt 1 -o $alooks -gt 1 ]; then
        if [ -f $dem_new_file ]; then
            echo "$dem_new_file exists."
        else
            printf "\n>>> Multilook geometry DEM file \n"
            looks.py -i $dem_old_file -o $dem_new_file -r $rlooks -a $alooks
        fi

        if [ -f $los_new_file ]; then
            echo "$los_new_file exists."
        else
            printf "\n>>> Multilook geometry LOS file \n"
            looks.py -i $los_old_file -o $los_new_file -r $rlooks -a $alooks
        fi
    elif [ $rlooks -eq 1 -a $alooks -eq 1 ]; then
        printf "\n>>> Just copy geometry files since multilook = 1 \n"
        if [ -f $dem_new_file ]; then
            echo "$dem_new_file exists."
        else
            cp $dem_old_file     $dem_new_file
            cp $dem_old_file.vrt $dem_new_file.vrt
            cp $dem_old_file.xml $dem_new_file.xml
            sed -i "s/$dem_old_file/$dem_new_file/"  $dem_new_file.xml
            sed -i "s/$dem_old_file/$dem_new_file/"  $dem_new_file.vrt
        fi
        if [ -f $los_new_file ]; then
            echo "$los_new_file exists."
        else
            cp $los_old_file     $los_new_file
            cp $los_old_file.vrt $los_new_file.vrt
            cp $los_old_file.xml $los_new_file.xml
            sed -i "s/$los_old_file/$los_new_file/"  $los_new_file.xml
            sed -i "s/$los_old_file/$los_new_file/"  $los_new_file.vrt
        fi
    fi
fi


## Copy DEM and LOS files across
if true; then
    cd $pair_dir/merged
    printf "\n>>> Copy DEM and LOS files to output directory \n"
    cp $dem_new_file* $los_new_file* $OUTDIR/$geodir/
    rm -rf *aux* */*aux*
fi


## generate geo-coord lat.geo and lon.geo files
if true; then
    cd $OUTDIR/$geodir/

    if [ -f $lat_geo ] && [ -f $lon_geo ] ; then
        echo "$lat_geo and $lon_geo exists."
    else
        printf "\n>>> Generate geo-coord lat and lon files \n"
        getlalo.py -t $dem_new_file.vrt --out-lat $lat_geo --out-lon $lon_geo
        if [ `cat $lat_geo.vrt | wc -l` -le "9" ]
        then
            sed -i "2 i $l2" $lat_geo.vrt
            sed -i "2 i $l1" $lat_geo.vrt
        fi

        gdal_translate $lat_geo.vrt $lat_geo -of ISCE
        sed -i 's/Coordinate1/coordinate1/'     $lat_geo.xml
        sed -i 's/Coordinate2/coordinate2/'     $lat_geo.xml
        sed -i 's/startingValue/startingvalue/' $lat_geo.xml

        if [ `cat $lon_geo.vrt | wc -l` -le "9" ]
        then
            sed -i "2 i $l2" $lon_geo.vrt
            sed -i "2 i $l1" $lon_geo.vrt
        fi
        gdal_translate $lon_geo.vrt $lon_geo -of ISCE
        sed -i 's/Coordinate1/coordinate1/'     $lon_geo.xml
        sed -i 's/Coordinate2/coordinate2/'     $lon_geo.xml
        sed -i 's/startingValue/startingvalue/' $lon_geo.xml
    fi
fi


## generate waterBody file with the same spatial dimension
if true; then
    cd $OUTDIR/$geodir/
    if [ -f $waterBody_new ] ; then
        echo "$waterBody_new exists."
    else
        printf "\n>>> Generate geocoded waterBody file \n"
        getwbd.py --lat $lat_geo --lon $lon_geo -i $waterBody_ori -o $waterBody_new
        if [ `cat $waterBody_new.vrt | wc -l` -le "9" ]
        then
            sed -i "2 i $l2" $waterBody_new.vrt
            sed -i "2 i $l1" $waterBody_new.vrt
        fi
        gdal_translate $waterBody_new.vrt $waterBody_new -of ISCE
        sed -i 's/Coordinate1/coordinate1/'     $waterBody_new.xml
        sed -i 's/Coordinate2/coordinate2/'     $waterBody_new.xml
        sed -i 's/startingValue/startingvalue/' $waterBody_new.xml
        rm -rf *aux* */*aux*
    fi
fi


## ============================ Folder: baselines ============================== ##
## Baselines computation all topsApp pairs
if false; then
    cd $process_dir
    printf "\n>>> Do baselines and pairs multilook... \n"
    echo "Computing baselines"
    getBaselines.py -d $basedir -r $refdir -s $secdir
fi


## ========================== Folder: interferograms =========================== ##
## Multilook all topsApp pairs
if false; then
    cd $process_dir
    echo "Multilooking all pairs; copy files if multilook=1"
    if [ $rlooks -ge 1 -o $alooks -ge 1 ]; then
        getLooks.py -d $ifgmdir -r $rlooks -a $alooks
    fi
fi


printf "\n>>> Preparations for loading topsApp products to MintPy is completed!\n"
