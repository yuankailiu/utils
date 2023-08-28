#! /bin/bash

## Use GDAL to resample the downloaded radar waterBody file to a smaller one
## The new waterBody matches the extent, dimension, resolutoin of the topsStack merged datasets
## using: isceobj.Alos2Proc.Alos2ProcPublic.waterBodyRadar(latFile, lonFile, wbdFile, wbdOutFile)
## Parameters defined in `params.json` as "wbd_orig", "wbd_dir".
## This waterBody is meant to be loaded into MintPy duing load_data.py step
## In `modify_network` step, MintPy automatically converts it to `waterMask.h5` for masking purposes
##
##  ykliu 2022.01.28

# =============== Read defined variables from json file ==================
my_json="./params.json"
declare -A dic
while IFS="=" read -r key value
do
    dic[$key]="$value"
done < <(jq -r 'to_entries|map("\(.key)=\(.value)")|.[]' $my_json)
# =============== ===================================== ==================

# Get parameters
wbd_orig="${dic['wbd_orig']}"
wbd_dir="${dic['wbd_dir']}"



# check the directory
mkdir -p $wbd_dir
#cd $wbd_dir

# get parent directory of the waterBody
if [[ $wbd_dir == */ ]] # if the wbd_dir ends with a /
then
    wbd_parentdir="$(echo $wbd_dir | rev | cut -d'/' -f3- | rev)"
else
    wbd_parentdir="$(echo $wbd_dir | rev | cut -d'/' -f2- | rev)"
fi

echo "Working on resmapling waterBody from: $wbd_parentdir"
echo "Resampled waterBody will be saved to : $wbd_dir"

# do gdal_translate to resampled
gdal_translate "$wbd_parentdir/lon.rdr.vrt" "$wbd_dir/lon.rdr" -of ISCE
gdal_translate "$wbd_parentdir/lat.rdr.vrt" "$wbd_dir/lat.rdr" -of ISCE
getwbd.py --lat "$wbd_dir/lat.rdr" --lon "$wbd_dir/lon.rdr" -i $wbd_orig -o "$wbd_dir/waterBody.rdr"

# finish
echo "Normal finish"
