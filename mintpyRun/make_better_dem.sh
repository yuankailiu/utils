#! /bin/bash

## Use GDAL to resample the orignal DEM to match the extent, dimension, and resolution of
## MintPy geocoded .h5 products.
## [This is optional], just to cover the full extent when using topsStack radar coord datsets
##  (when geocode geometryRadar.h5 to geometryGeo.h5, the height will have large gaps; not pretty)
## Should be run after having the geometryGeo.h5 file (must in geo-coord to allow reading lon lat)
## The output DEM is then saved separetly (defined in `params.json` as "dem_out")
## The output DEM is mainly for plotting purposes using view.py
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
txtfile="${dic['proc_home']}/${dic['hdf5_info']}"
geofile="${dic['proc_home']}/${dic['geom_file']}"
dem_orig="${dic['dem_orig']}"
dem_out="${dic['dem_out']}"
echo "HDF5 dimension info retireved from:" > $txtfile
echo "$geofile"                           >> $txtfile


# Declare an array of type of datasets
declare -a dSetArray=("latitude" "longitude" "height" )

# Iterate the datasets
for dset in ${dSetArray[@]}; do
    echo $dset
    shapes=$(info.py $geofile --dset $dset | sed -n -e 's/^.*dataset size: //p')
    minmax=$(info.py $geofile --dset $dset | sed -n -e 's/^.*dataset min \/ max: //p')
    printf "${dset}_shapes: $shapes\n"  >> $txtfile
    printf "${dset}_minmax: $minmax\n"  >> $txtfile
done

# Remove special characters in the txt
sed -i 's/(//g; s/)//g'  $txtfile
sed -i 's/, / /g'        $txtfile
sed -i 's/ \/ / /g'      $txtfile
echo "HDF5 dimension info have been stored: $txtfile "

# Get the following info
len="$(sed -n -e '/latitude_shapes/p'      $txtfile | cut -d' ' -f2)"
wid="$(sed -n -e '/latitude_shapes/p'      $txtfile | cut -d' ' -f3)"
lat_min="$(sed -n -e '/latitude_minmax/p'  $txtfile | cut -d' ' -f2)"
lat_max="$(sed -n -e '/latitude_minmax/p'  $txtfile | cut -d' ' -f3)"
lon_min="$(sed -n -e '/longitude_minmax/p' $txtfile | cut -d' ' -f2)"
lon_max="$(sed -n -e '/longitude_minmax/p' $txtfile | cut -d' ' -f3)"

echo "Dimension of the dataset: $len, $wid"
echo "$lon_min, $lat_min, $lon_max, $lat_max"


# Check the directory
dem_outdir="$(echo $dem_out | rev | cut -d'/' -f2- | rev)"
mkdir -p $dem_outdir

# do gdalwarp on teh orignal DEM and output it
gdalwarp $dem_orig $dem_out -te $lon_min $lat_min $lon_max $lat_max -ts $wid $len -of ISCE
fixImageXml.py -i $dem_out -f

echo "Normal finish"
