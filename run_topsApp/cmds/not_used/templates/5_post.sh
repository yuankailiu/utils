
IFS='-' read -r -a array <<< "$1"
MASTER=${array[0]}
SLAVE=${array[1]}

cd ${MASTER}-${SLAVE}

#processing interferogram without ionospheric correction
mv merged merged_ori
mkdir merged
#cp merged_ori/topophase_ori.flat ./merged/topophase.flat
cp merged_ori/topophase_noion.flat ./merged/topophase.flat
cp merged_ori/topophase.flat.vrt ./merged
cp merged_ori/topophase.flat.xml ./merged
topsApp.py ../example/topsApp_geo.xml --start=filter --end=geocode
rm ./merged/dem.crop*
mv merged merged_noion

#processing interferogram with ionospheric correction
mv merged_ori merged
topsApp.py ../example/topsApp_geo.xml --start=filter --end=geocode

cd ../
