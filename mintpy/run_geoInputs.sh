#! /bin/bash

config=./smallbaselineApp.cfg

cd inputs/


echo "Geocode inputs/geometryRadar.h5 inputs/ifgramStack.h5 to geo-coordinates"
geocode.py ./*.h5 -t ${config} --outdir ./geo/ --update


echo "Move original input files to ./inputs/radar/"
mkdir -p radar
mv geometryRadar.h5  ifgramStack.h5  ./radar/


echo "Finish geocoding the input files"
mv ./geo/*.h5 ./
mv geo_geometryRadar.h5 geometryGeo.h5
mv geo_ifgramStack.h5   ifgramStack.h5
rm -rf ./geo/

