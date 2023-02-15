
IFS='-' read -r -a array <<< "$1"
MASTER=${array[0]}
SLAVE=${array[1]}

cd ${MASTER}-${SLAVE}
topsApp.py ../example/topsApp.xml --start=fineresamp --end=unwrap2stage
topsApp.py ../example/topsApp_geo.xml --dostep=geocode
cd ../
