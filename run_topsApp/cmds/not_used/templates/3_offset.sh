
IFS='-' read -r -a array <<< "$1"
MASTER=${array[0]}
SLAVE=${array[1]}

cd ${MASTER}-${SLAVE}
topsApp.py ../example/topsApp.xml --end=fineoffsets
cd ../
