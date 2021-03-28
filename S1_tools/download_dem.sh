if [ $# -ne 4 ]; then
    echo ""
    echo "usage: $0 s n w e"
    echo "  s: south"
    echo "  n: north"
    echo "  w: west"
    echo "  e: east"
    echo ""
    echo "example of usage:"
    echo "$0 -32 -24 -70 -63"
    echo ""
    exit 1
fi

#set these before running
S=$1
N=$2
W=$3
E=$4

echo ""
echo "input parameters:"
echo "+++++++++++++++++++++++++++++++"
echo "  south: ${S}"
echo "  north: ${N}"
echo "  west:  ${W}"
echo "  east:  ${E}"
echo ""


#########################
#CHANGE THIS TO RUN
bbox="${S} ${N} ${W} ${E}"
#########################

#from: http://earthdef.caltech.edu/boards/4/topics/1845
#3 arcsec dem
mkdir dem_3_arcsec
cd dem_3_arcsec
dem.py -a stitch -b ${bbox} -k -s 3 -c -f -u http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL3.003/2000.02.11
fixImageXml.py -i demLat_*_*_Lon_*_*.dem.wgs84 -f
rm *.hgt* *.log demLat_*_*_Lon_*_*.dem demLat_*_*_Lon_*_*.dem.vrt demLat_*_*_Lon_*_*.dem.xml
cd ../

#1 arcsec dem
mkdir dem_1_arcsec
cd dem_1_arcsec
dem.py -a stitch -b ${bbox} -k -s 1 -c -f -u http://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11
fixImageXml.py -i demLat_*_*_Lon_*_*.dem.wgs84 -f
rm *.hgt* *.log demLat_*_*_Lon_*_*.dem demLat_*_*_Lon_*_*.dem.vrt demLat_*_*_Lon_*_*.dem.xml
cd ../

# 1 arcsec wbd
mkdir wbd_1_arcsec
cd wbd_1_arcsec
wbd.py ${bbox}
fixImageXml.py -i swbdLat_*_*_Lon_*_*.wbd -f
rm *.log
cd ../







