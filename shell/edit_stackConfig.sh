#!/bin/bash
#
# ***********************************************************************
#   For isce2 topsStack processor:
#   Quickly copy and edit the content of the config files under config/
#
#   Yuan-Kai Liu, 2022-3-3
# ***********************************************************************


config="config_filtIon_"

fkey1="20150714_20150726"
key1="20150714_20150726"
file1=${config}${fkey1}



fkey2_arr=(
'20150714_20150807'
'20150726_20150807'
'20150807_20150819'
'20160122_20160215'
'20160203_20160215'
'20160215_20160227'
)

key2_arr=(
'20150714_20150807'
'20150726_20150807'
'20150807_20150819'
'20160122_20160215'
'20160203_20160215'
'20160215_20160227'
)



for ((i=0;i<${#key2_arr[@]};i++)); do
    key2=${key2_arr[i]}
    fkey2=${fkey2_arr[i]}
    file2=${config}${fkey2}

    cp $file1 $file2
    sed -i "s/${key1}/${key2}/" $file2
done


echo "Normal end of the script."
~