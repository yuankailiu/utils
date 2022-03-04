#!/bin/bash

## Warning: this script can move or copy the entire parent folder of the specified pair:
## Watch out the list of files you specified!!
## execute at process/
## ykliu @ 2021.09.11


## Set default arguments
method=dry

## Help; example
if [ $# -ne 3 ]; then
    echo ""
    echo "Transfer or copy a list of files "
    echo "usage: `basename $0` file_list dest method"
    echo "  file_list:  a text containing the list of files you want to copy"
    echo "  dest:       the path of destination"
    echo "  method:     method of manipulation on the files (dry, cp, scp, mv, rsync, msrsync, rm)"
    echo ""
    echo "example of usage:"
    echo "  `basename $0` myfilelist ~/my/destination/folder/ rsync"
    echo ""
    echo "Note:"
    echo "  You need msrsync installed if you want to use it"
    echo ""
    exit 1
fi


## Set these before running
process_dir=$(pwd)
printf "Current path to find the files to be copied: $process_dir \n"
file_list=$1
dest=$2
method=$3

mapfile -t file_lists < $file_list


############ Asking for confirmation ########

echo "Copy these directories to $dest?"

for ((i=0;i<${#file_lists[@]};i++)); do
    echo " ${file_lists[i]}"
done


read -p "Are you sure? " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi


############ Run copying ####################

if [[ $method == dry ]]; then
    printf "This is a dry run; no action is done\n"

elif [[ $method == cp ]]; then
    for ((i=0;i<${#file_lists[@]};i++)); do
        cp -r ${file_lists[i]} $dest
        printf "cp the entire ${file_lists[i]} folder to $dest\n"
    done

elif [[ $method == scp ]]; then
    for ((i=0;i<${#file_lists[@]};i++)); do
        scp -r ${file_lists[i]} $dest
        printf "scp the entire ${file_lists[i]} folder to $dest\n"
    done


elif [[ $method == mv ]]; then
    for ((i=0;i<${#file_lists[@]};i++)); do
        mv ${file_lists[i]} $dest
        printf "mv the entire ${file_lists[i]} folder to $dest\n"
    done

elif [[ $method == rsync ]]; then
    for ((i=0;i<${#file_lists[@]};i++)); do
        rsync ${file_lists[i]} $dest
        printf "rsync the entire ${file_lists[i]} folder to $dest\n"
    done


elif [[ $method == msrsync ]]; then
    for ((i=0;i<${#file_lists[@]};i++)); do
        msrsync -P -p 4 --stats ${file_lists[i]} $dest
        printf "msrsync the entire ${file_lists[i]} folder to $dest\n"
    done

elif [[ $method == rm ]]; then
    for ((i=0;i<${#file_lists[@]};i++)); do
        rm -rf ${file_lists[i]}
        printf "rm the entire ${file_lists[i]} folder to $dest\n"
    done

fi


############ done  ####################
printf "Complete! End of the code"
