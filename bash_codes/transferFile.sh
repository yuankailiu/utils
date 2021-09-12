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
    echo "  method:     method of manipulation on the files (dry, cp, scp, mv, rsync, msrsync)"
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

mapfile -t insarpair < $file_list


############ Asking for confirmation ########

echo "Copy these directories to $dest?"

for ((i=0;i<${#insarpair[@]};i++)); do
    echo " ${insarpair[i]}"
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
    for ((i=0;i<${#insarpair[@]};i++)); do
        cp -r ${insarpair[i]} $dest
        printf "cp the entire ${insarpair[i]} folder to $dest\n"
    done

elif [[ $method == scp ]]; then
    for ((i=0;i<${#insarpair[@]};i++)); do
        scp -r ${insarpair[i]} $dest
        printf "scp the entire ${insarpair[i]} folder to $dest\n"
    done


elif [[ $method == mv ]]; then
    for ((i=0;i<${#insarpair[@]};i++)); do
        mv ${insarpair[i]} $dest
        printf "mv the entire ${insarpair[i]} folder to $dest\n"
    done

elif [[ $method == rsync ]]; then
    for ((i=0;i<${#insarpair[@]};i++)); do
        rsync ${insarpair[i]} $dest
        printf "rsync the entire ${insarpair[i]} folder to $dest\n"
    done


elif [[ $method == msrsync ]]; then
    for ((i=0;i<${#insarpair[@]};i++)); do
        msrsync -P -p 4 --stats ${insarpair[i]} $dest
        printf "msrsync the entire ${insarpair[i]} folder to $dest\n"
    done

fi


############ done  ####################
printf "Complete! End of the code"
