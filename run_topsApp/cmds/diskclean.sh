#!/bin/bash
# Delete the big useless files after topsApp processing
# ykliu @ Apr22 2021



#----------------------------------------------------------

logfile=cmd_diskclean.log

path="/central/scratch/ykliu/storedData"

# Set files to remove (1=remove;   0=not remove)
rm1=1    # fine_offsets/                     30 G
rm2=1    # geom_reference/                  100 G
rm3=1    # fine_coreg/                       30 G
rm4=1    # fine_interferogram/               58 G
rm5=1    # ion/lower/fine_interferogram/     30 G
rm6=1    # ion/upper/fine_interferogram/     30 G
rm7=1    # ion/ion_burst/                    15 G

#----------------------------------------------------------

start=`date +%s`

IFS='-' read -r -a array <<< "$1"
MASTER=${array[0]}
SLAVE=${array[1]}
pair=${MASTER}-${SLAVE}


## Start copying files
echo "Do not copy $pair to $path"
#rsync -ar --info=progress2 $pair $path



## Start deleting files
cd $pair
rm -rf ${logfile}
printf "Not copied pair $pair to $path \n" >> ${logfile}
printf "Delete the following big files for pair $pair :  \n" >> ${logfile}
now=`date`
echo "Deleting files for $pair"

# fine_offsets/
if [[ "$rm1" -eq 1 ]]; then
    printf "  >> ./fine_offsets/*/*.off \n" >> ${logfile}
    echo "Files deleted on $now" >> ./fine_offsets/deleted_files.txt
    ls ./fine_offsets/*/*.off >> ./fine_offsets/deleted_files.txt
    rm -f ./fine_offsets/*/*.off
fi

# geom_reference/
if [[ "$rm2" -eq 1 ]]; then
    printf "  >> ./geom_reference/*.rdr \n" >> ${logfile}
    echo "Files deleted on $now" >> ./geom_reference/deleted_files.txt
    ls ./geom_reference/*.rdr >> ./geom_reference/deleted_files.txt
    rm -f ./geom_reference/*.rdr
fi

# fine_coreg/
if [[ "$rm3" -eq 1 ]]; then
    printf "  >> ./fine_coreg/*/*.slc \n" >> ${logfile}
    echo "Files deleted on $now" >> ./fine_coreg/deleted_files.txt
    ls ./fine_coreg/*/*.slc >> ./fine_coreg/deleted_files.txt
    rm -f ./fine_coreg/*/*.slc
fi

# fine_interferogram/
if [[ "$rm4" -eq 1 ]]; then
    printf "  >> ./fine_interferogram/*/*.cor \n" >> ${logfile}
    printf "  >> ./fine_interferogram/*/*.int \n" >> ${logfile}
    echo "Files deleted on $now" >> ./fine_interferogram/deleted_files.txt
    ls ./fine_interferogram/*/*.cor >> ./fine_interferogram/deleted_files.txt
    ls ./fine_interferogram/*/*.int >> ./fine_interferogram/deleted_files.txt
    rm -f ./fine_interferogram/*/*.cor
    rm -f ./fine_interferogram/*/*.int
fi

# ion/lower/fine_interferogram/
if [[ "$rm5" -eq 1 ]]; then
    printf "  >> ./ion/lower/fine_interferogram/*/*.int \n" >> ${logfile}
    echo "Files deleted on $now" >> ./ion/lower/fine_interferogram/deleted_files.txt
    ls ./ion/lower/fine_interferogram/*/*.int >> ./ion/lower/fine_interferogram/deleted_files.txt
    rm -f ./ion/lower/fine_interferogram/*/*.int
fi

# ion/upper/fine_interferogram/
if [[ "$rm6" -eq 1 ]]; then
    printf "  >> ./ion/upper/fine_interferogram/*/*.int \n" >> ${logfile}
    echo "Files deleted on $now" >> ./ion/upper/fine_interferogram/deleted_files.txt
    ls ./ion/upper/fine_interferogram/*/*.int >> ./ion/upper/fine_interferogram/deleted_files.txt
    rm -f ./ion/upper/fine_interferogram/*/*.int
fi

# ion/ion_burst/
if [[ "$rm7" -eq 1 ]]; then
    printf "  >> ./ion/ion_burst/*/*.ion \n" >> ${logfile}
    echo "Files deleted on $now" >> ./ion/ion_burst/deleted_files.txt
    ls ./ion/ion_burst/*/*.ion >> ./ion/ion_burst/deleted_files.txt
    rm -f ./ion/ion_burst/*/*.ion
fi



## End of job
end=`date +%s`
runtime=$( echo "$end - $start" | bc -l )
Elaspsed="Total elapsed for file transfer and deletion: $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
printf "${Elaspsed} \n" >> ${logfile}
cd ../
