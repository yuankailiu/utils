#========= Define the pairs to process =================
insarpair=(
161223-170104
170104-170116
)
# Go check the list of all pairs in `list_pair.dat`
#-------------------------------------------------------


#======== Set GPU device & number of threads ===========
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=4
printf "CUDA GPU device no.: ${CUDA_VISIBLE_DEVICES}\n"
printf "Use no. of threads : ${OMP_NUM_THREADS}\n"
#-------------------------------------------------------

donelog="list_pair_done.dat"

#==== Backup filesin case of inadvertent changes ======
mkdir -p example/bak
cp example/*.xml example/bak
cp list_pair.dat example/bak
cp list_pair_done.dat example/bak
#-------------------------------------------------------



#=============== Ask for confirmation ==================
echo "Going to process the following pairs:"
for ((i=0;i<${#insarpair[@]};i++)); do
    echo " ${insarpair[i]}"
done
read -p "Are you sure? " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi
#-------------------------------------------------------



#===== Run topsApp.py steps in the cmds subroutine =====
for ((i=0;i<${#insarpair[@]};i++)); do
    printf "\n\n >>>>>>>>>>>> Working on pair: ${insarpair[i]} \n\n"
    start=`date +%s`
    cmds/2_unwrap.sh ${insarpair[i]}
    cmds/3_geocode.sh ${insarpair[i]}
    end=`date +%s`
    runtime=$( echo "$end - $start" | bc -l )
    Elaspsed="Total elapsed: $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
    printf "${insarpair[i]} stage2+3 finished     ${Elaspsed}\n" >> $donelog
done
#-------------------------------------------------------

pairstr=$(printf '%s\n' "${insarpair[@]}")
body=$(printf "Pairs has finished processing until geocode in job6:\n$pairstr")
printf "$body" | mail -s 'InSAR processing completed [job6 stage2+3]' ykliu@caltech.edu
