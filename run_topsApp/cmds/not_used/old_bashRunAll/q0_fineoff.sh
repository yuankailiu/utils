#========= Define the pairs to process =================
insarpair=(
201214-210107
)
# Go check the list of all pairs in `list_pair.dat`
#-------------------------------------------------------


#======== Set GPU device & number of threads ===========
export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=4
printf "CUDA GPU device no.: ${CUDA_VISIBLE_DEVICES}\n"
printf "Use no. of threads : ${OMP_NUM_THREADS}\n"
#-------------------------------------------------------



#==== Backup filesin case of inadvertent changes ======
mkdir -p example/bak
cp example/*.xml example/bak
cp list_pair.dat example/bak
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
    cmds/1_fineoffset.sh ${insarpair[i]}
    #cmds/2_unwrap.sh ${insarpair[i]}
    #cmds/3_geocode.sh ${insarpair[i]}
done
#-------------------------------------------------------
mail -s 'InSAR job0 finished [fineoffset]' ykliu@caltech.edu <<< "Processing has finished in job0, startup-fineoffset, CUDA_DEVICE ${CUDA_VISIBLE_DEVICES}"
