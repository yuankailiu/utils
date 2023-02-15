#========= Define the pairs to process =================
insarpair=(
141011-141023
141023-141104
141104-141116
141116-141128
141128-141210
141210-141222
141222-150103
150103-150115
150115-150127
150127-150208
150208-150220
150220-150304
150328-150409
150409-150421
150527-150608
150608-150702
150714-150726
150726-150807
)
# Go check the list of all pairs in `list_pair.dat`
#-------------------------------------------------------


#======== Set GPU device & number of threads ===========
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2
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
    cmds/9_DEMgeocode.sh ${insarpair[i]}
done
#-------------------------------------------------------
