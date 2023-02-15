#========= Define the pairs to process =================
insarpair=(
190729-190810
190810-190822
190822-190903
190903-190915
190915-190927
190927-191009
191009-191021
191021-191102
191102-191114
191114-191126
191126-191208
191208-191220
191220-200101
200101-200113
200113-200125
200125-200218
200218-200301
200301-200313
200313-200325
200325-200406
200406-200418
200418-200430
200430-200512
200512-200524
200524-200605
200605-200617
200617-200629
200629-200711
200711-200723
200723-200804
200804-200816
200816-200828
200828-200909
200909-200921
200921-201003
201003-201015
201015-201027
201027-201108
201108-201120
201120-201202
201214-201226
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
cp example/{reference.xml,topsApp.xml,topsApp_geocode.xml} example/bak
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
mail -s 'InSAR job2 finished [fineoffset]' ykliu@caltech.edu <<< "Processing has finished in job2, startup-fineoffset, CUDA_DEVICE ${CUDA_VISIBLE_DEVICES}"
