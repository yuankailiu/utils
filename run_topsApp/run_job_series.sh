#============ Modify the contents below ================

#### Define the pairs to be processed
# pair folders you see
# or go check the list of all pairs in `pair_log_*.txt`
insarpair=(
20230203-20230215
)

#### Resource and your email
email=ykliu@caltech.edu
export CUDA_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=8
printf "CUDA GPU device no.: ${CUDA_VISIBLE_DEVICES}\n"
printf "Use no. of threads : ${OMP_NUM_THREADS}\n"

#### topsApp processing steps
stage="runall"
#stage="mlk_filt"
#stage="offsets"


#============= No need to change below ================

#### Backing-up files
mkdir -p example/bak
cp example/*.xml example/bak
cp pair_log_*.txt example/bak


#### Ask for confirmation
echo "Going to process {${stage}} on the following pairs:"
for ((i=0;i<${#insarpair[@]};i++)); do
    echo " ${insarpair[i]}"
done
read -p "Are you sure? " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi


#### Run topsApp.py steps in the cmds subroutine
for ((i=0;i<${#insarpair[@]};i++)); do
    printf "\n\n >>>>>>>>>>>> Working on pair: ${insarpair[i]} \n\n"
    cmds/run_stage.sh ${insarpair[pairID]} ${stage}
    #cmds/diskclean.sh ${insarpair[pairID]}
    mail -s "InSAR job: ${insarpair[pairID]}" ${email} <<< "Processing {${stage}} done, CUDA_DEVICE ${CUDA_VISIBLE_DEVICES}"
done
