
## SLURM commands

[SLURM](https://slurm.schedmd.com/documentation.html) script is just a bash script with some special SLURM flags for resource management. There is definitely a lot of SLURM official documentation. But I would prefer to read the Caltech intro on that (or any other institution/university that has some SLURM HPC guide). So check these out if you haven’t:
- https://www.hpc.caltech.edu/documentation/faq/dependencies-and-pipelines
- https://www.hpc.caltech.edu/documentation/slurm-commands

The Caltech HPC SLURM example script is under “Setting up a pipeline of dependencies” in the first link. Also attached is an image of that example. Note that one typo there on that website is the “partition” flag, it wrote:
```
#SBATCH --partition=any
```
But the default partition for The Caltech Resnick HPCC will change from “any” (CentOS 7) to “expansion” (RHEL 9) on Tuesday, March 26th. So this should be either “expansion” (the default computational nodes without GPU hardware) or “gpu” (those nodes with GPUs). Check the [Caltech HPC nodes info](https://www.hpc.caltech.edu/resources) you can learn more.

Thus, for partition, you should write either:
```
#SBATCH --partition=expansion  (or you can just ignore this line and it will set to be this as default)
```
or
```
#SBATCH —partition=gpu   (if you want to use gpu)
#SBATCH --gres=gpu:2     (specify how many GPUs you need, here I want 2)
```


Now once you finish all these resource-allocating flags, you just put your regular Python execution line below these flags in the same SLURM script. Here are two examples.
- http://homeowmorphism.com/2017/04/18/Python-Slurm-Cluster-Five-Minutes
- https://stackoverflow.com/questions/71280783/running-python-scripts-in-slurm

After you finish your above SLURM script, you can name this file “mytask.sh” (the file extension does not matter, because there is a shebang in the script: `#! /bin/bash`. Also, later you will submit it with a [sbatch command](https://slurm.schedmd.com/sbatch.html)).

Anyway, I like to name it “mytask.job” or “mytask.sbatch” or “mytask.slurm” which I can quickly identify as a SLURM script simply from the filename.

After you have the file ready. You will submit your SLURM script by doing:
```
sbatch mytask.job
```

Now you can check the status (whether you are pending, runnning, etc.) of your submitted task by:
```
squeue -u your_user_name
```


### More useful links:
- https://www.hpc.caltech.edu/documentation/slurm-commands

- https://www.hpc.kaust.edu.sa/tips/running-multiple-parallel-jobs-simultaneously

- https://srcc.stanford.edu/sites/g/files/sbiybj25536/files/media/file/sherlock_onboarding-11-2022.pdf


### Some commands in terminal:
```bash
# check your jobs
squeue -u username

# check all jobs on hpc
squeue

# cancel your jobs
scancel -u username
scancel jobid

# show job
scontrol show jobid=12345678

# hold and release jobs
scontrol hold jobid
scontrol release jobid

# change dependency
scontrol update job=20284119 dependency="afterok:20292656"

# submit job2 with dependency on job1. 11254323 is submitted job1_ID, fo
sbatch --dependency=afterok:11254323 job2.sh

# requeue a job with the slurm jobfile
sbatch --requeue run_06_overlap_resample.sbatch

# reset time limit
scontrol update jobid=21255461_1 TimeLimit=5:00:00

## Checking Quotas for User and Group
# your /home/
mmlsquota -u ykliu --block-size auto central:home

# your /central/scratch/
mmlsquota -u ykliu --block-size auto central:scratch_independent

# your group space
mmlsquota -j simonsgroup --block-size auto central

## Check resources usage
# Check resources used by me
sreport  -T gres/gpu,cpu   cluster accountutilizationbyuser start=01/01/21T00:00:00 end=now    -t hours user=ykliu

# Check resources used by the whole group
sreport  -T gres/gpu,cpu   cluster accountutilizationbyuser start=01/01/21T00:00:00 end=now    -t hours account=simonsgroup
```



## Example of a SLUMR script

For each pair I want: 1 node, 1 GPU, 28 CPU cores (maximum cores)
```bash
#!/bin/bash

#Submit this script with: sbatch <this-filename>
#SBATCH --array=0-7
#SBATCH -A simonsgroup                  # pocket to charge
#SBATCH --time=6:00:00                  # walltime
##SBATCH --ntasks=1                     # number of processor cores (i.e. tasks)
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # tasks per node
#SBATCH --cpus-per-task=28              # CPU cores/threads per task
#SBATCH --gres=gpu:1                    # number of GPU per node
#SBATCH --mem-per-cpu=8G                # memory per CPU core
#SBATCH -J "topsApp"                    # job name
#SBATCH --mail-user=ykliu@caltech.edu   # my email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load cuda/11.2

#==== Backup files in case of inadvertent changes ======
mkdir -p example/bak
cp example/*.xml example/bak
cp list_pair.dat example/bak
cp submit_*.sh scripts/

#========= Define the pairs to process =================
insarpair=(
200921-201015
201003-201027
201015-201108
201027-201120
201108-201202
201120-201214
201202-201226
201214-210107
)

echo $SLURM_ARRAY_TASK_ID
pairID=`echo $SLURM_ARRAY_TASK_ID | awk '{printf($1)}'`
printf ">>>>>>>>>>>> Working on pair: ${insarpair[pairID]} \n"

#===== Run topsApp.py steps in the cmds subroutine =====
cmds/runsteps.sh ${insarpair[pairID]}
```


## Rates for computing

Usage exmaple upon login hpc:
```
--------------------------------------------------------------------------------
Cluster/Account/User Utilization 2021-01-01T00:00:00 - 2021-05-28T16:59:59 (12758400 secs)
Usage reported in TRES Hours
--------------------------------------------------------------------------------
  Cluster         Account     Login     Proper Name      TRES Name      Used
--------- --------------- --------- --------------- -------------- ---------
  central     simonsgroup                                      cpu    213866
  central     simonsgroup                                 gres/gpu      7638
  central     simonsgroup  olstephe Oliver L. (Oll+            cpu    140426
  central     simonsgroup  olstephe Oliver L. (Oll+       gres/gpu      5015
  central     simonsgroup     ykliu    Yuan Kai Liu            cpu     73440
  central     simonsgroup     ykliu    Yuan Kai Liu       gres/gpu      2623
```

Total computing units = cpu + 10 * gpu = 290246

Total fee = 290246 * 0.014 = $4,063

### Rate Structure

Latest rates: <https://www.hpc.caltech.edu/rates>

Total rates are broken into compute hours and additional storage costs. Rates are based on a tiered structure. Tiers are reset every fiscal year.

#### Core Hour Calculations

| Aggregate Spend | Fee per compute unit |
| --------------- | -------------------- |
| ≤ $6,500        | $0.014               |
| $6,501 - 24,000 | $0.008               |
| > $24,000       | $0.005               |

#### Compute Units

CPU = 1 computing unit
GPU = 10 computing units

#### Storage

Group Storage - 10TB initial allocation, up to 30TB available at no charge upon request from PI.
$6.40/TB/Month additional beyond the free allocation.
