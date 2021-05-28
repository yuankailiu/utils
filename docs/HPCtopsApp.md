# Running isce2 topsApp on HPC
This is a note and list of usefull things when running isce2 topsApp on the Caltech campus High Performing Computing (HPC) cluster. Details in this document include:
  - Installation of isce2 on HPC
  - Some command line examples
  - SLURM scripts and submit jobs
  - Resources usage
  - Short reports of processing speed and data transfering speed over HPC
  - Fee



## ISCE2 related links

Participate in discussions with the users/developers community!

- [JPL/ISCE2 GitHub repo](https://github.com/isce-framework/isce2)
- [ISCE Forum on Caltech Earthdef](http://earthdef.caltech.edu/projects/isce_forum/boards): May need login credentials based on a request to the admin

<br/>

## Reference of "the Experts"

- @[yunjunz](https://github.com/yunjunz)'s guidance: <https://github.com/yunjunz/conda_envs>
- @[lijun99](https://github.com/lijun99)'s guidance: <https://github.com/lijun99/isce2-install#linux-with-anaconda3--cmake>
- @[CunrenLiang](https://github.com/CunrenLiang)'s repo: https://github.com/CunrenLiang/isce2
- ISCE on KAMB wiki: <https://geo.caltech.edu/doku.php?id=local:insar:iscep-on-kamb> (May need login credentials based on a request to the admin)

<br/>

## Install isce2 on HPC:

```bash
# use tools directory to save files
mkdir -p ~/tools && cd ~/tools

# obtain Miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# install conda
sh ./Miniconda3-latest-Linux-x86_64.sh -b -p ~/tools/miniconda3

# initialize conda 
conda init bash

# restart bash by re-login or simply
bash

## First, set conda-forge as default conda channels 
conda config --prepend channels conda-forge
conda config --set channel_priority strict

# create a venv names isce2 (python 3.9 doesn't work with isce2)
conda create -n isce2 python=3.8

# activate isce2 
conda activate isce2

# install prerequisites 
conda install git cmake cython gdal h5py libgdal pytest numpy fftw scipy basemap opencv openmotif openmotif-dev xorg-libx11 xorg-libxt xorg-libxmu xorg-libxft libiconv xorg-libxrender xorg-libxau xorg-libxdmcp

# Load compilers, as well as cuda/nvcc compiler for GPU modules. 
# On HPC, CUDA compilers, the most recent version is 11.2,
# On HPC, use GCC 7.3.0 can work
module load cuda/11.2
module load gcc/7.3.0

# Download a source package of isce2 (download it to ~/tools/isce2/src)
mkdir -p $HOME/tools/isce2
cd $HOME/tools/isce2
mkdir -p build install src
cd src
git clone https://github.com/isce-framework/isce2.git

# Go to build/ directory and run CMake (install it under ~/tools/isce2/install)
cd ../build/
cmake ~/apps/isce2/src/isce2/ -DCMAKE_INSTALL_PREFIX=~/apps/isce2/install -DCMAKE_CUDA_FLAGS="-arch=sm_60" -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} -DCMAKE_BUILD_TYPE=Release

# Compile and install
make -j 16 # use multiple threads to accelerate
make install

## ---------------------- Installation finished ---------------------------
# Check: 
# You can see the following directories under ~/tools/isce2/install

## ----------- All you need to do when you use isce2 next time ------------
#                 (can put the following into ~/.bashrc)

module load cuda/11.2
conda activate isce2
# Path settings to where you installed isce2
export ISCE_ROOT=~/tools/isce2
export ISCE_HOME=$ISCE_ROOT/install/packages/isce
export PATH=${PATH}:${ISCE_ROOT}/install/bin
export PYTHONPATH=${PYTHONPATH}:${ISCE_ROOT}/install/packages
export PYTHONPATH=${PYTHONPATH}:${CONDA_PREFIX}/bin
```



## File transferring from KAMB to HPC

- Average speed:         300~400 Mb/sec
- Transfer 3T files:     2~3 hours

## How many pairs can HPC run concurrently

From my current experience, it can have **11–22 jobs** running at different nodes the same time, others are just pending for nodes

note: each job is one pair of SLC

```bash
## Here I submitted 40 SLC pairs in total, 11 of them got fired. Others are pending
 # I don't know why there is a (Priority) under the NODELIST(REASON)

JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
  14636041_[11-39]       any  topsApp    ykliu PD       0:00      1 (Priority)
       14636041_10       any  topsApp    ykliu  R    4:54:59      1 hpc-26-15
        14636041_9       any  topsApp    ykliu  R    4:55:59      1 hpc-26-21
        14636041_7       any  topsApp    ykliu  R    4:58:00      1 hpc-25-23
        14636041_8       any  topsApp    ykliu  R    4:58:00      1 hpc-26-14
        14636041_6       any  topsApp    ykliu  R    4:59:00      1 hpc-26-18
        14636041_5       any  topsApp    ykliu  R    4:59:30      1 hpc-26-17
        14636041_4       any  topsApp    ykliu  R    5:00:00      1 hpc-26-20
        14636041_3       any  topsApp    ykliu  R    5:02:31      1 hpc-25-24
        14636041_0       any  topsApp    ykliu  R    5:09:32      1 hpc-26-23
        14636041_1       any  topsApp    ykliu  R    5:09:32      1 hpc-26-24
        14636041_2       any  topsApp    ykliu  R    5:09:32      1 hpc-89-37
```

## Wait time

- 5.5 hours (for long acquisitions)

```bash
## steps and wait time of isce2 topsApp.py
 # This is estiamted for each pair. For my first testing on 8 SLC pairs, the wait times are all similar as below

## Recources for each pair: 1 node, 1 GPU, 28 CPU cores

use_steps=
('startup'                 # 0        min   ⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻|
'preprocess'               # 4        min             |
'computeBaselines'         # 0        min             |
'verifyDEM'                # 0        min             |
'topo'                     # 5        min (with GPU)  |
'subsetoverlaps'           # 0        min             |
'coarseoffsets'            # 0        min             |____ < 20 min
'coarseresamp'             # 0        min             |     
'overlapifg'               # 0        min             |
'prepesd'                  # 0        min             |
'esd'                      # 0        min             |
'rangecoreg'               # 0        min             |
'fineoffsets'              # 9        min (with GPU)__|
 
'fineresamp'               # 35       min (resampling;           no gpu)   ⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻|
'ion'                      # 175      min (resampling;           no gpu)                          |
'burstifg'                 # 60       min (sing-look igram, coh; no gpu)                          |
'mergebursts'              # 7        min                                                         |_____ ~315 min
'filter'                   # 1        min (filter strength=0)                                     |    
'unwrap'                   # 30       min                       (no gpu)                          |   
'unwrap2stage'             # 0        min                                                         |
'geocode'                  # 4-6      min    _____________________________________________________|
)

## Total runtime: ~5.5 hours for one pair
```

## Setting resources in SLURM script

For each pair

- 1 node, 1 GPU, 28 CPU cores (maximum cores)

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

## Resource usage by my account (command: `sreport`)

```bash
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

## Individual job information (command: `scontrol show job`)

```bash
## Below is a report from one of the SLC pair
	 
   JobId=14636053 ArrayJobId=14636041 ArrayTaskId=10 JobName=topsApp
   UserId=ykliu(21896) GroupId=grads(104) MCS_label=N/A
   Priority=6206 Nice=0 Account=simonsgroup QOS=normal
   JobState=RUNNING Reason=None Dependency=(null)
   Requeue=1 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
   RunTime=05:00:25 TimeLimit=06:00:00 TimeMin=N/A
   SubmitTime=2021-04-23T03:01:41 EligibleTime=2021-04-23T03:01:41
   AccrueTime=2021-04-23T03:01:41
   StartTime=2021-04-23T03:16:38 EndTime=2021-04-23T09:16:38 Deadline=N/A
   SuspendTime=None SecsPreSuspend=0 LastSchedEval=2021-04-23T03:16:38
   Partition=any AllocNode:Sid=login2:230052
   ReqNodeList=(null) ExcNodeList=(null)
   NodeList=hpc-26-15
   BatchHost=hpc-26-15
   NumNodes=1 NumCPUs=28 NumTasks=1 CPUs/Task=28 ReqB:S:C:T=0:0:*:*
   TRES=cpu=28,mem=224G,node=1,billing=28,gres/gpu=1
   Socks/Node=* NtasksPerN:B:S:C=1:0:*:* CoreSpec=*
   MinCPUsNode=28 MinMemoryCPU=8G MinTmpDiskNode=0
   Features=(null) DelayBoot=00:00:00
   OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
   Command=/central/groups/simonsgroup/ykliu/ykliu/aqaba/a087/process/s1a/submit_topsApp_0.sh
   WorkDir=/central/groups/simonsgroup/ykliu/ykliu/aqaba/a087/process/s1a
   StdErr=/central/groups/simonsgroup/ykliu/ykliu/aqaba/a087/process/s1a/slurm-14636041_10.out
   StdIn=/dev/null
   StdOut=/central/groups/simonsgroup/ykliu/ykliu/aqaba/a087/process/s1a/slurm-14636041_10.out
   Power=
   TresPerNode=gpu:1
   MailUser=ykliu@caltech.edu MailType=BEGIN,END,FAIL
```

## Disk usage by me on HPC (command: `mmlsquota`)

I am not putting all the files and running them under the group space `simonsgroup/`

There are 10T quota there. Since nobody is using it, I started working from there. Now I have used up ~8.3T and need to delete files

The alternative is to go to scratch/ and do the work there since it has 20T quota for each person (but any files not accessed in 14 days will be automatically purged)

```bash
## Block Limits                                 
Filesystem type         blocks      quota      limit   in_doubt    grace 
central    FILESET      8.368T        10T        12T     11.86G     none 

## File Limits
files   quota    limit in_doubt    grace  Remarks
87369       0        0      160     none central.ib.cluster
```

### My disk details

- All SLCs raw data:              2.9 T      (num of dates = 184 dates;    num of SLC zip files = 694)
- Processing directory:         5.8 T     (19 pairs completed; each completed pair occupies ~300G; I still have 170 pairs or so to process)

### Reducing files

Once `geocode` completed, we can delete most of the files. Especially those files run via GPU can be quickly re-generated if needed.

Each pair will take 3.3 G of disk space after deleting unnecessary files

```bash
## Here is a list of things Cunren suggests that we can delete after finishing all processing
 
 30G   fine_coreg/                             # can regenerate using GPUs
 58G   fine_interferogram/                     # can regenerate using GPUs
 30G   fine_offsets/                           # can regenerate using GPUs
100G   geom_reference/                         # can regenerate using GPUs
 30G   ion/lower/fine_interferogram/           # ion burst
 30G   ion/upper/fine_interferogram/           # ion burst
 15G   ion/ion_burst/                          # ion burst
```


## Rates

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

|Aggregate Spend	| Fee per compute unit|
|-----------------|---------------------|
| ≤ $6,500	      |   $0.014            |
| $6,501 - 24,000 |   $0.008            |
| > $24,000	      |   $0.005            |

#### Compute Units	
CPU = 1 computing unit
GPU = 10 computing units

#### Storage	
Group Storage - 10TB initial allocation, up to 30TB available at no charge upon request from PI.
$6.40/TB/Month additional beyond the free allocation.


## Useful commands

```bash
## Check and edit slurm submitted jobs

# check your jobs
squeue -u username

# check all jobs on hpc
squeue

# cancel your jobs
scancel -u username 
scancel jobid

# hold and release jobs
scontrol hold jobid
scontrol release jobid

## Checking Quotas for User and Group
# your /home/
mmlsquota -u ykliu --block-size auto central:home

# your /central/scratch/ 
mmlsquota -u ykliu --block-size auto central:scratch

# your group space
mmlsquota -j simonsgroup --block-size auto central

## Check resources usage 
# Check resources used by me
sreport  -T gres/gpu,cpu   cluster accountutilizationbyuser start=01/01/21T00:00:00 end=now    -t hours user=ykliu

# Check resources used by the whole group
sreport  -T gres/gpu,cpu   cluster accountutilizationbyuser start=01/01/21T00:00:00 end=now    -t hours account=simonsgroup
```

<br \>



## Supplementary:



#### Check isce output, how many threads are actually used

```bash

## topo

Max threads used: 4
#...
------------------ INITIALIZING GPU TOPO ------------------

    Loading slantrange and doppler data...
    Allocating host and general GPU memory...
    Copying general memory to GPU...
    Allocating block memory (99953172 pixels per image)...
    (NOTE: There will be 12 'empty' threads per image block).

## fineoffsets

Geo2rdr executing on 4 threads...
#...
Copying Orbit and Poly1d data to compatible arrays...
Calculating relevant GPU parameters...
NOTE: GPU will process image in 1 runs of 1495 lines

  ------------------ INITIALIZING GPU GEO2RDR ------------------

    Loading relevant geometry product data...
    Allocating memory...
    Done.
    Copying data to GPU...
    (NOTE: There will be 12 'empty' threads).
    Starting GPU Geo2rdr for run 0...
    GPU finished run 0 in 0.154889 s.
    Copying memory back to host...
    GPU finished run 0 (with memory copies) in 0.607357 s.
    Cleaning device memory and returning to main Geo2rdr function...
  Writing run 0 out asynchronously to image files...
  Finished writing to files!

  ------------------ EXITING GPU GEO2RDR ------------------

Finished!

## fineresamp

Number of threads:            4

## geocode
API open (WR): merged/dem.crop
API open (WR): merged/filt_topophase.unw.geo
GDAL open (R): merged/filt_topophase.unw.vrt
GDAL open (R): /home/ykliu/simonsgroup/ykliu/aqaba/a087/dem_3_arcsec/demLat_N26_N34_Lon_E033_E038.dem.wgs84.vrt
 Using nearest neighbor interpolation
 threads           4
 Starting Acquisition time:    56364.376972999999     
 Stop Acquisition time:    56466.753954521497     
 Azimuth line spacing in secs:    1.0277781499999991E-002
 Near Range in m:    799429.57602867822     
 Far  Range in m:    959470.49330962088     
 Range sample spacing in m:    46.591242294306461     
 Input Lines:         9962
 Input Width:         3436
 reading interferogram ...
 Geocoded Lines:          9601
 Geocoded Samples:        6001
 Initializing Nearest Neighbor Interpolator
 geocoding on            4  threads...
 Number of pixels with outside DEM:          6000
 Number of pixels outside the image:     32647905
 Number of pixels with valid data:       23439710
 elapsed time =    44.7890625      seconds
 Using nearest neighbor interpolation
```
