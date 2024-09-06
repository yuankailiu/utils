# Running isce2 topsApp on HPC

This is a note and list of useful things when running isce2 topsApp on the Caltech campus High Performing Computing (HPC) cluster. Details in this document include:

- [Useful commands](#useful-commands)
- [ISCE2 related links](#isce2-related-links)
- [Install isce2 on HPC](#install-isce2-on-HPC)
- [Basics](#basics)
- [Resource usage & SLURM scripts and submit jobs](#Resource-usage)
- [Rates](#Rates)


## New modifications
***Try to include these into the new Python workflow as well!***

Split large job arrays into smaller chunks that are concurrently running (ArrayTaskThrottle) to avoid jamming the I/O. Most of the steps (or stages) in topsStack can split into every 200 jobs (pairs) `--array=1-500%200`. For `17_subband_and_resamp_a014`, use even more conservative ArrayTaskThrottle `--array=1-500%100`.

For step `13_generate_burst_igram-25935841_49.out`, we may not need 80G of REM, need to test.

We read and report disk space on the 1st and the 300th job in all topsStack steps (or stages). Some of the steps only take 5 minutes in general, but reading the disk space will take 5 to 10 mins. 

On Caltech HPC, the upper limit of job array (MaxArraySize) we can submit is 1001.
You can check via ` scontrol show config | grep MaxArraySize`. 
So if we have more than 1001 jobs, the submission will fail. We need to slit large stages into two `.sbatch` scripts, specifying the job array within the MaxArraySize:

```bash
## Assume you have a stage that has 1047 jobs. You can split it to following:

#-----> within the first script
#SBATCH --array=1-500%200

#-----> within the second script
#SBATCH --array=1-547%200
SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID+500))
```

## SLURM commands

- https://www.hpc.caltech.edu/documentation/slurm-commands

- https://www.hpc.kaust.edu.sa/tips/running-multiple-parallel-jobs-simultaneously

- https://srcc.stanford.edu/sites/g/files/sbiybj25536/files/media/file/sherlock_onboarding-11-2022.pdf

```bash
## Slurm array batch script

# limit the number of jobs in the array that are concurrently running
# we have 100 jobs in the array but only 25 of them running concurrently
#SBATCH --array=1:100%25


## Check and edit slurm submitted jobs

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

</br>

## ISCE2 related links

Participate in discussions with the users/developers community!

- [JPL/ISCE2 GitHub repo](https://github.com/isce-framework/isce2)
- [ISCE Forum on Caltech Earthdef](http://earthdef.caltech.edu/projects/isce_forum/boards): May need login credentials based on a request to the admin

<br/>

### Reference to the experts

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

# Load CUDA and GCC compilers (*)
module load cuda/12.0
# module load gcc/7.3.0 (no need to load this, the default 4.8.5 can compile well)

# Download a source package of isce2 (download it to ~/tools/isce2/src)
mkdir -p $HOME/tools/isce2
cd $HOME/tools/isce2
mkdir -p build install src
cd src
git clone https://github.com/isce-framework/isce2.git

# Go to build/ directory and run CMake (install it under ~/tools/isce2/install)
cd ../build/
cmake $TOOL_DIR/isce2/src/isce2/ -DCMAKE_INSTALL_PREFIX=$TOOL_DIR/isce2/install -DCMAKE_CUDA_FLAGS="-arch=sm_60" -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} -DCMAKE_BUILD_TYPE=Release

# Compile and install
make -j 16 # use multiple threads to accelerate
make install

## ---------------------- Installation finished ---------------------------
# Check: 
# You can see the following directories under ~/tools/isce2/install
# Try if you can do this properly:
topsApp.py -h 

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


```bash
### Special notes for CUDA and GCC

# Load compilers, as well as cuda/nvcc compiler for GPU modules.
## Note: a cuda version determines up to which gcc is compatible
## Note: a newer gcc will require an even newer cuda version
## Check here for your machine, cuda, and gcc compatibilities:
## URL: https://developer.nvidia.com/cuda-toolkit-archive

### On HPC (CentOS 7.9; default gcc 4.8.5):
#   CUDA versions:
#	cuda/10.0         cuda/11.0         cuda/11.3         cuda/9.0
#	cuda/10.1         cuda/11.2         cuda/12.0         cuda/9.1
#	cuda/10.2         cuda/11.2-testing cuda/8.0
#   GCC versions:
#	gcc/10.3.0        gcc/11.2.0        gcc/6.4.0  
#	gcc/7.3.0         gcc/8.3.0         gcc/9.2.0
# + The most recent cuda version is cuda/12.0 (max supported GCC 12.1)
# + Watch out! 11.2 is dead; please use cuda 11.3, or 12.0
# + can use the default gcc 4.8.5
# + no need to module load gcc/7.3.0 on HPC
# + can also install newer gcc compilers for newer c++ features (for Altar)

### On KAMB (RHET 7.9; default gcc 4.8.5):
#   CUDA versions:
#	cuda/11.2         cuda/10.1         cuda/9.2          cuda/9.1
#   GCC versions:
#	gcc/4.8.5
# + The most recent cuda version is cuda/11.2 (max supported GCC 9.x)
# + can use the default gcc 4.8.5
# + can use "module load /home/geomod/apps/rhel7/modules/gcc/7.3.1"
# + can use conda gcc version <= 9.x


### On KAMB (UPDATED July 2023; RHET 9.2; default gcc 11.3.0; have not tested):
#   CUDA versions:
#	cuda/12.2
#   GCC versions:
#	gcc/11.3.0
# + The most recent cuda version is cuda/12.2 (supports GCC 6.x - 12.2)

```

## Check modules
Check here for updates on CUDA and GCC compatibility:
https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version
```bash
# ways to check linux system version
lsb_release -a
cat /etc/os-release
hostnamectl
uname -r

# check current loaded cuda version
nvcc --version 
which nvcc
nvidia-smi

# check current loaded gcc version
gcc --version
which gcc

# load & unload modules (CUDA or GCC)
module load
module unload

# check what is loaded
module list

# check available cuda versions
module avail cuda
module av | grep cuda
module avail gcc
```


### When install ISCE with CMAKE: mind the Bug (must load both `cuda` and `gcc`) :
1. Must use Kamb (or other machine with GPU modules, e.g., HPC) to compile
2. Must load both cuda nad gcc to compile. Otherwise, the GPU modules (**GPUgeo2rdr**, **PyCuAmpcor**, **GPUresampslc**, **GPUtopozero**) will not be complied!
+ Install with gcc/7.3.1
```bash
$ cmake blablabla
...<skip messages from above>...
-- Found Cython:  0.29.23
-- Performing Test C_FNO_COMMON
-- Performing Test C_FNO_COMMON - Success
-- Performing Test CXX_FNO_COMMON
-- Performing Test CXX_FNO_COMMON - Success
-- ISCE2s Stanford-licensed components will NOT be built.
-- Configuring done
-- Generating done
-- Build files have been written to: /home/ykliu/apps/isce2/build

$ make -j 16
...<skip messages from above>...
[100%] Built target mdx
[100%] Linking CXX shared module snaphu.so
[100%] Built target GPUgeo2rdr
[100%] Built target snaphu
[100%] Built target PyCuAmpcor
[100%] Linking CXX shared module GPUresampslc.so
[100%] Built target GPUresampslc
[100%] Linking CXX shared module GPUtopozero.so
[100%] Built target GPUtopozero

$ make install
...<skip messages from above>...
-- Installing: /home/ykliu/apps/isce2/install/packages/isce2/library/isceLib/__init__.py
-- Installing: /home/ykliu/apps/isce2/install/packages/isce2/library/__init__.py
-- Installing: /home/ykliu/apps/isce2/install/packages/isce2/__init__.py
-- Installing: /home/ykliu/apps/isce2/install/packages/isce2/release_history.py
-- Installing: /home/ykliu/apps/isce2/install/packages/isce2/helper

```

+ When install without loading gcc/7.3.1
```bash
$ cmake blablabla
...<skip messages from above>...
-- Found Cython:  0.29.23
-- Performing Test C_FNO_COMMON
-- Performing Test C_FNO_COMMON - Success
-- Performing Test CXX_FNO_COMMON
-- Performing Test CXX_FNO_COMMON - Success
-- ISCE2s Stanford-licensed components will NOT be built.
-- Configuring done
-- Generating done
CMake Warning:
  Manually-specified variables were not used by the project:

    CMAKE_CUDA_FLAGS

-- Build files have been written to: /home/ykliu/apps/isce2/build

$ make -j 16
...<skip messages from above>...
[100%] Linking CXX shared module snaphu.so
[100%] Built target snaphu
[100%] Linking Fortran executable mdx
[100%] Built target mdx

$ make install
...<skip messages from above>...
-- Installing: /home/ykliu/apps/isce2/install/packages/isce2/library/isceLib/__init__.py
-- Installing: /home/ykliu/apps/isce2/install/packages/isce2/library/__init__.py
-- Installing: /home/ykliu/apps/isce2/install/packages/isce2/__init__.py
-- Installing: /home/ykliu/apps/isce2/install/packages/isce2/release_history.py
-- Installing: /home/ykliu/apps/isce2/install/packages/isce2/helper
```



## InSAR config

```bash
##---------------------- ISCE2 ------------------------##
# load required modules for compiling ISCE-2
module load cuda/11.2
module load gcc/7.3.0

# root directory
export TOOL_DIR=~/tools

export ISCE_ROOT=${TOOL_DIR}/isce2
export ISCE_HOME=${ISCE_ROOT}/install/packages/isce
echo "load ISCE-2 core modules from "$ISCE_HOME

# make isce apps/libraries available/importable
export PATH=${PATH}:${ISCE_ROOT}/install/bin
export PYTHONPATH=${PYTHONPATH}:${ISCE_ROOT}/install/packages
export PYTHONPATH=${PYTHONPATH}:${CONDA_PREFIX}/bin


# source stack processors and PyCuAmpcor
export ISCE_STACK=${TOOL_DIR}/isce2/src/isce2/contrib/stack                     #set ISCE_STACK to the dev version
export PATH=${PATH}:${TOOL_DIR}/isce2/src/isce2/contrib/PyCuAmpcor/examples     #for cuDenseOffsets
export PYTHONPATH=${PYTHONPATH}:${ISCE_STACK}                                   #import tops/stripmapStack as python modules
export DEMDB=${DATA_DIR}/aux/DEM

alias load_tops_stack='export PATH=${PATH}:${ISCE_STACK}/topsStack; echo "load ISCE-2 topsStack from "${ISCE_STACK}/topsStack'

export OMP_NUM_THREADS=4
```

## Basics

### File transferring from KAMB to HPC

- Average speed:         300~400 Mb/sec
- Transfer 3T files:     2~3 hours

### Wait time

- 5.5 hours/pair (for long tracks)

```bash
## steps and wait time of isce2 topsApp.py
 # This is estiamted for each pair. For my first testing on 8 SLC pairs, the wait times are all similar as below

## Recources for each pair: 1 node, 1 GPU, 28 CPU cores

use_steps=
('startup'                 # 0   min   ⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻|
'preprocess'               # 4   min             |
'computeBaselines'         # 0   min             |
'verifyDEM'                # 0   min             |
'topo'                     # 5   min (with GPU)  |
'subsetoverlaps'           # 0   min             |
'coarseoffsets'            # 0   min             |____ < 20 min
'coarseresamp'             # 0   min             |     
'overlapifg'               # 0   min             |
'prepesd'                  # 0   min             |
'esd'                      # 0   min             |
'rangecoreg'               # 0   min             |
'fineoffsets'              # 9   min (with GPU)__|

'fineresamp'               # 35  min (resampling; no gpu)⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻|
'ion'                      # 175 min (resampling; no gpu)              |
'burstifg'                 # 60  min (sing-look igram, coh; no gpu)    |
'mergebursts'              # 7   min                                   |~315 min
'filter'                   # 1   min (filter strength=0)               |    
'unwrap'                   # 30  min (no gpu)                          |   
'unwrap2stage'             # 0   min                                   |
'geocode'                  # 4-6 min                    _______________|
)

## Total runtime: ~5.5 hours for one pair
```

## Resource usage

### Allocating resources in SLURM script

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

### Resource usage by my account (command: `sreport`)

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

### Disk usage by me on HPC (command: `mmlsquota`)

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
