# TopsApp SAR / InSAR processing code and notes

## 1. Installation of ISCE2
Not meant to be covered in full detials here... Please refer to below:

+ @[yunjunz](https://github.com/yunjunz)'s guidance: <https://github.com/yunjunz/conda_envs>
+ @[lijun99](https://github.com/lijun99)'s guidance: <https://github.com/lijun99/isce2-install#linux-with-anaconda3--cmake>
+ @[CunrenLiang](https://github.com/CunrenLiang)'s repo: https://github.com/CunrenLiang/isce2
+ ISCE on KAMB wiki: <https://geo.caltech.edu/doku.php?id=local:insar:iscep-on-kamb> (May need login credentials based on a request to the admin)

Yunjun's and Lijun's guide (using cmake) have been tested and they works. If it doesn't for you, then check this danger zone:

+ DANGER: Mind the compiler compatibilities

```bash
### Special notes for CUDA and GCC

# Load compilers, as well as cuda/nvcc compiler for GPU modules.
## Note: a cuda version determines up to which gcc is compatible
## Note: a newer gcc will require an even newer cuda version
## Check here for your machine, cuda, and gcc compatibilities.
## Must check:
## + CUDA Toolkit Archive: https://developer.nvidia.com/cuda-toolkit-archive
## Optional read:
## + a stackoverflow post: https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version
## + a github gist: https://gist.github.com/ax3l/9489132

### On HPC (CentOS 7.9; default gcc 4.8.5):
#   CUDA versions:
#	cuda/10.0         cuda/11.0         cuda/11.3         cuda/9.0
#	cuda/10.1         cuda/11.2         cuda/12.0         cuda/9.1
#	cuda/10.2         cuda/11.2-testing cuda/8.0
#   GCC versions:
#	gcc/10.3.0        gcc/11.2.0        gcc/6.4.0
#	gcc/7.3.0         gcc/8.3.0         gcc/9.2.0
# + The most recent cuda version is cuda/12.0 (max supported GCC 12.1)
# + Watch out! 11.2 is dead; please use cuda 12.0
# + can use the default gcc 4.8.5
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
```


## 2. Copy this folder to your track folder
```
# Copy teh scripts to the process folder
cp -r run_topsApp ./process

# make dir for SLCs
mkdir data & cd data

# download SLCs here

```


## 3. Modify the scripts/example templates
```
vi run_job_series.sh

vi example/reference.xml

vi example/topsApp.xml

vi example/topsApp_geocode.xml

vi example/topsAppDenseoffsets_geocode.xml

## Create pair folders
s1_pair.py -d ../data -x example/reference.xml
```


## 4. run the script
```
time bash run_job_series.sh
```
