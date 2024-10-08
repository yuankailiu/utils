
# ISCE2 related links

Participate in discussions with the users/developers community!

- [JPL/ISCE2 GitHub repo](https://github.com/isce-framework/isce2)
- [ISCE Forum on Caltech Earthdef](http://earthdef.caltech.edu/projects/isce_forum/boards): May need login credentials based on a request to the admin

<br/>

## Reference to the experts

- @[yunjunz](https://github.com/yunjunz)'s guidance: <https://github.com/yunjunz/conda-envs>
- @[lijun99](https://github.com/lijun99)'s guidance: <https://github.com/lijun99/Caltech_HPC/blob/main/isce2.md>
- @[CunrenLiang](https://github.com/CunrenLiang)'s repo: https://github.com/CunrenLiang/isce2
- ISCE on KAMB wiki: <https://geo.caltech.edu/doku.php?id=local:insar:iscep-on-kamb> (May need login credentials based on a request to the admin)

<br/>

## Install isce2 on HPC

**Main reference: [Yunjun's installation](https://github.com/yunjunz/conda-envs).**

1. [Install **conda**](https://github.com/yunjunz/conda-envs/tree/main?tab=readme-ov-file#1-install-conda)

2. Install **isce2 (development version) and mintpy (development version):**

    First, [install dependencies into an mamba environment `isce2` or `insar`](https://github.com/yunjunz/conda-envs/blob/main/isce2/README.md#b-install-dependencies-to-isce2-environment)

    While installing dependencies, watch out for dependency on your own OS system. Some potential incompatibilities may occur on Caltech HPC and Simons' Kamb.

    1. Add `pybind11`. PyCuAmpcor need pybind11 to compile since updates in 2022. If it is not present, PyCuAmpcor will be skipped. We still need cython for other modules. Pyind11 wrapper is probably only done for PyCuAmpcor.

    2. PyCuAmpcor is not compatible with newer versions of gcc and C compilers. So do not install new compilers from mamba.

    3. Don't install `openmotif` & `xorg-libs` for compiling `mdx`. Conda/mamba’s openmotif has some issues with Linux RHEL9 OS. Please avoid that installing it and just use the system’s lib (check with `ldconfig -p | grep libXm`).

    Thus, you will have to install `pybind11`:
    ```bash
    mamba install pybind11
    ```

    Then, ignoring/commeting those compilers in [isce2/requirements.txt](https://github.com/yunjunz/conda-envs/blob/main/isce2/requirements.txt):
    ```bash
    #for compilation [cross-platform names from conda-forge]
    #c-compiler
    #cxx-compiler
    #fortran-compiler
    #for mdx compile/install
    #openmotif
    #openmotif-dev
    #xorg-libxdmcp
    #xorg-libxft
    #xorg-libxmu
    #xorg-libxt
    ```

    and in [insar/requirements.txt](https://github.com/yunjunz/conda-envs/blob/main/insar/requirements.txt):
    ```bash
    # compilers
    #cxx-compiler      #for scalene installation on Linux
    #fortran-compiler  #for pysolid installation in development mode
    ```


3. **Install ISCE2 source code:**
    Make sure you have a system CUDA. You may need to load it manually `module load cuda`
    ```bash
    which nvcc      # check which cuda
    nvcc --version  # check version
    ```

    Make sure you loaded the system default compilers (before we did not get any compilers from conda/mamba).
    ```bash
    CC=/usr/bin/gcc; CXX=/usr/bin/g++; FC=/usr/bin/gfortran

    # you can put this in your .bashrc
    export CC=/usr/bin/gcc; export CXX=/usr/bin/g++; export FC=/usr/bin/gfortran
    ```

    Now, [install isce2 source code](https://github.com/yunjunz/conda-envs/blob/main/isce2/README.md#c-install-isce-2-to-isce2install_-folder) to `isce2/install_*` folder.


4. [Activate your mamba env and do the config setup in your `.bashrc`](https://github.com/yunjunz/conda-envs/blob/main/isce2/README.md#d-setup). So every time you login, it will be ready.

5. [Test the installation](https://github.com/yunjunz/conda-envs/blob/main/isce2/README.md#e-test-the-installation)
    ```bash
    topsApp.py -h
    cuDenseOffsets.py -h
    smallbaselineApp.py -h
    stackSentinel.py -h
    mdx.py
    ```

Note:
If you have mamba installed the c-compilers, openmotif, and xorg packages and you run into issues (either `PyCuAmpcor`, `mdx` is not properly compiled), you can uninstall them and try to [recompile](https://github.com/lijun99/Caltech_HPC/blob/main/isce2.md).
```bash
mamba remove c-compiler cxx-compiler fortran-compiler openmotif
```
But mamba errors may arise since you install a lot of other things that depends on them. In that case, simply erase the whole env and redo it from scratch.



### Module version commands
Check here for updates on [CUDA and GCC compatibility](https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version). But things get updated often.
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
module avail gcc
```

### Old notes for CUDA and GCC

```bash

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
