If you are using CUDA modules from pyre:

Install pyre from `pyre CUDA branch`
    + source code by Lijun Zhu to work with AlTar 2.0
    + not the one from Aïvázis, nor from pypi

General guide:
    + https://altar.readthedocs.io/en/cuda/cuda/Installation.html

Compile reminders:
    (https://altar.readthedocs.io/en/cuda/cuda/Installation.html#cmake-options)
    + built with cmake
    + -DCMAKE_INSTALL_PREFIX=YOUR_TARGET_DIR
    + -DCMAKE_CUDA_ARCHITECTURES="60"    (targeting NVIDIA Tesla P100 GPU)
    + -DCMAKE_CUDA_ARCHITECTURES="35;60" (targeting both K40/K80 and P100 GPUs)
    + -DCMAKE_CUDA_ARCHITECTURES="70"    (targeting V100 GPUs)

Example:
    + install to conda virtual env, with V100 GPUs on Kamb (Aug 2024)

    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CUDA_ARCHITECTURES="70"



Paths setting in .bash_profile:
    ##--------- pyre cuda branch -------##
    export PATH=/home/ykliu/apps/mambaforge/envs/insar/bin:${PATH}
    export LD_LIBRARY_PATH=/home/ykliu/apps/mambaforge/envs/insar/lib:${LD_LIBRARY_PATH}
    export PYTHONPATH=/home/ykliu/apps/mambaforge/envs/insar/packages:${PYTHONPATH}
