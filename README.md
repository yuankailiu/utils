# Miscellaneous utilities

For various purposes. Mostly InSAR-related.

## MintPy:

### `MintPyWrapper/`

Preparation, post-processing scripts, and configurations for MintPy.

## ISCE2:

### `run_topsApp/`

Scripts to run ISCE2 topsApp for Sentinel-1 in particular, modified from [Cunren Liang](https://github.com/CunrenLiang). This is meant for a single-interferogram workflow (e.g., a few coseismic observations). TopsApp can support GPUs.

For a stack of interferograms/coherence (e.g., secular time-series analysis, coherence time series), check the [ISCE2 stack processor](https://github.com/isce-framework/isce2/tree/main/contrib/stack) and [sar-proc](https://github.com/earthdef/sar-proc) for submitting a stack of jobs with [High Performance Computing](https://www.hpc.caltech.edu/resources) (HPC) with [SLURM commands](https://www.hpc.caltech.edu/documentation/slurm-commands).

### `s1_tools/`

Sentinel-1 scripts from [Cunren Liang](https://github.com/CunrenLiang) for ISCE2 pre-/post-processing.


## Submodules:
Good stuff mentioned in [Yunjun's insar conda-env installation guide](https://github.com/yunjunz/conda-envs).
### `SSARA/`

Links to [Seamless SAR Archvie (SSARA) Client project](https://web-services.unavco.org/brokered/ssara/).
Now the upstream is moved to [earthscope gitlab repository](https://gitlab.com/earthscope/public/sar/ssara_client). Other UNAVCO links (wiki, etc.) may have deprecated.
```
# clone from Yunjun's branch on gitlab
git clone https://gitlab.com/yunjunz/SSARA.git
```

### `sardem/`
A tool for making Digital Elevation Maps (DEMs) in binary data format (16-bit integers, little endian) for use in interferometric synthetic aperture radar (InSAR) processing. Upstream from [scottstanie's repo](https://github.com/scottstanie/sardem).
```
# clone from Yunjun's branch on gitlab
git clone https://github.com/yunjunz/sardem.git
```

## Others:

**`docs/`**: notes, installation guides, etc.

**`notebooks/`**: notebooks.

**`sarut/`**: some SAR and InSAR scripts.

**`shell/`**: bash shell scripts and my bashrc files.

**`junk/`**: unorganized stuff, temporary files.

**`requirements.txt`**: better get these once you have isce2 installed.

```
mamba install requirements.txt --yes
```
