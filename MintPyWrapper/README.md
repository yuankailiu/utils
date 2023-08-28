# Workflow to run mintpy on a track

Author: Yuan-Kai Liu, 2022

## Prerequisites:

### softwares:
- Mintpy [installed](https://github.com/insarlab/MintPy/blob/main/docs/installation.md)
- ISCE2 stack processor [installed](https://github.com/isce-framework/isce2/blob/e77d2073115725c08c395248a790d94c4a65ea9d/contrib/stack/topsStack/README.md)
- ICAMS [installed](https://github.com/yuankailiu/ICAMS)

### data:
- a stack of interferograms from e.g., ISCE2 topsStack
- DEM
- ECMWF data agreement signed for downloading weather models

## Steps:

1. copy `scripts/` and the related cfg files to the `mintpy/` folder of each track

2. go to the `mintpy/` folder of a track

3. run mintpy `smallbaselineApp.py` in a screen session:

    + generate the run files: `mintpyRuns.py -p *.par`

    + now execute run files in serial:

        -   run_0_prep

        -   run_1_network

        -   run_closurePhase.sh [optional]

        -   run_2_inversion

        -   run_3_corrections

        -   run_4_velocity

        -   run_icams.sh [optional]

        -   run_5_velocityPlot
