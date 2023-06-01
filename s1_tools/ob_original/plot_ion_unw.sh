#!/bin/bash

# Plot ionosphere and unw tiles for a quick survey
# TODO modify so we can run from the 'process' dir more easily
# Maybe add to the default processing chain so we finish with this product

cwd=$(pwd)
insar_dir='/marmot-nobak/olstephe/InSAR/Makran'
track='T86a'
track_dir=$insar_dir/$track/process
cd $track_dir


# Plot unw
mkdir ${track}_full_unw_test
cd ${track}_full_unw_test
plot_unw.py -dir .. -svg ${track}_full_unw.svg

# Plot ion
cd ..
mkdir ${track}_full_ion_test
cd ${track}_full_ion_test
plot_ion.py -dir .. -svg ${track}_full_ion.svg

cd $cwd
