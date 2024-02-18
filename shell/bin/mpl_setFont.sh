#!/bin/bash

# path where I store my own fonts
FONT_DIR='/net/kraken/bak/ykliu/fonts'
pattern='site-packages/matplotlib/mpl-data/fonts/ttf'

# copy fonts to conda base dir
echo $CONDA_BASE/lib/python3.*/$pattern | xargs -n 1 \cp -r $FONT_DIR/*/*.ttf

# copy fonts to other conda env dirs
echo $CONDA_BASE/envs/*/lib/python3.*/$pattern | xargs -n 1 \cp -r $FONT_DIR/*/*.ttf


# copy custom matplotlibrc (font priority) to the $MPLCONFIGDIR
cp $FONT_DIR/matplotlibrc $MPLCONFIGDIR

# delete the existing font list json
rm -rf $MPLCONFIGDIR/fontlist*

# Re-compile the font caches to create a new font list; and make a plot to check
python -v -c "import matplotlib.pyplot as plt; plt.figure(); plt.plot([1, 10]); plt.title('Matplotlib compile done'); plt.show()"

echo 'finished'

