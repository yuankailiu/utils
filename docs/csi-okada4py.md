# CSI installation manual improvements

The initial manual has typos and bugs.
https://www.geologie.ens.fr/~jolivet/csi/installation.html

## the `requirements.txt`
```
#python3      # you may have it already
#gcc          # to compile okada4py, but can use system default, export CC=/usr/bin/gcc
#numpy        # you may have it already
scipy
shapely
pyproj
matplotlib
cartopy
multiprocess
h5py
#okada4py     # (available on GitHub)
```

## okada4py

[available on GitHub](https://github.com/jolivetr/okada4py)

Lijun have fixed the setup.py issue, pull the branch from:
https://github.com/lijun99/okada4py/tree/setup

or simply clone his repo
```bash
git clone https://github.com/lijun99/okada4py.git
git checkout setup
```

### Compilation
```
export CC=/usr/bin/gcc
python setup.py build
```

### Linking
```
python setup.py install --user
```


### Test okada4py

Avoid running it under the source directory, which has a directory named okada4py. Try go to the test directory and run the test.

```bash
python3 -c "import okada4py"  # run import test

python3 test.py    # run the test script
```
