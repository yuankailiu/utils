# topsApp

## 1. modify xmls in `example/`


## 2. create pair directories
```
# relatvie path, so execute this line in the example/ folder
s1_pairs.py -d ../../data/ -x reference.xml

# then move the created folders and log to the parent dir
mv 20*-20* pair_log* ../
```

## 3. Modify `run_job_series.sh`
+ Add the pair number there
+ `export CUDA_VISIBLE_DEVICES=xx` (xx can be 0-7 on Kamb)
+ `export OMP_NUM_THREADS=yy` (yy can be 4, 8, 16, etc.)



## Build a stack.vrt for all files
```bash

ls ./merged/SLC/*/*.full > full_slc.txt

gdalbuildvrt -input_file_list full_slc.txt -separate full_slc.vrt

```


### topsApp example files:

```bash
#### merged/

## steps: 'startup' to 'fineoffset'
los.rdr.full.vrt         # all SLCs
lat.rdr.full.vrt         # [../geom_reference/IW1/lat_01.rdr.vrt, ...]; Size is 68508, 49801; Band 1 Block=128x128 Type=Float64, ColorInterp=Undefined
lon.rdr.full.vrt         # [../geom_reference/IW1/lon_01.rdr.vrt, ...]; Size is 68508, 49801; Band 1 Block=128x128 Type=Float64, ColorInterp=Undefined
z.rdr.full.vrt           # [../geom_reference/IW1/hgt_01.rdr.vrt, ...]; Size is 68508, 49801; Band 1 Block=128x128 Type=Float64, ColorInterp=Undefined
reference.slc.full.vrt   # all reference SLCs
secondary.slc.full.vrt   # all secondary SLCs
topophase.flat.full.vrt  # [../fine_interferogram/IW1/burst_01.int.vrt, ...]; Size is 68508, 49801; Band 1 Block=128x128 Type=Float32, ColorInterp=Undefined
topophase.cor.full.vrt   # [../fine_interferogram/IW1/burst_01.cor.vrt, ...]; Size is 68508, 49801; Band 1 Block=128x128 Type=Float32, ColorInterp=Undefined

## steps: 'finresample' to 'mergebursts'
#      (topophase.flat.full) -----> (topophase_ori.flat)
#  step 'ion': (topophase_ori.flat) - (topophase.ion) = (topophase.flat)
topophase_ori.flat       # ['band1'=complex] = original wrapped ifgram
los.rdr                  # ['incidenceAngle'=float, 'azimuthAngle'=float]
topophase.cor            # ['band1', 'band2'] = ['magnitude'=float, 'correlation'=float within 0 & 1]
topophase.ion            # ['band1'=float] = ['iono_phase_delay' = radian]
topophase.flat           # wrapped ifgram; ['band1'=complex]

## step: 'filter'
#     (topophase.flat) --> (filt_topophase.flat)
filt_topophase.flat      # filtered wrapped ifgram; ['band1'=complex]
phsig.cor                # phase sigma correlation; ['band1'=float within 0 and 1]

## step: 'unwrap' & 'unwrap2stage'
#     (filt_topophase.flat) --> (filt_topophase.unw)
filt_topophase.unw.conncomp    # filtered unwrapped connected component; ['band1'=float]
filt_topophase.unw             # filtered unwrapped ifgram; ['magnitude'=float, 'phase'=float]

## step: 'geocode'
#     (filt_topophase.unw) --> (filt_topophase.unw.geo)
filt_topophase.unw.geo   # filtered unwrapped geocoded ifgram; ['band1', 'band2'] = ['magnitude'=float, 'phase'=float]
phsig.cor.geo            # phase sigma correlation geocoded; ['band1'=float within 0 and 1]
dem.crop                 # DEM file; ['band1'=float]
filt_topophase.unw.conncomp.geo  # filtered unwrapped connected component geocoded ['band1'=float]
```

```bash
## check iono correction using imageMath.py
# this gives you the phase in radian (you need to wrap it from -pi to pi in python)
#     PYTHON: angle = np.rad2deg(np.arctan2(np.sin(phase), np.cos(phase)))
imageMath.py -e="arg(a)-b" -o iono_check.flat --a=topophase_ori.flat --b=topophase.ion

# to re-construct the pre-correction unwrap phase
imageMath.py -e="a_0;a_1+b" -o filt_topophase_ori.unw.geo --a=filt_topophase.unw.geo --b=topophase.ion.geo
```

---

# UNAVCO training class

## **The main github repo from ASF**

This builds the entire structure of directories and notebooks that we’ve been working for the week:

[https://github.com/asfadmin/asf-jupyter-notebooks](https://github.com/asfadmin/asf-jupyter-notebooks)

## Specific github repository

- isce-docs repo:

    [https://github.com/isce-framework/isce2-docs/tree/master/Notebooks/UNAVCO_2020](https://github.com/isce-framework/isce2-docs/tree/master/Notebooks/UNAVCO_2020)

- aria-tools repo:

    [https://github.com/aria-tools/ARIA-tools](https://github.com/aria-tools/ARIA-tools)

- Gareth’s repo:

    [https://github.com/geniusinaction](https://github.com/geniusinaction)

- MintPy:

    [https://github.com/insarlab/MintPy](https://github.com/insarlab/MintPy)

- FRInGe for time series

    [https://github.com/isce-framework/fringe](https://github.com/isce-framework/fringe)

- Previous year’s ARIA-tools course:

    [https://www.youtube.com/watch?v=ogLhijYy2ck](https://www.youtube.com/watch?v=ogLhijYy2ck)

## Other links:

- Satellites basic wavelength info

- Homework Deposit:

    [https://drive.google.com/drive/folders/13Wifso_EpVBDuBS7B6qTaZWXaErU5G9q](https://drive.google.com/drive/folders/13Wifso_EpVBDuBS7B6qTaZWXaErU5G9q)

## Recordings:

- Recording of Monday lectures:

    [https://jpl.webex.com/jpl/j.php?MTID=mf615b725ad7484d27d22fa69f84bc1e5](https://jpl.webex.com/jpl/j.php?MTID=mf615b725ad7484d27d22fa69f84bc1e5)

- Recording of Tuesday lectures:

    [https://jpl.webex.com/recordingservice/sites/jpl/recording/fe37d96d02504979b2ea4c1743de4a5e](https://jpl.webex.com/recordingservice/sites/jpl/recording/fe37d96d02504979b2ea4c1743de4a5e)

- JPL full Troposphere correction lectures (David Bekaert):

    [https://www.youtube.com/watch?v=WzUgtK84sqU&feature=youtu.be&t=13148](https://www.youtube.com/watch?v=WzUgtK84sqU&feature=youtu.be&t=13148)
