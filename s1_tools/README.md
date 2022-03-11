# ISCE2/Sentinel-1 tools

## Brief:

Scripts initially written by Cunren Liang 2018-2020

More scripts were added later by Ollie Stephenson and Yuan-Kai Liu @ 2021

These codes are now put together for usage:
- Place the codes in a common folder e.g., s1_tools/
- Place the s1_tools/ folder under ~/bin or an arbitray exported path defined in your ~/.bashrc
    - For example, add this line in ~/.bashrc: `export PATH=$PATH:$HOME/YOUR_OWN_PATH/s1_tools`
- Simply call these codes anywhere you like


## List of scripts:

```
 download_dem.sh            cunren
 s1_fetch_aux_cal.py        cunren
 s1_fetch_orbit.py          cunren
 s1_version.py              cunren, ollie
 s1_pairs.py                cunren, kai, ollie
 add_igram_pairs.py         ollie
 plot_ion.py                cunren
 plot_unw.py                cunren
 plot_ion_unw.sh            ollie
 plot_azshift.py            cunren
 rg_filter.sh               cunren
 rg_filter.c                cunren
 rg_filter                  cunren
 getBaselines.py            kai
 saveKml.py                 ollie
 check_runtime.py           kai
```


## TopsApp multiple pairs:
Author: Cunren Laing

ISCE2 package from Cunren Liang's repository:
https://github.com/CunrenLiang/isce2

#### Basic workflow
1. download SLC
https://search.asf.alaska.edu/#/

2. download orbit
https://qc.sentinel1.eo.esa.int/
`s1_fetch_orbit.py` (the url is obsoltete)

3. download auxiliary data
https://qc.sentinel1.eo.esa.int/
`s1_fetch_aux_cal.py` (the url is obsolete)

4. download DEM in 1 arcsec, 3 arcsec and water body
`download_dem.sh`

5. select SLCs:
`s1_version.py`
    - No gap (missing slices) in between slices, better to have same coverage
    - Reference and secondary have same starting ranges for all swaths; otherwise, the program has to estimate a phase difference between adjacent swaths, which might not be accurate enough. Therefore, reference and secondary better be acquired by the same satellite, e.g. Sentinel-1A.
    - All slice versions of an acquisition should be the same
   - Delete redundant slices (normally same slices of different versions).

6. Prepare input xml files topsApp.xml and reference/secondary.xml
See files in folder `./example`

7. Prepare pairs
`s1_pairs.py`

8. Process all pairs using `topsApp.py`
In each pair folder, do the following:
   - run this command for all pairs sequentially (process pair by pair)
    ```bash
    topsApp.py ../example/topsApp.xml --end=fineoffsets
    ```

   - run the following command concurrently (process a few pairs concurrently)
    ```bash
    topsApp.py ../example/topsApp.xml --start=fineresamp --end=unwrap2stage
    ```

    - run the following command concurrently (process a few pairs concurrently)
    (change 1 arcsec DEM to 3 arcsec DEM in the topsApp.xml to save time and space before running the command)
    ```bash
    topsApp.py ../example/topsApp.xml --dostep=geocode
    ```