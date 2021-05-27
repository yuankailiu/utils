# ISCE2/Sentinel-1 tools 

## Info:

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
```
