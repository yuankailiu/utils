
# [Stack Sentinel processor](https://github.com/yuankailiu/isce2/tree/test_liang/contrib/stack/topsStack)

### General workflow:
1. get Single Look Complex data
2. Write a script to get supplementary files:
	+ get AUX if needed, only once
	+ get DEM
	+ get orbits if needed (but `stackSentinel.py` can do it for you)
3. Somehow use a script to do pre-selection:
	+ Run s1_select_ion.py to select usable ionosphere SLCs (with the same starting ranges)
		- need to add to the code to report: number of slices: 1720; number of acquisitions: 252
		- need to restore some removed scenes, manually
	+ Run s1_version.py to select usable pairs (gaps, the previous step takes care of it?), s1_slice.txt?
	+ Remove or store the not-used SLCs, and write a log about these
4. Actual processing script
5. Move the processed files in between servers.
  + Download [msrsync](https://github.com/jbd/msrsync) for multi-stream rsync.
  +

### [Documentation](https://github.com/yuankailiu/isce2/tree/test_liang/contrib/stack/topsStack#41-example-workflow-coregistered-stack-of-slc)

#### Stack SLCs workflow

**run_01_unpack_topo_reference:**

Includes a command that refers to the config file of the stack reference, which includes configuration for running `topo` for the stack reference. Note that in the pair-wise processing strategy, one should run `topo` (mapping from range-Doppler to geo coordinate) for all pairs. However, with `stackSentinel.py`, `topo` needs to be run only one time for the reference in the stack. This stage will also unpack Sentinel-1 TOPS reference SLC. Reference geometry files are saved under `geom_reference/`. Reference burst SLCs are saved under `reference/`.

**run_02_unpack_secondary_slc:**

Unpack secondary Sentinel-1 TOPS SLCs using ISCE readers. For older SLCs which need antenna elevation pattern correction, the file is extracted and written to disk. For newer version of SLCs which don’t need the elevation antenna pattern correction, only a gdal virtual “vrt” file (and isce xml file) is generated. The “.vrt” file points to the Sentinel SLC file and reads them whenever required during the processing. If a user wants to write the “.vrt” SLC file to disk, it can be done easily using `gdal_translate` (e.g. `gdal_translate –of ENVI File.vrt File.slc`). Secondary burst SLCs are saved under `secondarys/`.

**run_03_average_baseline:**

Computes average baseline for the stack, saved under `baselines/`. These baselines are not used for processing anywhere. They are only an approximation and can be used for plotting purposes. A more precise baseline grid is estimated later in run_13_grid_baseline only for `-W slc` workflow.

**run_04_extract_burst_overlaps:**

Burst overlaps are extracted for estimating azimuth misregistration using NESD technique. If coregistration method is chosen to be “geometry”, then this run file won’t exist and the overlaps are not extracted. Saved under `reference/overlap/` and `geom_reference/overlap`.

**run_05_overlap_geo2rdr:**

Running geo2rdr to estimate geometrical offsets between secondary burst overlaps (`secondary/`) and the stack reference (`reference`) burst overlaps. Saved under `coreg_secondarys/YYYYMMDD/overlap`.

**run_06_overlap_resample:**

The secondary burst overlaps are then resampled to the stack reference burst overlaps. Saved under `coreg_secondarys/YYYYMMDD/overlap`.

**run_07_pairs_misreg:**

Using the coregistered stack burst overlaps generated from the previous step, differential overlap interferograms are generated and are used for estimating azimuth misregistration using Enhanced Spectral Diversity (ESD) technique. Saved under `misreg/azimuth/pairs/` and `misreg/range/pairs/`.

**run_08_timeseries_misreg:**

A time-series of azimuth and range misregistration is estimated with respect to the stack reference. The time-series is a least-squares estimation from the pair misregistration from the previous step. Saved under `misreg/azimuth/dates/` and `misreg/range/dates/`.

**run_09_fullBurst_geo2rdr:**

Using orbit and DEM, geometrical offsets among all secondary SLCs and the stack reference is computed. Saved under `coreg_secondarys/`.

**run_10_fullBurst_resample:**

The geometrical offsets, together with the misregistration time-series (from the previous step) are used for precise coregistration of each burst SLC by resampling to the stack reference burst SLC. Saved under `coreg_secondarys/`.

**run_11_extract_stack_valid_region:**

The valid region between burst SLCs at the overlap area of the bursts slightly changes for different acquisitions. Therefore, we need to keep track of these overlaps which will be used during merging bursts. Without these knowledges, lines of invalid data may appear in the merged products at the burst overlaps.

**run_12_merge_reference_secondary_slc:**

Merges all bursts for the reference and coregistered SLCs and apply multilooking to form full-scene SLCs (saved under  `merged/SLC/`). The geometry files are also merged including longitude, latitude, shadow and layer mask, line-of-sight files, etc. under `merged/geom_reference/`.

**run_13_grid_baseline:**

A coarse grid of baselines between each secondary SLC and the stack reference is generated. This is not used in any computation. Saved under `merged/baselines/`.

#### Interferogram workflow (follow-on steps)

~~**run_13_grid_baseline:**~~

This step does not exist in `-W interferogram` workflow.

**run_13_generate_burst_igram:**

Take the stack of coregistered burst SLCs (`reference` and  `coreg_secondary`) to generate burst interferograms. These burst-level interferograms are saved under `interferograms/`.

**run_14_merge_burst_igram:**

Merge the burst interferograms and apply multilooking to form a full-scene interferogram for each acquisition. Saved under `merged/interferograms/fine.int`

**run_15_filter_coherence:**

Use the full-scene SLCs in `merged/SLC/` to generate the complex coherence. Apply filtering to the full-scene interferograms and the coherence files. These files are saved as `fine.cor`, `filt_fine.int`, `filt_fine.cor` under `merged/interferograms/`.

**run_16_unwrap:**

Apply unwrapping to the multilooked and filtered interferograms `merged/interferograms/filt_fine.int`, generate the unwrapped files, `merged/interferograms/filt_fine.unw`.


**Cunren wrote these below**
-   run_ns+1_subband_and_resamp
-   run_ns+2_generateIgram_ion
-   run_ns+3_mergeBurstsIon
-   run_ns+4_unwrap_ion
-   run_ns+5_look_ion
-   run_ns+6_computeIon
-   run_ns+7_filtIon
-   run_ns+8_invertIon
-   run_ns+9_filtIonShift
-   run_ns+10_invertIonShift
-   run_ns+11_burstRampIon
-   run_ns+12_mergeBurstRampIon


# Getting data
## Single-Look complex zip files (SLCs)
### 1. asf_search multi-threads (ASF server)
```python
## Download SLCs multi-threads

import asf_search as asf

#SELECT AOI - edit point
aoi = 'POLYGON((36.099 39.0273,35.4648 35.8591,38.2659 35.457,39.0054 38.6309,36.099 39.0273))'


search_results = asf.geo_search(
    platform=asf.SENTINEL1,
    intersectsWith=aoi,
    start='2015-01-01',
    end='2023-02-07',
    processingLevel=asf.SLC,
    beamMode=asf.IW,
    relativeOrbit=21,#Change the path
    flightDirection=asf.DESCENDING,
)

session = asf.ASFSession().auth_with_creds(username, password)

print(f"--Downloading Results--")
search_results.download(
     path = './',
     session = session,
     processes = 10
  )

```

### 2. SSARA client multi-threads (ASF server)
```
## SSARA download data (https://web-services.unavco.org/brokered/ssara/)
ssara_federated_query.py --platform=SENTINEL-1A -i 'YOUR_POLYGON_HERE' -r 94 --flightDirection D --maxResults 5000 --download --parallel 8

# append --s1orbits: download orbits on the fly

### Aqaba polygon:
POLYGON((32.963 31.0769,32.8533 28.8243,33.8895 27.5921,34.466 27.7981,35.0044 27.9571,35.9853 26.7334,37.1809 26.5266,38.4394 32.8157,35.081 33.2915,34.4636 31.5101,32.963 31.0769))
```

### 3. ESA wget (ESA server)
```
## GUI: https://scihub.copernicus.eu/dhus/#/home

## wget SLC products from ESA Copernicus Open Access Hub
#  with super small bandwidth; slow

wget --content-disposition --continue --user={USERNAME} --password={PASSWORD} "https://scihub.copernicus.eu/dhus/odata/v1/Products('558e85b2-500a-4e24-acc5-ce82170bcf7e')/\$value"
```

### 4. Copernicus search viewer
[ESA Copernicus search](https://dataspace.copernicus.eu/browser/)


## Orbits
### 1. pre-download all available orbits from ESA
```
## wget the Sentinel-1 precise orbits from ASF archive
wget -r -l inf --no-remove-listing -nc --include aux_poeorb --execute robots=off --no-host-directories --cut-dirs=1 --reject="index.html*" --continue https://s1qc.asf.alaska.edu/aux_poeorb/
# -r,  --recursive
# -l,  --level=NUMBER       maximum recursion depth (inf or 0 for infinite).
```

### 2. isce2 stack processor `fetchOrbit.py`
```
fetchOrbit.py -d path/to/your/zipfiles/

usage: fetchOrbit.py [-h] [-i INPUT] [-d INDIR] [-o OUTDIR]

Fetch orbits corresponding to given SAFE package

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to SAFE package of interest
  -d INDIR, --indir INDIR
                        Directory to SAFE package(s) of interest
  -o OUTDIR, --output OUTDIR
                        Path to output directory
```


## Digital elevation model

### 1. SRTM link
```
## Cunren's script
download_dem.py

## stack Processor script
```

### 2. sardem
