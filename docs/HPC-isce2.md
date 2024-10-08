# Running isce2 topsApp on HPC

This is a note and list of useful things when running isce2 topsApp on the Caltech campus High Performing Computing (HPC) cluster.


## New modifications
***Try to include these into the new Python workflow as well!***

Split large job arrays into smaller chunks that are concurrently running (ArrayTaskThrottle) to avoid jamming the I/O. Most of the steps (or stages) in topsStack can split into every 200 jobs (pairs) `--array=1-500%200`. For `17_subband_and_resamp_a014`, use even more conservative ArrayTaskThrottle `--array=1-500%100`.

For step `13_generate_burst_igram-25935841_49.out`, we may not need 80G of REM, need to test.

We read and report disk space on the 1st and the 300th job in all topsStack steps (or stages). Some of the steps only take 5 minutes in general, but reading the disk space will take 5 to 10 mins.

On Caltech HPC, the upper limit of job array (MaxArraySize) we can submit is 1001.
You can check via ` scontrol show config | grep MaxArraySize`.
So if we have more than 1001 jobs, the submission will fail. We need to slit large stages into two `.sbatch` scripts, specifying the job array within the MaxArraySize:

```bash
## Assume you have a stage that has 1047 jobs. You can split it to following:

#-----> within the first script
#SBATCH --array=1-500%200

#-----> within the second script
#SBATCH --array=1-547%200
SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID+500))
```

## Basics

### File transferring from KAMB to HPC

- Average speed:         300~400 Mb/sec
- Transfer 3T files:     2~3 hours

### Wait time

- 5.5 hours/pair (for long tracks)

```bash
## steps and wait time of isce2 topsApp.py
 # This is estiamted for each pair. For my first testing on 8 SLC pairs, the wait times are all similar as below

## Recources for each pair: 1 node, 1 GPU, 28 CPU cores

use_steps=
('startup'                 # 0   min   ⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻|
'preprocess'               # 4   min             |
'computeBaselines'         # 0   min             |
'verifyDEM'                # 0   min             |
'topo'                     # 5   min (with GPU)  |
'subsetoverlaps'           # 0   min             |
'coarseoffsets'            # 0   min             |____ < 20 min
'coarseresamp'             # 0   min             |
'overlapifg'               # 0   min             |
'prepesd'                  # 0   min             |
'esd'                      # 0   min             |
'rangecoreg'               # 0   min             |
'fineoffsets'              # 9   min (with GPU)__|

'fineresamp'               # 35  min (resampling; no gpu)⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻⎻|
'ion'                      # 175 min (resampling; no gpu)              |
'burstifg'                 # 60  min (sing-look igram, coh; no gpu)    |
'mergebursts'              # 7   min                                   |~315 min
'filter'                   # 1   min (filter strength=0)               |
'unwrap'                   # 30  min (no gpu)                          |
'unwrap2stage'             # 0   min                                   |
'geocode'                  # 4-6 min                    _______________|
)

## Total runtime: ~5.5 hours for one pair
```

### Resource usage by my account (command: `sreport`)

```bash
--------------------------------------------------------------------------------
Cluster/Account/User Utilization 2021-01-01T00:00:00 - 2021-05-28T16:59:59 (12758400 secs)
Usage reported in TRES Hours
--------------------------------------------------------------------------------
  Cluster         Account     Login     Proper Name      TRES Name      Used
--------- --------------- --------- --------------- -------------- ---------
  central     simonsgroup                                      cpu    213866
  central     simonsgroup                                 gres/gpu      7638
  central     simonsgroup  olstephe Oliver L. (Oll+            cpu    140426
  central     simonsgroup  olstephe Oliver L. (Oll+       gres/gpu      5015
  central     simonsgroup     ykliu    Yuan Kai Liu            cpu     73440
  central     simonsgroup     ykliu    Yuan Kai Liu       gres/gpu      2623
```

### Disk usage by me on HPC (command: `mmlsquota`)

I am not putting all the files and running them under the group space `simonsgroup/`

There are 10T quota there. Since nobody is using it, I started working from there. Now I have used up ~8.3T and need to delete files

The alternative is to go to scratch/ and do the work there since it has 20T quota for each person (but any files not accessed in 14 days will be automatically purged)

```bash
## Block Limits
Filesystem type         blocks      quota      limit   in_doubt    grace
central    FILESET      8.368T        10T        12T     11.86G     none

## File Limits
files   quota    limit in_doubt    grace  Remarks
87369       0        0      160     none central.ib.cluster
```

### My disk details

- All SLCs raw data:              2.9 T      (num of dates = 184 dates;    num of SLC zip files = 694)
- Processing directory:         5.8 T     (19 pairs completed; each completed pair occupies ~300G; I still have 170 pairs or so to process)

### Reducing files

Once `geocode` completed, we can delete most of the files. Especially those files run via GPU can be quickly re-generated if needed.

Each pair will take 3.3 G of disk space after deleting unnecessary files

```bash
## Here is a list of things Cunren suggests that we can delete after finishing all processing

 30G   fine_coreg/                             # can regenerate using GPUs
 58G   fine_interferogram/                     # can regenerate using GPUs
 30G   fine_offsets/                           # can regenerate using GPUs
100G   geom_reference/                         # can regenerate using GPUs
 30G   ion/lower/fine_interferogram/           # ion burst
 30G   ion/upper/fine_interferogram/           # ion burst
 15G   ion/ion_burst/                          # ion burst
```
