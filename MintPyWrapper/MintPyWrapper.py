#!/usr/bin/env python3
############################################################
# This code is a recipe for MintPy  (Yunjun et al., 2019)  #
# Author: Yuan-Kai Liu, 2022; refactored 2026-09           #
############################################################
""" Generates mintpy cmd run files for each stage and to run in bash
REFERENCE:
    1. Yunjun, Z., Fattahi, H., & Amelung, F. (2019).
        Small baseline InSAR time series analysis: Unwrapping error correction and noise reduction.
        Computers & Geosciences, 133, 104331. https://doi.org/10.1016/j.cageo.2019.104331
    2. Zheng, Y., Fattahi, H., Agram, P., Simons, M., & Rosen, P. (2022).
        On Closure Phase and Systematic Bias in Multilooked SAR Interferometry.
        IEEE Transactions on Geoscience and Remote Sensing, 60, 1-11. https://doi.org/10.1109/TGRS.2022.3167648
    3. Cao, Y., Jónsson, S., & Li, Z. (2021).
        Advanced InSAR Tropospheric Corrections From Global Atmospheric Models that Incorporate Spatial Stochastic Properties of the Troposphere.
        Journal of Geophysical Research: Solid Earth, 126(5), e2020JB020952. https://doi.org/10.1029/2020JB020952
    4. Stephenson, O. L., Liu, Y.-K., Yunjun, Z., Simons, M., Rosen, P., & Xu, X. (2022).
        The Impact of Plate Motions on Long-Wavelength InSAR-Derived Velocity Fields.
        Geophysical Research Letters, 49(21), e2022GL099835. https://doi.org/10.1029/2022GL099835

Input: ONE custom MintPy template (e.g. ChileSenAT076.cfg) holding both the
mintpy.* options and the wrapper.* options below. A legacy `*.par` file is still
accepted (its keys are mapped to wrapper.*; its mintpy.template gives the cfg).

Re-run `MintPyWrapper.py <cfg>` after every edit of the cfg: the run files embed
reference point, time functions etc. at generation time.

Changes 2026-09 (see CLAUDE.md of the Chile project for the reasons):
  * --ref-date: `$(cat reference_date.txt)` when mintpy.reference.date = auto
  * ionBurstRamp / IonTotal branch removed (< 0.1 mm; its dem_error overwrote
    timeseriesResidual.h5 of the Ion branch)
  * date check before each `diff.py --force` (warns about skipped dates)
  * plate-motion model computed once; velocities corrected with diff.py
  * one temporal-coherence mask: MintPy's maskTempCoh.h5 (mintpy.networkInversion.minTempCoh)
  * wrapper.loader = geo: load_stack.py loads straight to geocoded inputs
  * run_x_bwAnalysis no longer generated
  * run_1: check_ref_closure.py (reference pixel on an isolated unwrapping patch?)
  * run_9: explicit view.py lines (MintPy figures minus ifgramStack); full --plot left commented
"""

import argparse
import glob
import os
import shutil
import sys
from types import SimpleNamespace

import numpy as np
from mintpy.utils import readfile

FILE_NAME = os.path.basename(__file__)
WRAPPER_DIR = os.path.dirname(os.path.abspath(__file__))


#############################################################################################################
def cmdLineParse():
    description = 'Generates mintpy command line run files for each stage and to run in bash'
    epilog = f"""Examples:
        {FILE_NAME} ChileSenAT076.cfg              # write run_0 ... run_8, run_all
        {FILE_NAME} ChileSenAT076.cfg -a dem_resamp
        {FILE_NAME} --check-dates timeseries_SET_ERA5.h5 inputs/ion.h5
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter, epilog=epilog)
    parser.add_argument('param_file', type=str, nargs='?',
            help='custom MintPy template (*.cfg) with wrapper.* keys, or a legacy *.par')
    parser.add_argument('-d', '--home', dest='proc_home', type=str, default='.',
            help='mintpy processing home directory')
    parser.add_argument('-a', '--action', dest='action', type=str, default='all',
            help='Choose either `all` or `dem_resamp`')
    parser.add_argument('--dem-out', dest='dem_out', type=str, default=None,
            help='[dem_resamp] Name of the output resampled DEM file')
    parser.add_argument('--geo-in', dest='geo_in', type=str, default=None,
            help='[dem_resamp] Name of input geometry file for area bbox')
    parser.add_argument('--dem-orig', dest='dem_orig', type=str, default=None,
            help='[dem_resamp] Name of input original DEM file')
    parser.add_argument('--dem-action', dest='dem_action', type=str, default='run',
            help='[dem_resamp] run or write')
    parser.add_argument('--icams', dest='icams', action='store_true', default=False,
            help='[Tropo] Switch on to run ICAMS (Cao et al. 2021) for tropospheric correction')
    parser.add_argument('--check-dates', dest='check_dates', nargs=2, metavar=('TS', 'CORR'),
            help='warn about dates of TS missing in CORR (they are skipped by diff.py --force)')

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#############################################################################################################
## parameters: wrapper.* keys in the cfg (defaults), and the legacy .par key mapping
#############################################################################################################
WRAPPER_DEFAULTS = {
    'wrapper.loader'          : 'geo',          # geo: load_stack.py (direct geocoded); mintpy: load_data + geocode.py
    'wrapper.nprocs'          : 8,              # processes for load_stack.py
    'wrapper.plateName'       : None,           # ITRF14 plate name for plate_motion.py, e.g. SouthAmerica
    'wrapper.ts2velo'         : None,           # extra options for timeseries2velocity.py
    'wrapper.bandwidth'       : 3,              # closure phase bias: bandwidth (Zheng et al. 2022)
    'wrapper.connLevel'       : 10,             # closure phase bias: connection level assumed unbiased
    'wrapper.wbdOrig'         : None,           # input water body
    'wrapper.demOrig'         : None,           # input DEM (for inputs/srtm.dem, plotting)
    'wrapper.velocityDir'     : './velocity_out/',
    'wrapper.picDir'          : './pic_supp/',
    'wrapper.customMask'      : 'maskPoly.h5',
    'wrapper.plot.shadeExag'  : 0.02,
    'wrapper.plot.shadeMin'   : -6000,
    'wrapper.plot.shadeMax'   : 4000,
    'wrapper.plot.velocityAlpha': 0.6,
    'wrapper.plot.velocityCmap' : 'RdYlBu_r',
    'wrapper.plot.velocityMsk'  : 'coh',        # water / coh / connComp / custom / no
    'wrapper.plot.veloUnit'     : 'mm',
    'wrapper.plot.dpi'          : 300,
    'wrapper.plot.lonMin'       : None,
    'wrapper.plot.lonMax'       : None,
    'wrapper.plot.latMin'       : None,
    'wrapper.plot.latMax'       : None,
    'wrapper.plot.vm_big'       : '-8,8',
    'wrapper.plot.vm_mid'       : '-5,5',
    'wrapper.plot.vm_sma'       : '-0.2,0.2',
    'wrapper.plot.vm_STD'       : '0,1.0',
    'wrapper.plot.vm_AMP'       : '0,16',
    'wrapper.plot.vm_SET'       : '-0.2,0.2',
    'wrapper.plot.vm_GAM'       : '-2,2',
}

LEGACY_PAR_KEYS = {   # old .par key -> wrapper key
    'mintpy.plateName'   : 'wrapper.plateName',
    'mintpy.ts2velo'     : 'wrapper.ts2velo',
    'mintpy.bandwidth'   : 'wrapper.bandwidth',
    'mintpy.connLevel'   : 'wrapper.connLevel',
    'path.wbdOrig'       : 'wrapper.wbdOrig',
    'path.demOrig'       : 'wrapper.demOrig',
    'path.velocityDir'   : 'wrapper.velocityDir',
    'path.extraPicDir'   : 'wrapper.picDir',
    'path.customMask'    : 'wrapper.customMask',
}


def _none(v):
    return None if v in (None, 'none', 'None', 'auto', '') else v


def read_params(param_file, home='.'):
    """Return (cfg_path, pDict with wrapper.* keys, iDict = full template)."""
    raw = readfile.read_template(param_file)
    if param_file.endswith('.par'):                         # legacy
        cfg = glob.glob(os.path.join(home, raw.get('mintpy.template', 'smallbaselineApp.cfg')))[0]
        pDict = {}
        for k, v in raw.items():
            if k in LEGACY_PAR_KEYS:
                pDict[LEGACY_PAR_KEYS[k]] = v
            elif k.startswith('plot.'):
                pDict['wrapper.' + k] = v
        if 'wrapper.plateName' not in pDict and raw.get('mintpy.itrfPlate'):
            pDict['wrapper.plateName'] = raw['mintpy.itrfPlate'].split()[-1]
        pDict.setdefault('wrapper.loader', 'mintpy')        # legacy tracks keep the old loader
        print(f'legacy parameter file {param_file} -> template {cfg}')
    else:
        cfg = param_file
        pDict = {k: v for k, v in raw.items() if k.startswith('wrapper.')}
    for k, v in WRAPPER_DEFAULTS.items():
        pDict.setdefault(k, v)
    unknown = [k for k in pDict if k not in WRAPPER_DEFAULTS]
    for k in unknown:
        print(f' ! unrecognized wrapper parameter: {k} (ignored)')
    iDict = readfile.read_template(cfg)
    return cfg, pDict, iDict


def check_dates(ts_file, corr_file):
    """Print a warning for dates of ts_file absent in corr_file (diff.py --force skips them)."""
    from mintpy.utils import readfile as rf
    d1 = rf.get_slice_list(ts_file); d2 = set(rf.get_slice_list(corr_file))
    d1 = [d.split('-')[-1] for d in d1]; d2 = {d.split('-')[-1] for d in d2}
    miss = [d for d in d1 if d not in d2]
    if miss:
        print(f'WARNING: {len(miss)} of {len(d1)} dates of {ts_file} are missing in {corr_file}; '
              f'diff.py --force leaves them UNCORRECTED: {" ".join(miss)}')
    else:
        print(f'date check OK: all {len(d1)} dates of {ts_file} are in {corr_file}')


#############################################################################################################
## class that writes the MintPy command lines
#############################################################################################################

class SBApp:

    def __init__(self, param_file, proc_home='.'):
        self.param_file = os.path.expanduser(param_file)
        self.home       = os.path.expanduser(proc_home)
        self.cwd        = os.getcwd()
        self.picdir     = os.path.join(self.home,  'pic')
        self.water_mask = os.path.join(self.home,  'waterMask.h5')
        self.coh_mask   = os.path.join(self.home,  'maskTempCoh.h5')
        self.conn_mask  = os.path.join(self.home,  'maskConnComp.h5')
        self.indir      = os.path.join(self.home,  'inputs')
        self.geom_file  = os.path.join(self.indir, 'geometryGeo.h5')
        self.ifg_stack  = os.path.join(self.indir, 'ifgramStack.h5')
        print(f'Current path: {self.cwd}')
        print(f'Reading parameters from: {self.param_file}')
        print(f'MintPy processing directory at: {self.home}')
        self.template, self.pDict, self.iDict = read_params(self.param_file, self.home)

    def create_run_file(self, file):
        self.run_files = getattr(self, 'run_files', []) + [file]
        self.f = open(file, 'w')
        self.f.write('#!/bin/bash\n\n')

    def get_template(self):
        print(f'Use template: {self.template}')
        for outdir in [self.indir, self.picdir]:        # backup copy of the cfg used for these run files
            os.makedirs(outdir, exist_ok=True)
            shutil.copyfile(self.template, os.path.join(outdir, os.path.basename(self.template)))
            print(f'copy {self.template} to {outdir}/')
        iD = self.iDict
        self.ram       = iD['mintpy.compute.maxMemory']
        self.cluster   = iD['mintpy.compute.cluster']
        self.numWorker = iD['mintpy.compute.numWorker']
        m_poly         = iD['mintpy.timeFunc.polynomial']
        m_peri         = iD['mintpy.timeFunc.periodic'].replace(',', ' ')
        self.time_func = f'--poly-order {m_poly} --periodic {m_peri}'
        self.refla, self.reflo = (x.strip() for x in iD['mintpy.reference.lalo'].split(','))
        rd = iD.get('mintpy.reference.date', 'auto')
        # auto / reference_date.txt -> read at RUN time (written by residual_RMS)
        self.ref_date = '$(cat reference_date.txt)' if rd in ('auto', 'reference_date.txt', 'no') else rd
        self.gam = iD.get('mintpy.troposphericDelay.weatherModel', 'auto')
        if self.gam == 'auto':
            self.gam = 'ERA5'
        self.plateName = _none(self.pDict['wrapper.plateName'])
        self.itrffile  = os.path.join(self.indir, f'ITRF14_{self.plateName}.h5') if self.plateName else None
        self.veldir    = self.pDict['wrapper.velocityDir']
        self.extrapic  = self.pDict['wrapper.picDir']

    def run_resampWbd(self, ftype='Body'):
        """ Make a water body file with the SAR dimension (needs ISCE) """
        os.chdir(self.home)
        geom_basedir = os.path.dirname(os.path.abspath(self.iDict['mintpy.load.demFile']))
        wbdOutFile   = os.path.join(geom_basedir, f'water{ftype}.rdr')
        os.chdir(self.cwd)
        if os.path.exists(wbdOutFile):
            print(f'water file exists: {wbdOutFile}; skip generating it')
            return
        import isce  # noqa: F401
        from applications.gdal2isce_xml import gdal2isce_xml
        from isceobj.Alos2Proc.Alos2ProcPublic import waterBodyRadar
        wbdFile = os.path.abspath(self.pDict['wrapper.wbdOrig'])
        latFile, lonFile = f'{geom_basedir}/lat.rdr', f'{geom_basedir}/lon.rdr'
        gdal2isce_xml(latFile + '.vrt'); gdal2isce_xml(lonFile + '.vrt')
        waterBodyRadar(latFile, lonFile, wbdFile, wbdOutFile)
        print(f'Resampled water file: {wbdOutFile}')
        os.system(f'fixImageXml.py -i {wbdOutFile} -f ')

    # ------------------------------------------------------------------ writers
    def write_smallbaselineApp(self, dostep=None, start=None, end=None):
        cmd = f'smallbaselineApp.py {self.template} '
        if dostep: cmd += f'--dostep {dostep} '
        if start:  cmd += f'--start {start} '
        if end:    cmd += f'--end {end} '
        self.f.write(cmd + '\n\n')

    def write_load_stack(self):
        """ Direct geocoded loading (no radar-coordinate stack is written) """
        self.f.write(f"{os.path.join(WRAPPER_DIR, 'load_stack.py')} {self.template} "
                     f"--outdir {self.indir} --nprocs {self.pDict['wrapper.nprocs']}\n\n")

    def write_radar2geo_inputs(self, lalo=None):
        """ Legacy loader: geocode ifgramStack.h5, ion.h5, geometryRadar.h5 and keep radar copies """
        geom_rdr = os.path.join(self.indir, 'geometryRadar.h5')
        rdr_dir  = os.path.join(self.indir, 'radar')
        if lalo is None:
            lalo = [float(n) for n in self.iDict['mintpy.geocode.laloStep'].replace(',', ' ').split()]
        file_list = ['geometryRadar.h5', 'ifgramStack.h5', 'ion.h5']
        files = [os.path.join(self.indir, f) for f in file_list]
        lalo9 = [f'{x:.9f}' for x in lalo]
        cmd  = f"geocode.py {' '.join(files)} -l {geom_rdr} --lalo {' '.join(lalo9)} --ram {self.ram} --update\n\n"
        cmd += f'mkdir -p {rdr_dir}\n'
        cmd += f"\\mv {' '.join(files)} {rdr_dir}\n"
        for file in file_list:
            dst = 'geometryGeo.h5' if file.startswith('geometry') else file
            cmd += f'\\mv geo_{file} {self.indir}/{dst} \n'
        self.f.write(cmd + '\n')

    def run_resamp_dem(self, dem_out, geo_file, dem_orig, action='run'):
        """ Resample the original DEM to the geocoded grid (inputs/srtm.dem, for plotting) """
        dem_out  = dem_out  or os.path.join(self.indir, 'srtm.dem')
        geo_file = geo_file or os.path.join(self.indir, 'geometryGeo.h5')
        dem_orig = dem_orig or self.pDict['wrapper.demOrig']
        atr = readfile.read_attribute(geo_file)
        x0, dx, y0, dy = (float(atr[k]) for k in ('X_FIRST', 'X_STEP', 'Y_FIRST', 'Y_STEP'))
        W, L = int(atr['WIDTH']), int(atr['LENGTH'])
        lon_min, lon_max = x0 + dx / 2, x0 + dx / 2 + dx * W
        lat_max = y0 - dx / 2
        lat_min = y0 - dx / 2 + dy * L
        cmd  = f"gdalwarp {dem_orig} {dem_out} -te {lon_min} {lat_min} {lon_max} {lat_max} -ts {W} {L} -of ISCE\n"
        cmd += f'fixImageXml.py -i {dem_out} -f\n\n'
        if action == 'write':
            self.f.write(cmd)
        else:
            print(cmd); os.system(cmd)

    def write_deramp_ifg(self, fname='ifgramStack.h5', dset='unwrapPhase', ramp_type='linear', mask_file='maskTempCoh.h5'):
        fname = os.path.join(self.indir, fname)
        fdir = os.path.dirname(fname)
        fbase = os.path.splitext(os.path.basename(fname))[0]
        self.f.write(f'remove_ramp.py {fname} -d {dset} -s {ramp_type} -m {mask_file} --save-ramp-coeff --update\n\n')
        self.f.write(f'\\mv {fdir}/rampCoeff_{fbase}.txt {fdir}/rampCoeff_{fbase}_{dset}.txt\n\n')

    def write_modify_network(self, file='ifgramStack.h5'):
        self.f.write(f'modify_network.py {os.path.join(self.indir, file)} -t {self.template}\n\n')

    def write_reference_point(self, files=(), ref_lalo=None, ref_yx=None):
        opt = ''
        if ref_lalo:
            opt = f'--lat {ref_lalo[0]} --lon {ref_lalo[1]}'
        elif ref_yx:
            opt = f'--row {ref_yx[0]} --col {ref_yx[1]}'
        for file in files:
            self.f.write(f'reference_point.py {file} -t {self.template} {opt}\n')
        self.f.write('\n')

    def write_plot_network(self, stacks, cmap_vlist=(0.2, 0.7, 1.0), arg=''):
        for file in stacks:
            cmd  = f'plot_network.py {os.path.join(self.indir, file)} -t {self.template} '
            cmd += f"--nodisplay --cmap-vlist {' '.join(map(str, cmap_vlist))} {arg}\n"
            cmd += f'mkdir -p {self.extrapic}\n'
            cmd += f'\\mv *.pdf *.png {self.extrapic}\n\n'
            self.f.write(cmd)

    def write_demErr(self, file, outfile, gfile=None):
        gfile = gfile or self.geom_file
        cmd  = f'dem_error.py {file} {self.time_func} -g {gfile} -o {outfile} '
        cmd += f'--cluster {self.cluster} --num-worker {self.numWorker} --ram {self.ram} --update\n\n'
        self.f.write(cmd)

    def write_check_dates(self, ts_file, corr_file):
        self.f.write(f'{os.path.abspath(__file__)} --check-dates {ts_file} {corr_file}\n')

    def write_diff(self, f1, f2, out, check=True):
        if check:
            self.write_check_dates(f1, f2)
        self.f.write(f'diff.py {f1} {f2} -o {out} --force\n\n')

    def write_closure_phase(self, ifile, nl, bw, action, nsig=3, neps=None, waterMask='waterMask.h5', outdir='.', ram=4, workers=4):
        if int(nl) < int(bw):
            sys.exit('--conn-level (assumed no bias) should be at least the --bandwidth of your analysis (Zheng et al. 2022)')
        cmd = f'closure_phase_bias.py -i {ifile} --nl {nl} --bw {bw} -a {action} --wm {waterMask} -o {outdir} '
        if nsig: cmd += f'--num-sigma {nsig} '
        if neps: cmd += f'--eps {neps} '
        cmd += f'--ram {ram} --num-worker {workers} \n\n'
        self.f.write(cmd)

    def write_ts2velo(self, tsfile, outfile, ts2velocmd=None, out_dir=None, update=True):
        outfile = os.path.join(out_dir or self.veldir, outfile)
        cmd  = f'timeseries2velocity.py {tsfile} {self.time_func} -o {outfile} '
        cmd += f'--ref-lalo {self.refla} {self.reflo} --ref-date {self.ref_date} '
        if _none(ts2velocmd):
            cmd += f' {ts2velocmd} '
        if update:
            cmd += '--update '
        self.f.write(cmd + '\n\n')

    def write_plate_motion_model(self):
        """ ITRF14 plate-motion LOS velocity, computed once (Stephenson et al. 2022) """
        if self.itrffile:
            self.f.write(f'plate_motion.py --geom {self.geom_file} --plate {self.plateName} -s {self.itrffile}\n\n')

    def write_remove_plate(self, vfile):
        """ velocity - plate-motion model; equals `plate_motion.py --velo` without recomputing the model """
        if self.itrffile:
            out = vfile.replace('.h5', '_ITRF14.h5')
            self.f.write(f'diff.py {vfile} {self.itrffile} -o {out}\n\n')

    def write_plot_velo(self, file, dataset, vlim='None,None', dem_file=None, title=None, outfile=None, picdir=None, mask=None, update=True):
        P = self.pDict
        picdir  = picdir or self.extrapic
        base    = os.path.basename(file).split('.')[0]
        title   = title or str(dataset)
        outfile = outfile or os.path.join(picdir, base + '.png')
        dem_file = dem_file or os.path.join(self.indir, 'srtm.dem')
        if not mask:
            m = _none(P['wrapper.plot.velocityMsk'])
            mask = {'water': self.water_mask, 'coh': self.coh_mask, 'connComp': self.conn_mask,
                    'custom': P['wrapper.customMask']}.get(m, 'no')
        vmin, vmax = str(vlim).split(',')
        cmd  = f"view.py {file} {dataset} -c {P['wrapper.plot.velocityCmap']} "
        cmd += f"--dem {dem_file} --alpha {P['wrapper.plot.velocityAlpha']} "
        cmd += (f"--dem-nocontour --shade-exag {P['wrapper.plot.shadeExag']} "
                f"--shade-min {P['wrapper.plot.shadeMin']} --shade-max {P['wrapper.plot.shadeMax']} ")
        cmd += f"--mask {mask} --zm --unit {P['wrapper.plot.veloUnit']} --ref-lalo {self.refla} {self.reflo} "
        if _none(vmin) is not None:
            cmd += f'--vlim {vmin} {vmax} '
        if _none(P['wrapper.plot.lonMin']) is not None:
            cmd += f"--sub-lon {P['wrapper.plot.lonMin']} {P['wrapper.plot.lonMax']} "
        if _none(P['wrapper.plot.latMin']) is not None:
            cmd += f"--sub-lat {P['wrapper.plot.latMin']} {P['wrapper.plot.latMax']} "
        cmd += f"--nodisplay --dpi {P['wrapper.plot.dpi']} --figtitle {title} -o {outfile} "
        if update:
            cmd += '--update '
        if getattr(self, 'parallel_plot', False):
            self.f.write(cmd + ' &\n[ $(jobs -rp | wc -l) -ge 6 ] && wait -n\n\n')   # at most 6 at once
        else:
            self.f.write(cmd + '\n\n')

    def write_closurePhase_Mask(self, bw, nl, nsig=3, ram=8, workers=4, clpdir='closurePhase', maskDict=None):
        """ Closure phase bias analysis (Zheng et al. 2022) """
        self.f.write(f'mkdir -p {clpdir} \n\n')
        self.f.write('## Do closure phase bias calculation\n\n')
        self.f.write(f'mask.py {self.ifg_stack} -m {self.water_mask} --fill 0 -o {self.ifg_stack_msk}\n\n')
        self.write_closure_phase(self.ifg_stack_msk, nl=nl, bw=bw, action='mask',           nsig=nsig, ram=ram, workers=workers)
        self.write_closure_phase(self.ifg_stack_msk, nl=nl, bw=bw, action='quick_estimate', nsig=nsig, ram=ram, workers=workers)
        self.f.write(f'\\mv maskClosurePhase.h5 avgCpxClosurePhase.h5 wratio.h5 timeseriesBiasApprox.h5 {clpdir}\n\n')
        mask = SimpleNamespace(**(maskDict or {'a': 'maskClp_tCoh.h5', 'b': 'maskTri.h5', 'c': 'maskClp_tCoh_Tri.h5'}))
        cm = self.pDict['wrapper.customMask']
        self.f.write('## Apply masking based on closure phase bias\n\n')
        self.f.write(f'mask.py {clpdir}/maskClosurePhase.h5 -m {self.coh_mask} --fill 0 -o {clpdir}/{mask.a}\n\n')
        self.f.write(f'test -f {cm} && mask.py {clpdir}/{mask.a} -m {cm} --fill 0 -o {clpdir}/{mask.a}\n\n')
        self.f.write(f'generate_mask.py numTriNonzeroIntAmbiguity.h5 -M 0 -o {mask.b}\n\n')
        self.f.write(f'mask.py {clpdir}/{mask.a} -m {mask.b} --fill 0 -o {clpdir}/{mask.c}\n\n')
        self.write_ts2velo(f'{clpdir}/timeseriesBiasApprox.h5', 'velocityAppBias.h5', ts2velocmd=self.pDict['wrapper.ts2velo'], update=False)
        veldir = os.path.normpath('./' + self.veldir)
        picdir = os.path.normpath('./' + self.extrapic)
        self.write_plot_velo(f'{veldir}/velocityAppBias.h5', 'velocity', self.pDict['wrapper.plot.vm_mid'],
                             mask=self.coh_mask, picdir=picdir, update=False)
        self.write_plot_velo(f'{veldir}/velocityAppBias.h5', 'velocityStd', self.pDict['wrapper.plot.vm_STD'],
                             mask=self.coh_mask, title='velocityStdBias',
                             outfile=os.path.join(picdir, 'velocityStdBias.png'), update=False)

    def write_icams(self, ref_ts_file, ts_icams=None, proj='los', nproc=4, method='sklm', icamdir='icams'):
        """ ICAMS global atmospheric model resampling (Cao et al. 2021) """
        outfile1 = f'timeseries_icams_{proj}_{method}.h5'
        outfile2 = f'timeseries_icamsCor_{proj}_{method}.h5'
        self.f.write('## Need to have ICAMS and dependencies installed before running\n\n')
        self.f.write(f'# rm -rf ./{icamdir}/{self.gam}/*.npy ./{icamdir}/{self.gam}/sar # remove old los results\n\n')
        self.f.write(f"mkdir -p {icamdir} && \\cp {self.iDict['mintpy.load.metaFile']} {icamdir}\n\n")
        self.f.write(f'tropo_icams.py {ref_ts_file} {self.geom_file} --sar-par {icamdir}/IW1.xml --ref-file {ref_ts_file} --project {proj} --method {method} --nproc {nproc}\n\n')
        self.f.write(f'\\mv {outfile1} {outfile2} ./{icamdir}\n\n')
        self.f.write(f'cd ./{icamdir}\n\n')
        ts_icams = ts_icams or f'{self.indir}/{self.gam}-{proj}-{method}.h5'
        self.f.write(f"image_math.py {outfile1} '*' -1.0 --output ../{ts_icams}\n\n")
        self.f.write(f'rm -rf {outfile1} {outfile2}\n\n')
        self.f.write(f'cd {self.cwd}\n\n')
        return ts_icams


#############################################################################################################
## Major function writing the workflow
#############################################################################################################

def main(proc, inps):
    proc.get_template()
    proc.run_resampWbd()
    ram, nproc, gam = proc.ram, proc.numWorker, proc.gam
    P = proc.pDict
    veldir, picdir = proc.veldir, proc.extrapic
    vm_mid = P['wrapper.plot.vm_mid']

    ########## Load data (geocoded) ##############
    proc.create_run_file('run_0_prep')
    if P['wrapper.loader'] == 'geo':
        proc.write_load_stack()
    else:
        proc.write_smallbaselineApp(dostep='load_data')
        proc.write_radar2geo_inputs()
    proc.f.write(f'{FILE_NAME} {proc.param_file} -a dem_resamp\n')
    proc.f.close()

    ########## Network modifications and plots ##############
    proc.create_run_file('run_1_network')
    proc.write_smallbaselineApp(dostep='modify_network')
    proc.write_smallbaselineApp(dostep='reference_point')
    # reference pixel on an isolated unwrapping patch? (warning only; see check_ref_closure.py -h)
    proc.f.write(f"{os.path.join(WRAPPER_DIR, 'check_ref_closure.py')} {proc.ifg_stack} --suggest 60\n\n")
    proc.write_smallbaselineApp(dostep='quick_overview')
    proc.write_plot_network(stacks=['ifgramStack.h5'], cmap_vlist=[0.2, 0.7, 1.0])
    proc.f.close()

    ################### Network inversion ##################
    # maskTempCoh.h5 is written by invert_network with mintpy.networkInversion.minTempCoh
    proc.create_run_file('run_2_inversion')
    proc.write_smallbaselineApp(dostep='correct_unwrap_error')
    proc.write_smallbaselineApp(dostep='invert_network')
    proc.f.close()

    ################## ICAMS #######################
    gam2, ts_icams = None, None
    if inps.icams:
        gam2 = f'{gam}Cao2021'
        proc.create_run_file('run_3_icams')
        ts_icams = proc.write_icams(ref_ts_file='timeseries.h5', proj='los', nproc=nproc, method='sklm')
        proc.f.write("echo 'Normal finish the ICAMS analysis'\n")
        proc.f.close()

    ################ Apply corrections ####################
    proc.create_run_file('run_4_corrections')
    proc.write_plate_motion_model()
    proc.write_smallbaselineApp(dostep='correct_LOD')
    proc.write_smallbaselineApp(dostep='correct_SET')
    proc.write_smallbaselineApp(dostep='correct_troposphere')
    proc.write_diff(f'timeseries_SET_{gam}.h5', 'inputs/ion.h5', f'timeseries_SET_{gam}_Ion.h5')
    proc.write_demErr(f'timeseries_SET_{gam}_Ion.h5', f'timeseries_SET_{gam}_Ion_demErr.h5')
    if inps.icams:
        proc.write_diff('timeseries_SET.h5', ts_icams, f'timeseries_SET_{gam2}.h5')
        proc.write_diff(f'timeseries_SET_{gam2}.h5', 'inputs/ion.h5', f'timeseries_SET_{gam2}_Ion.h5')
        # separate folder: dem_error writes timeseriesResidual.h5 next to its output
        proc.f.write('mkdir -p icams_demErr\n\n')
        proc.write_demErr(f'timeseries_SET_{gam2}_Ion.h5', f'icams_demErr/timeseries_SET_{gam2}_Ion_demErr.h5')
    proc.write_smallbaselineApp(dostep='residual_RMS')      # writes reference_date.txt
    proc.write_smallbaselineApp(dostep='deramp')
    proc.f.close()

    ################ Velocity estimation ###################
    proc.create_run_file('run_5_velocity')
    ts2veloDict = {
        'velocity'                    : ['timeseries'                     , vm_mid],
        'velocity_SET'                : ['timeseries_SET'                 , vm_mid],
        f'velocity_SET_{gam}'         : [f'timeseries_SET_{gam}'          , vm_mid],
        f'velocity_SET_{gam}_Ion'     : [f'timeseries_SET_{gam}_Ion'      , vm_mid],
        f'velocity_SET_{gam}_Ion_demErr' : [f'timeseries_SET_{gam}_Ion_demErr', vm_mid],
        f'velocity_SET_{gam}_Ion_demErr_ITRF14' : [None                   , vm_mid],
        'velocitySET'                 : ['inputs/SET'                     , P['wrapper.plot.vm_SET']],
        f'velocity{gam}'              : [f'inputs/{gam}'                  , P['wrapper.plot.vm_GAM']],
        'velocityIon'                 : ['inputs/ion'                     , vm_mid],
    }
    if inps.icams:
        ts2veloDict.update({
            'velocityICAMS-PyAPS'                  : [None                                  , P['wrapper.plot.vm_sma']],
            f'velocity_SET_{gam2}'                 : [f'timeseries_SET_{gam2}'              , vm_mid],
            f'velocity_SET_{gam2}_Ion'             : [f'timeseries_SET_{gam2}_Ion'          , vm_mid],
            f'velocity_SET_{gam2}_Ion_demErr'      : [f'icams_demErr/timeseries_SET_{gam2}_Ion_demErr', vm_mid],
            f'velocity_SET_{gam2}_Ion_demErr_ITRF14' : [None                                , vm_mid],
            f'velocity{gam2}'                      : [ts_icams.split('.h5')[0]              , P['wrapper.plot.vm_GAM']],
        })
    for key, item in ts2veloDict.items():
        if item[0]:
            proc.write_ts2velo(item[0] + '.h5', key + '.h5', ts2velocmd=P['wrapper.ts2velo'], update=False)
    proc.write_remove_plate(os.path.join(veldir, f'velocity_SET_{gam}_Ion_demErr.h5'))
    if inps.icams:
        proc.write_remove_plate(os.path.join(veldir, f'velocity_SET_{gam2}_Ion_demErr.h5'))
        proc.f.write(f'diff.py {veldir}velocity{gam2}.h5 {veldir}velocity{gam}.h5 -o {veldir}velocityICAMS-PyAPS.h5 \n\n')
    proc.f.close()

    ################## Plot Velocity #######################
    proc.create_run_file('run_6_velocityPlot')
    proc.f.write(f'mkdir -p {picdir}\n\n')
    proc.parallel_plot = True
    vfiles = sorted(set([os.path.join(veldir, x + '.h5') for x in ts2veloDict] + glob.glob(os.path.join(veldir, '*.h5'))))
    for vfile in vfiles:
        key = os.path.basename(vfile).split('.h5')[0]
        src, vlim = ts2veloDict.get(key, [None, vm_mid])
        title = 'velocity' + src.split('timeseries')[-1] if (src and src.startswith('timeseries_')) else key
        proc.write_plot_velo(vfile, 'velocity', vlim, title=title, outfile=os.path.join(picdir, key + '.png'), update=False)
    for suff in ['', gam, 'SET', 'Ion'] + ([gam2] if inps.icams else []):
        proc.write_plot_velo(os.path.join(veldir, f'velocity{suff}.h5'), 'velocityStd', P['wrapper.plot.vm_STD'],
                             title=f'velocityStd{suff}', outfile=os.path.join(picdir, f'velocityStd{suff}.png'), update=False)
    proc.f.write('wait\n')
    proc.parallel_plot = False
    proc.f.close()

    ################## Closure phase bias (testing) #######################
    proc.create_run_file('run_7_closurePhase')
    proc.ifg_stack_msk = os.path.join(proc.indir, 'ifgramStack_msk.h5')
    proc.write_closurePhase_Mask(int(P['wrapper.bandwidth']), int(P['wrapper.connLevel']), 3, ram, nproc, './closurePhase')
    proc.f.write("echo 'Normal finish the closure phase bias analysis'\n")
    proc.f.close()

    ########## generate more timeseries corrections for demo ##########
    proc.create_run_file('run_8_orderTS')
    if proc.itrffile:
        proc.write_diff('timeseries_SET.h5', proc.itrffile, 'timeseries_SET_ITRF14.h5', check=False)
        proc.write_diff('timeseries_SET_ITRF14.h5', f'inputs/{gam}.h5', f'timeseries_SET_ITRF14_{gam}.h5')
        ts2 = {'velocity_SET_ITRF14': 'timeseries_SET_ITRF14', f'velocity_SET_ITRF14_{gam}': f'timeseries_SET_ITRF14_{gam}'}
        for key, ts in ts2.items():
            proc.write_ts2velo(ts + '.h5', key + '.h5', ts2velocmd=P['wrapper.ts2velo'], update=False)
            vfile = os.path.join(veldir, key + '.h5'); suff = key.split('velocity')[-1]
            proc.write_plot_velo(vfile, 'velocity', vm_mid, title='velocity' + ts.split('timeseries')[-1],
                                 outfile=os.path.join(picdir, key + '.png'), update=False)
            proc.write_plot_velo(vfile, 'velocityStd', P['wrapper.plot.vm_STD'], title='velocityStd' + suff,
                                 outfile=os.path.join(picdir, 'velocityStd' + suff + '.png'), update=False)
    proc.f.close()

    ############# Compile all together in a script ############
    steps = list(proc.run_files)          # exactly the files written above

    ########## optional: MintPy's full default plot set (slow; replots every product) ##########
    proc.create_run_file('run_9_mintpyPlot')
    dpi = proc.iDict.get('mintpy.plot.dpi', 'auto'); dpi = 150 if dpi == 'auto' else dpi
    opt = f'--dpi {dpi} --noverbose --nodisplay --update --memory 4 --outdir pic'
    ts_opt = '--noaxis -u cm --wrap --wrap-range -5 5'
    J = ' &\n[ $(jobs -rp | wc -l) -ge 6 ] && wait -n\n'          # up to 6 plots at once
    proc.f.write('# MintPy default figures, without the ifgramStack ones (fast)\nmkdir -p pic\n\n')
    for args in [f'velocity.h5 --dem {proc.geom_file} --mask maskTempCoh.h5',
                 'temporalCoherence.h5 -c gray -v 0 1', 'maskTempCoh.h5 -c gray -v 0 1',
                 proc.geom_file, 'avgPhaseVelocity.h5', 'avgSpatialCoh.h5 -c gray -v 0 1',
                 'maskConnComp.h5 -c gray -v 0 1', 'numTriNonzeroIntAmbiguity.h5 --mask no',
                 f'velocity{gam}.h5 --mask no', 'numInvIfgram.h5 --mask no']:
        proc.f.write(f'view.py {opt} {args}{J}')
    proc.f.write(f'for f in timeseries*.h5; do\n  view.py {opt} $f {ts_opt}{J.replace(chr(10), chr(10) + "  ")}done\nwait\n\n')
    proc.f.write('# full MintPy figure set incl. every interferogram of ifgramStack.h5 (slow):\n')
    proc.f.write('# smallbaselineApp.py --plot\n')
    proc.f.close()
    proc.create_run_file('run_all')
    for rf in steps:
        proc.f.write(f'bash {rf} \n')
    proc.f.close()


#############################################################################################################

if __name__ == '__main__':
    inps = cmdLineParse()
    if inps.check_dates:
        check_dates(*inps.check_dates)
        sys.exit(0)
    proc = SBApp(inps.param_file, inps.proc_home)
    if inps.action == 'all':
        main(proc, inps)
    elif inps.action == 'dem_resamp':
        proc.get_template()
        proc.run_resamp_dem(inps.dem_out, inps.geo_in, inps.dem_orig, inps.dem_action)
    print('Finish writing the run files. Go ahead and run them sequentially.')
