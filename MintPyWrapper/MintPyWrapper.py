#!/usr/bin/env python3
############################################################
# This code is a recipe for MintPy  (Yunjun et al., 2019)  #
# Author: Yuan-Kai Liu, 2022                               #
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
"""

import argparse
import glob
import os
import shutil
import sys
from types import SimpleNamespace

# isce
import isce
# mintpy
import mintpy
import numpy as np
from applications.gdal2isce_xml import gdal2isce_xml
from isceobj.Alos2Proc.Alos2ProcPublic import waterBodyRadar
from mintpy.utils import readfile

FILE_NAME = os.path.basename(__file__)


#############################################################################################################
def cmdLineParse():
    """
    Command line parsers
    """
    description = 'Generates mintpy command line run files for each stage and to run in bash'

    EXAMPLE = f"""Examples:
        ## Specify the `*.par` and the process home directory under `mintpy/`.
        {FILE_NAME} AqabaSenAT087.par
    """
    epilog = EXAMPLE
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter,  epilog=epilog)

    parser.add_argument('param_file', type=str,
            help = 'TEXT file with custom specifications')
    parser.add_argument('-d', '--home', dest='proc_home', type=str, default='.',
            help = 'mintpy processing home directory')
    parser.add_argument('-a', '--action', dest='action', type=str, default='all',
            help = 'Choose either `all` or `dem_resamp`')
    parser.add_argument('--dem-out', dest='dem_out', type=str, default=None,
            help = '[dem_resamp] Name of the output resampled DEM file')
    parser.add_argument('--geo-in', dest='geo_in', type=str, default=None,
            help = '[dem_resamp] Name of input geometry file for area bbox')
    parser.add_argument('--dem-orig', dest='dem_orig', type=str, default=None,
            help = '[dem_resamp] Name of input original DEM file')
    parser.add_argument('--dem-action', dest='dem_action', type=str, default='run',
            help = '[dem_resamp] run or write')

    if len(sys.argv) <= 1:
        print('')
        parser.print_help()
        sys.exit(1)
    else:
        print('')
        return parser.parse_args()

#############################################################################################################
## class that calls the MintPy cli
#############################################################################################################

class SBApp:

    def __init__(self, param_file, proc_home='.'):
        """Initializing the SmallBaselineApp object in MintPy (Yunjun et al. 2019)
        """
        # set default file paths
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
        self.ion_stack  = os.path.join(self.indir, 'ionStack.h5')

        # print locations
        print(f'Current path: {self.cwd}')
        print(f'Reading MintPy custom paramters at: {self.param_file}')
        print(f'MintPy processing directory at: {self.home}')

        # read and check the parameter file
        self.pDict = check_parameter_txt(self.param_file)


    def create_run_file(self, file):
        self.f = open(file, 'w')
        self.f.write('#!/bin/bash\n\n')
        return


    def get_template(self):
        """ Define the template and save a copy
        """
        self.template    = self.pDict.get('mintpy.template'   , 'smallbaselineApp.cfg')
        self.template    = glob.glob(os.path.join(self.home, self.template))[0]

        print(f'Use template for regular     pairs: {self.template}')
        #print('Update smallbaselineApp.cfg based on {}'.format(self.template))
        #shutil.copyfile(self.template, os.path.join(self.indir, 'smallbaselineApp.cfg'))

        for outdir in [self.indir, self.picdir]:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            for cfg in [self.template]:
                print(f'copy {cfg} to {outdir}/{os.path.basename(cfg)}')
                shutil.copyfile(cfg, f'{outdir}/{os.path.basename(cfg)}')

        self.iDict     = readfile.read_template(self.template)
        self.ram       = self.iDict['mintpy.compute.maxMemory']
        self.cluster   = self.iDict['mintpy.compute.cluster']
        self.numWorker = self.iDict['mintpy.compute.numWorker']
        m_poly         = self.iDict['mintpy.timeFunc.polynomial']
        m_peri         = self.iDict['mintpy.timeFunc.periodic'].replace(',', ' ')
        self.time_func = f'--poly-order {m_poly} --periodic {m_peri}'
        self.refla     = self.iDict['mintpy.reference.lalo'].split(',')[0]
        self.reflo     = self.iDict['mintpy.reference.lalo'].split(',')[1]
        self.ref_date  = self.iDict['mintpy.reference.date']
        self.gam       = self.iDict['mintpy.troposphericDelay.weatherModel']  #[ERA5 / MERRA / NARR], auto for ERA5
        if self.gam == 'auto':
            self.gam = 'ERA5'


    def run_resampWbd(self, geom_basedir=None, wbdFile=None, ftype='Body'):
        """ Make a water body file with the SAR dimension
        """
        # get paths, filenames
        os.chdir(self.home)
        isce_geom_file = self.iDict['mintpy.load.demFile']
        geom_basedir   = os.path.dirname(os.path.abspath(isce_geom_file))
        wbdFile        = os.path.abspath(self.pDict['path.wbdOrig'])
        wbdOutFile     = os.path.join(geom_basedir, f'water{ftype}.rdr')
        os.chdir(self.cwd)

        if os.path.exists(wbdOutFile):
            print(f'water file exists: {wbdOutFile}')
            print('skip generating water file')
        else:
            print('water file does not exist')
            print(f'Working on resmapling water file from: {geom_basedir}')
            # do gdal_translate to re-generate lon.rdr.xml lat.rdr.xml
            latFile = f'{geom_basedir}/lat.rdr'
            lonFile = f'{geom_basedir}/lon.rdr'
            print('Generate ISCE xml file from gdal supported file')
            gdal2isce_xml(latFile+'.vrt')
            gdal2isce_xml(lonFile+'.vrt')
            print('Completed ISCE xml files')

            # resample the water file
            waterBodyRadar(latFile, lonFile, wbdFile, wbdOutFile)
            print(f'Resampled water file: {wbdOutFile}')
        os.system(f'fixImageXml.py -i {wbdOutFile} -f ')


    def write_smallbaselineApp(self, dostep=None, start=None, end=None):
        """ Write command line: `smallbaselineApp.py`
        """
        cmd = f'smallbaselineApp.py {self.template} '
        if dostep:
            cmd += f'--dostep {dostep} '
        if start:
            cmd += f'--start {start} '
        if end:
            cmd += f'--end {end} '
        self.f.write(cmd+'\n\n')


    def write_loaddata(self, project=None, datasets='-l ifg geom ion'):
        """ Write command line: `load_data.py`
        """
        cmd = f'load_data.py -t smallbaselineApp.cfg '
        if project:
            cmd += f'--project {project} '
        if datasets:
            cmd += datasets
        self.f.write(cmd+'\n\n')


    def write_radar2geo_inputs(self, lalo=None):
        """ Write command line: `geocode.py` to geocode ifgramStack.h5, ionStack.h5, geometryRadar.h5
        """
        geom_rdr = os.path.join(self.indir, 'geometryRadar.h5')
        rdr_dir  = os.path.join(self.indir, 'radar')

        if lalo is None: lalo = [float(n) for n in self.iDict['mintpy.geocode.laloStep'].replace(',', ' ').split()]

        # get the hdf5 files to be geocoded
        file_list = ['geometryRadar.h5', 'ifgramStack.h5', 'ion.h5', 'ionBurstRamp.h5']
        files = []
        for f in file_list:
            files.append(os.path.join(self.indir, f))

        # command line
        cmd = f"geocode.py {' '.join(map(str,files))} -l {geom_rdr} --lalo {' '.join(map(str,lalo))} --ram {self.ram} --update\n\n"

        # backup the radar coord files
        cmd += f'mkdir -p {rdr_dir}\n'
        cmd += f"mv {' '.join(map(str,files))} {rdr_dir}\n"

        # rename the geo files and store in inputs/
        for file in file_list:
            if file.startswith('geometry'):
                cmd += f'mv geo_{file} {self.indir}/geometryGeo.h5 \n'
            else:
                cmd += f'mv geo_{file} {self.indir}/{file} \n'

        self.f.write(cmd+'\n')


    def run_resamp_dem(self, dem_out, geo_file, dem_orig, action='run'):
        """
        Use GDAL to resample the orignal DEM to match the full extent of the isce2 interferograms.
        The extent, dimension, and resolution of the output DEM is the same as the interferograms.
        This is totally optional. After geocode geometryRadar.h5 to geometryGeo.h5, the height
        will have large holes; not pretty.
        Should be run after having the geometryGeo.h5 file (must be in geo-coord to allow reading lon lat)
        The output DEM is then saved separetly (inputs/srtm.dem)
        The output DEM is mainly for plotting purposes using view.py
        """
        if dem_out is None:
            dem_out   = os.path.join(self.indir, 'srtm.dem')
        if geo_file is None:
            geo_file  = os.path.join(self.indir, 'geometryGeo.h5')
        if dem_orig is None:
            dem_orig  = self.pDict['path.demOrig']

        # Read basic attributes from .h5
        atr = readfile.read_attribute(geo_file)

        # compute latitude and longitude min max
        lon_min = float(atr['X_FIRST']) + float(atr['X_STEP'])/2
        lon_max = float(atr['X_FIRST']) + float(atr['X_STEP'])/2 + float(atr['X_STEP']) * (int(atr['WIDTH']))
        lat_max = float(atr['Y_FIRST']) - float(atr['X_STEP'])/2
        lat_min = float(atr['Y_FIRST']) - float(atr['X_STEP'])/2 + float(atr['Y_STEP']) * (int(atr['LENGTH']))

        # do gdalwarp on the orignal DEM and output it
        cmd  = f"gdalwarp {dem_orig} {dem_out} -te {lon_min} {lat_min} {lon_max} {lat_max} "
        cmd += f"-ts {atr['WIDTH']} {atr['LENGTH']} -of ISCE\n"
        cmd += f'fixImageXml.py -i {dem_out} -f\n\n'
        if action == 'write':
            self.f.write(cmd)
        elif action == 'run':
            print('Do resample DEM file...')
            print('  Dimension of the dataset (length, width): {}, {}'.format(atr['LENGTH'], atr['WIDTH']))
            print(f'  S N W E: {lat_min} {lat_max} {lon_min} {lon_max}')
            print(cmd)
            os.system(cmd)


    def write_deramp_ifg(self, fname='ifgramStack.h5', dset='unwrapPhase', ramp_type='linear', mask_file='maskTempCoh.h5', coeff_file=None):
        """Deramp the interferograms for easier identifying unwrapping errors in plots
        """
        fname = os.path.join(self.indir, fname)

        # file/dir
        fdir = os.path.dirname(fname)
        fbase, fext = os.path.splitext(os.path.basename(fname))
        coeff_file_default = os.path.join(fdir, f'rampCoeff_{fbase}.txt')
        coeff_file = os.path.join(fdir, f'rampCoeff_{fbase}_{dset}.txt')

        self.f.write(f'remove_ramp.py {fname} -d {dset} -s {ramp_type} -m {mask_file} --save-ramp-coeff --update\n\n')
        self.f.write(f'mv {coeff_file_default} {coeff_file}\n\n')


    def run_inversion_misfit(self, stack_file, dset, ts_file, mask_file='maskTempCoh.h5', eval_at='pair'):
        """Compute the misfit of interferograms inversion, misfit = timeseires - ifg_stack
            Can evaluate at 1) each pair 2) each epoch
        """
        stack_file = os.path.join(self.indir, stack_file)



    def write_modify_network(self, file='ifgramStack.h5'):
        """ Write command line: `modify_network.py`
        """
        template = self.template
        self.f.write(f'modify_network.py {os.path.join(self.indir, file)} -t {template}\n\n')


    def write_reference_point(self, files=[], ref_lalo=None, ref_yx=None):
        if ref_lalo:
            opt = f'--lat {ref_lalo[0]} --lon {ref_lalo[1]}'
        if ref_yx:
            opt = f'--row {ref_yx[0]} --col {ref_yx[1]}'
        else:
            opt = ''
        for file in files:
            self.f.write(f'reference_point.py {file} -t {self.template} {opt}\n')
        self.f.write('\n')


    def write_plot_network(self, stacks, cmap_vlist=[0.2, 0.7, 1.0], arg=''):
        for file in stacks:
            cmd  = f'plot_network.py {os.path.join(self.indir, file)} -t {self.template} '
            cmd += f"--nodisplay --cmap-vlist {' '.join(map(str,cmap_vlist))} {arg}\n"
            cmd += f"mkdir -p {self.pDict['path.extraPicDir']}\n"
            cmd += f"mv *.pdf *.png {self.pDict['path.extraPicDir']}\n\n"
            self.f.write(cmd)


    def write_ifgram_inversion(self, stack, mask, weight='var'):
        cmd  = f'ifgram_inversion.py {os.path.join(self.indir, stack)} -m {mask} -w {weight} '
        cmd += f'--cluster {self.cluster} --num-worker {self.numWorker} --ram {self.ram} --update\n\n'
        self.f.write(cmd)


    def write_generate_mask(self, threshold=None, ctype='temporal'):
        if not threshold:
            threshold = self.pDict.get('mintpy.tempCohThreshold', 0.90)
        if ctype == 'temporal':
            cohfile       = os.path.join(self.home, 'temporalCoherence.h5')
            self.coh_mask = os.path.join(self.home, 'maskTempCoh_'+f'{threshold}.h5')
        if ctype == 'spatial':
            cohfile       = os.path.join(self.home, 'avgSpatialCoh.h5.h5')
            self.coh_mask = os.path.join(self.home, 'maskSpatCoh_'+f'{threshold}.h5')

        cmd   = f'generate_mask.py {cohfile} -m {threshold} -o {self.coh_mask} --update\n\n'
        self.threshold = threshold
        self.f.write(cmd)


    def write_demErr(self, file, outfile, gfile=None):
        if not gfile:
            gfile = self.geom_file
        cmd  = f'dem_error.py {file} {self.time_func} -g {gfile} -o {outfile} '
        cmd += f'--cluster {self.cluster} --num-worker {self.numWorker} --ram {self.ram} --update\n\n'
        self.f.write(cmd)


    def write_closure_phase(self, ifile, nl, bw, action, nsig=3, neps=None, waterMask='waterMask.h5', outdir='.', ram=4, workers=4):
        if int(nl) < int(bw):
            sys.exit('--conn-level (assumed no bias) should be at least the --bandwidth of your analysis (Zheng et al. 2022)')

        cmd = f'closure_phase_bias.py -i {ifile} --nl {nl} --bw {bw} -a {action} --wm {waterMask} -o {outdir} '
        if nsig:
            cmd += f'--num-sigma {nsig} '
        if neps:
            cmd += f'--eps {neps} '
        cmd += f'--ram {ram} --num-worker {workers} \n\n'
        self.f.write(cmd)


    def write_ts2velo(self, tsfile, outfile, ts2velocmd='', out_dir=None, update=True):
        if not out_dir:
            out_dir = self.pDict['path.velocityDir']
        outfile = os.path.join(out_dir, outfile)
        cmd  = f'timeseries2velocity.py {tsfile} {self.time_func} -o {outfile} '
        cmd += f'--ref-lalo {self.refla} {self.reflo} --ref-date {self.ref_date} '

        if (ts2velocmd is None) or (ts2velocmd == 'none'):
            ts2velocmd = ''
        cmd += f' {ts2velocmd} '
        if update:
            cmd += '--update '
        cmd += '\n\n'
        self.f.write(cmd)


    def write_plate_motion(self, gfile=None, itrffile=None, vfile=None):
        """ Create commands for plate motion model adjustment in MintPy (Stephenson et al. 2022)
        """
        if not gfile:
            gfile = self.geom_file

        cmd   = f'plate_motion.py --geom {gfile} {self.itrfPlate} '
        if vfile:
            cmd += f'--velo {vfile} '
        if itrffile:
            cmd += f'-s {itrffile} '
            cmd += '\n\n'
        self.f.write(cmd)


    def write_plot_velo(self, file, dataset, vlim='None,None', dem_file=None, title=None, outfile=None, picdir=None, mask=None, update=True):
        if not picdir:
            picdir = self.pDict['path.extraPicDir']
        base = os.path.basename(file).split('.')[0]
        if not title:   title   = str(dataset)
        if not outfile:
            outfile = os.path.join(picdir, base+'.png')

        if not dem_file:
            dem_file   = os.path.join(self.indir, 'srtm.dem')   # or read from self.geom_file
        shade_exag = self.pDict['plot.shadeExag']
        shade_min  = self.pDict['plot.shadeMin']
        shade_max  = self.pDict['plot.shadeMax']
        alpha      = self.pDict['plot.velocityAlpha']
        cmap       = self.pDict['plot.velocityCmap']
        unit       = self.pDict['plot.veloUnit']
        dpi        = self.pDict['plot.dpi']
        xmin       = self.pDict['plot.lonMin']
        xmax       = self.pDict['plot.lonMax']
        ymin       = self.pDict['plot.latMin']
        ymax       = self.pDict['plot.latMax']

        if not mask:
            mask   = self.pDict['plot.velocityMsk']
            if mask:
                if   mask in ['water','auto']: mask_file = self.water_mask               # e.g., waterMask.h5
                elif mask in ['coh']         : mask_file = self.coh_mask                 # e.g., maskTempCoh.h5
                elif mask in ['connComp']    : mask_file = self.conn_mask                # e.g., maskConnComp.h5
                elif mask in ['custom']      : mask_file = self.pDict['path.customMask'] # e.g., maskCustom.h5
                else                         : mask_file = 'no'
        else:
            mask_file = mask

        vmin, vmax = vlim.split(',')

        cmd  = f"view.py {file} {dataset} -c {cmap} "
        cmd += f"--dem {dem_file} --alpha {alpha} "
        cmd += f"--dem-nocontour --shade-exag {shade_exag} --shade-min {shade_min} --shade-max {shade_max} "
        cmd += f"--mask {mask_file} --unit {unit} --ref-lalo {self.refla} {self.reflo} "
        if (vmin is not None) and (vmin != 'none'): cmd += f'--vlim {vmin} {vmax} '
        if (xmin is not None) and (xmin != 'none'): cmd += f'--sub-lon {xmin} {xmax} '
        if (ymin is not None) and (ymin != 'none'): cmd += f'--sub-lat {ymin} {ymax} '
        cmd += f"--nodisplay --dpi {dpi} "
        cmd += f"--figtitle {title} -o {outfile} "

        if update: cmd += '--update '
        cmd += '\n\n'
        self.f.write(cmd)


    def write_closurePhase_Mask(self, bw, nl, nsig=3, ram=8, workers=4, threshold=0.9,
                                clpdir='closurePhase', maskDict=None):
        """ Create commands for closure phase bias analysis (Zheng et al. 2022)
        """
        self.f.write(f'mkdir -p {clpdir} \n\n')

        # calculate
        self.f.write('## Do closure phase bias calculation\n\n')
        self.f.write(f'mask.py {self.ifg_stack} -m {self.water_mask} --fill 0 -o {self.ifg_stack_msk}\n\n')
        self.write_closure_phase(self.ifg_stack_msk, nl=nl, bw=bw, action='mask',           nsig=nsig, ram=ram, workers=workers)
        self.write_closure_phase(self.ifg_stack_msk, nl=nl, bw=bw, action='quick_estimate', nsig=nsig, ram=ram, workers=workers)
        #self.f.write(f'modify_network.py {self.ifg_stack_msk} --max-conn-num {bw}\n\n')
        #self.write_closure_phase(self.ifg_stack_msk, nl=nl, bw=bw, action='estimate',       nsig=nsig, ram=ram, workers=workers)
        #self.f.write(f'\\mv maskClosurePhase.h5 avgCpxClosurePhase.h5 wratio.h5 timeseriesBiasApprox.h5 timeseriesBias.h5 {clpdir}\n\n')
        self.f.write(f'\\mv maskClosurePhase.h5 avgCpxClosurePhase.h5 wratio.h5 timeseriesBiasApprox.h5 {clpdir}\n\n')

        # create customed masks
        if maskDict is None:
            maskDict = {
                'a' : 'maskClp_tCoh.h5',
                'b' : 'maskTri.h5',
                'c' : 'maskClp_tCoh_Tri.h5',
            }
        mask = SimpleNamespace(**maskDict)
        self.f.write('## Apply masking based on closure phase bias\n\n')
        self.f.write(f'mask.py {clpdir}/maskClosurePhase.h5 -m maskTempCoh_{threshold}.h5 --fill 0 -o {clpdir}/{mask.a}\n\n')
        self.f.write(f"test -f {self.pDict['path.customMask']} && mask.py {clpdir}/{mask.a} -m {self.pDict['path.customMask']} --fill 0 -o {clpdir}/{mask.a}\n\n")
        self.f.write(f'generate_mask.py numTriNonzeroIntAmbiguity.h5 -M 0 -o {mask.b}\n\n')
        self.f.write(f'mask.py {clpdir}/{mask.a} -m {mask.b} --fill 0 -o {clpdir}/{mask.c}\n\n')


        # ts2velo fit to icams GAM timeseries
        veldir = os.path.normpath('./' + self.pDict['path.velocityDir'])
        #self.write_ts2velo(f'{clpdir}/timeseriesBias.h5', 'velocityBias.h5', ts2velocmd=self.pDict['mintpy.ts2velo'], update=False)
        self.write_ts2velo(f'{clpdir}/timeseriesBiasApprox.h5', 'velocityAppBias.h5', ts2velocmd=self.pDict['mintpy.ts2velo'], update=False)

        # plot the velocityBias
        picdir   = os.path.normpath('./' + self.pDict['path.extraPicDir'])
        dset     = 'velocity'
        dem_file = './inputs/srtm.dem'
        mask     = f'maskTempCoh_{threshold}.h5'
        vlim     = self.pDict['plot.vm_mid']
        #self.write_plot_velo(f'{veldir}/velocityBias.h5', dset, vlim, dem_file=dem_file, mask=mask, picdir=picdir, update=False)
        self.write_plot_velo(f'{veldir}/velocityAppBias.h5', dset, vlim, dem_file=dem_file, mask=mask, picdir=picdir, update=False)

        dset  = 'velocityStd'
        ofile = os.path.join(picdir, dset+'Bias'+'.png')
        title = dset+'Bias'
        vlim  = self.pDict['plot.vm_STD']
        #self.write_plot_velo(f'{veldir}/velocityBias.h5', dset, vlim, dem_file=dem_file, mask=mask, title=title, outfile=ofile, update=False)
        self.write_plot_velo(f'{veldir}/velocityAppBias.h5', dset, vlim, dem_file=dem_file, mask=mask, title=title, outfile=ofile, update=False)


    def write_icams(self, ref_ts_file, ts_icams=None, proj='los', nproc=4, method='sklm', icamdir='icams'):
        """Create commands for ICAMS global atmospheric model resampling (Cao et al. 2021)
        """
        outfile1 = f'timeseries_icams_{proj}_{method}.h5'      # ICAMS esimated GAM timeseries ("add" to MintPy timeseries*.h5 to correct)
        outfile2 = f'timeseries_icamsCor_{proj}_{method}.h5'   # ICAMS correced timeseries
        self.f.write('## Need to have ICAMS and dependencies installed before running\n\n')
        self.f.write(f'# rm -rf ./{icamdir}/{self.gam}/*.npy ./{icamdir}/{self.gam}/sar # remove old los results\n\n')
        self.f.write(f"mkdir -p {icamdir} && \\cp {self.iDict['mintpy.load.metaFile']} {icamdir}\n\n")
        self.f.write(f'tropo_icams.py {ref_ts_file} {self.geom_file} --sar-par {icamdir}/IW1.xml --ref-file {ref_ts_file} --project {proj} --method {method} --nproc {nproc}\n\n')
        self.f.write(f'mv {outfile1} {outfile2} ./{icamdir}\n\n')

        # prepare icams GAM timeseries save into inputs/
        self.f.write(f'cd ./{icamdir}\n\n')
        if ts_icams is None:
            ts_icams = f'{self.indir}/{self.gam}-{proj}-{method}.h5'
        self.f.write(f"image_math.py {outfile1} '*' -1.0 --output ../{ts_icams}\n\n") # ICAMS esimated GAM timeseries ("subtract" from MintPy timeseries*.h5 to correct)
        self.f.write(f"rm -rf {outfile1} {outfile2}\n\n")

        self.f.write(f'cd {self.cwd}\n\n')

        return ts_icams


    def write_bwAnalysis(self, bw, ts_icams='timeseriesICAMS.h5', clpdir='closurePhase', veldir='velocity_out'):
        """ Create commands for bandwidth analysis, and apply closure phase bias and ICAMS corrections
        """
        bw_dir = f'./bw{bw}'

        to_indir  = os.path.normpath('../'+self.indir)
        to_clpdir = os.path.normpath('../'+clpdir)

        # inversion on short-bandwidth analysis
        self.f.write('## Short-bandwidth analysis and corrections\n\n')
        self.f.write(f'modify_network.py {self.ifg_stack_msk} --max-conn-num {bw}\n\n')
        self.f.write(f'mkdir -p {bw_dir} && cd {bw_dir}\n\n')
        self.f.write(f"ifgram_inversion.py {os.path.normpath('../'+self.ifg_stack_msk)} -t {to_indir+'/smallbaselineApp.cfg'} --update\n\n")

        # apply corrections (SET, ERA5, ICAMS, Ion)
        self.f.write(f"diff.py timeseries.h5     {to_indir}/SET.h5  -o timeseries_SET.h5 --force\n\n")
        self.f.write(f"diff.py timeseries_SET.h5 {to_indir}/{self.gam}.h5 -o timeseries_SET_{self.gam}.h5 --force\n\n")
        self.f.write(f"diff.py timeseries_SET_{self.gam}.h5 {to_indir}/ion.h5 -o timeseries_SET_{self.gam}_IonSmooth.h5 --force\n\n")
        self.f.write(f"diff.py timeseries_SET_{self.gam}.h5 {to_indir}/ionTotal.h5 -o timeseries_SET_{self.gam}_Ion.h5 --force\n\n")   # end
        self.f.write(f'diff.py timeseries_SET.h5  ../icams/{ts_icams}  -o timeseries_SET_{self.gam}S.h5 --force\n\n')
        self.f.write(f'diff.py timeseries_SET_{self.gam}S.h5  {to_indir}/ionTotal.h5  -o timeseries_SET_{self.gam}S_Ion.h5 --force\n\n')  # end

        # apply closure phase bias correction
        self.f.write(f'diff.py timeseries_SET_{self.gam}_Ion.h5 {to_clpdir}/timeseriesBiasApprox.h5 -o timeseries_SET_{self.gam}_Ion_clpApprox.h5 --force\n\n')
        self.f.write(f'diff.py timeseries_SET_{self.gam}_Ion.h5 {to_clpdir}/timeseriesBias.h5 -o timeseries_SET_{self.gam}_Ion_clp.h5 --force\n\n')
        self.f.write(f'diff.py timeseries_SET_{self.gam}S_Ion.h5 {to_clpdir}/timeseriesBiasApprox.h5 -o timeseries_SET_{self.gam}S_Ion_clpApprox.h5 --force\n\n')
        self.f.write(f'diff.py timeseries_SET_{self.gam}S_Ion.h5 {to_clpdir}/timeseriesBias.h5 -o timeseries_SET_{self.gam}S_Ion_clp.h5 --force\n\n')

        # dem error and final residuals
        gfile = os.path.normpath('../'+self.geom_file)
        self.write_demErr(gfile=gfile, file=f'timeseries_SET_{self.gam}S_Ion_clp.h5',    outfile=f'timeseries_SET_{self.gam}S_Ion_clp_demErr.h5')
        self.f.write(f'timeseries_rms.py  timeseriesResidual.h5  --template ../smallbaselineApp.cfg\n\n')

        # velocity of corrected short-bw timeseries
        ts2velocmd = self.pDict['mintpy.ts2velo']
        out_dir    = os.path.normpath('../' + veldir)
        self.write_ts2velo(f'timeseries_SET_{self.gam}_IonSmooth.h5',      f'velocityBW{bw}_SET_{self.gam}_IonSmooth_short.h5', ts2velocmd=ts2velocmd, out_dir=out_dir, update=False)
        self.write_ts2velo(f'timeseries_SET_{self.gam}_IonSmooth.h5',      f'velocityBW{bw}_SET_{self.gam}_IonSmooth.h5',       ts2velocmd=ts2velocmd, out_dir=out_dir, update=False)
        self.write_ts2velo(f'timeseries_SET_{self.gam}_Ion.h5',            f'velocityBW{bw}_SET_{self.gam}_Ion.h5',             ts2velocmd=ts2velocmd, out_dir=out_dir, update=False)
        self.write_ts2velo(f'timeseries_SET_{self.gam}S_Ion.h5',           f'velocityBW{bw}_SET_{self.gam}S_Ion.h5',            ts2velocmd=ts2velocmd, out_dir=out_dir, update=False)
        self.write_ts2velo(f'timeseries_SET_{self.gam}S_Ion_clpApprox.h5', f'velocityBW{bw}_SET_{self.gam}S_Ion_clpApprox.h5',  ts2velocmd=ts2velocmd, out_dir=out_dir, update=False)
        self.write_ts2velo(f'timeseries_SET_{self.gam}S_Ion_clp.h5',       f'velocityBW{bw}_SET_{self.gam}S_Ion_clp.h5',        ts2velocmd=ts2velocmd, out_dir=out_dir, update=False)
        self.write_ts2velo(f'timeseries_SET_{self.gam}S_Ion_clp_demErr.h5',f'velocityBW{bw}_SET_{self.gam}S_Ion_clp_demErr.h5', ts2velocmd=ts2velocmd, out_dir=out_dir, update=False)

        # apply ITRF reference frame
        gfile = os.path.normpath('../' + self.geom_file)
        itrffile = os.path.normpath('../'+self.indir+'/ITRF14_'+self.plateName+'.h5')
        self.write_plate_motion(gfile=gfile, itrffile=itrffile, vfile=f'{out_dir}/velocityBW{bw}_SET_{self.gam}_IonSmooth_short.h5')
        self.write_plate_motion(gfile=gfile, itrffile=itrffile, vfile=f'{out_dir}/velocityBW{bw}_SET_{self.gam}_IonSmooth.h5')
        self.write_plate_motion(gfile=gfile, itrffile=itrffile, vfile=f'{out_dir}/velocityBW{bw}_SET_{self.gam}_Ion.h5')
        self.write_plate_motion(gfile=gfile, itrffile=itrffile, vfile=f'{out_dir}/velocityBW{bw}_SET_{self.gam}S_Ion.h5')
        self.write_plate_motion(gfile=gfile, itrffile=itrffile, vfile=f'{out_dir}/velocityBW{bw}_SET_{self.gam}S_Ion_clpApprox.h5')
        self.write_plate_motion(gfile=gfile, itrffile=itrffile, vfile=f'{out_dir}/velocityBW{bw}_SET_{self.gam}S_Ion_clp.h5')
        self.write_plate_motion(gfile=gfile, itrffile=itrffile, vfile=f'{out_dir}/velocityBW{bw}_SET_{self.gam}S_Ion_clp_demErr.h5')

        # plot the velocity
        picdir   = os.path.normpath('../' + self.pDict['path.extraPicDir'])
        dset     = 'velocity'
        vlim     = self.pDict['plot.vm_mid']
        dem_file = '../inputs/srtm.dem'
        mask     = f'../maskTempCoh_{self.threshold}.h5'
        self.write_plot_velo(f'{out_dir}/velocityBW{bw}_SET_{self.gam}_IonSmooth_short_ITRF14.h5',  dset, vlim, dem_file=dem_file, mask=mask, picdir=picdir, update=False)
        self.write_plot_velo(f'{out_dir}/velocityBW{bw}_SET_{self.gam}_IonSmooth_ITRF14.h5',        dset, vlim, dem_file=dem_file, mask=mask, picdir=picdir, update=False)
        self.write_plot_velo(f'{out_dir}/velocityBW{bw}_SET_{self.gam}_Ion_ITRF14.h5',              dset, vlim, dem_file=dem_file, mask=mask, picdir=picdir, update=False)
        self.write_plot_velo(f'{out_dir}/velocityBW{bw}_SET_{self.gam}S_Ion_ITRF14.h5',             dset, vlim, dem_file=dem_file, mask=mask, picdir=picdir, update=False)
        self.write_plot_velo(f'{out_dir}/velocityBW{bw}_SET_{self.gam}S_Ion_clpApprox_ITRF14.h5',   dset, vlim, dem_file=dem_file, mask=mask, picdir=picdir, update=False)
        self.write_plot_velo(f'{out_dir}/velocityBW{bw}_SET_{self.gam}S_Ion_clp_ITRF14.h5',         dset, vlim, dem_file=dem_file, mask=mask, picdir=picdir, update=False)
        self.write_plot_velo(f'{out_dir}/velocityBW{bw}_SET_{self.gam}S_Ion_clp_demErr_ITRF14.h5',  dset, vlim, dem_file=dem_file, mask=mask, picdir=picdir, update=False)

        # reset the ifgram network to all pairs, finish
        self.f.write(f'cd {self.cwd}\n\n')
        self.f.write(f'modify_network.py {self.ifg_stack_msk} --reset\n')
        self.f.write(f'modify_network.py {self.ifg_stack_msk} -t {self.template}\n\n')


###############################
# Utilities
###############################
## ----------------- parameter file descriptions
PARAM_DESCRIPTION = {
    'mintpy.template'         : '# used to run mintpy and copy saved as smallbaselineApp.cfg',
    'mintpy.tempCohThreshold' : '# create a mask from temporal coherence',
    'mintpy.itrfPlate'        : '# plate arg parameters in ITRF14 plate motion model',
    'mintpy.plateName'        : '# plate name in ITRF14 plate motion model',
    'mintpy.ts2velo'          : '# commands for timeseries2velocity, see timeseries2velocity -h',
    'mintpy.bandwidth'        : '# bandwidth of ifgram network analysis (Zheng et al. 2022)',
    'mintpy.connLevel'        : '# connection level of ifgram network analysis, assume no bias (Zheng et al. 2022)',
    'path.wbdOrig'            : '# input water body',
    'path.demOrig'            : '# input DEM',
    'path.velocityDir'        : '# velocity output folder',
    'path.extraPicDir'        : '# extra pics output directory (e.g. velocity plots)',
    'path.customMask'         : '# Given any custom mask',
    'plot.shadeExag'          : '# DEM shaded relief exageration',
    'plot.shadeMin'           : '# DEM shaded relief min',
    'plot.shadeMax'           : '# DEM shaded relief max',
    'plot.velocityAlpha'      : '# transparency of velocity plot',
    'plot.velocityCmap'       : '# colormap for the velocity plot',
    'plot.tempCohCmap'        : '# colormap for temporal coherence in the network plot',
    'plot.velocityMsk'        : '# ways to mask velocity plot',
    'plot.veloUnit'           : '# velocity plot unit',
    'plot.dpi'                : '# output figure dpi',
    'plot.lonMin'             : '# longitude min',
    'plot.lonMax'             : '# longitude max',
    'plot.latMin'             : '# latitute min',
    'plot.latMax'             : '# latitute max',
    'plot.vm_big'             : '# vlim for large range',
    'plot.vm_mid'             : '# vlim for middle range',
    'plot.vm_sma'             : '# vlim for small range',
    'plot.vm_STD'             : '# vlim for data Std',
    'plot.vm_AMP'             : '# vlim for seasonal amplitude',
    'plot.vm_SET'             : '# vlim for solid earth tides',
    'plot.vm_GAM'             : '# vlim for WEATHER model',
    }

def write_line(f, pDict, keys):
    for key in keys:
        value = pDict[key]
        if type(value) == int or type(value) == float:
            value = str(value)
        if isinstance(value, list):
            value = ','.join(str(k) for k in value)
        if value is None:
            value = 'none'
        f.write(f'{key:25s} = {value:20s} {PARAM_DESCRIPTION[key]}\n')


def check_parameter_txt(param_file):
    ## read from param file as dict
    pDict = readfile.read_template(fname=param_file)
    for inkey in pDict.keys():
        if not inkey in PARAM_DESCRIPTION.keys(): print(f' ! Unrecognized parameter: {inkey}, will not be used')

    ## use default value if missing
    # MintPy related
    pDict['mintpy.template']          = pDict.get('mintpy.template'         , 'smallbaselineApp.cfg')
    pDict['mintpy.tempCohThreshold']  = pDict.get('mintpy.tempCohThreshold' , 0.90)
    pDict['mintpy.itrfPlate']         = pDict.get('mintpy.itrfPlate'        , None)
    pDict['mintpy.plateName']         = pDict.get('mintpy.plateName'        , None)
    pDict['mintpy.ts2velo']           = pDict.get('mintpy.ts2velo'          , None)
    pDict['mintpy.bandwidth']         = pDict.get('mintpy.bandwidth'        , 3)
    pDict['mintpy.connLevel']         = pDict.get('mintpy.connLevel'        , 10)
    # File paths and locations
    pDict['path.wbdOrig']             = pDict.get('path.wbdOrig'            , None)
    pDict['path.demOrig']             = pDict.get('path.demOrig'            , None)
    pDict['path.velocityDir']         = pDict.get('path.velocityDir'        , './velocity_out/')
    pDict['path.extraPicDir']         = pDict.get('path.extraPicDir'        , './pic_supp/')
    pDict['path.customMask']          = pDict.get('path.customMask'         , 'maskPoly.h5')
    # Some plotting parameters
    pDict['plot.shadeExag']           = pDict.get('plot.shadeExag'          , 0.02)
    pDict['plot.shadeMin']            = pDict.get('plot.shadeMin'           , -6000)
    pDict['plot.shadeMax']            = pDict.get('plot.shadeMax'           , 4000)
    pDict['plot.velocityAlpha']       = pDict.get('plot.velocityAlpha'      , 0.6)
    pDict['plot.velocityCmap']        = pDict.get('plot.velocityCmap'       , 'RdYlBu_r')
    pDict['plot.tempCohCmap']         = pDict.get('plot.tempCohCmap'        , 'RdYlBu_r')
    pDict['plot.velocityMsk']         = pDict.get('plot.velocityMsk'        , None)
    pDict['plot.veloUnit']            = pDict.get('plot.veloUnit'           , 'mm')
    pDict['plot.dpi']                 = pDict.get('plot.dpi'                , 300)
    pDict['plot.lonMin']              = pDict.get('plot.lonMin'             , None)
    pDict['plot.lonMax']              = pDict.get('plot.lonMax'             , None)
    pDict['plot.latMin']              = pDict.get('plot.latMin'             , None)
    pDict['plot.latMax']              = pDict.get('plot.latMax'             , None)
    pDict['plot.vm_big']              = pDict.get('plot.vm_big'             , [-8,8])
    pDict['plot.vm_mid']              = pDict.get('plot.vm_mid'             , [-5,5])
    pDict['plot.vm_sma']              = pDict.get('plot.vm_sma'             , [-0.2,0.2])
    pDict['plot.vm_STD']              = pDict.get('plot.vm_STD'             , [0,1.0])
    pDict['plot.vm_AMP']              = pDict.get('plot.vm_AMP'             , [0,16])
    pDict['plot.vm_SET']              = pDict.get('plot.vm_SET'             , [-0.2,0.2])
    pDict['plot.vm_GAM']              = pDict.get('plot.vm_GAM'             , [-2,2])

    ## write the full params to txt
    full_file = os.path.splitext(param_file)[0]+'.full.par'
    with open(full_file, 'w') as f:
        keys = list(np.array(list(pDict.keys()))[[k.startswith('mintpy') for k in pDict.keys()]])
        f.write('## MintPy related\n')
        write_line(f, pDict, keys)
        keys = list(np.array(list(pDict.keys()))[[k.startswith('path') for k in pDict.keys()]])
        f.write('\n## File paths and locations\n')
        write_line(f, pDict, keys)
        keys = list(np.array(list(pDict.keys()))[[k.startswith('plot') for k in pDict.keys()]])
        f.write('\n## Some plotting parameters\n')
        write_line(f, pDict, keys)

    ## read from the full params file
    pDict = readfile.read_template(fname=full_file)
    return pDict

#############################################################################################################
## Major function wrting the workflow
#############################################################################################################

def main(proc, inps):
    ########## Initializing the MintPy process ##############

    proc.get_template()
    proc.run_resampWbd()
    ram   = proc.ram
    nproc = proc.numWorker

    ########## Load data and geocode stack DEM ##############
    proc.create_run_file('run_0_prep')

    proc.write_smallbaselineApp(dostep='load_data')
    proc.write_radar2geo_inputs()
    proc.f.write(f'{FILE_NAME} {proc.param_file} -a dem_resamp\n')

    proc.f.close()


    ########## Network modifications and plots ##############
    proc.create_run_file('run_1_network')

    proc.write_smallbaselineApp(dostep='modify_network')
    proc.write_smallbaselineApp(dostep='reference_point')
    proc.write_smallbaselineApp(dostep='quick_overview')
    proc.write_plot_network(stacks=['ifgramStack.h5'], cmap_vlist=[0.2, 0.7, 1.0])
    proc.f.write('smallbaselineApp.py --plot \n\n')

    proc.f.close()


    ################### Network inversion ##################
    proc.create_run_file('run_2_inversion')

    proc.write_smallbaselineApp(dostep='correct_unwrap_error')
    proc.write_smallbaselineApp(dostep='invert_network')
    proc.write_generate_mask()

    proc.f.close()



    ################## ICAMS #######################
    proc.create_run_file('run_3_icams')

    icams_proj   = 'los'
    icams_method = 'sklm'
    ts_icams = proc.write_icams(ref_ts_file='timeseries.h5', proj=icams_proj, nproc=nproc, method=icams_method)
    proc.f.write("echo 'Normal finish the ICAMS analysis'\n")
    proc.f.close()


    ################ Apply corrections ####################
    proc.create_run_file('run_4_corrections')

    proc.itrfPlate = proc.pDict['mintpy.itrfPlate']
    proc.plateName = proc.pDict['mintpy.plateName']
    gam = proc.gam

    if proc.plateName == 'none':
        proc.plateName = proc.itrfPlate.split()[-1]

    itrffile = os.path.join(proc.indir, f'ITRF14_{proc.plateName}.h5')
    proc.write_plate_motion(itrffile=itrffile)
    proc.f.write('add.py  inputs/ion.h5   inputs/ionBurstRamp.h5  -o inputs/ionTotal.h5  --force\n\n')

    proc.write_smallbaselineApp(dostep='correct_LOD')
    proc.write_smallbaselineApp(dostep='correct_SET')
    proc.write_smallbaselineApp(dostep='correct_troposphere')

    # try correct troposphere with ICAMS, with file suffix ERA5S, "S" for Stochastic
    proc.f.write(f'diff.py timeseries_SET.h5 {ts_icams} -o timeseries_SET_{gam}S.h5  --force\n\n')

    # correct for ionosphere, both smooth iono and the iono burst ramps (can be dangerous)
    proc.f.write(f'diff.py timeseries_SET_{gam}.h5  inputs/ion.h5       -o timeseries_SET_{gam}_Ion.h5  --force\n\n')
    proc.f.write(f'diff.py timeseries_SET_{gam}.h5  inputs/ionTotal.h5  -o timeseries_SET_{gam}_IonTotal.h5  --force\n\n')
    proc.f.write(f'diff.py timeseries_SET_{gam}S.h5  inputs/ion.h5       -o timeseries_SET_{gam}S_Ion.h5  --force\n\n')
    proc.f.write(f'diff.py timeseries_SET_{gam}S.h5  inputs/ionTotal.h5  -o timeseries_SET_{gam}S_IonTotal.h5  --force\n\n')

    # correct for any topographic term (due to look angle error, commonly referred to as DEM error)
    # then compute timeseriesResidual.h5 against some functional fit
    proc.write_demErr(f'timeseries_SET_{gam}_Ion.h5', f'timeseries_SET_{gam}_Ion_demErr.h5') # operate on the smooth iono corrected file
    proc.write_demErr(f'timeseries_SET_{gam}S_Ion.h5', f'timeseries_SET_{gam}S_Ion_demErr.h5') # operate on the smooth iono corrected file

    # compute RMS of the residuals
    proc.write_smallbaselineApp(dostep='residual_RMS')

    # whether to deramp timeseries?
    proc.write_smallbaselineApp(dostep='deramp')

    proc.f.close()


    ################ Velocity estimation ###################
    proc.create_run_file('run_5_velocity')

    # velocity fitting & plotting dictionary
    ts2veloDict = {
        # velocity fit to the stages of time series
        f'velocity'                              : [f'timeseries'                       , proc.pDict['plot.vm_mid']],
        f'velocity_SET'                          : [f'timeseries_SET'                   , proc.pDict['plot.vm_mid']],
        f'velocity_SET_{gam}'                    : [f'timeseries_SET_{gam}'             , proc.pDict['plot.vm_mid']],
        f'velocity_SET_{gam}_Ion'                : [f'timeseries_SET_{gam}_Ion'         , proc.pDict['plot.vm_mid']],
        f'velocity_SET_{gam}_IonTotal'           : [f'timeseries_SET_{gam}_IonTotal'    , proc.pDict['plot.vm_mid']],
        f'velocity_SET_{gam}_Ion_demErr'         : [f'timeseries_SET_{gam}_Ion_demErr'  , proc.pDict['plot.vm_mid']],
        f'velocity_SET_{gam}_Ion_demErr_ITRF14'  : [None                                , proc.pDict['plot.vm_mid']],
        f'velocity_SET_{gam}S'                   : [f'timeseries_SET_{gam}S'            , proc.pDict['plot.vm_mid']],
        f'velocity_SET_{gam}S_Ion'               : [f'timeseries_SET_{gam}S_Ion'        , proc.pDict['plot.vm_mid']],
        f'velocity_SET_{gam}S_IonTotal'          : [f'timeseries_SET_{gam}S_IonTotal'   , proc.pDict['plot.vm_mid']],
        f'velocity_SET_{gam}S_Ion_demErr'        : [f'timeseries_SET_{gam}S_Ion_demErr' , proc.pDict['plot.vm_mid']],
        f'velocity_SET_{gam}S_Ion_demErr_ITRF14' : [None                                , proc.pDict['plot.vm_mid']],
        f'velocityICAMS-PyAPS'                   : [None                                , proc.pDict['plot.vm_sma']],

        # apparent velocity fit of each correction screen
        f'velocitySET'                           : [f'inputs/SET'                       , proc.pDict['plot.vm_SET']],
        f'velocity{gam}'                         : [f'inputs/{gam}'                     , proc.pDict['plot.vm_GAM']],
        f'velocity{gam}S'                        : [ts_icams.split('.h5')[0]            , proc.pDict['plot.vm_GAM']],
        f'velocityIon'                           : [f'inputs/ion'                       , proc.pDict['plot.vm_mid']],
        f'velocityIonBurstRamp'                  : [f'inputs/ionBurstRamp'              , proc.pDict['plot.vm_sma']],
        f'velocityIonTotal'                      : [f'inputs/ionTotal'                  , proc.pDict['plot.vm_mid']],
        }

    # fit on these time series
    for key, item in ts2veloDict.items():
        if not item[0]: continue
        vfile = key + '.h5'
        tfile = item[0] + '.h5'
        proc.write_ts2velo(tfile, vfile, ts2velocmd=proc.pDict['mintpy.ts2velo'], update=False)

    # velocity difference between ICAMS vs PyAPS
    veldir = proc.pDict['path.velocityDir']
    proc.f.write(f'diff.py {veldir}velocity{gam}S.h5 {veldir}velocity{gam}.h5 -o {veldir}velocityICAMS-PyAPS.h5 \n\n')

    # plate motion removal (reference frame adjustment) on these velocity
    itrffile = os.path.join(proc.indir, f'ITRF14_{proc.plateName}.h5')
    vfile    = os.path.join(veldir, f'velocity_SET_{gam}_Ion_demErr.h5')
    proc.write_plate_motion(vfile=vfile, itrffile=itrffile)
    vfile    = os.path.join(veldir, f'velocity_SET_{gam}S_Ion_demErr.h5')
    proc.write_plate_motion(vfile=vfile, itrffile=itrffile)

    proc.f.close()


    ################## Plot Velocity #######################
    proc.create_run_file('run_6_velocityPlot')

    picdir = proc.pDict['path.extraPicDir']
    veldir = proc.pDict['path.velocityDir']
    proc.f.write(f'mkdir -p {picdir}\n\n')

    dset = 'velocity'
    vfiles = [os.path.join(veldir,x+'.h5') for x in ts2veloDict.keys()]
    vfiles += glob.glob(os.path.join(veldir,'*.h5'))
    vfiles = sorted(list(set(vfiles)))

    for vfile in vfiles:
        key = os.path.basename(vfile).split('.h5')[0]
        ofile = os.path.join(picdir, key + '.png')
        if key in ts2veloDict.keys():
            vlim  = ts2veloDict[key][1]
            if ts2veloDict[key][0]:
                if ts2veloDict[key][0].startswith('timeseries_'):
                    title = dset+ts2veloDict[key][0].split('timeseries')[-1]
                else:
                    title = key
            else:
                title = key
        else:
            vlim = proc.pDict['plot.vm_mid']
            title = str(key)
        proc.write_plot_velo(vfile, dset, vlim, title=title, outfile=ofile, update=False)

    dset = 'velocityStd'
    suffs = ['', gam, gam+'S', 'SET', 'Ion']
    for suff in suffs:
        vfile = os.path.join(veldir, 'velocity'+suff+'.h5')
        ofile = os.path.join(picdir, dset+suff+'.png')
        title = dset+suff
        proc.write_plot_velo(vfile, dset, proc.pDict['plot.vm_STD'], title=title, outfile=ofile, update=False)

    proc.f.write('smallbaselineApp.py --plot \n\n')
    proc.f.close()


    ################## Closure phase bias ####################### Testing, need re-factor
    proc.create_run_file('run_7_closurePhase')

    clpdir    = './closurePhase'
    nsig      = 3
    bw        = int(proc.pDict['mintpy.bandwidth'])
    nl        = int(proc.pDict['mintpy.connLevel'])
    threshold = proc.pDict.get('mintpy.tempCohThreshold', 0.90)
    maskDict = {
        'a' : 'maskClp_tCoh.h5',
        'b' : 'maskTri.h5',
        'c' : 'maskClp_tCoh_Tri.h5',
    }
    proc.ifg_stack_msk = os.path.join(proc.indir,'ifgramStack_msk.h5')
    proc.write_closurePhase_Mask(bw, nl, nsig, ram, nproc, threshold, clpdir, maskDict)
    proc.f.write("echo 'Normal finish the closure phase bias analysis'\n")
    proc.f.close()


    ########## generate more timeseries corrections for demo ##########
    proc.create_run_file('run_8_orderTS')

    # model long-wavelength first
    proc.f.write(f'diff.py timeseries_SET.h5 {itrffile} -o timeseries_SET_ITRF14.h5  --force\n\n')
    proc.f.write(f'diff.py timeseries_SET_ITRF14.h5 inputs/ERA5.h5 -o timeseries_SET_ITRF14_{gam}.h5  --force\n\n')
    proc.f.write(f'diff.py timeseries_SET_ITRF14.h5 {ts_icams} -o timeseries_SET_ITRF14_{gam}S.h5  --force\n\n')
    proc.f.write(f'diff.py timeseries_SET_ITRF14_{gam}S.h5 inputs/ion.h5 -o timeseries_SET_ITRF14_{gam}S_Ion.h5  --force\n\n')
    proc.f.write(f'diff.py timeseries_SET_ITRF14_{gam}S.h5 inputs/ionTotal.h5 -o timeseries_SET_ITRF14_{gam}S_IonTotal.h5  --force\n\n')
    proc.f.write(f'diff.py timeseries_SET_ITRF14_{gam}S_Ion.h5 closurePhase/timeseriesBiasApprox.h5 -o timeseries_SET_ITRF14_{gam}S_Ion_Cpb.h5  --force\n\n')


    # velocity fitting & plotting dictionary
    ts2veloDict = {
        # velocity fit to the stages of time series
        f'velocity_SET_ITRF14'                 : [f'timeseries_SET_ITRF14'                 , proc.pDict['plot.vm_mid']],
        f'velocity_SET_ITRF14_{gam}'           : [f'timeseries_SET_ITRF14_{gam}'           , proc.pDict['plot.vm_mid']],
        f'velocity_SET_ITRF14_{gam}S'          : [f'timeseries_SET_ITRF14_{gam}S'          , proc.pDict['plot.vm_mid']],
        f'velocity_SET_ITRF14_{gam}S_Ion'      : [f'timeseries_SET_ITRF14_{gam}S_Ion'      , proc.pDict['plot.vm_mid']],
        f'velocity_SET_ITRF14_{gam}S_IonTotal' : [f'timeseries_SET_ITRF14_{gam}S_IonTotal' , proc.pDict['plot.vm_mid']],
        f'velocity_SET_ITRF14_{gam}S_Ion_Cpb'  : [f'timeseries_SET_ITRF14_{gam}S_Ion_Cpb'  , proc.pDict['plot.vm_mid']],
        }

    # fit on these time series
    for key, item in ts2veloDict.items():
        if not item[0]: continue
        vfile = key + '.h5'
        tfile = item[0] + '.h5'
        proc.write_ts2velo(tfile, vfile, ts2velocmd=proc.pDict['mintpy.ts2velo'], update=False)

    # plot
    vfiles = [os.path.join(veldir,x+'.h5') for x in ts2veloDict.keys()]
    vfiles = sorted(list(set(vfiles)))

    for vfile in vfiles:
        dset  = 'velocity'
        key   = os.path.basename(vfile).split('.h5')[0]
        suff  = key.split(dset)[-1]
        ofile = os.path.join(picdir, key + '.png')
        vlim  = ts2veloDict[key][1]
        title = dset+ts2veloDict[key][0].split('timeseries')[-1]
        proc.write_plot_velo(vfile, dset, vlim, title=title, outfile=ofile, update=False)

        dset  = 'velocityStd'
        ofile = os.path.join(picdir, dset+suff+'.png')
        vlim  = proc.pDict['plot.vm_STD']
        title = dset+suff
        proc.write_plot_velo(vfile, dset, vlim, title=title, outfile=ofile, update=False)

    proc.f.close()


    ############# Compile all together in a script ############
    proc.create_run_file('run_all')
    runfiles = sorted(glob.glob(os.path.join(inps.proc_home, 'run_*_*')))
    for rf in runfiles:
        proc.f.write(f"bash {rf} \n")
    proc.f.close()



    ################## BW-analysis ####################### Testing, need re-factor
    proc.create_run_file('run_x_bwAnalysis')

    proc.write_bwAnalysis(bw, ts_icams, clpdir=clpdir, veldir=veldir)
    proc.f.write("echo 'Normal finish the short BW analysis'\n")
    proc.f.close()


#############################################################################################################
#############################################################################################################

if __name__ == '__main__':

    # get user inputs
    inps = cmdLineParse()

    # initialize the process
    proc = SBApp(inps.param_file, inps.proc_home)

    # run it
    if inps.action == 'all':
        main(proc, inps)
    elif inps.action == 'dem_resamp':
        proc.f = open('run_0_prep', 'a+') # append after the runfile
        proc.run_resamp_dem(inps.dem_out, inps.geo_in, inps.dem_orig, inps.dem_action)
        proc.f.close()

    print('Finish writing the run files. Go ahead and run them sequentially.')