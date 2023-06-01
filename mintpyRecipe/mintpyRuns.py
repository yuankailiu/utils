#!/usr/bin/env python3
############################################################
# This code is a recipe for MintPy processing              #
# Author: Yuan-Kai Liu, 2022                               #
############################################################
# It generates mintpy cmd run files for each stage and to run in bash

import argparse
import glob
import os
import shutil
import sys

# isce
import isce
# mintpy
import mintpy
import numpy as np
from applications.gdal2isce_xml import gdal2isce_xml
from isceobj.Alos2Proc.Alos2ProcPublic import waterBodyRadar
from mintpy.utils import readfile


def cmdLineParse():
    '''
    Command line parsers
    '''
    description = 'Generates mintpy command line run files for each stage and to run in bash'

    EXAMPLE = """Examples:
        ## Specify the `params.json` and the process home directory under `mintpy/`.
        python mintpyRuns.py -j params.json
    """
    epilog = EXAMPLE
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter,  epilog=epilog)

    parser.add_argument('-p', '--param', dest='param_file', type=str, required=True,
            help = 'TEXT file contrains the parameters for processing')
    parser.add_argument('-d', '--home', dest='proc_home', type=str, default='.',
            help = 'mintpy processing home directory')
    parser.add_argument('-a', '--action', dest='action', type=str, default='all',
            help = 'Allow either `all` or `dem_resamp`')

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
        """Initializing the SmallBaselineApp object
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


    def write_loaddata(self, project=None, only_geom=False):
        """ Write command line: `load_data.py`
        """
        cmd = f'load_data.py -t smallbaselineApp.cfg '
        if project:
            cmd += f'--project {project} '
        if only_geom:
            cmd += '--geom '
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
        cmd = f"geocode.py {' '.join(map(str,files))} -l {geom_rdr} --lalo {' '.join(map(str,lalo))} --ram {self.ram} --update\n"

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


    def run_resamp_dem(self, action='run'):
        """
        Use GDAL to resample the orignal DEM to match the extent, dimension, and resolution of
        [This is optional], just to cover the full extent when using topsStack radar coord datsets
        (when geocode geometryRadar.h5 to geometryGeo.h5, the height will have large gaps; not pretty)
        Should be run after having the geometryGeo.h5 file (must in geo-coord to allow reading lon lat)
        The output DEM is then saved separetly (inputs/srtm.dem)
        The output DEM is mainly for plotting purposes using view.py
        """
        #proc_home = os.path.expanduser(jobj.proc_home)
        dem_out   = os.path.join(self.indir, 'srtm.dem')
        geo_file  = os.path.join(self.indir, 'geometryGeo.h5')
        dem_orig  = self.pDict['path.demOrig']

        # Read basic attributes from .h5
        atr = readfile.read(geo_file, datasetName='height')[1]

        # compute latitude and longitude min max
        lon_min = float(atr['X_FIRST']) - float(atr['X_STEP'])
        lon_max = float(atr['X_FIRST']) + float(atr['X_STEP']) * (int(atr['WIDTH'])+1)
        lat_max = float(atr['Y_FIRST']) - float(atr['Y_STEP'])
        lat_min = float(atr['Y_FIRST']) + float(atr['Y_STEP']) * (int(atr['LENGTH'])+1)

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
        self.f.write(cmd)


    def write_demErr(self, file, outfile):
        cmd  = f'dem_error.py {file} {self.time_func} -g {self.geom_file} -o {outfile} '
        cmd += f'--cluster {self.cluster} --num-worker {self.numWorker} --ram {self.ram} --update\n\n'
        self.f.write(cmd)


    def write_ts2velo(self, tsfile, outfile, ts2velocmd='', update=True):
        outfile = os.path.join(self.pDict['path.velocityDir'], outfile)
        cmd  = f'timeseries2velocity.py {tsfile} {self.time_func} -o {outfile} '
        cmd += f'--ref-lalo {self.refla} {self.reflo} --ref-date {self.ref_date} '

        if (ts2velocmd is None) or (ts2velocmd == 'none'):
            ts2velocmd = ''
        cmd += f'{ts2velocmd} '
        if update:
            cmd += '--update '
        cmd += '\n\n'
        self.f.write(cmd)


    def write_plate_motion(self, vfile=None):
        self.platename = self.pDict['mintpy.itrfPlate']
        cmd   = f'plate_motion.py --geom {self.geom_file} --plate {self.platename} --velo {vfile}\n\n'
        self.f.write(cmd)


    def write_plot_velo(self, file, dataset, vlim='None,None', title=None, outfile=None, mask=None, update=True):
        picdir = self.pDict['path.extraPicDir']
        base = os.path.basename(file).split('.')[0]
        if not title:   title   = str(dataset)
        if not outfile: outfile = os.path.join(picdir, base+'.png')

        #dem_file = self.geom_file
        dem_file   = os.path.join(self.indir, 'srtm.dem')
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
        mask       = self.pDict['plot.velocityMsk']

        if mask:
            if   mask in ['water','auto']: mask_file = self.water_mask               # e.g., waterMask.h5
            elif mask in ['coh']         : mask_file = self.coh_mask                 # e.g., maskTempCoh.h5
            elif mask in ['connComp']    : mask_file = self.conn_mask                # e.g., maskConnComp.h5
            elif mask in ['custom']      : mask_file = self.pDict['path.customMask'] # e.g., maskCustom.h5
            else                         : mask_file = 'no'

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


###############################
# Utilities
###############################
def check_parameter_txt(param_file):
    ## parameter file comment
    COMMENT = {'mintpy.template'          : '# used to run mintpy and copy saved as smallbaselineApp.cfg',
               'mintpy.tempCohThreshold'  : '# create a mask from temporal coherence',
               'mintpy.itrfPlate'         : '# plate name in ITRF14 plate motion model',
               'mintpy.ts2velo'           : '# commands for timeseries2velocity, see timeseries2velocity -h',
               'path.wbdOrig'             : '# input water body',
               'path.demOrig'             : '# input DEM',
               'path.velocityDir'         : '# velocity output folder',
               'path.extraPicDir'         : '# extra pics output directory (e.g. velocity plots)',
               'path.customMask'          : '# Given any custom mask',
               'plot.shadeExag'           : '# DEM shaded relief exageration',
               'plot.shadeMin'            : '# DEM shaded relief min',
               'plot.shadeMax'            : '# DEM shaded relief max',
               'plot.velocityAlpha'       : '# transparency of velocity plot',
               'plot.velocityCmap'        : '# colormap for the velocity plot',
               'plot.tempCohCmap'         : '# colormap for temporal coherence in the network plot',
               'plot.velocityMsk'         : '# ways to mask velocity plot',
               'plot.veloUnit'            : '# velocity plot unit',
               'plot.dpi'                 : '# output figure dpi',
               'plot.lonMin'              : '# longitude min',
               'plot.lonMax'              : '# longitude max',
               'plot.latMin'              : '# latitute min',
               'plot.latMax'              : '# latitute max',
               'plot.vm_big'              : '# vlim for large range',
               'plot.vm_mid'              : '# vlim for middle range',
               'plot.vm_sma'              : '# vlim for small range',
               'plot.vm_STD'              : '# vlim for data Std',
               'plot.vm_AMP'              : '# vlim for seasonal amplitude',
               'plot.vm_SET'              : '# vlim for solid earth tides',
               'plot.vm_ERA'              : '# vlim for WEATHER model',
               }

    pDict = readfile.read_template(fname=param_file)

    ## read parameters, use default value if missing
    pDict['mintpy.template']          = pDict.get('mintpy.template'         , 'smallbaselineApp.cfg')
    pDict['mintpy.tempCohThreshold']  = pDict.get('mintpy.tempCohThreshold' , 0.90)
    pDict['mintpy.itrfPlate']         = pDict.get('mintpy.itrfPlate'        , None)
    pDict['mintpy.ts2velo']           = pDict.get('mintpy.ts2velo'          , None)

    pDict['path.wbdOrig']             = pDict.get('path.wbdOrig'            , None)
    pDict['path.demOrig']             = pDict.get('path.demOrig'            , None)
    pDict['path.velocityDir']         = pDict.get('path.velocityDir'        , './velocity_out/')
    pDict['path.extraPicDir']         = pDict.get('path.extraPicDir'        , './pic_supp/')
    pDict['path.customMask']          = pDict.get('path.customMask'         , None)

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
    pDict['plot.vm_ERA']              = pDict.get('plot.vm_ERA'             , [-2,2])

    # write the full params to txt
    full_file = os.path.splitext(param_file)[0]+'_full.par'
    with open(full_file, 'w') as f:
        def write_line(f, pDict, keys):
            for key in keys:
                value = pDict[key]
                if type(value) == int or type(value) == float:
                    value = str(value)
                if isinstance(value, list):
                    value = ','.join(str(k) for k in value)
                if value is None:
                    value = 'none'
                f.write(f'{key:25s} = {value:30s} {COMMENT[key]}\n')
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


    ########## Load data and geocode stack DEM ##############
    proc.f = open('run_0_prep', 'w')

    proc.write_smallbaselineApp(dostep='load_data')
    proc.write_loaddata(only_geom=True)
    proc.write_radar2geo_inputs()
    proc.f.write(f'{os.path.basename(__file__)} -p {proc.param_file} -a dem_resamp\n')

    proc.f.close()


    ########## Network modifications and plots ##############
    proc.f = open('run_1_network', 'w')

    proc.write_smallbaselineApp(dostep='modify_network')
    #proc.write_modify_network(file='ionStack.h5')
    proc.write_smallbaselineApp(dostep='reference_point')
    #proc.write_reference_point(files=['inputs/ionStack.h5'])
    proc.write_smallbaselineApp(dostep='quick_overview')
    proc.write_plot_network(stacks=['ifgramStack.h5'], cmap_vlist=[0.2, 0.7, 1.0])
    proc.f.write('smallbaselineApp.py --plot \n\n')

    proc.f.close()


    ################### Network inversion ##################
    proc.f = open('run_2_inversion', 'w')

    proc.write_smallbaselineApp(dostep='correct_unwrap_error')
    proc.write_smallbaselineApp(dostep='invert_network')
    #proc.write_ifgram_inversion('ionStack.h5', mask='waterMask.h5', weight='no')
    proc.write_generate_mask()

    proc.f.close()


    ################ Apply corrections ####################
    proc.f = open('run_3_corrections', 'w')

    proc.write_smallbaselineApp(dostep='correct_LOD')
    proc.write_smallbaselineApp(dostep='correct_SET')
    proc.write_smallbaselineApp(dostep='correct_troposphere')
    proc.f.write('add.py  inputs/ion.h5             inputs/ionBurstRamp.h5  -o inputs/ionTotal.h5          --force\n\n')
    proc.f.write('diff.py timeseries_SET_ERA5.h5    inputs/ionTotal.h5      -o timeseries_SET_ERA5_Ion.h5  --force\n\n')

    #proc.write_smallbaselineApp(dostep='correct_topography')
    proc.write_demErr('timeseries_SET_ERA5_Ion.h5', 'timeseries_SET_ERA5_Ion_demErr.h5')
    proc.write_smallbaselineApp(dostep='residual_RMS')
    proc.write_smallbaselineApp(dostep='deramp')

    proc.f.close()


    ################ Velocity estimation ###################

    ts2veloDict = { 'velocity'              : ['timeseries'                     , proc.pDict['plot.vm_mid']],
                    'velocity2'             : ['timeseries_SET'                 , proc.pDict['plot.vm_mid']],
                    'velocity3'             : ['timeseries_SET_ERA5'            , proc.pDict['plot.vm_mid']],
                    'velocity4'             : ['timeseries_SET_ERA5_Ion'        , proc.pDict['plot.vm_mid']],
                    'velocity5'             : ['timeseries_SET_ERA5_Ion_demErr' , proc.pDict['plot.vm_mid']],
                    'velocitySET'           : ['inputs/SET'                     , proc.pDict['plot.vm_SET']],
                    'velocityERA5'          : ['inputs/ERA5'                    , proc.pDict['plot.vm_ERA']],
                    'velocityIon'           : ['inputs/ion'                     , proc.pDict['plot.vm_mid']],
                    'velocityIonBurstRamp'  : ['inputs/ionBurstRamp'            , proc.pDict['plot.vm_sma']],
                    'velocityIonTotal'      : ['inputs/ionTotal'                , proc.pDict['plot.vm_mid']],
                    }

    proc.f = open('run_4_velocity', 'w')

    cmd = proc.pDict['mintpy.ts2velo']
    for key, item in ts2veloDict.items():
        vfile = key + '.h5'
        tfile = item[0] + '.h5'
        proc.write_ts2velo(tfile, vfile, ts2velocmd=cmd, update=False)

    proc.write_plate_motion(vfile=os.path.join(proc.pDict['path.velocityDir'], 'velocity5.h5'))

    proc.f.close()


    ################## Plot Velocity #######################
    proc.f = open('run_5_velocityPlot', 'w')

    outdir = proc.pDict['path.extraPicDir']
    veldir = proc.pDict['path.velocityDir']
    proc.f.write(f'mkdir -p {outdir}\n\n')

    dset = 'velocity'
    vfiles = sorted(glob.glob(os.path.join(veldir+'/*.h5')))
    for vfile in vfiles:
        key = os.path.basename(vfile).split('.h5')[0]
        ofile = os.path.join(outdir, key + '.png')
        if key in ts2veloDict.keys():
            vlim  = ts2veloDict[key][1]
            if ts2veloDict[key][0].startswith('timeseries_'):
                title = dset+ts2veloDict[key][0].split('timeseries')[-1]
            else:
                title = key
        else:
            vlim = proc.pDict['plot.vm_mid']
            title = str(key)
        proc.write_plot_velo(vfile, dset, vlim, title, ofile, update=False)

    dset = 'velocityStd'
    suffs = ['','ERA5','SET','Ion']
    for suff in suffs:
        vfile = os.path.join(veldir, 'velocity'+suff+'.h5')
        ofile = os.path.join(outdir, dset+suff+'.png')
        title = dset+suff
        proc.write_plot_velo(vfile, dset, proc.pDict['plot.vm_STD'], title, ofile, update=False)

    proc.f.write('smallbaselineApp.py --plot \n\n')
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
        proc.f = open('run_0_prep', 'a') # append after the runfile
        proc.run_resamp_dem(action='run')
        proc.f.close()

    print('Finish writing the run files. Go ahead and run them sequentially.')
