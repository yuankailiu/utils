#!/usr/bin/env python3
############################################################
# This code is a recipe for MintPy processing              #
# Author: Yuan-Kai Liu, 2022                               #
############################################################
# It generates mintpy cmd run files for each stage and to run in bash

import argparse
import glob
import json
import os
import shutil
import sys

# isce
import isce
# mintpy
import mintpy
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

    parser.add_argument('-j', '--json', dest='json_file', type=str, required=True,
            help = 'JSON file contrains the parameters for processing')
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

    def __init__(self, json_file, proc_home='.'):
        """Initializing the SmallBaselineApp object
        """
        # set default file paths
        self.json       = os.path.expanduser(json_file)
        self.home       = os.path.expanduser(proc_home)
        self.cwd        = os.getcwd()
        self.picdir     = os.path.join(self.home,  'pic')
        self.picdir2    = os.path.join(self.home,  'pic_supp')
        self.water_mask = os.path.join(self.home,  'waterMask.h5')
        self.coh_mask   = os.path.join(self.home,  'maskTempCoh.h5')
        self.conn_mask  = os.path.join(self.home,  'maskConnComp.h5')
        self.indir      = os.path.join(self.home,  'inputs')
        self.geom_file  = os.path.join(self.indir, 'geometryGeo.h5')
        self.ifg_stack  = os.path.join(self.indir, 'ifgramStack.h5')
        self.ion_stack  = os.path.join(self.indir, 'ionStack.h5')

        # print locations
        print(f'Current path: {self.cwd}')
        print(f'Reading MintPy custom json at: {self.json}')
        print(f'MintPy processing directory at: {self.home}')

        with open(self.json) as f:
            self.jdic = json.load(f)


    def get_template(self):
        """ Define the template and save a copy
        """
        self.template    = self.jdic.get('template',   'smallbaselineApp.cfg')
        self.templateIon = self.jdic.get('templateIon', self.template)
        self.template    = glob.glob(os.path.join(self.home, self.template))[0]
        self.templateIon = glob.glob(os.path.join(self.home, self.templateIon))[0]

        print(f'Use template for regular     pairs: {self.template}')
        print(f'Use template for ionospheric pairs: {self.templateIon}')
        #print('Update smallbaselineApp.cfg based on {}'.format(self.template))
        #shutil.copyfile(self.template, os.path.join(self.indir, 'smallbaselineApp.cfg'))

        for outdir in [self.indir, self.picdir]:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            for cfg in [self.template, self.templateIon]:
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
        wbdFile        = os.path.abspath(self.jdic['wbd_orig'])
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
        processor = self.iDict['mintpy.load.processor']
        cmd = f'load_data.py -t {self.template} --processor {processor} '
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
        file_list = ['geometryRadar.h5', 'ifgramStack.h5', 'ionStack.h5']
        files = []
        for f in file_list:
            files.append(os.path.join(self.indir, f))

        # command line
        cmd = f"geocode.py {' '.join(map(str,files))} -l {geom_rdr} --lalo {' '.join(map(str,lalo))} --ram {self.ram} --update\n"

        # manage files
        cmd += f'mkdir -p {rdr_dir}\n'
        files = glob.glob(os.path.join(self.indir,'geo_*.h5'))
        for file in files:
            basefile = os.path.basename(file).split('geo_')[-1]
            cmd += f'mv {basefile} {os.path.join(rdr_dir, basefile)}\n'
            if basefile.startswith('geometry'):
                outfile = 'geometryGeo.h5'
            else:
                outfile = str(basefile)
            cmd += f'mv {file} {os.path.join(self.indir, outfile)}\n'
        self.f.write(cmd+'\n')


    def write_resamp_dem(self):
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
        dem_orig  = self.jdic['dem_orig']

        # Read basic attributes from .h5
        atr = readfile.read(geo_file, datasetName='height')[1]

        # compute latitude and longitude min max
        lon_min = float(atr['X_FIRST']) - float(atr['X_STEP'])
        lon_max = float(atr['X_FIRST']) + float(atr['X_STEP']) * (int(atr['WIDTH'])+1)
        lat_max = float(atr['Y_FIRST']) - float(atr['Y_STEP'])
        lat_min = float(atr['Y_FIRST']) + float(atr['Y_STEP']) * (int(atr['LENGTH'])+1)
        #print('Dimension of the dataset (length, width): {}, {}'.format(atr['LENGTH'], atr['WIDTH']))
        #print('S N W E: {} {} {} {}'.format(lat_min, lat_max, lon_min, lon_max))

        # do gdalwarp on the orignal DEM and output it
        cmd  = f"gdalwarp {dem_orig} {dem_out} -te {lon_min} {lat_min} {lon_max} {lat_max} "
        cmd += f"-ts {atr['WIDTH']} {atr['LENGTH']} -of ISCE\n"
        cmd += f'fixImageXml.py -i {dem_out} -f\n\n'
        self.f.write(cmd)


    def write_modify_network(self, file='ifgramStack.h5'):
        """ Write command line: `modify_network.py`
        """
        if file.startswith('ion'): template = self.templateIon
        else: template = self.template
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
            cmd += f'mkdir -p {self.picdir2}\n'
            cmd += f'mv *.pdf *.png {self.picdir2}\n\n'
            self.f.write(cmd)


    def write_ifgram_inversion(self, stack, mask, weight='var'):
        cmd  = f'ifgram_inversion.py {os.path.join(self.indir, stack)} -m {mask} -w {weight} '
        cmd += f'--cluster {self.cluster} --num-worker {self.numWorker} --ram {self.ram} --update\n\n'
        self.f.write(cmd)


    def write_generate_mask(self, threshold=None, ctype='temporal'):
        if not threshold:
            threshold = self.jdic.get('tcoh_threshold', 0.90)
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


    def write_ts2velo(self, tsfile, outfile):
        outfile = os.path.join(self.jdic['velo_dir'], outfile)
        cmd  = f'timeseries2velocity.py {tsfile} {self.time_func} -o {outfile} '
        cmd += f'--ref-lalo {self.refla} {self.reflo} --ref-date {self.ref_date} --update\n\n'
        self.f.write(cmd)


    def write_plate_motion(self, vfile=None):
        self.platename = self.jdic['platename']
        cmd   = f'plate_motion.py --geom {self.geom_file} --plate {self.platename} --velo {vfile}\n\n'
        self.f.write(cmd)


    def write_plot_velo(self, file, dataset, vlim=[None,None], title=None, outfile=None, mask=None, update=True):
        picdir = self.jdic['velo_pic']
        base = os.path.basename(file).split('.')[0]
        if not title:   title   = str(dataset)
        if not outfile: outfile = os.path.join(picdir, base+'.png')

        #dem_file = self.geom_file
        dem_file   = os.path.join(self.indir, 'srtm.dem')
        shade_exag = self.jdic.get('shade_exag', 0.02)
        shade_min  = self.jdic.get('shade_min', -6000)
        shade_max  = self.jdic.get('shade_max',  4000)
        alpha      = self.jdic.get('velo_alpha', 0.6)
        cmap       = self.jdic.get('velo_cmap', 'RdYlBu_r')
        unit       = self.jdic.get('velo_unit', 'mm')
        dpi        = self.jdic.get('dpi',       400)
        xmin       = self.jdic.get('lon_min',   None)
        xmax       = self.jdic.get('lon_max',   None)
        ymin       = self.jdic.get('lat_min',   None)
        ymax       = self.jdic.get('lat_max',   None)

        if not mask:
            try:
                mask = jobj.velo_msk
                if   mask in ['water','auto']:  mask_file = self.water_mask          # e.g., waterMask.h5
                elif mask in ['coh']:           mask_file = self.coh_mask            # e.g., maskTempCoh.h5
                elif mask in ['connComp']:      mask_file = self.conn_mask           # e.g., maskConnComp.h5
                elif mask in ['custom']:        mask_file = self.jdic['cust_mask']   # e.g., maskCustom.h5
                else:                           mask_file = 'no'
            except:
                mask_file = 'no'

        cmd  = f"view.py {file} {dataset} -c {cmap} "
        cmd += f"--dem {dem_file} --alpha {alpha} "
        cmd += f"--dem-nocontour --shade-exag {shade_exag} --shade-min {shade_min} --shade-max {shade_max} "
        cmd += f"--mask {mask_file} --unit {unit} --ref-lalo {self.refla} {self.reflo} "
        cmd += f"--vlim {vlim[0]} {vlim[1]} "
        cmd += f"-o {outfile} --figtitle {title} "
        cmd += f"--nodisplay --dpi {dpi} "
        if xmin is not None: cmd += f'--sub-lon {xmin} {xmax} '
        if ymin is not None: cmd += f'--sub-lat {ymin} {ymax} '
        if update: cmd += '--update '
        cmd += '\n\n'
        self.f.write(cmd)


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

    proc.f.close()


    ########## Network modifications and plots ##############
    proc.f = open('run_1_network', 'w')

    proc.write_smallbaselineApp(dostep='modify_network')
    proc.write_modify_network(file='ionStack.h5')
    proc.write_smallbaselineApp(dostep='reference_point')
    proc.write_reference_point(files=['inputs/ionStack.h5'])
    proc.write_smallbaselineApp(dostep='quick_overview')
    proc.write_plot_network(stacks=['ifgramStack.h5','ionStack.h5'], cmap_vlist=[0.2, 0.7, 1.0])
    proc.f.write('bash plot_smallbaselineApp.sh\n\n')

    proc.f.close()


    ################### Network inversion ##################
    proc.f = open('run_2_inversion', 'w')

    proc.write_smallbaselineApp(dostep='correct_unwrap_error')
    proc.write_smallbaselineApp(dostep='invert_network')
    proc.write_ifgram_inversion('ionStack.h5', mask='waterMask.h5', weight='no')
    proc.write_generate_mask()

    proc.f.close()


    ################ Apply corrections ####################
    proc.f = open('run_3_corrections', 'w')

    proc.write_smallbaselineApp(dostep='correct_LOD')
    proc.write_smallbaselineApp(dostep='correct_SET')
    proc.write_smallbaselineApp(dostep='correct_troposphere')
    proc.f.write('diff.py timeseries_SET_ERA5.h5 timeseriesIon.h5 -o timeseries_SET_ERA5_Ion.h5 --force\n\n')

    #proc.write_smallbaselineApp(dostep='correct_topography')
    proc.write_demErr('timeseries_SET_ERA5_Ion.h5', 'timeseries_SET_ERA5_Ion_demErr.h5')
    proc.write_smallbaselineApp(dostep='residual_RMS')
    proc.write_smallbaselineApp(dostep='deramp')

    proc.f.close()


    ################ Velocity estimation ###################
    proc.f = open('run_4_velocity', 'w')

    tsfiles = ['timeseries_SET_ERA5_Ion_demErr.h5', 'inputs/ERA5.h5', 'inputs/SET.h5', 'timeseriesIon.h5']
    outfiles = ['velocity.h5', 'velocityERA5.h5', 'velocitySET.h5', 'velocityIon.h5']
    for (tsfile, outfile) in zip(tsfiles, outfiles):
        proc.write_ts2velo(tsfile, outfile)

    proc.write_plate_motion(vfile=os.path.join(proc.jdic['velo_dir'], 'velocity.h5'))

    proc.f.close()


    ################## Plot Velocity #######################
    proc.f = open('run_5_velocityPlot', 'w')

    outdir = proc.jdic['velo_pic']
    veldir = proc.jdic['velo_dir']
    proc.f.write(f'mkdir -p {outdir}\n\n')

    suffs = ['','ERA5','SET','Ion','_ITRF14']
    vlims = [proc.jdic['vm_mid'],
            proc.jdic['vm_ERA'],
            proc.jdic['vm_SET'],
            proc.jdic['vm_mid'],
            proc.jdic['vm_mid']]
    dset = 'velocity'
    for suff, vlim in zip(suffs, vlims):
        infile  = os.path.join(veldir, 'velocity'+suff+'.h5')
        outfile = os.path.join(outdir, dset+suff+'.png')
        title   = dset+suff
        proc.write_plot_velo(infile, dset, vlim, title, outfile)

    suffs = ['','ERA5','SET','Ion']
    dset = 'velocityStd'
    for suff in suffs:
        infile  = os.path.join(veldir, 'velocity'+suff+'.h5')
        outfile = os.path.join(outdir, dset+suff+'.png')
        title   = dset+suff
        proc.write_plot_velo(infile, dset, proc.jdic['vm_STD'], title, outfile)

    proc.f.close()


#############################################################################################################
#############################################################################################################

if __name__ == '__main__':

    # get user inputs
    inps = cmdLineParse()

    # initialize the process
    proc = SBApp(inps.json_file, inps.proc_home)

    # run it
    if inps.action == 'all':
        main(proc, inps)
    elif inps.action == 'dem_resamp':
        proc.f = open('run_0_prep', 'a') # append after the runfile
        proc.write_resamp_dem()
        proc.f.close()

    print('Finish writing the run files. Go ahead and run them sequentially.')
