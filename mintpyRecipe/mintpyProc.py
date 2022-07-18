#!/usr/bin/env python3
############################################################
# This code is a recipe for MintPy processing              #
# The better way will be to generates mintpy cmd run files #
# for each stage and to run in bash                        #
# Author: Yuan-Kai Liu, 2022                               #
############################################################


import os
import sys
import json
import glob
import shutil
import argparse
import numpy as np
from datetime import datetime
from types import SimpleNamespace

# suppress DEBUG message for import mintpy.workflow
import logging
modules = ['matplotlib', 'fiona', 'asyncio', 'shapely', 'gdal']
for mod in modules:
    logging.getLogger(mod).setLevel(logging.WARNING)

# gdal
from osgeo import gdal

# isce
import isce
from applications.gdal2isce_xml import gdal2isce_xml
from isceobj.Alos2Proc.Alos2ProcPublic import waterBodyRadar

# mintpy
import mintpy
import mintpy.workflow
from mintpy import smallbaselineApp, bulk_plate_motion
from mintpy.objects import datasetUnitDict
from mintpy.utils import readfile, writefile


STEPS =     {   'load'          : ['load', 'loading', 'load_data'],
                'network'       : ['network', 'modify_network'],
                'inversion'     : ['inversion', 'invert', 'network_inversion'],
                'corrections'   : ['corrections'],
                'topographic'   : ['demErr', 'topo', 'topographic', 'topographic_residual'],
                'residual'      : ['residual', 'residual_RMS'],
                'deramp'        : ['deramp', 'deramping'],
                'ts2velo'       : ['ts2velo', 'timeseries2velocity'],
                'PMM'           : ['bulk_plate_motion', 'plate_motion', 'PMM'],
            }

STEPS_SUPP= {   'plot_network'  : ['plot_network'],
                'plot_loaded'   : ['plot_loaded'],
                'plot_velo'     : ['plot_velo'],
            }

stepmsg = ''
for i, (key, value) in enumerate(STEPS.items()):
    stepmsg += '{}.  {:20s}{} \n'.format(i+1, key, str(value))
STEP_HELP = f"""Command line options for steps processing with names are chosen from the following list:

{stepmsg}

"""

stepmsg = ''
for i, (key, value) in enumerate(STEPS_SUPP.items()):
    stepmsg += '{}.  {:20s}{} \n'.format(i+1, key, str(value))
STEP_HELP_MORE = f"""Command line options for supplementary steps (e.g., plotting):

{stepmsg}

"""


def create_parser():
    description = ' This code is a recipe for MintPy processing '
    ## basic input/output files and paths
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(dest='json_file', type=str,
            help='custom json with specifications of running MintPym, e.g., params.json')
    parser.add_argument('-i', '--indir', dest='inDir', type=str, default='./inputs',
            help = 'Inputs directory containing geometryRadar.h5, ifgramStack.h5.')
    step = parser.add_argument_group('Steps of processing (dosteps)', STEP_HELP+STEP_HELP_MORE)
    step.add_argument('--do','--dosteps', dest='dosteps', type=str, nargs='+', default=[],
                      help='run processing at the named step(s)')
    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    if (inps.json_file) and (os.path.exists(inps.json_file) is False):
        print('Parameter JSON file {} does not exist'.format(inps.json_file))
        parser.print_help()
        sys.exit(1)
    if len(inps.dosteps) == 0:
        print('No steps specified: {}'.format(inps.dosteps))
        sys.exit(1)
    return inps



#################################################################
#################################################################


def get_template(jobj):
    default_cfg = list(set(glob.glob(jobj.template))-set(glob.glob('*_*.cfg')))
    try:
        jobj.template = default_cfg[0]
    except:
        try:
            jobj.template = glob.glob('./smallbaselineApp.cfg')[0]
        except:
            print('Error: No MintPy template file found!')
            sys.exit(1)
    jobj.template = os.path.basename(jobj.template)
    print('Use template for regular pairs: {}'.format(jobj.template))

    try:
        jobj.templateIon = glob.glob(jobj.templateIon)[0]
    except:
        try:
            jobj.templateIon = glob.glob(jobj.template)[0]
        except:
            try:
                jobj.templateIon = glob.glob('./smallbaselineApp.cfg')[0]
            except:
                print('Error: No MintPy template for ionosphere file found!')
                sys.exit(1)
    jobj.templateIon = os.path.basename(jobj.templateIon)
    print('Use template for ionospheric pairs: {}'.format(jobj.templateIon))

    for outdir in ['inputs', 'pic']:
        if not os.path.exists(outdir):  os.makedirs(outdir)
        for cfg in [jobj.template, jobj.templateIon]:
            print('copy {} to {}/{}'.format(cfg, outdir, cfg))
            shutil.copyfile(cfg, f'{outdir}/{cfg}')
    return jobj


def read_template_compute(template, key):
    iDict = readfile.read_template(template)
    cluster  = iDict['mintpy.compute.cluster']
    n_worker = iDict['mintpy.compute.numWorker']
    ram      = iDict['mintpy.compute.maxMemory']
    if cluster  == 'auto': cluster  = 'none'
    if n_worker == 'auto': n_worker = 4
    if ram      == 'auto': ram      = 4.0
    n_worker = int(n_worker)
    ram      = float(ram)
    if key == 'cluster' : return cluster
    if key == 'n_worker': return n_worker
    if key == 'ram'     : return ram


def run_resamp_wbd(jobj):
    # get paths, filenames
    iDict = readfile.read_template(jobj.template)
    geom_basedir = os.path.dirname(os.path.abspath(iDict['mintpy.load.demFile']))
    wbdFile      = os.path.abspath(jobj.wbd_orig)
    wbdOutFile   = f'{geom_basedir}/waterBody.rdr'

    if os.path.exists(wbdOutFile):
        print('waterBody exists: {}'.format(wbdOutFile))
        print('skip generating waterBody')
    else:
        print('waterBody does not exist')
        print('Working on resmapling waterBody from: {}'.format(geom_basedir))
        # do gdal_translate to re-generate lon.rdr.xml lat.rdr.xml
        latFile = f'{geom_basedir}/lat.rdr'
        lonFile = f'{geom_basedir}/lon.rdr'
        print('Generate ISCE xml file from gdal supported file')
        gdal2isce_xml(latFile+'.vrt')
        gdal2isce_xml(lonFile+'.vrt')
        print('Completed ISCE xml files')

        # resample the waterbody
        waterBodyRadar(latFile, lonFile, wbdFile, wbdOutFile)
        print('Resampled waterBody: {}'.format(wbdOutFile))
    os.system('fixImageXml.py -i {} -f '.format(wbdOutFile))
    return


def run_resamp_dem(jobj, inps):
    """
    Use GDAL to resample the orignal DEM to match the extent, dimension, and resolution of
    [This is optional], just to cover the full extent when using topsStack radar coord datsets
    (when geocode geometryRadar.h5 to geometryGeo.h5, the height will have large gaps; not pretty)
    Should be run after having the geometryGeo.h5 file (must in geo-coord to allow reading lon lat)
    The output DEM is then saved separetly (defined in `params.json` as "dem_out")
    The output DEM is mainly for plotting purposes using view.py
    """
    #proc_home = os.path.expanduser(jobj.proc_home)
    dem_out   = inps.inDir+'/srtm.dem'          # '{}'.format(jobj.dem_out)
    geo_file  = inps.inDir+'/geometryGeo.h5'    # '{}/{}'.format(proc_home, jobj.geom_file)
    dem_orig  = '{}'.format(jobj.dem_orig)

    # Check the directory
    outdir = os.path.dirname(dem_out)
    if not os.path.exists(outdir):  os.makedirs(outdir)

    # Read basic attributes from .h5
    atr = readfile.read(geo_file, datasetName='height')[1]

    # compute latitude and longitude min max
    lon_min = float(atr['X_FIRST']) - float(atr['X_STEP'])
    lon_max = float(atr['X_FIRST']) + float(atr['X_STEP']) * (int(atr['WIDTH'])+1)
    lat_max = float(atr['Y_FIRST']) - float(atr['Y_STEP'])
    lat_min = float(atr['Y_FIRST']) + float(atr['Y_STEP']) * (int(atr['LENGTH'])+1)
    print('Dimension of the dataset (length, width): {}, {}'.format(atr['LENGTH'], atr['WIDTH']))
    print('S N W E: {} {} {} {}'.format(lat_min, lat_max, lon_min, lon_max))

    # do gdalwarp on the orignal DEM and output it
    cmd = 'gdalwarp {} {} -te {} {} {} {} -ts {} {} -of ISCE '.format(
            dem_orig, dem_out, lon_min, lat_min, lon_max, lat_max, atr['WIDTH'], atr['LENGTH'])
    os.system(cmd)
    cmd = 'fixImageXml.py -i {} -f '.format(dem_out)
    os.system(cmd)
    return


def run_smallbaselineApp(template, dostep=None, start=None, end=None):
    iargs = [template, '--dostep', str(dostep)]
    smallbaselineApp.main(iargs=iargs)
    return


def run_load_data(template, project, processor, only_geom=False):
    iargs = ['-t', template, '--project', str(project), '--processor', processor]
    if only_geom:
        iargs += ['--geom']
    mintpy.load_data.main(iargs=iargs)
    return


def run_radar2geo_inputs(jobj, inps, lalo=None, ram=None):
    """
    Use MintPy geocode.py to do radar to geo coord conversion for:
        1. ifgramStack.h5
        2. ionStack.h5
        3. geometryRadar.h5
    """
    in_dir    = os.path.abspath(os.path.expanduser(inps.inDir))
    print('Go to the directory: {}'.format(in_dir))
    os.chdir(in_dir)
    geom_rdr  = os.path.expanduser('geometryRadar.h5')
    rdr_dir   = os.path.expanduser('./radar')

    # read xystep and ram from template
    iDict = readfile.read_template(jobj.template)
    inps = mintpy.geocode.read_template2inps(jobj.template, inps)
    if ram is None:  ram = inps.maxMemory
    if lalo is None: lalo = [float(n) for n in iDict['mintpy.geocode.laloStep'].replace(',', ' ').split()]

    # get the hdf5 files to be geocoded
    hdf5_list = glob.glob('*.h5')
    files = []
    for f in hdf5_list:
        if not os.path.basename(f).startswith('geo_'): files.append(f)

    # print message
    print('Files to be geocoded: ', *files)
    print('Latitude/longitude lookup table: ', geom_rdr)
    print('Latitude/longitude resolution: ', *lalo)

    # build iargs and command
    iargs = [*files, '-l', geom_rdr, '--lalo', *lalo, '--ram', ram, '--update']
    iargs = list(map(str,iargs))
    cmd   = 'geocode.py '+' '.join(iargs)
    print()
    print(cmd)

    # execute mintpy geocode
    mintpy.geocode.main(iargs=iargs)

    # manage files
    if not os.path.exists(rdr_dir):  os.makedirs(rdr_dir)
    files = glob.glob('geo_*.h5')
    for file in files:
        basefile = os.path.basename(file).split('geo_')[-1]
        os.replace(basefile, os.path.expanduser(rdr_dir+'/'+basefile))
        if basefile.startswith('geometry'):
            outfile = 'geometryGeo.h5'
        else:
            outfile = str(basefile)
        os.replace(file, outfile)

    print('Go back to the directory: {}'.format(jobj.proc_home))
    os.chdir(jobj.proc_home)
    return


def run_modify_network(template, file):
    iargs = [file, '-t', template]
    iargs = list(map(str,iargs))
    cmd   = 'modify_network.py '+' '.join(iargs)
    print()
    print(cmd)
    mintpy.modify_network.main(iargs=iargs)
    return


def run_reference_point(template, files, ref_lalo=None, ref_yx=None):
    if ref_lalo:
        opt = ['--lat', ref_lalo[0], '--lon', ref_lalo[1]]
    if ref_yx:
        opt = ['--row', ref_yx[0], '--col', ref_yx[1]]
    else: opt = []

    for file in files:
        iargs = [file, '-t', template] + opt
        iargs = list(map(str,iargs))
        cmd   = 'reference_point.py '+' '.join(iargs)
        print()
        print(cmd)
        mintpy.reference_point.main(iargs=iargs)
    return


def run_plot_network(template, ifgstacks, cmap_vlist=[0.2, 0.7, 1.0]):
    for file in ifgstacks:
        iargs = [file, '-t', template, '--nodisplay', '--cmap-vlist', *cmap_vlist]
        iargs = list(map(str,iargs))
        cmd   = 'plot_network.py '+' '.join(iargs)
        print()
        print(cmd)
        mintpy.plot_network.main(iargs=iargs)

        exts = ('*.pdf', '*.png')
        outdir = './pic_orig'
        files = []
        for ext in exts:
            files.extend(glob.glob(ext))
        if not os.path.exists(outdir):  os.makedirs(outdir)
        for f in files:
            shutil.copyfile(f, '{}/{}'.format(outdir, f))
    return


def run_plot_SBApp(script='./2_plot_SBApp.sh'):
    # This could be improved by calling the python plotting script in MintPy
    os.system('bash {}'.format(script))
    return


def run_ifgram_inversion(template, ifstack, mask, weight='var', cluster=None, n_worker=None, ram=None):
    if not cluster:  cluster  = read_template_compute(template, 'cluster')
    if not n_worker: n_worker = read_template_compute(template, 'n_worker')
    if not ram:      ram      = read_template_compute(template, 'ram')

    print('Multiprocessing: cluster: {}  num_workers: {}'.format(cluster, n_worker))
    iargs = [ifstack, '-m', mask, '-w', weight, '--cluster', cluster, '--num-worker', n_worker, '--ram', ram, '--update']
    iargs = list(map(str,iargs))
    cmd   = 'ifgram_inversion.py '+' '.join(iargs)
    print()
    print(cmd)
    mintpy.ifgram_inversion.main(iargs=iargs)


def run_generate_mask(jobj, coherence=None, threshold=None, outfile=None):
    proc_home = os.path.expanduser(jobj.proc_home)
    if not coherence:
        coherence = '{}/{}'.format(proc_home, jobj.tcoh_file)
    if not threshold:
        threshold =    '{}'.format(jobj.tcoh_threshold)
    if not outfile:
        outfile   = '{}/{}'.format(proc_home, jobj.tcoh_mask)
    iargs = [coherence, '-m', threshold, '-o', outfile, '--update']
    iargs = list(map(str,iargs))
    cmd   = 'generate_mask.py '+' '.join(iargs)
    print()
    print(cmd)
    mintpy.generate_mask.main(iargs=iargs)


def run_diff(file1, file2, outfile):
    iargs = [file1, file2, '-o', outfile, '--force']
    iargs = list(map(str,iargs))
    cmd   = 'diff.py '+' '.join(iargs)
    print()
    print(cmd)
    mintpy.diff.main(iargs=iargs)


def run_demErr(jobj, file, outfile, cluster=None, n_worker=None, ram=None, bak_dir=False):
    iDict     = readfile.read_template(jobj.template)
    proc_home = os.path.expanduser(jobj.proc_home)
    geom_file = os.path.expanduser(proc_home+'/'+jobj.geom_file)
    m_poly    = int(iDict['mintpy.timeFunc.polynomial'])
    m_peri    = [float(n) for n in iDict['mintpy.timeFunc.periodic'].replace(',', ' ').split()]
    velo_model = ['--poly-order', m_poly, '--periodic', *m_peri]

    if not cluster:  cluster  = read_template_compute(jobj.template, 'cluster')
    if not n_worker: n_worker = read_template_compute(jobj.template, 'n_worker')
    if not ram:      ram      = read_template_compute(jobj.template, 'ram')

    if bak_dir:
        rms_dir = os.path.expanduser(proc_home+'/'+jobj.rms_dir)
        ex_dir  = os.path.expanduser(rms_dir+'/ex/')
        if not os.path.exists(ex_dir): os.makedirs(ex_dir)

    opt = [*velo_model, '-g', geom_file, '--cluster', cluster, '--num-worker', n_worker, '--ram', ram, '--update']
    iargs = [file, '-o', outfile] + opt
    iargs = list(map(str,iargs))
    cmd   = 'dem_error.py '+' '.join(iargs)
    print()
    print(cmd)
    mintpy.dem_error.main(iargs=iargs)
    return


def run_timeseries_rms(jobj, file, maskfile=None, rms_cutoff=None, deramp='no', figsize=[10,5]):
    proc_home = os.path.expanduser(jobj.proc_home)
    if not maskfile:
        maskfile = '{}/{}'.format(proc_home, jobj.tcoh_mask)
    if not rms_cutoff:
        iDict = readfile.read_template(jobj.template)
        rms_cutoff = iDict['mintpy.residualRMS.cutoff']
        if rms_cutoff == 'auto':
            rms_cutoff = 3.0
        rms_cutoff = float(rms_cutoff)

    opt = ['-m', maskfile, '--cutoff', rms_cutoff, '--figsize', *figsize]
    iargs = [file, '-r', deramp] + opt
    iargs = list(map(str,iargs))
    cmd   = 'timeseries_rms.py '+' '.join(iargs)
    print()
    print(cmd)
    mintpy.timeseries_rms.main(iargs=iargs)
    return


def run_deramp(jobj, file, outfile, kind='quadratic', save_coeff=True):
    if not maskfile:
        proc_home = os.path.expanduser(jobj.proc_home)
        maskfile  = '{}/{}'.format(proc_home, jobj.tcoh_mask)

    iargs = [file, '-m', maskfile, '-s', kind, '-o', outfile, '--update']
    if save_coeff:
        iargs += ['--save-ramp-coeff']
    iargs = list(map(str,iargs))
    cmd   = 'remove_ramp.py '+' '.join(iargs)
    print()
    print(cmd)
    mintpy.remove_ramp.main(iargs=iargs)
    return


def run_timeseries2velocity(jobj, tsfile, outfile, velo_model=None):
    iDict = readfile.read_template(jobj.template)
    refla, reflo = [float(n) for n in iDict['mintpy.reference.lalo'].replace(',', ' ').split()]
    ref_date     = iDict['mintpy.reference.date']

    if not velo_model:
        m_poly     = int(iDict['mintpy.timeFunc.polynomial'])
        m_peri     = [float(n) for n in iDict['mintpy.timeFunc.periodic'].replace(',', ' ').split()]
        velo_model = ['--poly-order', m_poly, '--periodic', *m_peri]

    iargs = [tsfile, *velo_model, '-o', outfile, '--update']
    if all(elem != 'auto' for elem in [refla, reflo]):
        iargs += ['--ref-lalo', refla, reflo]
    if ref_date != 'auto':
        iargs += ['--ref-date', ref_date]

    iargs = list(map(str,iargs))
    cmd   = 'timeseries2velocity.py '+' '.join(iargs)
    print()
    print(cmd)
    mintpy.timeseries2velocity.main(iargs=iargs)
    return


def run_bulk_plate_motion(jobj, vfile=None):
    om_cart = jobj.euler_cartesian
    iargs = ['--geom', jobj.geom_file, '--om-cart', *om_cart, '--velo', vfile]
    iargs = list(map(str,iargs))
    cmd   = 'bulk_plate_motion.py '+' '.join(iargs)
    print()
    print(cmd)
    bulk_plate_motion.main(iargs=iargs)
    return


def run_plot_velo(jobj, file, dataset, vlim=[None,None], title=None, outfile=None, mask=None, update=False):
    picdir = os.path.dirname(outfile)
    base, ext = os.path.basename(file).split('.')
    if not os.path.exists(picdir):
        os.makedirs(picdir)

    if title is None:   title = str(dataset)
    if outfile is None: outfile = str(base+'.png')

    iDict = readfile.read_template(jobj.template)
    xmin     = jobj.lon_min
    xmax     = jobj.lon_max
    ymin     = jobj.lat_min
    ymax     = jobj.lat_max
    dem_file = jobj.geom_file   # 'inputs/geometryGeo.h5'
    dem_file = jobj.dem_out     # 'inputs/srtm.dem'
    refla, reflo = [float(n) for n in iDict['mintpy.reference.lalo'].replace(',', ' ').split()]

    if not mask:
        try:
            mask = jobj.velo_msk
            if   mask in ['water','auto']:  mask_file = jobj.water_mask  # e.g., waterMask.h5
            elif mask in ['tempCoh']:       mask_file = jobj.tcoh_mask   # e.g., maskTempCoh.h5
            elif mask in ['connComp']:      mask_file = jobj.conn_mask   # e.g., maskConnComp.h5
            elif mask in ['custom']:        mask_file = jobj.cust_mask   # e.g., maskCustom.h5
            else:                           mask_file = 'no'
        except:
            mask_file = 'no'

    iargs  = [file, dataset, '--nodisplay', '--dpi', 300, '-c', jobj.velo_cmap]
    iargs += ['--dem', dem_file, '--alpha', jobj.velo_alpha, '--dem-nocontour' ,'--shade-exag', jobj.shade_exag]
    iargs += ['--mask', mask_file, '--unit', 'mm', '--ref-lalo', refla, reflo]
    iargs += ['--vlim', vlim[0], vlim[1]]
    iargs += ['-o', outfile, '--figtitle', title]

    if all(elem != 'None' for elem in [xmin, xmax]):
        iargs += ['--sub-lon', xmin, xmax]
    if all(elem != 'None' for elem in [ymin, ymax]):
        iargs += ['--sub-lat', ymin, ymax]
    if update:
        iargs += ['--update']

    iargs = list(map(str,iargs))
    cmd = 'view.py '+' '.join(iargs)
    print()
    print(cmd)
    mintpy.view.main(iargs=iargs)
    return


#################################################################
#################################################################


def main(iargs=None):
    # Read parser arguments
    inps = cmd_line_parse(iargs)


    ## Begin timing
    t_start = datetime.now()
    t_string = t_start.strftime("%Y-%m-%d %H:%M:%S")
    print('\n\n###################################################################')
    print('#### MintPy recipe | step: {} | Start'.format(inps.dosteps))
    print('#### Time: {}'.format(t_string))
    print('###################################################################\n')
    print('Run the follow stages in sequence: {}'.format(inps.dosteps))


    # Other parameters from the JSON file
    with open(inps.json_file) as f:
        jdic = json.load(f)
        jobj = SimpleNamespace(**jdic)
        jobj = get_template(jobj)
        jobj.proc_home = os.path.abspath(jobj.proc_home)
        jobj.proc_name = jobj.template.split('.')[0]
    print('Current directory: {}'.format(jobj.proc_home))
    os.chdir(jobj.proc_home)


    #------------------ 1. Load data and prepare stack ------------------
    if any(key in inps.dosteps for key in STEPS['load']):

        run_resamp_wbd(jobj)

        run_smallbaselineApp(jobj.template, dostep='load_data')
        #run_load_data(template=jobj.template, project=jobj.proc_name, processor='isce', only_geom=True)  # only read the geometry

        run_radar2geo_inputs(jobj, inps, ram=16)

        run_resamp_dem(jobj, inps)


    #------------------ 2. Network modification / quick review ------------------
    if any(key in inps.dosteps for key in STEPS['network']):

        run_smallbaselineApp(jobj.template, dostep='modify_network')

        run_modify_network(jobj.templateIon, file='inputs/ionStack.h5')

        run_smallbaselineApp(jobj.template, dostep='reference_point')

        run_reference_point(jobj.template, files=['inputs/ionStack.h5'], ref_lalo=None, ref_yx=None)

        run_smallbaselineApp(jobj.template, dostep='quick_overview')

        run_plot_network(jobj.template, ifgstacks=['inputs/ifgramStack.h5','inputs/ionStack.h5'], cmap_vlist=[0.2, 0.7, 1.0])


    #------------------ supp. just plot network and loaded data ----------
    if any(key in inps.dosteps for key in STEPS_SUPP['plot_network']):
        run_plot_network(jobj.template, ifgstacks=['inputs/ifgramStack.h5','inputs/ionStack.h5'], cmap_vlist=[0.2, 0.7, 1.0])

    if any(key in inps.dosteps for key in STEPS_SUPP['plot_loaded']):
        run_plot_SBApp(script='./2_plot_SBApp.sh')  # can be improved later


    #------------------ 3. Network inversion ----------------------------
    if any(key in inps.dosteps for key in STEPS['inversion']):

        run_smallbaselineApp(jobj.template, dostep='correct_unwrap_error')

        run_smallbaselineApp(jobj.template, dostep='invert_network')

        run_ifgram_inversion(jobj.template, 'inputs/ionStack.h5', mask='waterMask.h5', weight='no')

        run_generate_mask(jobj)


    #------------------ 4. Applying corrections -------------------------
    if any(key in inps.dosteps for key in STEPS['corrections']):

        run_smallbaselineApp(jobj.template, dostep='correct_LOD')

        run_smallbaselineApp(jobj.template, dostep='correct_SET')

        run_smallbaselineApp(jobj.template, dostep='correct_troposphere')

        run_diff(file1='timeseries_SET_ERA5.h5', file2='timeseriesIon.h5', outfile='timeseries_SET_ERA5_Ion.h5')


    #------------------ 5. Estimate topographic error -------------------
    if any(key in inps.dosteps for key in STEPS['topographic']):
        # choose one from below two:
        #run_smallbaselineApp(jobj.template, dostep='correct_topography')
        run_demErr(jobj, file='timeseries_SET_ERA5_Ion.h5', outfile='timeseries_SET_ERA5_Ion_demErr.h5')


    #------------- 6. Estimate RMS from other noise sources -------------
    if any(key in inps.dosteps for key in STEPS['residual']):
        # choose one from below two:
        run_smallbaselineApp(jobj.template, dostep='residual_RMS')
        #run_timeseries_rms(jobj, file='timeseriesResidual.h5', deramp='no')


    #----------------------- 7. Deramp ----------------------------------
    if any(key in inps.dosteps for key in STEPS['deramp']):
        # choose one from below two; or no need to deramp
        run_smallbaselineApp(jobj.template, dostep='deramp')
        #run_deramp(jobj, file='timeseries_SET_ERA5_Ion_demErr.h5', outfile='timeseries_SET_ERA5_Ion_demErr_rampl.h5', kind='linear')
        #run_deramp(jobj, file='timeseries_SET_ERA5_Ion_demErr.h5', outfile='timeseries_SET_ERA5_Ion_demErr_rampq.h5', kind='quadratic')


    #----------------------- 8. Time functions --------------------------
    if any(key in inps.dosteps for key in STEPS['ts2velo']):
        veloDir = jobj.velo_dir
        tsfiles = ['timeseries_SET_ERA5_Ion_demErr.h5', 'inputs/ERA5.h5', 'inputs/SET.h5', 'timeseriesIon.h5']
        outfiles = ['velocity.h5', 'velocityERA5.h5', 'velocitySET.h5', 'velocityIon.h5']
        for i, (tsfile, outfile) in enumerate(zip(tsfiles, outfiles)):
            run_timeseries2velocity(jobj, tsfile, f'{veloDir}/{outfile}')


    #----------------------- 9. Plate motion correction -----------------
    if any(key in inps.dosteps for key in STEPS['PMM']):
        run_bulk_plate_motion(jobj, vfile='{}/{}'.format(veloDir, 'velocity.h5'))


    #----------------------- supp. Plot velocity fields ---------------------
    if any(key in inps.dosteps for key in STEPS_SUPP['plot_velo']):
        outdir = jobj.velo_pic
        inf    = os.path.expanduser('{}/velocity'.format(jobj.velo_dir))

        dset = 'velocity'
        outf = outdir+dset
        for suff, vlim in zip(['','ERA5','SET','Ion','_ITRF14'], [jobj.vm_mid, jobj.vm_ERA, jobj.vm_SET, jobj.vm_mid, jobj.vm_mid]):
            run_plot_velo(jobj, inf+suff+'.h5', dset, vlim=vlim, title=dset, outfile=outf+suff+'.png')

        dset = 'velocityStd'
        outf = outdir+dset
        for suff in ['','ERA5','SET','Ion']:
            run_plot_velo(jobj, inf+suff+'.h5', dset, vlim=jobj.vm_STD, title=dset,  outfile=outf+suff+'.png')


    ## End of the workflow
    t_end = datetime.now()
    t_string = t_end.strftime("%Y-%m-%d %H:%M:%S")
    print('\n###################################################################')
    print('#### MintPy recipe | step: {} | Normal finish.'.format(inps.dosteps))
    print('#### Time: {}'.format(t_string))
    print('#### Elapsed time: {}'.format(t_end-t_start))
    print('###################################################################\n\n')


#################################################################

if __name__ == '__main__':
    main(sys.argv[1:])
