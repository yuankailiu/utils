#!/usr/bin/env python3

############################################################
# This code it meant to examine/plot the products from MintPy
# Yuan-Kai Liu @ 2022.06.22
############################################################

import numpy as np

import os
import sys
import json
import glob
import argparse
from types import SimpleNamespace

# gdal
from osgeo import gdal, osr

# isce
import isce
from applications.gdal2isce_xml import gdal2isce_xml
from isceobj.Alos2Proc.Alos2ProcPublic import waterBodyRadar

# mintpy
import mintpy
import mintpy.workflow
from mintpy import smallbaselineApp
from mintpy.objects import datasetUnitDict
from mintpy.utils import readfile, writefile


def create_parser():
    description = ' Use MintPy geocode.py to do radar2geo coord conversion for ifgramStack.h5, ionStack.h5, geometryRadar.h5 '
    ## basic input/output files and paths
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(dest='json_file', type=str,
            help='Custom json with specifications of running MintPym, e.g., params.json')
    parser.add_argument('-i', '--indir', dest='inDir', type=str, default='./inputs',
            help = 'Inputs directory containing geometryRadar.h5, ifgramStack.h5.')
    parser.add_argument('-t', '--temp', dest='template_file', type=str, default='./smallbaselineApp.cfg',
            help = 'Default template file')
    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    if (inps.json_file) and (os.path.exists(inps.json_file) is False):
        print('Parameter JSON file {} does not exist'.format(inps.json_file))
        parser.print_help()
        sys.exit(1)
    return inps


def array2raster(array, rasterName, rasterFormat, rasterOrigin, xStep, yStep, width, length):
    """
    tested, don't use it:
    gdalwarp cmd cannot generate the files, but it report no errors...
    """
    rasterName_tmp = rasterName+'.tmp'
    # transform info
    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]
    transform = (originX, xStep, 0, originY, 0, yStep)

    # write
    driver = gdal.GetDriverByName(rasterFormat)
    print('initiate GDAL driver: {}'.format(driver.LongName))

    print('create raster band')
    print('raster row / column number: {}, {}'.format(rows, cols))
    print('raster transform info: {}'.format(transform))
    outRaster = driver.Create(rasterName_tmp, cols, rows, 1, gdal.GDT_Float64)
    #outRaster.SetGeoTransform(transform)

    print('write data to raster band')
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    print('finished writing to {}'.format(rasterName_tmp))

    cmd = 'gdalwarp {} {} -ts {} {} -of ISCE -to SRC_METHOD=NO_GEOTRANSFORM -to DST_METHOD=NO_GEOTRANSFORM -ot Float64 '.format(rasterName_tmp, rasterName, str(width), str(length))
    print(cmd)
    #os.system(cmd)
    #print('finished writing to {}'.format(rasterName))
    #for file in glob.glob(os.path.dirname(rasterName_tmp)+'/*tmp*'):
    #    print('deleting {}'.format(os.path.abspath(file)))
    #    os.remove(os.path.abspath(file))
    #os.system('fixImageXml.py -i {} -f '.format(rasterName))
    return


def convert_geom_lalo():
    """
    use to call array2raster() to operate on the geometry file
    """
    for i, (dname, key) in enumerate(zip(['latitude','longitude'], ['lat','lon'])):
        array, attr  = readfile.read('inputs/radar/geometryRadar.h5', datasetName=dname)
        rasterName   = os.path.abspath('inputs/radar/{}.rdr'.format(key))
        rasterFormat = 'ISCE'
        rasterOrigin = [0,0]
        xStep  = 1
        yStep  = 1
        width  = 3436
        length = 11653
        array2raster(array, rasterName, rasterFormat, rasterOrigin, xStep, yStep, width, length)


def copy_add_dset2h5(file1, file2, copy_dset, outfile):
    """
    Copy a dataset from file1 and add it to file2
    Will save to a new outfile, using the attr of file2
    """
    # search dset from file1
    dsets1 = readfile.get_dataset_list(file1)
    for d in dsets1:
        if d == copy_dset:
            print('copy {} from {} to {}'.format(d, file1, file2))
            copy  = readfile.read(file1, datasetName=d)[0]

    # get all dsets from file2
    dsets2  = readfile.get_dataset_list(file2)
    outdict = dict()
    for d in dsets2:
        outdict[d] = readfile.read(file2, datasetName=d)[0]
        length, width = outdict[d].shape

    # add copy dset to outdict
    outdict[copy_dset] = copy

    # metadata and the units
    attr1 = readfile.read_attribute(file1)
    attr2 = readfile.read_attribute(file2)
    if attr1 != attr2: print('Attribute from the two files are different, use file2 attributes for output.')
    out_attr = attr2
    ds_name_dict0 = {
        "azimuthAngle"       : [np.float32, (length, width), None],
        "height"             : [np.float32, (length, width), None],
        "incidenceAngle"     : [np.float32, (length, width), None],
        "slantRangeDistance" : [np.float32, (length, width), None],
        "latitude"           : [np.float32, (length, width), None],
        "longitude"          : [np.float32, (length, width), None],
        "shadowMask"         : [bool,       (length, width), None],
        "waterMask"          : [bool,       (length, width), None],
        }
    unit_dict = dict()
    ds_name_dict = dict()
    for d in outdict.keys():
        if d in datasetUnitDict.keys():
            unit_dict[d] = datasetUnitDict[d]
        else:
            unit_dict[d] = '1'
        if d in ds_name_dict0:
            ds_name_dict[d] = ds_name_dict0[d]
        else:
            ds_name_dict[d] = [np.float32, (length, width), None]


    # instantiate the output file
    writefile.layout_hdf5(outfile, ds_name_dict=ds_name_dict, metadata=out_attr, ds_unit_dict=unit_dict)

    # output the datasets
    for key, value in outdict.items():
        print('writing {}  {}'.format(key, value.shape))
        writefile.write_hdf5_block(outfile, data=value, datasetName=key)
    return


def run_resamp_wbd(jobj, inps):
    # get paths, filenames
    iDict = readfile.read_template(inps.template_file)
    geom_basedir = os.path.dirname(os.path.abspath(iDict['mintpy.load.demFile']))
    wbdFile      = os.path.abspath(jobj.wbd_orig)
    wbdOutFile   = f'{geom_basedir}/waterBody.rdr'
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
    proc_home = os.path.abspath(jobj.proc_home)
    geom_rdr  = os.path.expanduser('geometryRadar.h5')
    rdr_dir   = os.path.expanduser('./radar')

    # read xystep and ram from template
    try:
        cfg_file = glob.glob(proc_home+'/'+jobj.config)[0]
    except:
        cfg_file = 'smallbaselineApp.cfg'
    if os.path.exists(cfg_file):
        inps = mintpy.geocode.read_template2inps(cfg_file, inps)
    else:
        print('Template file {} does not exist!'.format(cfg_file))
        sys.exit(1)
    if ram is None:  ram = inps.maxMemory

    # get xystep
    if lalo is None:  lalo = inps.lalo

    # get the hdf5 files to be geocoded
    globs = glob.glob('*.h5')
    files = []
    for f in globs:
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
    return


def run_ifgram_inversion(template, ifstack, mask, weight):
    iargs = [ifstack, '-m', mask, '-w', weight, '--update']
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


def run_demErr(jobj, file, outfile):
    proc_home = os.path.expanduser(jobj.proc_home)
    rms_dir   = os.path.expanduser(proc_home+'/'+jobj.rms2_dir)
    geom_file = os.path.expanduser(proc_home+'/'+jobj.geom_file)
    velo_model= jobj.velo_model.split()
    n_worker  = int(jobj.num_worker)

    ex_dir = os.path.expanduser(rms_dir+'/ex/')
    if not os.path.exists(ex_dir):
        os.makedirs(ex_dir)

    opt = [*velo_model, '-g', geom_file, '--num-worker', n_worker, '--update']
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
        rms_cutoff = float(jobj.rms_cutoff)

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
    if not velo_model:
        velo_model = jobj.velo_model.split()
    iargs = [tsfile, *velo_model, '-o', outfile, '--update']
    iargs = list(map(str,iargs))
    cmd   = 'timeseries2velocity.py '+' '.join(iargs)
    print()
    print(cmd)
    mintpy.timeseries2velocity.main(iargs=iargs)
    return


#################################################################

def main(iargs=None):
    # Read parser arguments
    inps = cmd_line_parse(iargs)

    # Other parameters from the JSON file
    with open(inps.json_file) as f:
        jdic = json.load(f)
        jobj = SimpleNamespace(**jdic)


    ## Beginning of the workflow

    #------------------ 0. prepare geom_reference waterBody -------------
    run_resamp_wbd(jobj, inps)

    #------------------ 1. Load data and prepare stack ------------------
    run_smallbaselineApp(jobj.config, dostep='load_data')

    run_radar2geo_inputs(jobj, inps, ram=16)

    run_resamp_dem(jobj, inps)

    run_smallbaselineApp(jobj.config, dostep='modify_network')

    run_modify_network(jobj.config, file='inputs/ionStack.h5')

    run_smallbaselineApp(jobj.config, dostep='reference_point')

    run_reference_point(jobj.config, files=['inputs/ionStack.h5'], ref_lalo=None, ref_yx=None)

    run_smallbaselineApp(jobj.config, dostep='quick_overview')

    run_plot_network(jobj.config, ifgstacks=['inputs/ifgramStack.h5','inputs/ionStack.h5'], cmap_vlist=[0.2, 0.7, 1.0])


    #------------------ 2. Network inversion ----------------------------
    run_smallbaselineApp(jobj.config, dostep='correct_unwrap_error')

    run_smallbaselineApp(jobj.config, dostep='invert_network')

    run_ifgram_inversion(jobj.config, 'inputs/ionStack.h5', mask='waterMask.h5', weight='no')

    run_generate_mask(jobj)


    #------------------ 3. Applying corrections -------------------------
    run_smallbaselineApp(jobj.config, dostep='correct_LOD')

    run_smallbaselineApp(jobj.config, dostep='correct_SET')

    run_smallbaselineApp(jobj.config, dostep='correct_troposphere')

    run_diff(file1='timeseries_SET_ERA5.h5', file2='timeseriesIon.h5', outfile='timeseries_SET_ERA5_Ion.h5')


    #------------------ 4. Estimate topographic error -------------------
    # choose one from below two
    run_smallbaselineApp(jobj.config, dostep='correct_topography')
    run_demErr(jobj, file='timeseries_SET_ERA5_Ion.h5', outfile='timeseries_SET_ERA5_Ion_demErr.h5')


    #------------- 5. Estimate RMS from other noise sources -------------
    # choose one from below two
    run_smallbaselineApp(jobj.config, dostep='residual_RMS')
    run_timeseries_rms(jobj, file='timeseriesResidual.h5', deramp='no')


    #----------------------- 6. Deramp ----------------------------------
    # choose one from below two ; or no need to deramp
    #run_smallbaselineApp(jobj.config, dostep='deramp')
    #run_deramp(jobj, file='timeseries_SET_ERA5_Ion_demErr.h5', outfile='timeseries_SET_ERA5_Ion_demErr_rampl.h5', kind='linear')
    #run_deramp(jobj, file='timeseries_SET_ERA5_Ion_demErr.h5', outfile='timeseries_SET_ERA5_Ion_demErr_rampq.h5', kind='quadratic')


    #----------------------- 7. Time functions --------------------------
    veloDir = './velocity_out'
    tsfiles = ['timeseries_SET_ERA5_Ion_demErr.h5', 'inputs/ERA5.h5', 'inputs/SET.h5', 'timeseriesIon.h5']
    outfiles = ['velocity.h5', 'velocityERA5.h5', 'velocitySET.h5', 'velocityIon.h5']
    for i, (tsfile, outfile) in enumerate(zip(tsfiles, outfiles)):
        run_timeseries2velocity(jobj, tsfile, f'{veloDir}/{outfile}')


    ## End of the workflow
    print('Normal finish.')


#################################################################
#if __name__ == '__main__':
#    main(sys.argv[1:])
