#!/usr/bin/env python3
"""Load an ISCE-2 topsStack into GEOCODED MintPy inputs, reading each file once.

Replaces `load_data.py` (radar coordinates) + `geocode.py` + moving files to
inputs/radar/. Every interferogram / ionosphere epoch is read from the ISCE
files and resampled on the fly, so no large radar-coordinate stack is written.
Reading, metadata and resampling reuse MintPy's own objects
(load_data.read_inps_dict2*_object, objects.resample), so the output equals the
standard two-step product.

Datasets (--only, default all that the cfg defines):
    geom   inputs/geometryGeo.h5, plus inputs/radar/geometryRadar.h5 (30-40 MB,
           the lookup table; kept)
    ifg    inputs/ifgramStack.h5
    ion    inputs/ion.h5            <- mintpy.load.ionFile          or --ion-dir
    ramp   inputs/ionBurstRamp.h5   <- mintpy.load.ionBurstRampFile or --ramp-dir
           (loaded for optional manual use; the run files do not apply it)
Without geom, the existing inputs/radar/geometryRadar.h5 is the lookup table.

Examples:
    load_stack.py ChileSenAT076.cfg                                  # everything
    load_stack.py ChileSenAT076.cfg --only ion ramp --suffix hpc0926 # -> ion_hpc0926.h5, ionBurstRamp_hpc0926.h5
    load_stack.py ChileSenAT018.cfg --only ion --ion-dir ../hpc_topsStack/ion_dates_irlsE --suffix irlsE
"""
import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np
from mintpy import load_data as ld
from mintpy.cli.load_data import cmd_line_parse
from mintpy.objects.resample import resample
from mintpy.objects.stackDict import IFGRAM_DSET_NAMES, TIMESERIES_DSET_NAMES
from mintpy.utils import attribute as attr, readfile

_OBJ = None   # per-worker handle on the dict object (set by the pool initializer)


def _init(obj):
    global _OBJ
    _OBJ = obj


ION_DIRS = ('ion_dates', 'ion_burst_ramp_merged_dates')


def _read(args):
    key, ds_name, box, xstep, ystep, mli, nodata, resize2shape = args
    item = _OBJ.pairsDict[key] if hasattr(_OBJ, 'pairsDict') else _OBJ.datesDict[key]
    data = item.read(ds_name, box=box, xstep=xstep, ystep=ystep, mli_method=mli,
                     no_data_values=nodata, resize2shape=resize2shape)[0]
    # GOTCHA: mintpy.objects.stackDict.timeseriesDict.read picks the phase->range sign
    # from the FOLDER NAME: +lambda/4pi only for ion_dates/ and ion_burst_ramp_merged_dates/,
    # -lambda/4pi otherwise. Ionosphere epochs from any other folder (ion_dates_new/, ...)
    # would come out sign-flipped. Force the ionosphere convention.
    if hasattr(_OBJ, 'datesDict') and getattr(item, 'data_unit', None) == 'radian':
        folder = os.path.basename(os.path.dirname(item.datasetDict[ds_name]))
        if folder not in ION_DIRS:
            data *= -1
    return data


def geocode_stack(obj, keys, ds_names, f, res, iDict, nprocs, batch, resize2shape=None):
    """Read `ds_names` for all `keys` of a stack/timeseries dict object, resample, write into h5 f."""
    n = len(keys)
    for ds_name in ds_names:
        dtype = np.int16 if ds_name == 'connectComponent' else np.float32
        comp = 'lzf' if ds_name == 'connectComponent' else None
        mli = 'nearest' if ds_name == 'connectComponent' else iDict['method']
        ds = f.create_dataset(ds_name, shape=(n, res.length, res.width), dtype=dtype,
                              maxshape=(None, res.length, res.width), chunks=True, compression=comp)
        t0 = time.time()
        args = [(k, ds_name, iDict['box'], iDict['xstep'], iDict['ystep'], mli,
                 iDict['noDataVal'], resize2shape) for k in keys]
        with Pool(nprocs, initializer=_init, initargs=(obj,)) as pool:
            for b0 in range(0, n, batch):
                b1 = min(n, b0 + batch)
                data = np.stack(pool.map(_read, args[b0:b1]))
                for i in range(res.num_box):
                    sb, db = res.src_box_list[i], res.dest_box_list[i]
                    out = res.run_resample(src_data=data[:, sb[1]:sb[3], sb[0]:sb[2]], box_ind=i, print_msg=False)
                    ds[b0:b1, db[1]:db[3], db[0]:db[2]] = out
                print(f'  {ds_name}: {b1}/{n}  {time.time() - t0:.0f} s', flush=True)
        ds.attrs['MODIFICATION_TIME'] = str(time.time())


def geo_meta(meta, extra, iDict, res, file_type, resize2shape=None):
    meta = dict(meta)
    meta.update(extra)
    if resize2shape:
        meta = attr.update_attribute4resize(meta, resize2shape)
    if iDict['box']:
        meta = attr.update_attribute4subset(meta, iDict['box'])
    if iDict['xstep'] * iDict['ystep'] > 1:
        meta = attr.update_attribute4multilook(meta, iDict['ystep'], iDict['xstep'])
    meta['FILE_TYPE'] = file_type
    return attr.update_attribute4radar2geo(meta, res_obj=res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('template')
    ap.add_argument('--outdir', default='inputs')
    ap.add_argument('--nprocs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=16, help='pairs resampled per block')
    ap.add_argument('--only', nargs='+', choices=['geom', 'ifg', 'ion', 'ramp'],
                    default=['geom', 'ifg', 'ion', 'ramp'], help='datasets to load (default: all)')
    ap.add_argument('--ion-dir', help='directory of *.ion to load instead of mintpy.load.ionFile')
    ap.add_argument('--ramp-dir', help='directory of *.float to load instead of mintpy.load.ionBurstRampFile')
    ap.add_argument('--suffix', default='', help='output name suffix for ion/ramp, e.g. hpc0926 -> ion_hpc0926.h5')
    ap.add_argument('--num-box', default='auto',
                    help='resampling boxes. auto: the count geocode.py used for this track\'s radar '
                         'ifgramStack (ceil(n*L*W*24 B / maxMemory)) if inputs/radar/ifgramStack.h5 exists, '
                         'so new ion files map pixels exactly like the existing stack; else 1')
    ap.add_argument('--force', action='store_true', help='overwrite existing outputs')
    a = ap.parse_args()
    t_start = time.time()

    # MintPy's default smallbaselineApp.cfg first, then the custom template (later overrides earlier)
    import mintpy
    default_cfg = os.path.join(os.path.dirname(mintpy.__file__), 'defaults', 'smallbaselineApp.cfg')
    inps = cmd_line_parse(['-t', default_cfg, a.template, '-l', 'ifg', 'geom', 'ion'])
    iDict = ld.read_inps2dict(inps)
    if a.ion_dir:
        iDict['mintpy.load.ionFile'] = os.path.join(a.ion_dir, '*.ion')
    if a.ramp_dir:
        iDict['mintpy.load.ionBurstRampFile'] = os.path.join(a.ramp_dir, '*.float')
    ld.prepare_metadata(iDict)
    extra = ld.get_extra_metadata(iDict)
    iDict = ld.read_subset_box(iDict)
    if a.ion_dir:
        iDict['mintpy.load.ionFile'] = os.path.join(a.ion_dir, '*.ion')
    if a.ramp_dir:
        iDict['mintpy.load.ionBurstRampFile'] = os.path.join(a.ramp_dir, '*.float')
    iDict.update(load_geom=True, load_ifg='ifg' in a.only, load_ion=bool({'ion', 'ramp'} & set(a.only)))
    tmpl = readfile.read_template(a.template)
    lalo = [float(x) for x in tmpl['mintpy.geocode.laloStep'].replace(',', ' ').split()]

    out = os.path.abspath(a.outdir); rdr = os.path.join(out, 'radar')
    os.makedirs(rdr, exist_ok=True)
    sfx = f'_{a.suffix}' if a.suffix else ''
    names = {'geom': 'geometryGeo.h5', 'ifg': 'ifgramStack.h5', 'ion': f'ion{sfx}.h5', 'ramp': f'ionBurstRamp{sfx}.h5'}
    targets = [os.path.join(out, names[k]) for k in a.only]
    if not a.force and any(os.path.exists(t) for t in targets):
        sys.exit(f'outputs exist in {out}; use --force to overwrite')

    # 1. geometry: radar (small, = lookup table) -> geocode.py -> geometryGeo.h5
    geom_rdr = os.path.join(rdr, 'geometryRadar.h5')
    _, geom_radar_obj = ld.read_inps_dict2geometry_dict_object(
        iDict, {**ld.GEOM_DSET_NAME2TEMPLATE_KEY, **ld.IFG_DSET_NAME2TEMPLATE_KEY, **ld.OFF_DSET_NAME2TEMPLATE_KEY})
    if 'geom' in a.only:
        geom_radar_obj.write2hdf5(outputFile=geom_rdr, access_mode='w', box=iDict['box'], xstep=iDict['xstep'],
                                  ystep=iDict['ystep'], compression='lzf', extra_metadata=extra)
        lalo_s = [f'{x:.9f}' for x in lalo]
        subprocess.run(['geocode.py', geom_rdr, '-l', geom_rdr, '--lalo', *lalo_s,
                        '-o', os.path.join(out, 'geometryGeo.h5')], check=True)
    elif not os.path.isfile(geom_rdr):
        sys.exit(f'{geom_rdr} not found: load geom first (it is the lookup table)')
    else:
        print(f'use existing lookup table {geom_rdr}')

    # resample object, identical to what geocode.py builds
    # nprocs=1: pyresample's parallel neighbour query picks different nearest neighbours on
    # ~0.7 % of pixels (not reproducible); reading stays parallel (--nprocs).
    ram = float(tmpl.get('mintpy.compute.maxMemory', 16))
    res = resample(lut_file=geom_rdr, src_file=geom_rdr, lalo_step=lalo, interp_method='nearest',
                   fill_value=np.nan, nprocs=1, max_memory=ram, software='pyresample', print_msg=True)
    res.open()
    if a.num_box == 'auto':
        legacy = os.path.join(rdr, 'ifgramStack.h5')
        if os.path.isfile(legacy):
            with h5py.File(legacy, 'r') as f:
                n, L, W = f['unwrapPhase'].shape
            res.num_box = int(np.ceil(n * L * W * 4 * 6 / (ram * 1024**3)))
            print(f'num_box auto: {res.num_box}, as geocode.py for {legacy}')
        else:
            res.num_box = 1
    else:
        res.num_box = int(a.num_box)
    res.prepare()
    print(f'resampling in {res.num_box} box(es)')

    # 2. interferogram stack
    stack = ld.read_inps_dict2ifgram_stack_dict_object(iDict, ld.IFG_DSET_NAME2TEMPLATE_KEY) if 'ifg' in a.only else None
    if stack is not None:
        write_ifg(stack, out, res, iDict, extra, a)

    # 3. ionosphere and ionosphere burst ramp (timeseries)
    for tag, key in (('ion', 'mintpy.load.ionFile'), ('ramp', 'mintpy.load.ionBurstRampFile')):
        if tag in a.only and iDict.get(key, 'auto') not in ('auto', 'no', None):
            write_ts(iDict, key, os.path.join(out, names[tag]), res, extra, geom_radar_obj, a)

    m, s = divmod(time.time() - t_start, 60)
    print(f'load_stack done in {m:.0f} min {s:.0f} s')


def write_ifg(stack, out, res, iDict, extra, a):
    pairs = sorted(stack.pairsDict.keys())
    ds_names = [d for d in IFGRAM_DSET_NAMES if d in stack.pairsDict[pairs[0]].datasetDict]
    print(f'ifgramStack: {len(pairs)} pairs, datasets {ds_names}')
    with h5py.File(os.path.join(out, 'ifgramStack.h5'), 'w') as f:
        geocode_stack(stack, pairs, ds_names, f, res, iDict, a.nprocs, a.batch)
        f.create_dataset('date', data=np.array(pairs, dtype=np.bytes_))
        stack.pairs = pairs
        f.create_dataset('bperp', data=np.array(
            [stack.pairsDict[p].get_perp_baseline(family=stack.dsName0) for p in pairs], dtype=np.float32))
        f.create_dataset('dropIfgram', data=np.ones(len(pairs), dtype=np.bool_))
        for k, v in geo_meta(stack.get_metadata(), extra, iDict, res, 'ifgramStack').items():
            f.attrs[k] = v



def write_ts(iDict, key, fname, res, extra, geom_radar_obj, a):
    """ionosphere-type timeseries (*.ion, *.float) -> geocoded timeseries file"""
    ts = ld.read_inps_dict2timeseries_dict_object(iDict, {TIMESERIES_DSET_NAMES[0]: key})
    dates = sorted(ts.get_date_list())
    resize2shape = None
    if ts.get_size()[1:] != geom_radar_obj.get_size():
        print(f'{fname}: lower resolution {ts.get_size()[1:]} -> resize to {geom_radar_obj.get_size()}')
        resize2shape = geom_radar_obj.get_size()
    print(f'{os.path.basename(fname)}: {len(dates)} epochs from {iDict[key]}')
    with h5py.File(fname, 'w') as f:
        geocode_stack(ts, dates, [TIMESERIES_DSET_NAMES[0]], f, res, iDict, a.nprocs, a.batch, resize2shape)
        f.create_dataset('date', data=np.array(dates, dtype=np.bytes_))
        for k, v in geo_meta(ts.get_metadata(), extra, iDict, res, 'timeseries', resize2shape).items():
            f.attrs[k] = v


if __name__ == '__main__':
    main()
