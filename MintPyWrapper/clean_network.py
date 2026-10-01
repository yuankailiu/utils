#!/usr/bin/env python3
############################################################
# Author: Yuan-Kai Liu, 2026                               #
############################################################
"""Drop interferograms that break triplet closure (integer unwrapping inconsistencies).

For every triplet (t1, t2, t3) of the current network (dropIfgram = True) and every pixel,
    C = phi12 + phi23 - phi13 (each referenced to REF_Y/REF_X),   K = floor((C + pi) / 2 pi)
K != 0 means one of the three pairs carries a different 2 pi ambiguity. Wrapped closure is ~0 for
any signal that is a function of date (deformation, coseismic steps, troposphere), so K != 0 is an
unwrapping error, not deformation.

    f(triplet) = share of mask pixels with K != 0 (pixels valid in all three pairs)
    score(pair) = mean f over the pair's network triplets

A bad pair corrupts every triplet it is in, a good pair only the few it shares with a bad one, so
the score isolates the bad pair. Pairs with score > --thr are dropped worst first, scores computed
once on the input network; a pair is kept if dropping it would disconnect the network or remove an
acquisition date.

Reference values (chile a018 full span, 2026-09-29/30, maskTempCoh pixels): curated pairs median
score 0.008 (p90 0.019), added long pairs p90 0.187; --thr 0.1 dropped 120 of 983 pairs and gave
Tratio > 0.05 in the mask 23.6 -> 3.6 %, a018-a120 overlap RMS 2.58 -> 2.35 mm/yr.
(The rule is the same as chile.qc.fullspan.clean, reimplemented standalone.)

Mask: --mask FILE (e.g. maskTempCoh.h5 of a previous inversion); default maskTempCoh.h5 if it exists,
else waterMask.h5 AND avgSpatialCoh.h5 >= --min-coh (before the first inversion). The mask is
combined with waterMask.h5 when that exists.

Writes dropIfgram in place (networks before / after in <stack>_dropIfgram_{before,after}_clean.txt,
summary in the stack attribute CLEAN_NETWORK), and a report <outdir>/clean_network.txt with every
pair's score. Re-running on a stack whose network is still the previous clean result restarts from the
saved pre-clean network, so the step is idempotent.
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np

TWO_PI = np.float32(2 * np.pi)


def cmd_line_parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stack', help='MintPy ifgramStack.h5 (modified in place: dropIfgram)')
    p.add_argument('--thr', type=float, default=0.1, help='score threshold (default: %(default)s)')
    p.add_argument('-d', '--dset', default='unwrapPhase', help='dataset to score (default: %(default)s)')
    p.add_argument('--mask', default=None, help='pixel mask (default: maskTempCoh.h5, else waterMask & avgSpatialCoh)')
    p.add_argument('--water-mask', default='waterMask.h5', help='land mask, 1 = land (default: %(default)s)')
    p.add_argument('--coh-file', default='avgSpatialCoh.h5', help='used when no --mask / maskTempCoh.h5 (default: %(default)s)')
    p.add_argument('--min-coh', type=float, default=0.7, help='threshold on --coh-file (default: %(default)s)')
    p.add_argument('--outdir', default='.', help='directory of the report (default: %(default)s)')
    p.add_argument('--dry-run', action='store_true', help='report only, do not change dropIfgram')
    return p.parse_args()


def read_mask(f):
    with h5py.File(f, 'r') as h:
        ds = 'mask' if 'mask' in h else ('waterMask' if 'waterMask' in h else list(h.keys())[0])
        return h[ds][:] > 0


def pixel_mask(inps, shape):
    if inps.mask:
        m, src = read_mask(inps.mask), inps.mask
    elif os.path.isfile('maskTempCoh.h5'):
        m, src = read_mask('maskTempCoh.h5'), 'maskTempCoh.h5'
    elif os.path.isfile(inps.coh_file):
        with h5py.File(inps.coh_file, 'r') as h:
            m = h[list(h.keys())[0]][:] >= inps.min_coh
        src = f'{inps.coh_file} >= {inps.min_coh}'
    else:
        m, src = np.ones(shape, bool), 'all pixels'
    if os.path.isfile(inps.water_mask):
        m &= read_mask(inps.water_mask); src += f' AND {inps.water_mask}'
    print(f'pixel mask: {src}  ({m.sum()} px, {m.mean() * 100:.1f} %)')
    return m, src


def triplets(date12):
    """Index triples (i12, i23, i13) of all triangles among date12 (MintPy's C matrix)."""
    from mintpy.objects import ifgramStack
    C = ifgramStack.get_design_matrix4triplet(list(date12))
    if C is None or len(C) == 0:
        return np.zeros((3, 0), int)
    i12 = np.empty(C.shape[0], int); i23 = i12.copy(); i13 = i12.copy()
    for k, row in enumerate(C):
        pp = np.flatnonzero(row == 1); nn = np.flatnonzero(row == -1)
        i12[k], i23[k] = pp[0], pp[1]; i13[k] = nn[0]
    return np.stack([i12, i23, i13])


def closure_counts(stack, dset, kept, tri, mask, rows=40, tb=256):
    """Per triplet: number of mask pixels with K != 0 and number of mask pixels valid in all three."""
    with h5py.File(stack, 'r') as f:
        ds = f[dset]
        ry, rx = int(f.attrs['REF_Y']), int(f.attrs['REF_X'])
        L, W = ds.shape[1:]
        ref = ds[:, ry, rx][kept].astype(np.float32)
        step = ds.chunks[1] * max(1, int(round(rows / ds.chunks[1]))) if ds.chunks else rows
        ntri = tri.shape[1]; nz = np.zeros(ntri, np.int64); nv = np.zeros(ntri, np.int64)
        t0 = time.time()
        for r0 in range(0, L, step):
            r1 = min(L, r0 + step)
            mk = mask[r0:r1].ravel()
            if not mk.any():
                continue
            u = ds[:, r0:r1, :][kept].reshape(len(kept), -1)[:, mk]
            valid = u != 0
            u = np.where(valid, u.astype(np.float32) - ref[:, None], np.float32(0))
            for k0 in range(0, ntri, tb):
                a, b, c = tri[:, k0:k0 + tb]
                cp = u[a] + u[b] - u[c]
                K = np.floor((cp + np.pi) / TWO_PI) != 0
                v = valid[a] & valid[b] & valid[c]
                nz[k0:k0 + tb] += (K & v).sum(1); nv[k0:k0 + tb] += v.sum(1)
            print(f'  closure rows {r1}/{L}  {time.time() - t0:.0f} s', end='\r', flush=True)
        print()
    return nz, nv


def n_components(date12, keep):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pairs = [p for p, k in zip(date12, keep) if k]
    ds = sorted({x for p in pairs for x in p.split('_')})
    ix = {d: i for i, d in enumerate(ds)}
    e = np.array([(ix[p[:8]], ix[p[9:]]) for p in pairs])
    n = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(ds),) * 2),
                             directed=False)[0]
    return len(ds), n


def main():
    inps = cmd_line_parse()
    bak = os.path.splitext(inps.stack)[0] + '_dropIfgram_before_clean.txt'
    aft = os.path.splitext(inps.stack)[0] + '_dropIfgram_after_clean.txt'
    with h5py.File(inps.stack, 'r') as f:
        date12 = np.array(['_'.join(x.decode() for x in r) for r in f['date'][:]])
        keep0 = f['dropIfgram'][:].astype(bool)
        L, W = f[inps.dset].shape[1:]
    # idempotent: if the current network is exactly the result of a previous clean, score the
    # pre-clean network again (a network changed since, e.g. by modify_network, is taken as is)
    if os.path.isfile(bak) and os.path.isfile(aft):
        b, a = np.loadtxt(bak, dtype=str), np.loadtxt(aft, dtype=str)
        if (b[:, 0] == date12).all() and (a[:, 1].astype(int).astype(bool) == keep0).all():
            keep0 = b[:, 1].astype(int).astype(bool)
            print(f'{inps.stack} is the result of a previous clean: restart from the network in {bak}')
    kept = np.flatnonzero(keep0)
    print(f'{inps.stack}: {len(date12)} pairs, {len(kept)} in the network')
    mask, msrc = pixel_mask(inps, (L, W))
    tri_local = triplets(date12[kept])                      # indices into `kept`
    print(f'{tri_local.shape[1]} network triplets')
    nz, nv = closure_counts(inps.stack, inps.dset, kept, tri_local, mask)
    fr = np.where(nv > 0, nz / np.maximum(nv, 1), np.nan)

    # per-pair score (indices of the full stack)
    s = np.zeros(len(date12)); c = np.zeros(len(date12))
    for t in range(tri_local.shape[1]):
        if np.isfinite(fr[t]):
            for k in kept[tri_local[:, t]]:
                s[k] += fr[t]; c[k] += 1
    score = np.where(c > 0, s / np.maximum(c, 1), np.nan)

    # drop worst first, keep connectivity and every date
    keep = keep0.copy()
    cand = keep & np.isfinite(score) & (score > inps.thr)
    nd0, nc0 = n_components(date12, keep)
    dropped = []
    for k in np.argsort(-np.nan_to_num(np.where(cand, score, -1))):
        if not cand[k]:
            break
        keep[k] = False
        nd, nc = n_components(date12, keep)
        if nc > nc0 or nd < nd0:
            keep[k] = True
        else:
            dropped.append(date12[k])
    nd, nc = n_components(date12, keep)
    msg = (f'clean_network thr {inps.thr} (mask: {msrc}): {int(cand.sum())} candidates, dropped {len(dropped)}, '
           f'kept {int(keep.sum())} of {int(keep0.sum())} pairs, {nd} dates, {nc} component(s); '
           f'median triplet non-closure {np.nanmedian(fr):.4f}, triplets f > 0.1: {np.nanmean(fr > 0.1) * 100:.1f} %')
    print(msg)

    os.makedirs(inps.outdir, exist_ok=True)
    rep = os.path.join(inps.outdir, 'clean_network.txt')
    with open(rep, 'w') as fh:
        fh.write(f'# {msg}\n# stack {os.path.abspath(inps.stack)}  dataset {inps.dset}\n')
        fh.write('# date12  score  n_triplets  in_network_before  in_network_after\n')
        for k in np.argsort(-np.nan_to_num(score, nan=-1)):
            fh.write(f'{date12[k]}  {score[k]:.4f}  {int(c[k])}  {int(keep0[k])}  {int(keep[k])}\n')
    print(f'report: {rep}')
    if inps.dry_run:
        print('dry run: dropIfgram unchanged'); return
    np.savetxt(bak, np.c_[date12, keep0.astype(int)], fmt='%s')
    np.savetxt(aft, np.c_[date12, keep.astype(int)], fmt='%s')
    with h5py.File(inps.stack, 'r+') as f:
        f['dropIfgram'][:] = keep
        f.attrs['CLEAN_NETWORK'] = msg
    print(f'dropIfgram updated in {inps.stack} (previous network saved in {bak})')


if __name__ == '__main__':
    sys.exit(main())
