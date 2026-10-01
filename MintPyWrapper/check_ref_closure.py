#!/usr/bin/env python3
"""Check that the reference pixel is unwrapped consistently with the scene.

For every triplet (t1,t2,t3) of the kept network the closure
    C = phi12 + phi23 - phi13   (after subtracting the reference pixel, as MintPy)
has an integer part K = round((C - wrap C) / 2pi). If the reference pixel sits
on an isolated patch that is unwrapped one cycle away from the rest of the
scene in some interferograms, K is non-zero for that triplet over (almost) the
WHOLE scene. Such "scene-wide" triplets are counted here from a sample of rows.

For a per-pixel linear inversion this error is common-mode (it cancels in
relative velocities) but it lowers temporal coherence everywhere and becomes
spatially variable where pixels use different interferogram subsets. The cure is
to move the reference point, not to re-unwrap.

Exit status 0 always (warning only). Optional: list better candidates nearby.

Usage:
    check_ref_closure.py inputs/ifgramStack.h5 [--frac 0.8] [--suggest 60]
"""
import argparse
import sys

import h5py
import numpy as np
from mintpy.objects import ifgramStack


def kint(u, a, b, c):
    cp = u[a] + u[b] - u[c]
    return np.floor((cp + np.pi) / (2 * np.pi))       # = round((cp - wrap cp) / 2pi), MintPy wrap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stack')
    ap.add_argument('--frac', type=float, default=0.8, help='scene fraction defining a scene-wide triplet')
    ap.add_argument('--nbands', type=int, default=8, help='row bands sampled along the track')
    ap.add_argument('--suggest', type=int, default=0, metavar='RADIUS_PX',
                    help='if scene-wide triplets exist, list consistent pixels within this radius')
    a = ap.parse_args()

    obj = ifgramStack(a.stack); obj.open(print_msg=False)
    d12 = obj.get_date12_list(dropIfgram=True)
    keep = obj.dropIfgram
    C = ifgramStack.get_design_matrix4triplet(d12)
    if C is None:
        print('no triplets in the kept network; skip'); return
    ia = np.array([np.flatnonzero(r == 1)[0] for r in C])
    ib = np.array([np.flatnonzero(r == 1)[1] for r in C])
    ic = np.array([np.flatnonzero(r == -1)[0] for r in C])
    idx = np.flatnonzero(keep)
    with h5py.File(a.stack, 'r') as f:
        ds = f['unwrapPhase']
        L, W = ds.shape[1:]
        ry, rx = int(f.attrs['REF_Y']), int(f.attrs['REF_X'])
        ref = ds[:, ry, rx][idx].astype(np.float32)

        def rows(r0, r1):
            u = ds[:, r0:r1, :][idx].astype(np.float32)
            v = u != 0
            return np.where(v, u - ref[:, None, None], 0), v

        ch = ds.chunks[1] if ds.chunks else 64
        starts = np.linspace(0, max(L - ch, 0), a.nbands).astype(int) // ch * ch
        nz = np.zeros(len(ia)); nv = np.zeros(len(ia))
        for s in starts:
            u, v = rows(s, min(L, s + ch))
            u = u[:, ::4, ::4].reshape(len(idx), -1); v = v[:, ::4, ::4].reshape(len(idx), -1)
            ok = v[ia] & v[ib] & v[ic]
            K = kint(u, ia, ib, ic)
            nz += np.sum(ok & (K != 0), axis=1); nv += ok.sum(1)
    F = nz / np.maximum(nv, 1)
    bad = np.flatnonzero(F > a.frac)
    lat = obj.metadata.get('REF_LAT'); lon = obj.metadata.get('REF_LON')
    print(f'reference pixel y,x = {ry},{rx} (lat,lon = {lat},{lon}); '
          f'{len(ia)} triplets in the kept network')
    if len(bad) == 0:
        print(f'OK: 0 scene-wide triplets (K != 0 over > {a.frac:.0%} of sampled pixels)')
        return
    print(f'WARNING: {len(bad)} of {len(ia)} triplets ({100*len(bad)/len(ia):.1f} %) are inconsistent over '
          f'> {a.frac:.0%} of the scene: the reference pixel is likely on an isolated unwrapping patch. '
          f'Consider moving mintpy.reference.lalo/yx (re-referencing cures it; re-unwrapping is not needed).')
    worst = {}
    for j in bad:
        for i in (ia[j], ib[j], ic[j]):
            worst[d12[i]] = worst.get(d12[i], 0) + 1
    print('  interferograms most involved: ' + ', '.join(f'{k}({v})' for k, v in
                                                        sorted(worst.items(), key=lambda x: -x[1])[:8]))
    if a.suggest:
        R = a.suggest
        with h5py.File(a.stack, 'r') as f:
            y0, y1 = max(ry - R, 0), min(ry + R + 1, L); x0, x1 = max(rx - R, 0), min(rx + R + 1, W)
            u, v = rows(y0, y1); u = u[:, :, x0:x1]; v = v[:, :, x0:x1]
        h, w = u.shape[1:]
        uu = u.reshape(len(idx), -1); vv = v.reshape(len(idx), -1)
        mode = (F > 0.5).astype(float)      # scene majority: K != 0 where most of the scene disagrees with REF
        K = kint(uu, ia, ib, ic)
        ok = vv[ia] & vv[ib] & vv[ic]
        Kn = (K != 0)
        mism = np.sum(ok & (Kn != mode[:, None].astype(bool)), axis=0).reshape(h, w)
        allvalid = np.all(vv, axis=0).reshape(h, w)
        cand = np.flatnonzero((allvalid & (mism <= mism[allvalid].min() + 2)).ravel()) if allvalid.any() else []
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.hypot(yy - ry, xx - rx).ravel()
        cand = sorted(cand, key=lambda k: dist[k])[:5]
        print(f'  closest consistent, all-valid pixels within {R} px (y,x: mismatching triplets):')
        for k in cand:
            print(f'    {yy.ravel()[k]},{xx.ravel()[k]}: {mism.ravel()[k]}')


if __name__ == '__main__':
    sys.exit(main())
