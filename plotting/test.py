#!/usr/bin/env python3
"""
Quick diagnostic for the LRD plotting script.
Run:  python3 plotting/diagnose_bh_fields.py
Tells us how BHMaxaccretionRate / BHEddingtonRateLimit are stored
and where the non-zero accretion events actually live.
"""
import glob, sys
import h5py
import numpy as np

pattern = sys.argv[1] if len(sys.argv) > 1 else './output/millennium/model_*.hdf5'
files = sorted(glob.glob(pattern))
if not files:
    print(f'No files match {pattern}'); sys.exit(1)

f = files[0]
print(f'Inspecting: {f}\n')

with h5py.File(f, 'r') as hf:
    snaps = sorted([k for k in hf.keys() if k.startswith('Snap_')],
                   key=lambda s: int(s.split('_')[1]))
    print(f'Snapshot groups present: {snaps[0]} ... {snaps[-1]}  ({len(snaps)} total)\n')

    # Pick the last snapshot group (z=0-ish, most populated) and snap 10
    for target in ('Snap_62', 'Snap_10', snaps[-1]):
        if target not in hf:
            continue
        grp = hf[target]
        ngal = grp['BlackHoleMass'].shape[0]
        print(f'=== {target}  (Ngal = {ngal:,}) ===')

        for field in ('BlackHoleMass', 'StellarMass',
                      'BHMaxaccretionRate', 'BHEddingtonRateLimit',
                      'BHAccretionType', 'BHMassatAccretion', 'dt'):
            if field not in grp:
                print(f'  {field:24s}  --- NOT PRESENT ---')
                continue
            d = grp[field]
            flat = d[:].ravel()
            nz = np.sum(flat > 0)
            print(f'  {field:24s}  shape={str(d.shape):20s}  '
                  f'nonzero={nz:,}/{flat.size:,}  '
                  f'max={flat.max():.3e}')

        # If BHMaxaccretionRate is 2D, show which columns hold the events
        if 'BHMaxaccretionRate' in grp:
            arr = grp['BHMaxaccretionRate'][:]
            if arr.ndim == 2:
                col_nz = (arr > 0).sum(axis=0)
                live = np.where(col_nz > 0)[0]
                print(f'\n  BHMaxaccretionRate columns with ANY nonzero entry:')
                print(f'    columns: {live.tolist()}')
                if len(live):
                    print(f'    per-column counts: '
                          f'{ {int(c): int(col_nz[c]) for c in live} }')
        print()