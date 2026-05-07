#!/usr/bin/env python3
"""
Diagnostic script to check BH seed mass vs initial snapshot contributions.
"""
import argparse
import glob
import h5py
import numpy as np
import sys
from pathlib import Path

def read_simulation_params(filepath):
    try:
        with h5py.File(filepath, 'r') as hf:
            header = hf['Header/Simulation'].attrs if 'Header/Simulation' in hf else hf['Header'].attrs
            return {
                'Hubble_h': header.get('hubble_h', header.get('HubbleParam', 0.681)),
                'latest_snapshot': 62,
            }
    except:
        return {'Hubble_h': 0.681, 'latest_snapshot': 62}

def find_id_field(file_list, snap_num):
    candidates = ['GalaxyIndex', 'GalaxyID', 'ID', 'galaxy_id', 'id']
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f"Snap_{snap_num}"
            if snap_key in hf:
                for c in candidates:
                    if c in hf[snap_key]:
                        return c
    return None

def main():
    parser = argparse.ArgumentParser(description='Check BH seed mass vs first snapshot contributions')
    parser.add_argument('-i', '--input-pattern', default='./output/millennium/model_*.hdf5',
                        help='Glob pattern for input HDF5 files')
    parser.add_argument('-s', '--snapshot', type=int, default=None,
                        help='Snapshot to read from (default: latest)')
    parser.add_argument('-n', '--count', type=int, default=50,
                        help='Number of galaxy IDs to check (default: 5)')
    
    args = parser.parse_args()
    file_list = sorted(glob.glob(args.input_pattern))

    if not file_list:
        print(f"Error: No files found matching pattern '{args.input_pattern}'")
        sys.exit(1)

    sim = read_simulation_params(file_list[0])
    h_h = sim['Hubble_h']
    snap_num = args.snapshot if args.snapshot is not None else sim['latest_snapshot']
    id_field = find_id_field(file_list, snap_num)

    if not id_field:
        print("Error: Could not find Galaxy ID field.")
        sys.exit(1)

    print(f"Checking first {args.count} galaxies from snapshot {snap_num}...")
    print(f"Hubble_h: {h_h}\n")
    
    # Growth channel fields
    channels = [
        'MergerDrivenBHaccretionMass',
        'InstabilityDrivenBHaccretionMass',
        'RadioModeBHaccretionMass',
        'BHMergerMass'
    ]

    found_count = 0
    
    # Header for the table
    header = f"{'GalaxyID':<15} | {'Seed Mass':<12} | {'Snap':<4} | {'MD':<10} | {'ID':<10} | {'RM':<10} | {'BM':<10}"
    print(header)
    print("-" * len(header))

    for f in file_list:
        if found_count >= args.count:
            break
            
        with h5py.File(f, 'r') as hf:
            snap_key = f"Snap_{snap_num}"
            if snap_key not in hf:
                continue
            
            grp = hf[snap_key]
            ids = grp[id_field][:]
            
            # Optional fields
            bh_mass = grp['BlackHoleMass'][:] if 'BlackHoleMass' in grp else np.zeros(len(ids))
            
            # Identify indices with BHs
            bh_indices = np.where(bh_mass > 0)[0]
            
            for idx in bh_indices:
                if found_count >= args.count:
                    break
                
                gid = ids[idx]
                seed = (grp['BHSeedMass'][idx] if 'BHSeedMass' in grp else 0.0) * 1e10 / h_h
                
                # We need to find the FIRST contribution across snapshots.
                # In your context, these fields are often 2D (Ngal x MAXSNAPS)
                # or flattened. We'll reconstruct the first non-zero snapshot.
                
                contribs = {}
                first_snap = -1
                
                for ch in channels:
                    if ch in grp:
                        val = grp[ch][idx]
                        # If it's an array (full history), find the first non-zero frame
                        if isinstance(val, (np.ndarray, list)) or (has_len := hasattr(val, '__len__') and len(val) > 1):
                            # Ensure it's treated as numeric array
                            arr = np.array(val)
                            # Find first snapshot index where this channel > 0
                            nonzero = np.where(arr > 0)[0]
                            if len(nonzero) > 0:
                                s_idx = nonzero[0]
                                if first_snap == -1 or s_idx < first_snap:
                                    first_snap = s_idx
                                contribs[ch] = arr * 1e10 / h_h
                            else:
                                contribs[ch] = np.zeros_like(arr)
                        else:
                            # Scalar value - assume it's the current snapshot
                            contribs[ch] = float(val) * 1e10 / h_h
                            if contribs[ch] > 0 and first_snap == -1:
                                first_snap = snap_num
                    else:
                        contribs[ch] = 0.0

                # Print logic: if we found a "first" snap, print values at that snap
                # Otherwise print current snap values
                display_snap = first_snap if first_snap != -1 else snap_num
                
                row_vals = []
                for ch in channels:
                    v = contribs[ch]
                    if isinstance(v, np.ndarray):
                        val_at_snap = v[display_snap] if display_snap < len(v) else v[-1]
                        row_vals.append(f"{val_at_snap:1.2e}")
                    else:
                        row_vals.append(f"{v:1.2e}")

                print(f"{gid:<15} | {seed:1.2e} | {display_snap:<4} | {' | '.join(row_vals)}")
                found_count += 1

    print(f"\nCompleted check for {found_count} galaxies.")

if __name__ == "__main__":
    main()
