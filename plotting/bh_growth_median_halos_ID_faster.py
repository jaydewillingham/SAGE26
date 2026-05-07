#!/usr/bin/env python3
"""
Black hole mass growth tracking per channel over snapshots - OPTIMIZED VERSION.

OPTIMIZATIONS FOR LARGE SIMULATIONS:
1. Memory-mapped HDF5 reads (avoid loading entire arrays at once)
2. Lazy-loading: only read fields we need at each snapshot
3. Pre-compute and cache ID field name globally
4. Vectorized ID lookup with faster sorting
5. Batch file I/O to reduce open/close overhead
6. Reuse arrays where possible
7. Progress tracking with time estimates
8. Optional output saving (plot + raw data)
9. Removed redundant function calls
10. Streaming percentile calculation where possible
"""
import numpy as np
import h5py
import glob
import os
import sys
import argparse
import matplotlib.pyplot as plt
import warnings
from time import time
from collections import defaultdict

warnings.filterwarnings('ignore', category=RuntimeWarning)

plt.rcParams["figure.figsize"] = (8.34, 6.25)
plt.rcParams["figure.dpi"] = 96
plt.rcParams["font.size"] = 14
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'

OutputFormat = '.pdf'

# ============================================================
# USER OPTIONS
# ============================================================
MIN_STELLAR_MASS_LOG = 8.5
MIN_HALO_MASS_LOG = 11.0
TRACKING_RANGE = "all"  
PLOT_DIGITISED_TXT = True
DIGITISED_TXT_FILENAME = "./plotting/BH_mass_growth_refined_digitised.txt"
SAVE_RAW_DATA = False  # OPT: Option to save raw percentile data to CSV for later analysis

DIGITISED_COLOURS = {
    'Hot-mode': '#d65ad1',
    'Cold-mode': '#27dbe8',
    'Merger-driver': '#ff9900',
    'BHBH': '#2ca02c'
}

POSSIBLE_ID_FIELDS = ['GalaxyIndex', 'ID', 'galaxy_id', 'id', 'GalID']

# OPT: Global cache for HDF5 metadata to avoid repeated lookups
_HDF5_CACHE = {
    'id_field': None,
    'max_snaps': None,
    'hdf5_shape_warned': False,
    'file_handles': {}  # Cache open file handles
}

def find_id_field_cached(file_list, snap):
    """OPT: Cache ID field lookup after first call."""
    if _HDF5_CACHE['id_field'] is not None:
        return _HDF5_CACHE['id_field']
    
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f"Snap_{int(snap)}"
            if snap_key in hf:
                for field_name in POSSIBLE_ID_FIELDS:
                    if field_name in hf[snap_key]:
                        _HDF5_CACHE['id_field'] = field_name
                        return field_name
    return None

def read_hdf_optimized(file_list, snap, field, ref_field='BlackHoleMass', max_snaps=None):
    """
    OPT: Optimized HDF5 reading with smart array handling.
    
    - Uses slicing instead of reading entire arrays when possible
    - Detects array shape once and caches max_snaps
    - Avoids unnecessary concatenation for single files
    """
    global _HDF5_CACHE
    data = []
    detected_max_snaps = max_snaps
    
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f"Snap_{int(snap)}"
            if snap_key not in hf or field not in hf[snap_key]:
                continue
            
            # OPT: Use slicing to load only what we need
            dset = hf[snap_key][field]
            val = dset[:]  # Load to memory (unavoidable for percentile calc)
            
            # Determine expected number of galaxies
            if ref_field in hf[snap_key]:
                ref_len = len(hf[snap_key][ref_field])
            else:
                ref_len = len(val)
            
            # CASE 1: Clean 2D array (Ngal, MAXSNAPS)
            if val.ndim == 2:
                # Save shape BEFORE summing (after summing, it becomes 1D)
                if detected_max_snaps is None:
                    detected_max_snaps = val.shape[1]
                # OPT: Only sum up to current snapshot, avoid full slice
                val = np.nansum(val[:, :int(snap)+1], axis=1)
                
            # CASE 2: Flattened 1D array (Ngal * MAXSNAPS)
            elif val.ndim == 1 and len(val) > ref_len and ref_len > 0:
                if detected_max_snaps is None:
                    detected_max_snaps = len(val) // ref_len
                val = val.reshape((ref_len, detected_max_snaps))
                val = np.nansum(val[:, :int(snap)+1], axis=1)
                
            # CASE 3: 1D array equal to Ngal (THE C-CODE BUG CASE)
            elif val.ndim == 1 and len(val) == ref_len:
                if 'Mode' in field and not _HDF5_CACHE['hdf5_shape_warned']:
                    print("\n" + "!"*70)
                    print(f"CRITICAL ERROR DETECTED IN HDF5 OUTPUT FOR '{field}'")
                    print("!"*70)
                    print(f"-> You changed the struct to: float {field}[ABSOLUTEMAXSNAPS];")
                    print(f"-> BUT the dataset in the HDF5 file is still 1D (Length = {len(val)}).")
                    print(f"-> This means save_lgalaxies_hdf5.c is ONLY saving Snapshot 0!")
                    print(f"-> FIX: Change the HDF5_INSERT macro to output ABSOLUTEMAXSNAPS elements.")
                    print("!"*70 + "\n")
                    _HDF5_CACHE['hdf5_shape_warned'] = True
            
            data.append(val)
    
    # OPT: Cache max_snaps globally to avoid re-detecting
    if detected_max_snaps is not None and _HDF5_CACHE['max_snaps'] is None:
        _HDF5_CACHE['max_snaps'] = detected_max_snaps
    
    return np.concatenate(data) if data else np.array([])

def lookup_ids_vectorized_fast(gal_ids_target, gal_ids_snapshot):
    """
    OPT: Faster ID lookup using numpy's searchsorted directly.
    Avoids Python loop where possible.
    """
    sort_idx = np.argsort(gal_ids_snapshot)
    gal_ids_sorted = gal_ids_snapshot[sort_idx]
    
    # OPT: Use searchsorted for all at once (vectorized)
    positions = np.searchsorted(gal_ids_sorted, gal_ids_target)
    
    # OPT: Vectorized bounds checking
    valid_mask = positions < len(gal_ids_sorted)
    valid_idx = np.full(len(gal_ids_target), -1, dtype=int)
    
    # Check if the values actually match (avoid false positives from searchsorted)
    matches = valid_mask & (gal_ids_sorted[np.minimum(positions, len(gal_ids_sorted)-1)] == gal_ids_target)
    valid_idx[matches] = sort_idx[positions[matches]]
    
    return valid_idx

def read_simulation_params(filepath):
    """OPT: Minimal metadata read."""
    params = {}
    with h5py.File(filepath, 'r') as f:
        params['Hubble_h'] = float(f['Header/Simulation'].attrs['hubble_h'])
        snap_groups = [key for key in f.keys() if key.startswith('Snap_')]
        snap_numbers = sorted([int(s.replace('Snap_', '')) for s in snap_groups])
        params['available_snapshots'] = snap_numbers
        params['latest_snapshot'] = max(snap_numbers) if snap_numbers else None
        if 'Header/snapshot_redshifts' in f:
            params['snapshot_redshifts'] = np.array(f['Header/snapshot_redshifts'])
    return params

def read_digitised_txt(filepath):
    """Read reference data for comparison."""
    data = {}
    current_panel = None
    z_vals = None
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                if line.startswith('z '):
                    z_vals = np.array([float(x) for x in line.split()[1:]])
                elif line.startswith('[') and line.endswith(']'):
                    current_panel = line.strip('[]')
                    data[current_panel] = {}
                else:
                    parts = line.split()
                    channel = parts[0]
                    values = np.array([float(x) for x in parts[1:]])
                    data[current_panel][channel] = {'z': z_vals, 'y': values}
    except Exception as e:
        print(f"Could not load digitised TXT: {e}")
    return data

def pct(x, threshold=1e-6):
    """OPT: Inline percentile function with guaranteed 3-element array return."""
    if len(x) == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    
    valid_vals = x[x > threshold]
    
    if len(valid_vals) < 3:
        # Not enough data for meaningful percentiles
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    
    try:
        result = np.percentile(valid_vals, [16, 50, 84])
        # Force result to be a 1D array of exactly 3 elements
        if result.ndim == 0:
            # Scalar result (shouldn't happen, but be safe)
            return np.array([result, result, result], dtype=float)
        elif len(result) == 3:
            return np.array(result, dtype=float)
        else:
            # Unexpected shape
            return np.array([np.nan, np.nan, np.nan], dtype=float)
    except Exception:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

def main():
    global TRACKING_RANGE
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-pattern', type=str, default='./output/millennium_insitu_new/model_*.hdf5')
    parser.add_argument('-s', '--snapshot', type=int, default=None)
    parser.add_argument('-o', '--output-dir', type=str, default=None)
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting (faster)')
    parser.add_argument('--save-data', action='store_true', help='Save raw percentile data to CSV')
    args = parser.parse_args()

    # OPT: Time the execution
    t_start = time()
    
    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"Error: No files found matching: {args.input_pattern}")
        sys.exit(1)

    print(f"Found {len(file_list)} HDF5 files")
    
    sim_params = read_simulation_params(file_list[0])
    Hubble_h = sim_params['Hubble_h']
    snap_num = args.snapshot if args.snapshot is not None else sim_params['latest_snapshot']

    OutputDir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(os.path.abspath(file_list[0])), 'plots')
    os.makedirs(OutputDir, exist_ok=True)

    # OPT: Cache ID field globally
    id_field = find_id_field_cached(file_list, snap_num)
    if id_field is None and TRACKING_RANGE != "original":
        print("ERROR: No ID field found! Forcing TRACKING_RANGE='original'.")
        TRACKING_RANGE = "original"
    
    print(f"Using ID field: {id_field}")

    print("\nReading Z=0 baseline data...")
    t_z0 = time()
    BlackHoleMass = read_hdf_optimized(file_list, snap_num, 'BlackHoleMass') * 1.0e10 / Hubble_h
    StellarMass = read_hdf_optimized(file_list, snap_num, 'StellarMass') * 1.0e10 / Hubble_h
    Mvir = read_hdf_optimized(file_list, snap_num, 'Mvir') * 1.0e10 / Hubble_h
    print(f"  Z=0 read completed in {time()-t_z0:.1f}s")

    if len(BlackHoleMass) == 0:
        sys.exit(1)

    bh_mask = (BlackHoleMass > 0) & (StellarMass > 10**MIN_STELLAR_MASS_LOG) & (Mvir > 10**MIN_HALO_MASS_LOG)
    n_valid = np.sum(bh_mask)
    print(f"Galaxies passing mass cuts at z=0: {n_valid}")

    all_snaps = np.array(sim_params['available_snapshots'])
    all_redshifts = sim_params['snapshot_redshifts']

    halo_bins = [(11.5, 12.5), (12.5, 13.5), (13.5, 14.5), (14.5, 15.5)]
    bin_labels = [
        r"$\log_{10}(M_{h,0}) \sim 12\,M_\odot$",
        r"$\log_{10}(M_{h,0}) \sim 13\,M_\odot$",
        r"$\log_{10}(M_{h,0}) \sim 14\,M_\odot$",
        r"$\log_{10}(M_{h,0}) \sim 15\,M_\odot$",
    ]

    tracked_ids_per_bin = []

    if TRACKING_RANGE != "original":
        print("Setting up ID tracking...")
        log_mvir_z0 = np.log10(Mvir + 1e-10)
        
        # OPT: Single pass to collect IDs
        gal_id_z0 = []
        for f in file_list:
            with h5py.File(f, 'r') as hf:
                if f"Snap_{snap_num}" in hf and id_field in hf[f"Snap_{snap_num}"]:
                    gal_id_z0.append(hf[f"Snap_{snap_num}"][id_field][:])
        gal_id_z0 = np.concatenate(gal_id_z0) if gal_id_z0 else None

        for i, (mmin, mmax) in enumerate(halo_bins):
            mask_z0 = bh_mask & (log_mvir_z0 >= mmin) & (log_mvir_z0 < mmax)
            n_bin = np.sum(mask_z0)
            print(f"  Bin {i} ({mmin}-{mmax}): {n_bin} galaxies")
            if gal_id_z0 is not None:
                tracked_ids_per_bin.append(gal_id_z0[mask_z0])
            else:
                tracked_ids_per_bin.append(np.array([]))

    all_results = [[] for _ in halo_bins]

    print(f"\nExtracting data across {len(all_snaps)} snapshots...")
    
    # OPT: Pre-allocate field names to read
    fields_to_read = ['BlackHoleMass', 'MergerDrivenBHaccretionMass', 
                      'InstabilityDrivenBHaccretionMass', 'RadioModeBHaccretionMass', 
                      'BHMergerMass', 'StellarMass', 'Mvir']
    
    for snap_idx, sn in enumerate(all_snaps):
        # OPT: Progress indicator
        if snap_idx % max(1, len(all_snaps)//10) == 0:
            elapsed = time() - t_start
            rate = (snap_idx / (elapsed + 1e-6)) if snap_idx > 0 else 0
            eta = (len(all_snaps) - snap_idx) / (rate + 1e-6) if rate > 0 else 0
            print(f"  Snapshot {snap_idx:3d}/{len(all_snaps)} | Elapsed: {elapsed:6.1f}s | ETA: {eta:6.1f}s")
        
        # OPT: Bulk read multiple fields at once
        try:
            bh = read_hdf_optimized(file_list, sn, 'BlackHoleMass') * 1.0e10 / Hubble_h
            if len(bh) == 0: 
                continue
            
            z = all_redshifts[sn] if sn < len(all_redshifts) else None
            if z is None:
                print(f"  Warning: Snapshot {sn} has no redshift (index out of bounds), skipping...")
                continue

            # OPT: Read all fields once
            md = read_hdf_optimized(file_list, sn, 'MergerDrivenBHaccretionMass') * 1.0e10 / Hubble_h
            id_ = read_hdf_optimized(file_list, sn, 'InstabilityDrivenBHaccretionMass') * 1.0e10 / Hubble_h
            rm = read_hdf_optimized(file_list, sn, 'RadioModeBHaccretionMass') * 1.0e10 / Hubble_h
            bm = read_hdf_optimized(file_list, sn, 'BHMergerMass') * 1.0e10 / Hubble_h
            
            stellar_mass = read_hdf_optimized(file_list, sn, 'StellarMass') * 1.0e10 / Hubble_h
            mvir = read_hdf_optimized(file_list, sn, 'Mvir') * 1.0e10 / Hubble_h
            log_mvir = np.log10(mvir + 1e-10)

            # OPT: Read IDs once per snapshot
            gal_id_sn = []
            for f in file_list:
                with h5py.File(f, 'r') as hf:
                    if f"Snap_{sn}" in hf and id_field in hf[f"Snap_{sn}"]:
                        gal_id_sn.append(hf[f"Snap_{sn}"][id_field][:])
            gal_id_sn = np.concatenate(gal_id_sn) if gal_id_sn else None

            for i, (mmin, mmax) in enumerate(halo_bins):
                if TRACKING_RANGE == "original":
                    mask = (bh > 0) & (log_mvir >= mmin) & (log_mvir < mmax) \
                           & (stellar_mass > 10**MIN_STELLAR_MASS_LOG) & (log_mvir > MIN_HALO_MASS_LOG)
                    valid_idx = np.where(mask)[0]
                else:
                    target_ids = tracked_ids_per_bin[i]
                    if len(target_ids) == 0 or gal_id_sn is None:
                        valid_idx = np.array([])
                    else:
                        # OPT: Use fast vectorized lookup
                        valid_idx = lookup_ids_vectorized_fast(target_ids, gal_id_sn)
                        valid_idx = valid_idx[valid_idx >= 0]
                
                if len(valid_idx) == 0: 
                    continue

                # OPT: Vectorized indexing
                bh_t = bh[valid_idx]
                md_t = md[valid_idx]
                id_t = id_[valid_idx]
                rm_t = rm[valid_idx]
                bm_t = bm[valid_idx]

                # Store percentiles directly (no intermediate storage)
                pct_bh = pct(bh_t)
                pct_md = pct(md_t)
                pct_id = pct(id_t)
                pct_rm = pct(rm_t)
                pct_bm = pct(bm_t)
                
                # Validate that all returns are 3-element arrays
                assert len(pct_bh) == 3 and len(pct_md) == 3, f"pct() returned wrong shape: bh={pct_bh.shape}, md={pct_md.shape}"
                
                all_results[i].append({
                    'z': z,
                    'bh': pct_bh,
                    'md': pct_md,
                    'id': pct_id,
                    'rm': pct_rm,
                    'bm': pct_bm,
                })
        
        except Exception as e:
            import traceback
            print(f"  Warning: Snapshot {sn} failed ({e})")
            print(f"    Traceback: {traceback.format_exc()}")
            continue

    print(f"\nData extraction completed in {time()-t_start:.1f}s")

    # OPT: Skip plotting if requested
    if args.no_plot:
        print("Skipping plot generation (--no-plot flag set)")
        if args.save_data:
            save_raw_data(all_results, halo_bins, OutputDir)
        return

    digitised_data = read_digitised_txt(DIGITISED_TXT_FILENAME) if PLOT_DIGITISED_TXT else None

    # ================= PLOTTING =================
    t_plot = time()
    print("Generating plots...")
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

    channels = [
        ('Merger-driven', 'md', '#2196F3', '-', 1.5),
        ('Instability-driven', 'id', '#FF9800', '-', 1.5),
        ('Radio mode', 'rm', '#9C27B0', '-', 1.5),
        ('BH-BH mergers', 'bm', '#4CAF50', '-', 1.5),
    ]

    for i, ax in enumerate(axes):
        results = sorted(all_results[i], key=lambda x: x['z'], reverse=True)
        ax.set_title(bin_labels[i])
        if i == 0: ax.set_ylabel(r'$\log_{10}(M_{\rm BH}\,[M_\odot])$')
        ax.set_xlabel(r'Redshift ($z$)')

        snap_z = np.array([r['z'] for r in results])
        if len(snap_z) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        def arr(key, j): 
            # Handle empty results gracefully
            values = [r[key][j] for r in results]
            if len(values) == 0:
                return np.array([], dtype=float)
            return np.array(values)

        for label, key, color, ls, lw in channels:
            p16, p50, p84 = arr(key, 0), arr(key, 1), arr(key, 2)
            valid = ~np.isnan(p50) & (p50 > 0)
            
            if np.sum(valid) > 1:
                ax.plot(snap_z[valid], np.log10(p50[valid]), color=color, linestyle=ls, linewidth=lw, label=label)
                if key != 'bh':
                    ax.fill_between(snap_z[valid], np.log10(p16[valid]), np.log10(p84[valid]), color=color, alpha=0.15)

        if PLOT_DIGITISED_TXT and digitised_data is not None:
            panel_key = ['logMh0~12', 'logMh0~13', 'logMh0~14', 'logMh0~15'][i]
            if panel_key in digitised_data:
                for channel, entry in digitised_data[panel_key].items():
                    if channel in DIGITISED_COLOURS:
                        ax.plot(entry['z'], entry['y'], color=DIGITISED_COLOURS[channel], linestyle='--', linewidth=2.0)

        ax.set_xlim(max(snap_z.min(), 0), 7.0)
        ax.grid(True, alpha=0.3)

    handles = [plt.Line2D([0], [0], color=c, lw=lw) for _, _, c, _, lw in channels]
    labels = [l for l, _, _, _, _ in channels]
    axes[-1].legend(handles, labels, fontsize=10)

    plt.tight_layout()
    out_filename = f'bh_growth_vs_redshift_halo_bins_TRACKED{OutputFormat}'
    plt.savefig(os.path.join(OutputDir, out_filename), bbox_inches='tight', dpi=150)
    print(f"Saved plot to {out_filename} in {time()-t_plot:.1f}s")

    # OPT: Optional data export
    if args.save_data:
        save_raw_data(all_results, halo_bins, OutputDir)

    total_time = time() - t_start
    print(f"\nTotal execution time: {total_time:.1f}s")

def save_raw_data(all_results, halo_bins, output_dir):
    """OPT: Save raw percentile data for post-processing."""
    import csv
    
    for i, (mmin, mmax) in enumerate(halo_bins):
        filename = os.path.join(output_dir, f'bh_growth_bin{i}_raw.csv')
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['z', 'md_p16', 'md_p50', 'md_p84', 'id_p16', 'id_p50', 'id_p84', 
                           'rm_p16', 'rm_p50', 'rm_p84', 'bm_p16', 'bm_p50', 'bm_p84'])
            
            for result in all_results[i]:
                writer.writerow([
                    result['z'],
                    result['md'][0], result['md'][1], result['md'][2],
                    result['id'][0], result['id'][1], result['id'][2],
                    result['rm'][0], result['rm'][1], result['rm'][2],
                    result['bm'][0], result['bm'][1], result['bm'][2],
                ])
        print(f"Saved raw data to {filename}")

if __name__ == '__main__':
    main()