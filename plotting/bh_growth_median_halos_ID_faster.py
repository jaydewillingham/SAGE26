#!/usr/bin/env python3
"""
Black hole mass growth tracking per channel over snapshots.

UPDATED: 
Intelligently handles [ABSOLUTEMAXSNAPS] arrays.
Auto-detects HDF5 struct dimensions and reconstructs the cumulative sums.
Includes diagnostics to tell you if your C-code HDF5 output is broken.
"""
import numpy as np
import h5py
import glob
import os
import sys
import argparse
import matplotlib.pyplot as plt
import warnings

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

DIGITISED_COLOURS = {
    'Hot-mode': '#d65ad1',
    'Cold-mode': '#27dbe8',
    'Merger-driver': '#ff9900',
    'BHBH': '#2ca02c'
}

POSSIBLE_ID_FIELDS = ['GalaxyIndex', 'ID', 'galaxy_id', 'id', 'GalID']

# Global flag to ensure we only warn about broken HDF5 dimensions once
HDF5_SHAPE_WARNED = False

def read_hdf(file_list, snap, field, ref_field='BlackHoleMass'):
    """Reads a field and intelligently handles L-Galaxies 2D/1D arrays."""
    global HDF5_SHAPE_WARNED
    data = []
    
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f"Snap_{int(snap)}"
            if snap_key not in hf or field not in hf[snap_key]:
                continue
                
            val = hf[snap_key][field][:]
            
            # Determine expected number of galaxies
            ref_len = len(hf[snap_key][ref_field][:]) if ref_field in hf[snap_key] else len(val)
            
            # CASE 1: Clean 2D array (Ngal, MAXSNAPS)
            if val.ndim == 2:
                # Sum incrementally up to current snapshot frame
                val = np.nansum(val[:, :int(snap)+1], axis=1)
                
            # CASE 2: Flattened 1D array (Ngal * MAXSNAPS)
            elif val.ndim == 1 and len(val) > ref_len and ref_len > 0:
                max_snaps = len(val) // ref_len
                val = val.reshape((ref_len, max_snaps))
                val = np.nansum(val[:, :int(snap)+1], axis=1)
                
            # CASE 3: 1D array equal to Ngal. (THE C-CODE BUG CASE)
            elif val.ndim == 1 and len(val) == ref_len:
                if 'Mode' in field and not HDF5_SHAPE_WARNED:
                    print("\n" + "!"*70)
                    print(f"CRITICAL ERROR DETECTED IN HDF5 OUTPUT FOR '{field}'")
                    print("!"*70)
                    print(f"-> You changed the struct to: float {field}[ABSOLUTEMAXSNAPS];")
                    print(f"-> BUT the dataset in the HDF5 file is still 1D (Length = {len(val)}).")
                    print(f"-> This means save_lgalaxies_hdf5.c is ONLY saving Snapshot 0, which is 0.0!")
                    print(f"-> FIX: Change the HDF5_INSERT macro for this field in your C code to")
                    print(f"   output ABSOLUTEMAXSNAPS elements instead of 1.")
                    print("!"*70 + "\n")
                    HDF5_SHAPE_WARNED = True
            
            data.append(val)
            
    return np.concatenate(data) if data else np.array([])


def find_id_field(file_list, snap):
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f"Snap_{int(snap)}"
            if snap_key in hf:
                for field_name in POSSIBLE_ID_FIELDS:
                    if field_name in hf[snap_key]:
                        return field_name
    return None

def read_simulation_params(filepath):
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

def lookup_ids_vectorized(gal_ids_target, gal_ids_snapshot):
    sort_idx = np.argsort(gal_ids_snapshot)
    gal_ids_sorted = gal_ids_snapshot[sort_idx]
    indices = np.full(len(gal_ids_target), -1, dtype=int)
    for i, target_id in enumerate(gal_ids_target):
        pos = np.searchsorted(gal_ids_sorted, target_id)
        if pos < len(gal_ids_sorted) and gal_ids_sorted[pos] == target_id:
            indices[i] = sort_idx[pos]
    return indices

def main():
    global TRACKING_RANGE  # <--- FIXED: Moved to the very top of main()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-pattern', type=str, default='./output/millennium_insitu_new/model_*.hdf5')
    parser.add_argument('-s', '--snapshot', type=int, default=None)
    parser.add_argument('-o', '--output-dir', type=str, default=None)
    args = parser.parse_args()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"Error: No files found matching: {args.input_pattern}")
        sys.exit(1)

    sim_params = read_simulation_params(file_list[0])
    Hubble_h = sim_params['Hubble_h']
    snap_num = args.snapshot if args.snapshot is not None else sim_params['latest_snapshot']

    OutputDir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(os.path.abspath(file_list[0])), 'plots')
    os.makedirs(OutputDir, exist_ok=True)

    id_field = find_id_field(file_list, snap_num)
    if id_field is None and TRACKING_RANGE != "original":
        print("ERROR: No ID field found! Forcing TRACKING_RANGE='original'.")
        TRACKING_RANGE = "original"

    print("\nReading Z=0 baseline data...")
    BlackHoleMass = read_hdf(file_list, snap_num, 'BlackHoleMass') * 1.0e10 / Hubble_h
    StellarMass = read_hdf(file_list, snap_num, 'StellarMass') * 1.0e10 / Hubble_h
    Mvir = read_hdf(file_list, snap_num, 'Mvir') * 1.0e10 / Hubble_h

    if len(BlackHoleMass) == 0:
        sys.exit(1)

    bh_mask = (BlackHoleMass > 0) & (StellarMass > 10**MIN_STELLAR_MASS_LOG) & (Mvir > 10**MIN_HALO_MASS_LOG)
    print(f"Galaxies passing mass cuts at z=0: {np.sum(bh_mask)}")

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
        log_mvir_z0 = np.log10(Mvir + 1e-10)
        
        # Read across all files for IDs
        gal_id_z0 = []
        for f in file_list:
             with h5py.File(f, 'r') as hf:
                 if f"Snap_{snap_num}" in hf and id_field in hf[f"Snap_{snap_num}"]:
                     gal_id_z0.append(hf[f"Snap_{snap_num}"][id_field][:])
        gal_id_z0 = np.concatenate(gal_id_z0) if gal_id_z0 else None

        for i, (mmin, mmax) in enumerate(halo_bins):
            mask_z0 = bh_mask & (log_mvir_z0 >= mmin) & (log_mvir_z0 < mmax)
            if gal_id_z0 is not None:
                tracked_ids_per_bin.append(gal_id_z0[mask_z0])
            else:
                tracked_ids_per_bin.append(np.array([]))

    all_results = [[] for _ in halo_bins]

    print("\nExtracting data across snapshots...")
    for sn in all_snaps:
        bh = read_hdf(file_list, sn, 'BlackHoleMass') * 1.0e10 / Hubble_h
        if len(bh) == 0: continue
        z = all_redshifts[sn] if sn < len(all_redshifts) else None
        if z is None: continue

        def safe_read(field):
            arr = read_hdf(file_list, sn, field)
            return arr * 1.0e10 / Hubble_h if len(arr) > 0 else np.zeros(len(bh))

        md = safe_read('MergerDrivenBHaccretionMass')
        id_ = safe_read('InstabilityDrivenBHaccretionMass')
        rm = safe_read('RadioModeBHaccretionMass')
        bm = safe_read('BHMergerMass')
        
        stellar_mass = safe_read('StellarMass')
        mvir = safe_read('Mvir')
        log_mvir = np.log10(mvir + 1e-10)

        # Grab IDs
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
                    valid_idx = lookup_ids_vectorized(target_ids, gal_id_sn)
                    valid_idx = valid_idx[valid_idx >= 0]
            
            if len(valid_idx) == 0: continue

            bh_t = bh[valid_idx]
            md_t = md[valid_idx]
            id_t = id_[valid_idx]
            rm_t = rm[valid_idx]
            bm_t = bm[valid_idx]

            # Diagnostic print for one specific snap (z ~ 0.5) to see if values are 0
            if sn == all_snaps[-5] and i == 1:
                print(f"  [Diag Z={z:.2f} Bin 2] Medians -> BH: {np.median(bh_t):.1e}, RM: {np.median(rm_t):.1e}")

            def pct(x):
                # Filter exact zeros so log10 doesn't crash, but keep them for diagnostics
                valid_vals = x[x > 1e-6] 
                return np.percentile(valid_vals, [16, 50, 84]) if len(valid_vals) > 2 else [np.nan]*3

            all_results[i].append({
                'z': z,
                'bh': pct(bh_t),
                'md': pct(md_t),
                'id': pct(id_t),
                'rm': pct(rm_t),
                'bm': pct(bm_t),
            })

    digitised_data = read_digitised_txt(DIGITISED_TXT_FILENAME) if PLOT_DIGITISED_TXT else None

    # ================= PLOTTING =================
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

        def arr(key, j): return np.array([r[key][j] for r in results])

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
    plt.savefig(os.path.join(OutputDir, out_filename), bbox_inches='tight')
    print(f"\nSaved plot to {out_filename}")

if __name__ == '__main__':
    main()