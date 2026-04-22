#!/usr/bin/env python3
"""
Black hole growth tracking: SINGLE MODEL DIAGNOSTICS (TOTAL SUMS)

This script evaluates a single model output. It dynamically tracks galaxies 
backward in time using Vectorized ID tracking, and internally verifies 
mass conservation.

**MODIFICATION**: All plots have been converted to show Absolute Total Mass (Sums) 
instead of Medians, and outputs are saved to a specific subfolder.
"""
import numpy as np
import h5py
import glob
import os
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["figure.figsize"] = (18, 5)
plt.rcParams["figure.dpi"] = 96
plt.rcParams["font.size"] = 14
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'

OutputFormat = '.pdf'

# ============================================================
# USER OPTIONS
# ============================================================
USE_MASKS = False  # <--- SET TO False TO DISABLE MASS CUTS
MIN_STELLAR_MASS_LOG = 8.5
MIN_HALO_MASS_LOG = 11.0

# Options: "original", "all", "1-sigma", "iqr", "middle_20", "middle_10"
TRACKING_RANGE = "all"  

# Disabled because the digitised reference curves are Medians, not Sums!
PLOT_DIGITISED_TXT = False  

POSSIBLE_ID_FIELDS = ['GalaxyIndex', 'ID', 'galaxy_id', 'id', 'GalID']

def read_hdf(file_list, snap, field):
    data = []
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f"Snap_{int(snap)}"
            if snap_key in hf and field in hf[snap_key]:
                data.append(hf[snap_key][field][:])
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

def lookup_ids_vectorized(gal_ids_target, gal_ids_snapshot):
    sort_idx = np.argsort(gal_ids_snapshot)
    gal_ids_sorted = gal_ids_snapshot[sort_idx]
    indices = np.full(len(gal_ids_target), -1, dtype=int)
    for i, target_id in enumerate(gal_ids_target):
        pos = np.searchsorted(gal_ids_sorted, target_id)
        if pos < len(gal_ids_sorted) and gal_ids_sorted[pos] == target_id:
            indices[i] = sort_idx[pos]
    return indices

def process_single_dataset(input_pattern, snapshot=None):
    print(f"\n{'='*60}")
    print(f"PROCESSING MODEL: {input_pattern}")
    print(f"{'='*60}")
    
    file_list = sorted(glob.glob(input_pattern))
    if not file_list:
        print(f"Error: No files found matching: {input_pattern}")
        return None

    sim_params = read_simulation_params(file_list[0])
    Hubble_h = sim_params['Hubble_h']
    snap_num = snapshot if snapshot is not None else sim_params['latest_snapshot']
    
    id_field = find_id_field(file_list, snap_num)

    all_snaps = np.array(sim_params['available_snapshots'])
    all_redshifts = sim_params['snapshot_redshifts']

    range_options = {
        "original": (None, "Original Cross-Sectional"),
        "all": ([0, 50, 100], "All z=0 Valid Galaxies"),
        "1-sigma": ([16, 50, 84], "1-sigma (16%-84%)"),
        "iqr": ([25, 50, 75], "IQR (25%-75%)"),
        "middle_20": ([40, 50, 60], "Middle 20%"),
        "middle_10": ([45, 50, 55], "Middle 10%")
    }
    chosen_range = TRACKING_RANGE if TRACKING_RANGE in range_options else "original"
    pct_vals, range_name = range_options[chosen_range]
    
    halo_bins = [
        (11.5, 12.5), 
        (12.5, 13.5), 
        (13.5, 14.5), 
        (14.5, 15.5),
        (MIN_HALO_MASS_LOG, 99.0) # Bin 4 is the GLOBAL bin
    ]
    tracked_ids_per_bin = []

    if chosen_range != "original":
        z0_snap = all_snaps[np.argmin(all_redshifts[all_snaps])]
        bh_z0 = read_hdf(file_list, z0_snap, 'BlackHoleMass') * 1.0e10 / Hubble_h
        mvir_z0 = read_hdf(file_list, z0_snap, 'Mvir') * 1.0e10 / Hubble_h
        stellar_mass_z0 = read_hdf(file_list, z0_snap, 'StellarMass') * 1.0e10 / Hubble_h
        type_z0 = read_hdf(file_list, z0_snap, 'Type')
        log_mvir_z0 = np.log10(mvir_z0 + 1e-10)
        gal_id_z0 = read_hdf(file_list, z0_snap, id_field)

        for i, (mmin, mmax) in enumerate(halo_bins):
            if USE_MASKS:
                mask_z0 = (bh_z0 > 0) & (log_mvir_z0 >= mmin) & (log_mvir_z0 < mmax) \
                          & (stellar_mass_z0 > 10**MIN_STELLAR_MASS_LOG) & (type_z0 == 0)
            else:
                mask_z0 = (bh_z0 > 0) & (log_mvir_z0 >= mmin) & (log_mvir_z0 < mmax) & (type_z0 == 0)
                      
            if np.sum(mask_z0) > 0:
                bh_in_bin = bh_z0[mask_z0]
                p_lower, p50, p_upper = np.percentile(bh_in_bin, pct_vals)
                in_band = mask_z0 & (bh_z0 >= p_lower) & (bh_z0 <= p_upper)
                tracked_ids = gal_id_z0[in_band]
                tracked_ids_per_bin.append(tracked_ids)
            else:
                tracked_ids_per_bin.append(np.array([], dtype=gal_id_z0.dtype))

    all_results = [[] for _ in halo_bins]
    
    print(f"  Extracting historical data across snapshots...")
    for sn in all_snaps:
        bh = read_hdf(file_list, sn, 'BlackHoleMass') * 1.0e10 / Hubble_h
        if len(bh) == 0: continue
        z = all_redshifts[sn] if sn < len(all_redshifts) else None
        if z is None: continue

        def safe_read(field):
            arr = read_hdf(file_list, sn, field) * 1.0e10 / Hubble_h
            return arr if len(arr) > 0 else np.zeros(len(bh))

        qm = safe_read('QuasarModeBHaccretionMass')
        md = safe_read('MergerDrivenBHaccretionMass')
        id_ = safe_read('InstabilityDrivenBHaccretionMass')
        rm = safe_read('RadioModeBHaccretionMass')
        bm = safe_read('BHMergerMass')
        sd = safe_read('BHSeedMass')
        
        stellar_mass = safe_read('StellarMass')
        mvir = safe_read('Mvir')
        log_mvir = np.log10(mvir + 1e-10)
        gal_id_sn = read_hdf(file_list, sn, id_field) if id_field else np.arange(len(bh))

        for i, (mmin, mmax) in enumerate(halo_bins):
            if chosen_range == "original":
                if USE_MASKS:
                    mask = (bh > 0) & (log_mvir >= mmin) & (log_mvir < mmax) \
                           & (stellar_mass > 10**MIN_STELLAR_MASS_LOG)
                else:
                    mask = (bh > 0) & (log_mvir >= mmin) & (log_mvir < mmax)
                valid_idx = np.where(mask)[0]
            else:
                target_ids = tracked_ids_per_bin[i]
                if len(target_ids) == 0:
                    valid_idx = np.array([], dtype=int)
                else:
                    valid_idx = lookup_ids_vectorized(target_ids, gal_id_sn)
                    valid_idx = valid_idx[valid_idx >= 0]
                
            if len(valid_idx) == 0: continue

            bh_t = bh[valid_idx]
            qm_t = qm[valid_idx] if len(qm)>0 else np.zeros(len(valid_idx))
            md_t = md[valid_idx] if len(md)>0 else np.zeros(len(valid_idx))
            id_t = id_[valid_idx] if len(id_)>0 else np.zeros(len(valid_idx))
            rm_t = rm[valid_idx] if len(rm)>0 else np.zeros(len(valid_idx))
            bm_t = bm[valid_idx] if len(bm)>0 else np.zeros(len(valid_idx))
            sd_t = sd[valid_idx] if len(sd)>0 else np.zeros(len(valid_idx))

            # --- INTERNAL DIAGNOSTIC ARRAYS ---
            insitu_implied = bh_t - bm_t
            insitu_explicit = qm_t + rm_t + sd_t

            all_results[i].append({
                'z': z,
                'bh_sum': np.sum(bh_t),
                'md_sum': np.sum(md_t),
                'id_sum': np.sum(id_t),
                'rm_sum': np.sum(rm_t),
                'bm_sum': np.sum(bm_t),
                'qm_sum': np.sum(qm_t),
                'sd_sum': np.sum(sd_t),
                'insitu_implied_sum': np.sum(insitu_implied),
                'insitu_explicit_sum': np.sum(insitu_explicit)
            })

    return all_results, file_list, Hubble_h, sim_params

def generate_global_budget_table(files, snap_num, Hubble_h):
    """Generates a terminal table validating internal mass conservation at z=0"""
    print(f"\n{'='*85}")
    print(f"INTERNAL MASS BUDGET VALIDATION AT Z=0 (No Masks Applied)")
    print(f"{'='*85}")
    
    bh = read_hdf(files, snap_num, 'BlackHoleMass').sum() * 1.0e10 / Hubble_h
    qm = read_hdf(files, snap_num, 'QuasarModeBHaccretionMass').sum() * 1.0e10 / Hubble_h
    md = read_hdf(files, snap_num, 'MergerDrivenBHaccretionMass').sum() * 1.0e10 / Hubble_h
    id_ = read_hdf(files, snap_num, 'InstabilityDrivenBHaccretionMass').sum() * 1.0e10 / Hubble_h
    rm = read_hdf(files, snap_num, 'RadioModeBHaccretionMass').sum() * 1.0e10 / Hubble_h
    bm = read_hdf(files, snap_num, 'BHMergerMass').sum() * 1.0e10 / Hubble_h
    sd = read_hdf(files, snap_num, 'BHSeedMass').sum() * 1.0e10 / Hubble_h

    print(f"Total BH Mass              : {bh:.4e} M_sun")
    print(f"Quasar Mode                : {qm:.4e} M_sun")
    print(f"  -> Merger-Driven         : {md:.4e} M_sun")
    print(f"  -> Instab-Driven         : {id_:.4e} M_sun")
    print(f"Radio Mode                 : {rm:.4e} M_sun")
    print(f"BH-BH Mergers (Ex-Situ)    : {bm:.4e} M_sun")
    print(f"Seed Mass                  : {sd:.4e} M_sun")
    
    print("-" * 85)
    implied_insitu = bh - bm
    explicit_insitu = qm + rm + sd
    discrepancy = implied_insitu - explicit_insitu
    
    print(f"Implied In-Situ (Total-BM) : {implied_insitu:.4e} M_sun")
    print(f"Explicit In-Situ (Q+R+S)   : {explicit_insitu:.4e} M_sun")
    print(f"DISCREPANCY                : {discrepancy:.4e} M_sun")
    print("=" * 85 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Internal BH Growth Tracking (Absolute Sums)')
    parser.add_argument('-i', '--input-pattern', type=str, default='./output/millennium_insitu/model_*.hdf5', help='Path to model files')
    parser.add_argument('-o', '--output-dir', type=str, default='./plots', help='Output directory')
    args = parser.parse_args()

    # --- Create the new Subfolder for the plots ---
    os.makedirs(args.output_dir, exist_ok=True)
    OutSubDir = os.path.join(args.output_dir, 'total_mass_tracking')
    os.makedirs(OutSubDir, exist_ok=True)

    # Process the single dataset
    res_data, files, hubble_h, sim_params = process_single_dataset(args.input_pattern)

    if res_data is None:
        print("Failed to load dataset. Exiting.")
        sys.exit(1)

    snap_num = sim_params['latest_snapshot']
    generate_global_budget_table(files, snap_num, hubble_h)

    print(f"Generating plots in {OutSubDir}...")

    channels = [
        ('Merger-driven', 'md', '#2196F3'),
        ('Instability-driven', 'id', '#FF9800'),
        ('Radio mode', 'rm', '#9C27B0'),
        ('BH-BH mergers', 'bm', '#4CAF50'),
    ]
    
    bin_labels = [r"$\log_{10}(M_{h,0}) \sim 12$", r"$\log_{10}(M_{h,0}) \sim 13$", r"$\log_{10}(M_{h,0}) \sim 14$", r"$\log_{10}(M_{h,0}) \sim 15$"]

    # -------------------------------------------------------
    # PLOT 1: THE STANDARD 4-PANEL ABSOLUTE SUM PLOT
    # -------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

    for i, ax in enumerate(axes):
        ax.set_title(bin_labels[i])
        if i == 0: ax.set_ylabel(r'$\log_{10}(\Sigma M_{\rm BH}\,[M_\odot])$')
        ax.set_xlabel(r'Redshift ($z$)')

        data = sorted(res_data[i], key=lambda x: x['z'], reverse=True)

        if data:
            z_arr = np.array([r['z'] for r in data])
            
            # Plot Total BH Mass as reference
            sum_tot = np.array([r['bh_sum'] for r in data])
            valid_tot = (~np.isnan(sum_tot)) & (sum_tot > 0)
            if np.sum(valid_tot) > 1:
                ax.plot(z_arr[valid_tot], np.log10(sum_tot[valid_tot]), color='black', linestyle='-', linewidth=2.0, label='Total BH Mass')
            
            for label, key, color in channels:
                sum_val = np.array([r[f'{key}_sum'] for r in data])
                valid = (~np.isnan(sum_val)) & (sum_val > 0)
                if np.sum(valid) > 1:
                    ax.plot(z_arr[valid], np.log10(sum_val[valid]), color=color, linestyle='-', linewidth=2.0, label=label)

        ax.set_xlim(7.0, -0.2)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[-1].legend(by_label.values(), by_label.keys(), fontsize=10, loc='best')

    plt.suptitle(f"BH Growth Channels Absolute Sums (Tracking: {TRACKING_RANGE})", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OutSubDir, f'bh_growth_4panel_sum{OutputFormat}'), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------
    # PLOT 2: GLOBAL SUMS PLOT (No Halo Bins)
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Global Population Absolute Sum (All Halo Masses)")
    ax.set_ylabel(r'$\log_{10}(\Sigma M_{\rm BH}\,[M_\odot])$')
    ax.set_xlabel(r'Redshift ($z$)')
    
    data_glob = sorted(res_data[4], key=lambda x: x['z'], reverse=True) # Bin 4 is Global
    
    if data_glob:
        z_arr = np.array([r['z'] for r in data_glob])
        
        sum_tot = np.array([r['bh_sum'] for r in data_glob])
        valid = sum_tot > 0
        if np.sum(valid) > 1:
            ax.plot(z_arr[valid], np.log10(sum_tot[valid]), color='black', lw=2.0, label='Total BH Mass')
            
        for label, key, color in channels:
            sum_val = np.array([r[f'{key}_sum'] for r in data_glob])
            valid = sum_val > 0
            if np.sum(valid) > 1:
                ax.plot(z_arr[valid], np.log10(sum_val[valid]), color=color, lw=2.0, label=label)

    ax.set_xlim(7.0, -0.2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(OutSubDir, f'bh_growth_GLOBAL_sum{OutputFormat}'), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------
    # PLOT 3: INTERNAL IN-SITU ISOLATION DIAGNOSTIC (SUMS ONLY)
    # -------------------------------------------------------
    print("Generating Internal In-Situ Isolation Diagnostic Plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if data_glob:
        z = np.array([r['z'] for r in data_glob])
        
        # Total BH Mass
        val_bh = np.array([r['bh_sum'] for r in data_glob])
        valid = (~np.isnan(val_bh)) & (val_bh > 0)
        if np.sum(valid) > 1:
            ax.plot(z[valid], np.log10(val_bh[valid]), 'k-', lw=2, label='Total BH Mass')
        
        # Implied In-Situ (Total - Mergers)
        val_implied = np.array([r['insitu_implied_sum'] for r in data_glob])
        valid = (~np.isnan(val_implied)) & (val_implied > 0)
        if np.sum(valid) > 1:
            ax.plot(z[valid], np.log10(val_implied[valid]), 'b--', lw=3, label='Implied In-Situ (Total - BM)')

        # Explicit In-Situ (Q + R + S)
        val_explicit = np.array([r['insitu_explicit_sum'] for r in data_glob])
        valid = (~np.isnan(val_explicit)) & (val_explicit > 0)
        if np.sum(valid) > 1:
            ax.plot(z[valid], np.log10(val_explicit[valid]), 'r:', lw=2, label='Explicit In-Situ (Q+R+S)')
            
        # Ex-Situ (BM)
        val_bm = np.array([r['bm_sum'] for r in data_glob])
        valid = (~np.isnan(val_bm)) & (val_bm > 0)
        if np.sum(valid) > 1:
            ax.plot(z[valid], np.log10(val_bm[valid]), 'g-', lw=2, label='Ex-Situ (BH-BH Mergers)')

    ax.set_title("Global Absolute Sums (Isolation Diagnostic)")
    ax.set_ylabel(r'$\log_{10}(\Sigma M_{\rm BH}\,[M_\odot])$')
    ax.set_xlim(7.0, -0.2)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'Redshift ($z$)')
    ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(OutSubDir, f'bh_internal_insitu_isolation_sum{OutputFormat}'), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------
    # PLOT 4: IN-SITU DIFFERENCE PLOT (Implied vs Explicit)
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    if data_glob:
        z_arr = np.array([r['z'] for r in data_glob])

        sum_implied = np.array([r['insitu_implied_sum'] for r in data_glob])
        sum_explicit = np.array([r['insitu_explicit_sum'] for r in data_glob])
        valid_sum = (sum_implied > 0) & (sum_explicit > 0)
        
        if np.sum(valid_sum) > 1:
            # Difference in millions of M_sun
            delta_sum = (sum_implied[valid_sum] - sum_explicit[valid_sum]) / 1e6
            ax.plot(z_arr[valid_sum], delta_sum, 'b-', lw=2.5, label=r'$\Delta$ Sum (Implied - Explicit)')

    ax.axhline(0, color='k', linestyle='--', lw=1.5)
    ax.set_xlim(7.0, -0.2)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'Redshift ($z$)')
    ax.legend(loc='best')

    ax.set_title("Difference in Global Absolute Sums")
    ax.set_ylabel(r'Difference ($10^6\ M_\odot$) [Implied - Explicit]')

    # ---> ADD THIS LINE TO STOP THE AUTO-ZOOM <---
    ax.set_ylim(-1.0, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(OutSubDir, f'bh_internal_insitu_difference_sum{OutputFormat}'), bbox_inches='tight')
    plt.close()
    
    print("\nDone.")

if __name__ == '__main__':
    main()