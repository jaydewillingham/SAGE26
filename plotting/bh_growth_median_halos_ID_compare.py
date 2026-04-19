#!/usr/bin/env python3
"""
Black hole growth tracking: TOTAL vs IN-SITU COMPARISON & DIAGNOSTICS

This script evaluates both the original 'Total' tracking outputs and 
the new 'In-situ' outputs. 
It features GLOBAL diagnostic modes, DIFFERENCE plots, and an ISOLATION
plot to pinpoint exact mass conservation issues.
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

PLOT_DIGITISED_TXT = True
DIGITISED_TXT_FILENAME = "BH_mass_growth_refined_digitised.txt"

DIGITISED_COLOURS = {
    'Hot-mode': '#d65ad1',
    'Cold-mode': '#27dbe8',
    'Merger-driver': '#ff9900',
    'BHBH': '#2ca02c'
}

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

def read_digitised_txt(filepath):
    data = {}
    current_panel = None
    z_vals = None
    if not os.path.exists(filepath): return None
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

def process_dataset(name, input_pattern, is_insitu, snapshot=None):
    print(f"\n{'='*60}")
    print(f"PROCESSING DATASET: {name}")
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
    
    # ADDED A 5TH BIN FOR THE GLOBAL POPULATION
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

        md = safe_read('MergerDrivenBHaccretionMass')
        id_ = safe_read('InstabilityDrivenBHaccretionMass')
        rm = safe_read('RadioModeBHaccretionMass')
        bm = safe_read('BHMergerMass')
        
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
            md_t = md[valid_idx] if len(md)>0 else np.zeros(len(valid_idx))
            id_t = id_[valid_idx] if len(id_)>0 else np.zeros(len(valid_idx))
            rm_t = rm[valid_idx] if len(rm)>0 else np.zeros(len(valid_idx))
            bm_t = bm[valid_idx] if len(bm)>0 else np.zeros(len(valid_idx))

            # --- ISOLATION DIAGNOSTIC ARRAYS ---
            insitu_implied = bh_t - bm_t
            insitu_channels = md_t + id_t + rm_t

            def pct(x):
                valid = x[x > 0]
                return np.percentile(valid, [16, 50, 84]) if len(valid) > 0 else [np.nan]*3

            # Calculate sums and medians for the tracked group
            all_results[i].append({
                'z': z,
                'bh': pct(bh_t), 'bh_sum': np.sum(bh_t),
                'md': pct(md_t), 'md_sum': np.sum(md_t),
                'id': pct(id_t), 'id_sum': np.sum(id_t),
                'rm': pct(rm_t), 'rm_sum': np.sum(rm_t),
                'bm': pct(bm_t), 'bm_sum': np.sum(bm_t),
                'insitu_implied': pct(insitu_implied), 'insitu_implied_sum': np.sum(insitu_implied),
                'insitu_channels': pct(insitu_channels), 'insitu_channels_sum': np.sum(insitu_channels)
            })

    return all_results, file_list, Hubble_h, sim_params

def generate_global_budget_table(res_total_files, res_insitu_files, snap_num, Hubble_h):
    """Generates a side-by-side terminal table comparing all mass channels globally at z=0"""
    print(f"\n{'='*85}")
    print(f"GLOBAL MASS BUDGET COMPARISON AT Z=0 (No Masks Applied)")
    print(f"{'='*85}")
    
    def get_sums(files):
        bh = read_hdf(files, snap_num, 'BlackHoleMass').sum() * 1.0e10 / Hubble_h
        qm = read_hdf(files, snap_num, 'QuasarModeBHaccretionMass').sum() * 1.0e10 / Hubble_h
        md = read_hdf(files, snap_num, 'MergerDrivenBHaccretionMass').sum() * 1.0e10 / Hubble_h
        id_ = read_hdf(files, snap_num, 'InstabilityDrivenBHaccretionMass').sum() * 1.0e10 / Hubble_h
        rm = read_hdf(files, snap_num, 'RadioModeBHaccretionMass').sum() * 1.0e10 / Hubble_h
        bm = read_hdf(files, snap_num, 'BHMergerMass').sum() * 1.0e10 / Hubble_h
        sd = read_hdf(files, snap_num, 'BHSeedMass').sum() * 1.0e10 / Hubble_h
        return bh, qm, md, id_, rm, bm, sd

    tot_bh, tot_qm, tot_md, tot_id, tot_rm, tot_bm, tot_sd = get_sums(res_total_files)
    ins_bh, ins_qm, ins_md, ins_id, ins_rm, ins_bm, ins_sd = get_sums(res_insitu_files)

    print(f"{'Channel':<26} | {'TOTAL (Old) [Msun]':<18} | {'IN-SITU (New) [Msun]':<18} | {'Diff (Tot - Ins)':<18}")
    print("-" * 85)
    print(f"{'Total BH Mass':<26} | {tot_bh:<18.4e} | {ins_bh:<18.4e} | {tot_bh - ins_bh:<18.4e}")
    print(f"{'Quasar Mode':<26} | {tot_qm:<18.4e} | {ins_qm:<18.4e} | {tot_qm - ins_qm:<18.4e}")
    print(f"{'  -> Merger-Driven':<26} | {tot_md:<18.4e} | {ins_md:<18.4e} | {tot_md - ins_md:<18.4e}")
    print(f"{'  -> Instab-Driven':<26} | {tot_id:<18.4e} | {ins_id:<18.4e} | {tot_id - ins_id:<18.4e}")
    print(f"{'Radio Mode':<26} | {tot_rm:<18.4e} | {ins_rm:<18.4e} | {tot_rm - ins_rm:<18.4e}")
    print(f"{'BH-BH Mergers':<26} | {tot_bm:<18.4e} | {ins_bm:<18.4e} | {tot_bm - ins_bm:<18.4e}")
    print(f"{'Seed Mass':<26} | {tot_sd:<18.4e} | {ins_sd:<18.4e} | {tot_sd - ins_sd:<18.4e}")
    print("-" * 85)
    
    tot_sum = tot_qm + tot_rm + tot_sd
    ins_sum = ins_qm + ins_rm + ins_sd + ins_bm
    print(f"{'Growth Budget SUM':<26} | {tot_sum:<18.4e} | {ins_sum:<18.4e} | {tot_sum - ins_sum:<18.4e}")
    
    # =======================================================
    # ISOLATING THE IN-SITU DISCREPANCY & ADDITIONAL CHECKS
    # =======================================================
    print("-" * 85)
    
    # In the old code, true in-situ growth is Total Mass minus the mergers it underwent
    old_implied_insitu = tot_bh - tot_bm
    
    # In the new code, Quasar + Radio + Seed strictly represents in-situ growth
    new_explicit_insitu = ins_qm + ins_rm + ins_sd
    discrepancy = old_implied_insitu - new_explicit_insitu
    
    # Additional requested checks
    tot_qr = tot_qm + tot_rm
    ins_qr = ins_qm + ins_rm
    
    tot_bh_minus_qr = tot_bh - tot_qm - tot_rm
    ins_bh_minus_qr = ins_bh - ins_qm - ins_rm
    
    print(f"{'Old Implied In-Situ (BH-BM)':<26} | {old_implied_insitu:<18.4e} | {'-':<18} | {'-':<18}")
    print(f"{'New Explicit In-Situ (Q+R+S)':<26} | {'-':<18} | {new_explicit_insitu:<18.4e} | {'-':<18}")
    print(f"{'DISCREPANCY (Old - New)':<26} | {'-':<18} | {'-':<18} | {discrepancy:<18.4e}")
    print("-" * 85)
    print(f"{'Quasar + Radio (Q+R)':<26} | {tot_qr:<18.4e} | {ins_qr:<18.4e} | {tot_qr - ins_qr:<18.4e}")
    print(f"{'Total - Quasar - Radio':<26} | {tot_bh_minus_qr:<18.4e} | {ins_bh_minus_qr:<18.4e} | {tot_bh_minus_qr - ins_bh_minus_qr:<18.4e}")
    print("=" * 85 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Compare Total vs In-Situ BH Growth')
    parser.add_argument('--total', type=str, default='./output/millennium/model_*.hdf5', help='Path to Total tracking files')
    parser.add_argument('--insitu', type=str, default='./output/millennium_insitu/model_*.hdf5', help='Path to In-situ tracking files')
    parser.add_argument('-o', '--output-dir', type=str, default='./plots', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Process both datasets
    res_total, f_tot, h_tot, sim_params = process_dataset("TOTAL (Original)", args.total, is_insitu=False)
    res_insitu, f_ins, h_ins, _ = process_dataset("IN-SITU (Modified)", args.insitu, is_insitu=True)

    if res_total is None or res_insitu is None:
        print("Failed to load one or both datasets. Exiting.")
        sys.exit(1)

    snap_num = sim_params['latest_snapshot']
    generate_global_budget_table(f_tot, f_ins, snap_num, h_tot)

    print(f"Generating plots in {args.output_dir}...")
    digitised_data = read_digitised_txt(DIGITISED_TXT_FILENAME) if PLOT_DIGITISED_TXT else None

    channels = [
        ('Merger-driven', 'md', '#2196F3'),
        ('Instability-driven', 'id', '#FF9800'),
        ('Radio mode', 'rm', '#9C27B0'),
        ('BH-BH mergers', 'bm', '#4CAF50'),
    ]

    custom_lines = [Line2D([0], [0], color=c, lw=2) for _, _, c in channels]
    custom_labels = [l for l, _, _ in channels]
    custom_lines.extend([Line2D([0], [0], color='k', linestyle='-', lw=2), Line2D([0], [0], color='k', linestyle='--', lw=2.5)])
    custom_labels.extend(['TOTAL (Solid)', 'IN-SITU (Dashed)'])

    bin_labels = [r"$\log_{10}(M_{h,0}) \sim 12$", r"$\log_{10}(M_{h,0}) \sim 13$", r"$\log_{10}(M_{h,0}) \sim 14$", r"$\log_{10}(M_{h,0}) \sim 15$"]

    # -------------------------------------------------------
    # PLOT 1: THE STANDARD 4-PANEL MEDIAN PLOT
    # -------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

    for i, ax in enumerate(axes):
        ax.set_title(bin_labels[i])
        if i == 0: ax.set_ylabel(r'$\log_{10}(M_{\rm BH}\,[M_\odot])$')
        ax.set_xlabel(r'Redshift ($z$)')

        data_tot = sorted(res_total[i], key=lambda x: x['z'], reverse=True)
        data_ins = sorted(res_insitu[i], key=lambda x: x['z'], reverse=True)

        if data_tot:
            z_tot = np.array([r['z'] for r in data_tot])
            for label, key, color in channels:
                p16 = np.array([r[key][0] for r in data_tot])
                p50 = np.array([r[key][1] for r in data_tot])
                p84 = np.array([r[key][2] for r in data_tot])
                valid = ~np.isnan(p50) & (p50 > 0)
                if np.sum(valid) > 1:
                    ax.plot(z_tot[valid], np.log10(p50[valid]), color=color, linestyle='-', linewidth=2.0)
                    ax.fill_between(z_tot[valid], np.log10(p16[valid]), np.log10(p84[valid]), color=color, alpha=0.15)

        if data_ins:
            z_ins = np.array([r['z'] for r in data_ins])
            for label, key, color in channels:
                p50 = np.array([r[key][1] for r in data_ins])
                valid = ~np.isnan(p50) & (p50 > 0)
                if np.sum(valid) > 1:
                    ax.plot(z_ins[valid], np.log10(p50[valid]), color=color, linestyle='--', linewidth=2.5)

        if digitised_data:
            panel_key = ['logMh0~12', 'logMh0~13', 'logMh0~14', 'logMh0~15'][i]
            if panel_key in digitised_data:
                for channel, entry in digitised_data[panel_key].items():
                    if channel in DIGITISED_COLOURS:
                        ax.plot(entry['z'], entry['y'], color=DIGITISED_COLOURS[channel], linestyle=':', linewidth=1.5, alpha=0.7)

        ax.set_xlim(7.0, -0.2)
        ax.grid(True, alpha=0.3)

    axes[-1].legend(custom_lines, custom_labels, fontsize=10, loc='best')
    plt.suptitle("BH Growth Comparison: Total vs In-Situ", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'bh_comparison_4panel_median{OutputFormat}'), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------
    # PLOT 2: THE 4-PANEL DIFFERENCE PLOT
    # -------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

    for i, ax in enumerate(axes):
        ax.set_title(bin_labels[i])
        if i == 0: ax.set_ylabel(r'$\Delta \log_{10}(M_{\rm BH})$ (Total - In-Situ)')
        ax.set_xlabel(r'Redshift ($z$)')

        data_tot = sorted(res_total[i], key=lambda x: x['z'], reverse=True)
        data_ins = sorted(res_insitu[i], key=lambda x: x['z'], reverse=True)

        if not data_tot or not data_ins:
            ax.set_xlim(7.0, -0.2)
            continue

        z_tot = np.array([r['z'] for r in data_tot])
        z_ins = np.array([r['z'] for r in data_ins])

        if len(z_tot) == len(z_ins) and np.allclose(z_tot, z_ins):
            for label, key, color in channels:
                p50_tot = np.array([r[key][1] for r in data_tot])
                p50_ins = np.array([r[key][1] for r in data_ins])
                valid = (~np.isnan(p50_tot) & ~np.isnan(p50_ins) & (p50_tot > 0) & (p50_ins > 0))

                if np.sum(valid) > 1:
                    delta = np.log10(p50_tot[valid]) - np.log10(p50_ins[valid])
                    ax.plot(z_tot[valid], delta, color=color, linewidth=2.0)

        ax.axhline(0, color='k', linestyle='--', linewidth=1)
        ax.set_xlim(7.0, -0.2)
        ax.grid(True, alpha=0.3)

    custom_lines_diff = [Line2D([0], [0], color=c, lw=2) for _, _, c in channels]
    custom_labels_diff = [l for l, _, _ in channels]
    axes[-1].legend(custom_lines_diff, custom_labels_diff, fontsize=10, loc='best')

    plt.suptitle("Missing Mass: Log Ratio of Total to In-Situ (Halo Bins)", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'bh_comparison_4panel_difference{OutputFormat}'), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------
    # PLOT 3: GLOBAL MEDIAN PLOT (No Halo Bins)
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Global Population Median (All Halo Masses)")
    ax.set_ylabel(r'$\log_{10}(M_{\rm BH}\,[M_\odot])$')
    ax.set_xlabel(r'Redshift ($z$)')

    data_tot = sorted(res_total[4], key=lambda x: x['z'], reverse=True) # Bin 4 is Global
    data_ins = sorted(res_insitu[4], key=lambda x: x['z'], reverse=True)

    if data_tot:
        z_tot = np.array([r['z'] for r in data_tot])
        for label, key, color in channels:
            p16 = np.array([r[key][0] for r in data_tot])
            p50 = np.array([r[key][1] for r in data_tot])
            p84 = np.array([r[key][2] for r in data_tot])
            valid = ~np.isnan(p50) & (p50 > 0)
            if np.sum(valid) > 1:
                ax.plot(z_tot[valid], np.log10(p50[valid]), color=color, linestyle='-', linewidth=2.0)
                ax.fill_between(z_tot[valid], np.log10(p16[valid]), np.log10(p84[valid]), color=color, alpha=0.15)

    if data_ins:
        z_ins = np.array([r['z'] for r in data_ins])
        for label, key, color in channels:
            p50 = np.array([r[key][1] for r in data_ins])
            valid = ~np.isnan(p50) & (p50 > 0)
            if np.sum(valid) > 1:
                ax.plot(z_ins[valid], np.log10(p50[valid]), color=color, linestyle='--', linewidth=2.5)

    ax.set_xlim(7.0, -0.2)
    ax.grid(True, alpha=0.3)
    ax.legend(custom_lines, custom_labels, fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'bh_comparison_GLOBAL_median{OutputFormat}'), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------
    # PLOT 4: GLOBAL ABSOLUTE SUM PLOT (Diagnosis)
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Global Absolute Sum in Simulation Box (Conservation Check)")
    ax.set_ylabel(r'$\log_{10}(\Sigma M_{\rm BH}\,[M_\odot])$')
    ax.set_xlabel(r'Redshift ($z$)')

    if data_tot:
        z_tot = np.array([r['z'] for r in data_tot])
        for label, key, color in channels:
            sum_val = np.array([r[f'{key}_sum'] for r in data_tot])
            valid = sum_val > 0
            if np.sum(valid) > 1:
                ax.plot(z_tot[valid], np.log10(sum_val[valid]), color=color, linestyle='-', linewidth=2.0)

    if data_ins:
        z_ins = np.array([r['z'] for r in data_ins])
        for label, key, color in channels:
            sum_val = np.array([r[f'{key}_sum'] for r in data_ins])
            valid = sum_val > 0
            if np.sum(valid) > 1:
                ax.plot(z_ins[valid], np.log10(sum_val[valid]), color=color, linestyle='--', linewidth=2.5)

    ax.set_xlim(7.0, -0.2)
    ax.grid(True, alpha=0.3)
    ax.legend(custom_lines, custom_labels, fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'bh_comparison_GLOBAL_sum{OutputFormat}'), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------
    # PLOT 5: IN-SITU ISOLATION DIAGNOSTIC
    # -------------------------------------------------------
    print("Generating In-Situ Isolation Diagnostic Plot...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    def plot_diagnostic(ax, is_sum):
        key_suffix = '_sum' if is_sum else ''
        
        if data_tot:
            z = np.array([r['z'] for r in data_tot])
            
            # Old: Total BH Mass
            #val_bh = np.array([r['bh' + key_suffix] if is_sum else r['bh'][1] for r in data_tot])
            #valid = (~np.isnan(val_bh)) & (val_bh > 0)
            #ax.plot(z[valid], np.log10(val_bh[valid]), 'k-', lw=2, label='OLD: Total BH Mass')
            
            # Old: Total - Merger (Implied In-Situ)
            val_insitu = np.array([r['insitu_implied' + key_suffix] if is_sum else r['insitu_implied'][1] for r in data_tot])
            valid = (~np.isnan(val_insitu)) & (val_insitu > 0)
            ax.plot(z[valid], np.log10(val_insitu[valid]), 'b-', lw=2, label='OLD: Total - Merger (Implied In-Situ)')

        if data_ins:
            z = np.array([r['z'] for r in data_ins])
            
            # New: Total BH Mass
            #val_bh = np.array([r['bh' + key_suffix] if is_sum else r['bh'][1] for r in data_ins])
            #valid = (~np.isnan(val_bh)) & (val_bh > 0)
            #ax.plot(z[valid], np.log10(val_bh[valid]), 'k--', lw=2.5, label='NEW: Total BH Mass')

            # New: Total - Merger
            #val_insitu = np.array([r['insitu_implied' + key_suffix] if is_sum else r['insitu_implied'][1] for r in data_ins])
            #valid = (~np.isnan(val_insitu)) & (val_insitu > 0)
            #ax.plot(z[valid], np.log10(val_insitu[valid]), 'c--', lw=2.5, label='NEW: Total - Merger')

            # New: In-Situ Channels (Q+R)
            val_channels = np.array([r['insitu_channels' + key_suffix] if is_sum else r['insitu_channels'][1] for r in data_ins])
            valid = (~np.isnan(val_channels)) & (val_channels > 0)
            ax.plot(z[valid], np.log10(val_channels[valid]), 'r:', lw=3, label='NEW: In-Situ Channels Sum (Q+R)')
            
        ax.set_xlim(7.0, -0.2)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r'Redshift ($z$)')

    axes[0].set_title("Global Absolute Sums")
    axes[0].set_ylabel(r'$\log_{10}(\Sigma M_{\rm BH}\,[M_\odot])$')
    axes[0].set_ylim(7.0, 11.0)
    plot_diagnostic(axes[0], is_sum=True)
    
    axes[1].set_title("Global Medians")
    axes[1].set_ylabel(r'$\log_{10}(\widetilde{M}_{\rm BH}\,[M_\odot])$')
    axes[1].set_ylim(5.5, 6.4)
    plot_diagnostic(axes[1], is_sum=False)
    axes[1].legend(fontsize=9, loc='best')

    plt.tight_layout()
    out_filename = f'bh_comparison_GLOBAL_insitu_isolation{OutputFormat}'
    plt.savefig(os.path.join(args.output_dir, out_filename), bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {out_filename}")

    # -------------------------------------------------------
    # PLOT 5: IN-SITU DIFFERENCE PLOT (Old Implied In-Situ vs New In-Situ)
    # -------------------------------------------------------
    print("Generating In-Situ Difference Plot (Old vs New)...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if data_tot and data_ins:
        z_tot = np.array([r['z'] for r in data_tot])
        z_ins = np.array([r['z'] for r in data_ins])

        if len(z_tot) == len(z_ins) and np.allclose(z_tot, z_ins):
            # Panel 1: Sums Difference
            tot_sum = np.array([r['insitu_implied_sum'] for r in data_tot])
            ins_sum = np.array([r['insitu_implied_sum'] for r in data_ins])
            valid_sum = (tot_sum > 0) & (ins_sum > 0)
            
            if np.sum(valid_sum) > 1:
                delta_sum = (tot_sum[valid_sum] - ins_sum[valid_sum]) / 1e6  # Divide by 1 million
                axes[0].plot(z_tot[valid_sum], delta_sum, 'b-', lw=2.5, label=r'$\Delta$ Sum (Old - New)')

            # Panel 2: Medians Difference
            tot_med = np.array([r['insitu_implied'][1] for r in data_tot])
            ins_med = np.array([r['insitu_implied'][1] for r in data_ins])
            valid_med = (~np.isnan(tot_med)) & (~np.isnan(ins_med)) & (tot_med > 0) & (ins_med > 0)
            
            if np.sum(valid_med) > 1:
                delta_med = np.log10(tot_med[valid_med]) - np.log10(ins_med[valid_med])
                axes[1].plot(z_tot[valid_med], delta_med, 'b-', lw=2.5, label=r'$\Delta$ Median (Old - New)')

    for ax in axes:
        ax.axhline(0, color='k', linestyle='--', lw=1.5)
        ax.set_xlim(7.0, -0.2)
        ax.set_ylim(-0.000001, 0.000001)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r'Redshift ($z$)')
        ax.legend(loc='best')

    axes[0].set_title("Difference in Global Absolute Sums")
    axes[0].set_ylabel(r'Difference ($10^6\ M_\odot$)')
    
    axes[1].set_title("Difference in Global Medians")
    axes[1].set_ylabel(r'$\Delta \log_{10}(\widetilde{M}_{\rm In-Situ})$')

    plt.suptitle("In-Situ Discrepancy: Old Implied In-Situ vs New In-Situ", y=1.05)
    plt.tight_layout()
    out_filename = f'bh_comparison_GLOBAL_insitu_difference{OutputFormat}'
    plt.savefig(os.path.join(args.output_dir, out_filename), bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {out_filename}")
    print("Done.")

if __name__ == '__main__':
    main()