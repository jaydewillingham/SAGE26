#!/usr/bin/env python3
"""
Black hole growth tracking: validates that the three accretion channels
(quasar mode, radio mode, BH-BH mergers) sum to the total BlackHoleMass,
and plots the MEDIAN relative contributions across galaxies.

NOTE: This version applies strict cuts: 
- log10(M*) > 8.5
- log10(M_halo) > 11.0

It includes a universal toggle to either use the Original Cross-Sectional
method, or track a specific statistical sample of galaxies back from z=0.
"""
import numpy as np
import h5py
import glob
import os
import sys
import argparse
import matplotlib.pyplot as plt

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

# Choose how the script evaluates galaxies across time:
#   "original"  -> Recreates the original script: evaluates ALL active galaxies at EACH snapshot independently (No z=0 tracking)
#   "all"       -> Tracks 100% of the valid z=0 galaxies backward in time
#   "1-sigma"   -> Tracks the middle 68% (16th to 84th percentile of z=0 BH mass)
#   "iqr"       -> Tracks the middle 50% (25th to 75th percentile)
#   "middle_20" -> Tracks the middle 20% (40th to 60th percentile)
#   "middle_10" -> Tracks the middle 10% (45th to 55th percentile - Very tight to median)

TRACKING_RANGE = "all"  # <--- SET YOUR CHOICE HERE

PLOT_DIGITISED_TXT = True
DIGITISED_TXT_FILENAME = "BH_mass_growth_refined_digitised.txt"

DIGITISED_COLOURS = {
    'Hot-mode': '#d65ad1',         # magenta
    'Cold-mode': '#27dbe8',        # cyan
    'Merger-driver': '#ff9900',    # orange
    'BHBH': '#2ca02c'              # green
}

def read_hdf(file_list, snap, field):
    """Read a field from multiple HDF5 files."""
    data = []
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f"Snap_{int(snap)}"
            if snap_key in hf and field in hf[snap_key]:
                data.append(hf[snap_key][field][:])
    return np.concatenate(data) if data else np.array([])

def read_simulation_params(filepath):
    """Read simulation parameters from HDF5 header."""
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
    """Reads digitised BH growth curves from the provided TXT file."""
    data = {}
    current_panel = None
    z_vals = None
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

def main():
    parser = argparse.ArgumentParser(description='Black hole growth tracking validation and plots')
    parser.add_argument('-i', '--input-pattern', type=str, default='./output/millennium/model_*.hdf5')
    parser.add_argument('-s', '--snapshot', type=int, default=None)
    parser.add_argument('-o', '--output-dir', type=str, default=None)
    parser.add_argument('--validation-snap', type=int, default=None)
    args = parser.parse_args()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"Error: No files found matching: {args.input_pattern}")
        sys.exit(1)

    print(f"Found {len(file_list)} model files.")
    sim_params = read_simulation_params(file_list[0])
    Hubble_h = sim_params['Hubble_h']

    snap_num = args.snapshot if args.snapshot is not None else sim_params['latest_snapshot']
    print(f"Using snapshot: {snap_num}")
    
    validation_snap = args.validation_snap if args.validation_snap is not None else snap_num

    if 'snapshot_redshifts' in sim_params and snap_num < len(sim_params['snapshot_redshifts']):
        print(f"Redshift: {sim_params['snapshot_redshifts'][snap_num]:.4f}")

    OutputDir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(os.path.abspath(file_list[0])), 'plots')
    os.makedirs(OutputDir, exist_ok=True)

    # Read data
    print("Reading black hole data...")
    BlackHoleMass = read_hdf(file_list, snap_num, 'BlackHoleMass') * 1.0e10 / Hubble_h
    QuasarMode = read_hdf(file_list, snap_num, 'QuasarModeBHaccretionMass') * 1.0e10 / Hubble_h
    MergerDriven = read_hdf(file_list, snap_num, 'MergerDrivenBHaccretionMass') * 1.0e10 / Hubble_h
    InstabilityDriven = read_hdf(file_list, snap_num, 'InstabilityDrivenBHaccretionMass') * 1.0e10 / Hubble_h
    TorqueDriven = read_hdf(file_list, snap_num, 'TorqueDrivenBHaccretionMass') * 1.0e10 / Hubble_h
    SeedModeAccretion = read_hdf(file_list, snap_num, 'SeedModeBHaccretionMass') * 1.0e10 / Hubble_h
    RadioMode = read_hdf(file_list, snap_num, 'RadioModeBHaccretionMass') * 1.0e10 / Hubble_h
    BHMerger = read_hdf(file_list, snap_num, 'BHMergerMass') * 1.0e10 / Hubble_h
    BHSeedMass = read_hdf(file_list, snap_num, 'BHSeedMass') * 1.0e10 / Hubble_h
    StellarMass = read_hdf(file_list, snap_num, 'StellarMass') * 1.0e10 / Hubble_h
    BulgeMass = read_hdf(file_list, snap_num, 'BulgeMass') * 1.0e10 / Hubble_h
    Mvir = read_hdf(file_list, snap_num, 'Mvir') * 1.0e10 / Hubble_h

    if len(BlackHoleMass) == 0:
        print("No galaxies found!")
        sys.exit(1)

    ngal = len(BlackHoleMass)
    if len(TorqueDriven) == 0: TorqueDriven = np.zeros(ngal)
    if len(SeedModeAccretion) == 0: SeedModeAccretion = np.zeros(ngal)
    if len(BHSeedMass) == 0: BHSeedMass = np.zeros(ngal)

    # --- STELLAR AND HALO MASS CUTS APPLIED HERE ---
    bh_mask = (BlackHoleMass > 0) & (StellarMass > 10**MIN_STELLAR_MASS_LOG) & (Mvir > 10**MIN_HALO_MASS_LOG)
    n_bh = np.sum(bh_mask)
    print(f"Galaxies with BH, log10(M*) > {MIN_STELLAR_MASS_LOG}, & log10(Mh) > {MIN_HALO_MASS_LOG}: {n_bh} ({100*n_bh/len(BlackHoleMass):.1f}%)")
    
    if n_bh > 0:
        log_stellar = np.log10(StellarMass[bh_mask])
        log_halo = np.log10(Mvir[bh_mask])
        print(f"  --> Minimum log10(M*) of sample:  {np.min(log_stellar):.2f}")
        print(f"  --> Minimum log10(Mh) of sample: {np.min(log_halo):.2f}")

    # ===================== VALIDATION =====================
    print("\n" + "="*60)
    print("VALIDATION: Channel sum vs BlackHoleMass (Filtered Galaxies)")
    print("="*60)

    growth_sum = QuasarMode + RadioMode + BHSeedMass
    residual = BlackHoleMass - growth_sum

    if n_bh > 0:
        bh = BlackHoleMass[bh_mask]
        gs = growth_sum[bh_mask]
        res = residual[bh_mask]
        frac_res = res / bh

        print(f"\nBlackHoleMass total:  {bh.sum():.6e} M_sun")
        print(f"Growth sum total:     {gs.sum():.6e} M_sun")
        print(f"  Quasar mode total:  {QuasarMode[bh_mask].sum():.6e} M_sun")
        print(f"    Merger-driven:    {MergerDriven[bh_mask].sum():.6e} M_sun")
        print(f"    Instability:      {InstabilityDriven[bh_mask].sum():.6e} M_sun")
        print(f"  Radio mode:         {RadioMode[bh_mask].sum():.6e} M_sun")
        
        bad = np.abs(frac_res) > 0.01
        if np.sum(bad) > 0:
            print(f"\n  PASS/FAIL: WARNING — {np.sum(bad)} galaxies have >1% residual")
        else:
            print(f"\n  PASS: All galaxies have <1% residual")

    # ===================== PLOTS =====================
    print(f"\nGenerating plots in {OutputDir}...")

    # ================= MULTI HALO MASS PLOT =================
    all_snaps = np.array(sim_params['available_snapshots'])
    all_redshifts = sim_params['snapshot_redshifts']

    range_options = {
        "original": (None, "Original Cross-Sectional (Independent at each z)"),
        "all": ([0, 50, 100], "All z=0 Valid Galaxies (0%-100%)"),
        "1-sigma": ([16, 50, 84], "1-sigma (16%-84%)"),
        "iqr": ([25, 50, 75], "IQR (25%-75%)"),
        "middle_20": ([40, 50, 60], "Middle 20% (40%-60%)"),
        "middle_10": ([45, 50, 55], "Middle 10% (45%-55%)")
    }
    
    chosen_range = TRACKING_RANGE if TRACKING_RANGE in range_options else "original"
    pct_vals, range_name = range_options[chosen_range]
    
    halo_bins = [
        (11.5, 12.5),
        (12.5, 13.5),
        (13.5, 14.5),
        (14.5, 15.5),
    ]
    bin_labels = [
        r"$\log_{10}(M_{h,0}) \sim 12\,M_\odot$",
        r"$\log_{10}(M_{h,0}) \sim 13\,M_\odot$",
        r"$\log_{10}(M_{h,0}) \sim 14\,M_\odot$",
        r"$\log_{10}(M_{h,0}) \sim 15\,M_\odot$",
    ]

    tracked_indices_per_bin = []

    # --- Step 1: Define Samples at z=0 (If tracking is enabled) ---
    if chosen_range != "original":
        print(f"  Defining {range_name} samples based on Total BH Mass at z=0...")
        z0_snap = all_snaps[np.argmin(all_redshifts[all_snaps])]
        bh_z0 = read_hdf(file_list, z0_snap, 'BlackHoleMass') * 1.0e10 / Hubble_h
        mvir_z0 = read_hdf(file_list, z0_snap, 'Mvir') * 1.0e10 / Hubble_h
        stellar_mass_z0 = read_hdf(file_list, z0_snap, 'StellarMass') * 1.0e10 / Hubble_h
        log_mvir_z0 = np.log10(mvir_z0 + 1e-10)

        for i, (mmin, mmax) in enumerate(halo_bins):
            # 1. Mask galaxies by criteria
            mask_z0 = (bh_z0 > 0) & (log_mvir_z0 >= mmin) & (log_mvir_z0 < mmax) \
                      & (stellar_mass_z0 > 10**MIN_STELLAR_MASS_LOG) \
                      & (log_mvir_z0 > MIN_HALO_MASS_LOG)
                      
            if np.sum(mask_z0) > 0:
                bh_in_bin = bh_z0[mask_z0]
                p_lower, p50, p_upper = np.percentile(bh_in_bin, pct_vals)
                
                in_band = mask_z0 & (bh_z0 >= p_lower) & (bh_z0 <= p_upper)
                indices = np.where(in_band)[0]
                tracked_indices_per_bin.append(indices)
                print(f"    Bin {i+1}: BH Mass range [{p_lower:.2e} - {p_upper:.2e}], tracking {len(indices)} galaxies.")
            else:
                tracked_indices_per_bin.append(np.array([], dtype=int))
                print(f"    Bin {i+1}: 0 galaxies found matching cuts.")
    else:
        print(f"  Using Original Cross-Sectional method (evaluating populations independently at each snapshot)...")

    # --- Step 2: Extract Data Across Time ---
    all_results = [[] for _ in halo_bins]
    
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

        for i, (mmin, mmax) in enumerate(halo_bins):
            
            if chosen_range == "original":
                # Independent snapshot evaluation (The Original Method)
                mask = (bh > 0) & (log_mvir >= mmin) & (log_mvir < mmax) \
                       & (stellar_mass > 10**MIN_STELLAR_MASS_LOG) \
                       & (log_mvir > MIN_HALO_MASS_LOG)
                valid_idx = np.where(mask)[0]
            else:
                # Tracing exact row indices from z=0
                idx = tracked_indices_per_bin[i]
                valid_idx = idx[idx < len(bh)]
                
            if len(valid_idx) == 0: continue

            # Extract data for the valid galaxies
            bh_t = bh[valid_idx]
            md_t = md[valid_idx]
            id_t = id_[valid_idx]
            rm_t = rm[valid_idx]
            bm_t = bm[valid_idx]

            def pct(x):
                valid = x[x > 0]
                return np.percentile(valid, [16, 50, 84]) if len(valid) > 0 else [np.nan]*3

            all_results[i].append({
                'z': z,
                'bh': pct(bh_t),
                'md': pct(md_t),
                'id': pct(id_t),
                'rm': pct(rm_t),
                'bm': pct(bm_t),
            })

    # ------------------------------------------------------------
    # Load digitised TXT data (if requested)
    # ------------------------------------------------------------
    digitised_data = None
    if PLOT_DIGITISED_TXT:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        txt_path = os.path.join(script_dir, DIGITISED_TXT_FILENAME)
        if os.path.exists(txt_path):
            digitised_data = read_digitised_txt(txt_path)

    # ================= PLOTTING =================
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

    channels = [
        #('Total BH Mass', 'bh', 'black', '-', 2.0),
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

        if len(results) < 2:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.set_xlim(7.0, -0.2)
            continue

        snap_z = np.array([r['z'] for r in results])
        def arr(key, j): return np.array([r[key][j] for r in results])

        for label, key, color, ls, lw in channels:
            p16, p50, p84 = arr(key, 0), arr(key, 1), arr(key, 2)
            valid = ~np.isnan(p50)
            if np.sum(valid) < 2: continue

            log_p16 = np.log10(p16[valid])
            log_p50 = np.log10(p50[valid])
            log_p84 = np.log10(p84[valid])

            ax.plot(snap_z[valid], log_p50, color=color, linestyle=ls, linewidth=lw, label=label)
            
            if key != 'bh':
                ax.fill_between(snap_z[valid], log_p16, log_p84, color=color, alpha=0.15)

        # Overlay digitised reference curves
        if PLOT_DIGITISED_TXT and digitised_data is not None:
            panel_key = ['logMh0~12', 'logMh0~13', 'logMh0~14', 'logMh0~15'][i]
            if panel_key in digitised_data:
                for channel, entry in digitised_data[panel_key].items():
                    if channel in DIGITISED_COLOURS:
                        ax.plot(entry['z'], entry['y'], color=DIGITISED_COLOURS[channel],
                                linestyle='--', linewidth=2.0, alpha=0.9)

        ax.set_xlim(snap_z.min(), 7.0)
        ax.grid(True, alpha=0.3)

    handles = [plt.Line2D([0], [0], color=c, lw=lw) for _, _, c, _, lw in channels]
    labels = [l for l, _, _, _, _ in channels]
    axes[-1].legend(handles, labels, fontsize=10)

    #plt.suptitle(f"BH Growth Channels: {range_name}", y=1.05)
    plt.tight_layout()
    
    # Save with unique filename based on the option used
    out_filename = f'bh_growth_vs_redshift_halo_bins_{chosen_range}{OutputFormat}'
    plt.savefig(os.path.join(OutputDir, out_filename), bbox_inches='tight')
    plt.close()

    print(f"\n  Saved: {out_filename}")
    print("Done.")

if __name__ == '__main__':
    main()