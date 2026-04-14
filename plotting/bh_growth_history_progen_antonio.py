#!/usr/bin/env python3
"""
Black hole growth tracking: validates that the three accretion channels
(quasar mode, radio mode, BH-BH mergers) sum to the total BlackHoleMass,
and plots the MEDIAN relative contributions across galaxies.

Now features the "Top N Most Massive Central Galaxies" proxy 
for tracking main-branch physics inside Halo Mass bins.
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
# Optional: overlay digitised reference data from TXT file
# ============================================================

PLOT_DIGITISED_TXT = True   # <--- MASTER SWITCH (True / False)

DIGITISED_TXT_FILENAME = "BH_mass_growth_refined_digitised.txt"

# Colour scheme matching the paper
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
            # Robust check to prevent numpy.int64 crashes
            snap_key = snap if isinstance(snap, str) else f"Snap_{int(snap)}"
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


def compute_percentiles(data, percentiles=[16, 50, 84]):
    """Compute percentiles for non-zero values, return NaN if no valid data."""
    valid = data[data > 0]
    if len(valid) == 0:
        return [np.nan] * len(percentiles)
    return np.percentile(valid, percentiles)

def read_digitised_txt(filepath):
    """Reads digitised BH growth curves from the provided TXT file."""
    data = {}
    current_panel = None
    z_vals = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('z '):
                z_vals = np.array([float(x) for x in line.split()[1:]])
            elif line.startswith('[') and line.endswith(']'):
                current_panel = line.strip('[]')
                data[current_panel] = {}
            else:
                parts = line.split()
                channel = parts[0]
                values = np.array([float(x) for x in parts[1:]])
                data[current_panel][channel] = {
                    'z': z_vals,
                    'y': values
                }
    return data


def main():
    parser = argparse.ArgumentParser(description='Black hole growth tracking validation and plots')
    parser.add_argument('-i', '--input-pattern', type=str,
                        default='./output/millennium/model_*.hdf5',
                        help='Path pattern to model HDF5 files')
    parser.add_argument('-s', '--snapshot', type=int, default=None,
                        help='Snapshot number (default: latest)')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory for plots')
    parser.add_argument('--validation-snap', type=int, default=None,
                        help='Snapshot to create validation histograms for (default: latest)')
    parser.add_argument('-n', '--top-n', type=int, default=100,
                        help='Number of most massive galaxies to track per bin (default: 100)')
    args = parser.parse_args()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"Error: No files found matching: {args.input_pattern}")
        sys.exit(1)

    print(f"Found {len(file_list)} model files.")

    sim_params = read_simulation_params(file_list[0])
    Hubble_h = sim_params['Hubble_h']

    if args.snapshot is not None:
        snap_num = args.snapshot
    else:
        snap_num = sim_params['latest_snapshot']
    print(f"Using snapshot: {snap_num}")

    if args.validation_snap is not None:
        validation_snap = args.validation_snap
    else:
        validation_snap = snap_num
    print(f"Validation snapshot for histograms: {validation_snap}")

    if 'snapshot_redshifts' in sim_params and snap_num < len(sim_params['snapshot_redshifts']):
        print(f"Redshift: {sim_params['snapshot_redshifts'][snap_num]:.4f}")

    if args.output_dir:
        OutputDir = args.output_dir
    else:
        input_dir = os.path.dirname(os.path.abspath(file_list[0]))
        OutputDir = os.path.join(input_dir, 'plots')
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
    Type = read_hdf(file_list, snap_num, 'Type')

    if len(BlackHoleMass) == 0:
        print("No galaxies found!")
        sys.exit(1)

    # Handle missing fields from older output files (default to zeros)
    ngal = len(BlackHoleMass)
    if len(TorqueDriven) == 0:
        TorqueDriven = np.zeros(ngal)
    if len(SeedModeAccretion) == 0:
        SeedModeAccretion = np.zeros(ngal)
    if len(BHSeedMass) == 0:
        BHSeedMass = np.zeros(ngal)

    print(f"Total galaxies: {len(BlackHoleMass)}")

    bh_mask = BlackHoleMass > 0
    n_bh = np.sum(bh_mask)
    print(f"Galaxies with BH: {n_bh} ({100*n_bh/len(BlackHoleMass):.1f}%)")

    # ===================== VALIDATION =====================
    print("\n" + "="*60)
    print("VALIDATION: Channel sum vs BlackHoleMass")
    print("="*60)

    growth_sum = QuasarMode + RadioMode + BHSeedMass
    residual = BlackHoleMass - growth_sum

    if n_bh > 0:
        bh = BlackHoleMass[bh_mask]
        gs = growth_sum[bh_mask]
        res = residual[bh_mask]
        frac_res = res / bh

        print(f"\nBlackHoleMass total:  {bh.sum():.6e} M_sun")
        print(f"Growth sum total:     {gs.sum():.6e} M_sun  (QuasarMode + RadioMode + BHSeedMass)")
        print(f"  Quasar mode total:  {QuasarMode[bh_mask].sum():.6e} M_sun  ({100*QuasarMode[bh_mask].sum()/bh.sum():.1f}%)")
        print(f"    Merger-driven:    {MergerDriven[bh_mask].sum():.6e} M_sun  ({100*MergerDriven[bh_mask].sum()/bh.sum():.1f}%)")
        print(f"    Instability:      {InstabilityDriven[bh_mask].sum():.6e} M_sun  ({100*InstabilityDriven[bh_mask].sum()/bh.sum():.1f}%)")
        print(f"    Torque-driven:    {TorqueDriven[bh_mask].sum():.6e} M_sun  ({100*TorqueDriven[bh_mask].sum()/bh.sum():.1f}%)")
        print(f"    Seed-mode:        {SeedModeAccretion[bh_mask].sum():.6e} M_sun  ({100*SeedModeAccretion[bh_mask].sum()/bh.sum():.1f}%)")
        print(f"  Radio mode:         {RadioMode[bh_mask].sum():.6e} M_sun  ({100*RadioMode[bh_mask].sum()/bh.sum():.1f}%)")
        print(f"  BH seed mass:       {BHSeedMass[bh_mask].sum():.6e} M_sun  ({100*BHSeedMass[bh_mask].sum()/bh.sum():.1f}%)")
        print(f"\n  BH-BH mergers:      {BHMerger[bh_mask].sum():.6e} M_sun  (diagnostic — mass received via coalescence)")
        print(f"\nResidual (BH - growth sum):  {res.sum():.6e} M_sun")
        print(f"\nPer-galaxy fractional residual (BH - growth sum) / BH:")
        print(f"  Median: {np.median(frac_res):.6f}")
        print(f"  Max:    {np.max(np.abs(frac_res)):.6f}")
        print(f"  99th percentile: {np.percentile(np.abs(frac_res), 99):.6f}")

        bad = np.abs(frac_res) > 0.01
        if np.sum(bad) > 0:
            print(f"\n  PASS/FAIL: WARNING — {np.sum(bad)} galaxies have >1% residual")
        else:
            print(f"\n  PASS: All galaxies have <1% residual")

    # ===================== STATISTICS =====================
    print("\n" + "="*60)
    print("CHANNEL STATISTICS (galaxies with BH > 0)")
    print("="*60)

    if n_bh > 0:
        md = MergerDriven[bh_mask]
        id_ = InstabilityDriven[bh_mask]
        td = TorqueDriven[bh_mask]
        sm = SeedModeAccretion[bh_mask]
        rm = RadioMode[bh_mask]
        bm = BHMerger[bh_mask]
        sd = BHSeedMass[bh_mask]

        has_md = md > 0
        has_id = id_ > 0
        has_td = td > 0
        has_sm = sm > 0
        has_rm = rm > 0
        has_bm = bm > 0
        has_sd = sd > 0

        print(f"\nGalaxies with merger-driven accretion:      {np.sum(has_md)} ({100*np.sum(has_md)/n_bh:.1f}%)")
        print(f"Galaxies with instability-driven accretion: {np.sum(has_id)} ({100*np.sum(has_id)/n_bh:.1f}%)")
        print(f"Galaxies with torque-driven accretion:      {np.sum(has_td)} ({100*np.sum(has_td)/n_bh:.1f}%)")
        print(f"Galaxies with seed-mode accretion:          {np.sum(has_sm)} ({100*np.sum(has_sm)/n_bh:.1f}%)")
        print(f"Galaxies with radio mode accretion:         {np.sum(has_rm)} ({100*np.sum(has_rm)/n_bh:.1f}%)")
        print(f"Galaxies with BH-BH mergers:                {np.sum(has_bm)} ({100*np.sum(has_bm)/n_bh:.1f}%)")
        print(f"Galaxies with BH seed mass:                 {np.sum(has_sd)} ({100*np.sum(has_sd)/n_bh:.1f}%)")

        dominant = np.argmax(np.column_stack([md, id_, td, sm, rm]), axis=1)
        labels = ['Merger-driven', 'Instability-driven', 'Torque-driven', 'Seed-mode', 'Radio mode']
        for i, lab in enumerate(labels):
            n = np.sum(dominant == i)
            print(f"  Dominant growth channel = {lab}: {n} ({100*n/n_bh:.1f}%)")

    # ===================== PLOTS =====================
    print(f"\nGenerating plots in {OutputDir}...")

    # ================= MULTI HALO MASS PLOT =================
    all_snaps = np.array(sim_params['available_snapshots'])
    all_redshifts = sim_params['snapshot_redshifts']
    print(f"  Computing Top {args.top_n} Most Massive Central Galaxies in Halo Mass Bins...")

    # --- Define halo mass bins (log10 Mvir) ---
    halo_bins = [
        (11.5, 12.5),
        (12.5, 13.5),
        (13.5, 14.5),
        (14.5, 15.5),
    ]

    bin_labels = [
        r"$\log_{10}(M_{h}) \sim 12\,M_\odot$",
        r"$\log_{10}(M_{h}) \sim 13\,M_\odot$",
        r"$\log_{10}(M_{h}) \sim 14\,M_\odot$",
        r"$\log_{10}(M_{h}) \sim 15\,M_\odot$",
    ]

    # --- Storage per bin ---
    all_results = [[] for _ in halo_bins]

    # Loop chronologically so the timeline plots correctly
    for sn in sorted(all_snaps):
        bh = read_hdf(file_list, sn, 'BlackHoleMass') * 1.0e10 / Hubble_h
        if len(bh) == 0:
            continue

        z = all_redshifts[sn] if sn < len(all_redshifts) else None
        if z is None:
            continue

        def safe_read(field):
            arr = read_hdf(file_list, sn, field) * 1.0e10 / Hubble_h
            return arr if len(arr) > 0 else np.zeros(len(bh))

        mvir = safe_read('Mvir')
        stellar_mass = safe_read('StellarMass')
        
        md = safe_read('MergerDrivenBHaccretionMass')
        id_ = safe_read('InstabilityDrivenBHaccretionMass')
        rm = safe_read('RadioModeBHaccretionMass')
        bm = safe_read('BHMergerMass')

        # Read Type without dividing by Hubble_h
        gal_type = read_hdf(file_list, sn, 'Type')
        if len(gal_type) == 0: gal_type = np.zeros(len(bh))

        log_mvir = np.log10(mvir + 1e-10)

        for i, (mmin, mmax) in enumerate(halo_bins):
            
            # --- THE "TOP N CENTRAL GALAXY" PROXY ---
            # 1. Isolate centrals in this halo mass bin that have a black hole
            valid_idx = np.where((bh > 0) & (gal_type == 0) & (log_mvir >= mmin) & (log_mvir < mmax) & (stellar_mass > 0))[0]

            if len(valid_idx) < 3:
                continue

            # 2. Sort by Stellar Mass and take the Top N
            sorted_idx = valid_idx[np.argsort(stellar_mass[valid_idx])]
            top_idx = sorted_idx[-args.top_n:]

            def pct(x):
                valid = x[x > 0]
                return np.percentile(valid, [16, 50, 84]) if len(valid) > 0 else [np.nan, np.nan, np.nan]

            all_results[i].append({
                'z': z,
                'bh': pct(bh[top_idx]),
                'md': pct(md[top_idx]),
                'id': pct(id_[top_idx]),
                'rm': pct(rm[top_idx]),
                'bm': pct(bm[top_idx]),
            })

    # ------------------------------------------------------------
    # Load digitised TXT data (if requested)
    # ------------------------------------------------------------
    digitised_data = None
    if PLOT_DIGITISED_TXT:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        txt_path = os.path.join(script_dir, DIGITISED_TXT_FILENAME)

        if os.path.exists(txt_path):
            print(f"  Loading digitised reference curves from: {txt_path}")
            digitised_data = read_digitised_txt(txt_path)
        else:
            print(f"  WARNING: Digitised TXT file not found: {txt_path}")
            print("  Disabling digitised reference plotting.")
            digitised_data = None

    # ================= PLOTTING =================
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

    channels = [
        ('Total BH Mass', 'bh', 'black', '-', 2.0),
        ('Merger-driven', 'md', '#2196F3', '-', 1.5),
        ('Instability-driven', 'id', '#FF9800', '-', 1.5),
        ('Radio mode', 'rm', '#9C27B0', '-', 1.5),
        ('BH-BH mergers', 'bm', '#4CAF50', '-', 1.5),
    ]

    for i, ax in enumerate(axes):
        # Sort chronologically, descending Redshift (time moves forward left-to-right)
        results = sorted(all_results[i], key=lambda x: x['z'], reverse=True)

        ax.set_title(bin_labels[i])

        if i == 0:
            ax.set_ylabel(r'$\log_{10}(M_{\rm BH}\,[M_\odot])$')
        ax.set_xlabel(r'Redshift ($z$)')

        if len(results) < 2:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.set_xlim(7.0, 0.0) # Maintain reversed bounds even if empty
            continue

        snap_z = np.array([r['z'] for r in results])

        def arr(key, j):
            return np.array([r[key][j] for r in results])

        for label, key, color, ls, lw in channels:
            p16, p50, p84 = arr(key, 0), arr(key, 1), arr(key, 2)
            valid = ~np.isnan(p50)

            if np.sum(valid) > 1:
                log_p16 = np.log10(p16[valid])
                log_p50 = np.log10(p50[valid])
                log_p84 = np.log10(p84[valid])

                ax.plot(snap_z[valid], log_p50, color=color, linestyle=ls, linewidth=lw, label=label)
                
                # Shade the spread for growth channels (but keep Total BH line clean)
                if key != 'bh':
                    ax.fill_between(snap_z[valid], log_p16, log_p84, color=color, alpha=0.15)

        # ----------------------------------------------------
        # Overlay digitised reference curves (optional)
        # ----------------------------------------------------
        if PLOT_DIGITISED_TXT and digitised_data is not None:
            panel_key = ['logMh0~12', 'logMh0~13', 'logMh0~14', 'logMh0~15'][i]
            if panel_key in digitised_data:
                for channel, entry in digitised_data[panel_key].items():
                    if channel not in DIGITISED_COLOURS:
                        continue

                    ax.plot(
                        entry['z'], entry['y'],
                        color=DIGITISED_COLOURS[channel],
                        linestyle='--', linewidth=2.0, alpha=0.9
                    )

        # Reverse X-axis so Time moves forward Left -> Right
        ax.set_xlim(snap_z.min(), 7.0)
        ax.grid(True, alpha=0.3)

    # Legend (only once on the last panel)
    handles = [plt.Line2D([0], [0], color=c, lw=lw) for _, _, c, _, lw in channels]
    labels = [l for l, _, _, _, _ in channels]
    axes[-1].legend(handles, labels, fontsize=10)

    plt.tight_layout()
    output_filename = os.path.join(OutputDir, f'bh_growth_vs_redshift_halo_bins_top{args.top_n}{OutputFormat}')
    plt.savefig(output_filename)
    plt.close()

    print(f"  Saved: bh_growth_vs_redshift_halo_bins_top{args.top_n}{OutputFormat}")
    print("\nDone.")

if __name__ == '__main__':
    main()