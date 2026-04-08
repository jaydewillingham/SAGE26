#!/usr/bin/env python3
"""
Black hole growth tracking: validates that the three accretion channels
(quasar mode, radio mode, BH-BH mergers) sum to the total BlackHoleMass,
and plots the MEDIAN relative contributions across galaxies.

MODIFIED VERSION:
Instead of plotting median statistics (which average out individual galaxy effects),
this version:
1. Tracks the growth trajectory of EACH INDIVIDUAL GALAXY across all snapshots
2. Plots all galaxy trajectories in light, semi-transparent colors
3. Highlights the galaxy that is at the 50th percentile (median) at z=0 in darker colors
This reveals the diversity of growth histories and shows which galaxy follows the "typical" path.

python bh_growth_tracking_modified.py \
    -i ./output/millennium/model_*.hdf5 \
    -o ./plots \
    -s <latest_snap> \
    --validation-snap <snapshot>
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


def compute_percentiles(data, percentiles=[16, 50, 84]):
    """Compute percentiles for non-zero values, return NaN if no valid data."""
    valid = data[data > 0]
    if len(valid) == 0:
        return [np.nan] * len(percentiles)
    return np.percentile(valid, percentiles)


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

    # Growth budget: QuasarMode + RadioMode + BHSeedMass = BlackHoleMass
    growth_sum = QuasarMode + RadioMode + BHSeedMass
    residual = BlackHoleMass - growth_sum

    # Only check galaxies with BHs
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

        # Flag any large discrepancies
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

        # Dominant growth channel per galaxy (excluding BH mergers and seed mass as they're not growth channels)
        dominant = np.argmax(np.column_stack([md, id_, td, sm, rm]), axis=1)
        labels = ['Merger-driven', 'Instability-driven', 'Torque-driven', 'Seed-mode', 'Radio mode']
        for i, lab in enumerate(labels):
            n = np.sum(dominant == i)
            print(f"  Dominant growth channel = {lab}: {n} ({100*n/n_bh:.1f}%)")

    # ===================== PLOTS =====================
    print(f"\nGenerating plots in {OutputDir}...")

    # -- Plot individual galaxy trajectories --
    if 'snapshot_redshifts' in sim_params:
        print("  Computing individual galaxy growth trajectories across all snapshots...")

        all_snaps = np.array(sim_params['available_snapshots'])
        all_redshifts = sim_params['snapshot_redshifts']

        # --- Cosmology ---
        with h5py.File(file_list[0], 'r') as f:
            omega_m = float(f['Header/Simulation'].attrs['omega_matter'])
            omega_l = float(f['Header/Simulation'].attrs['omega_lambda'])

        def redshift_to_age_gyr(z, H0=Hubble_h*100, Om=omega_m, Ol=omega_l):
            from scipy.integrate import quad
            H0_per_gyr = H0 * 1.0222e-3
            integrand = lambda a: 1.0 / (a * np.sqrt(Om / a**3 + Ol))
            age, _ = quad(integrand, 0, 1.0 / (1.0 + z))
            return age / H0_per_gyr

# ========== NEW APPROACH ==========
        # Step 1: Identify the z=0 snapshot (lowest redshift)
        z0_snap = all_snaps[np.argmin(all_redshifts[all_snaps])]
        print(f"  z=0 reference snapshot: {z0_snap} (z = {all_redshifts[z0_snap]:.6f})")

        # Step 2: Read z=0 data and find the median galaxy for each channel
        bh_z0 = read_hdf(file_list, z0_snap, 'BlackHoleMass') * 1.0e10 / Hubble_h

        def safe_read_z0(field):
            arr = read_hdf(file_list, z0_snap, field) * 1.0e10 / Hubble_h
            return arr if len(arr) > 0 else np.zeros(len(bh_z0))

        md_z0 = safe_read_z0('MergerDrivenBHaccretionMass')
        id_z0 = safe_read_z0('InstabilityDrivenBHaccretionMass')
        td_z0 = safe_read_z0('TorqueDrivenBHaccretionMass')
        sm_z0 = safe_read_z0('SeedModeBHaccretionMass')
        rm_z0 = safe_read_z0('RadioModeBHaccretionMass')
        bm_z0 = safe_read_z0('BHMergerMass')

        # FIX: We pass the FULL z0 array here (not the masked one)
        channels_info = [
            ('Merger-driven', 'md', '#2196F3', 'o', md_z0),
            ('Instability-driven', 'id', '#FF9800', 'D', id_z0),
            ('Radio mode', 'rm', '#9C27B0', 's', rm_z0),
            ('BH-BH mergers', 'bm', '#4CAF50', '^', bm_z0),
        ]

        # For each channel, find the galaxy index that has the median z=0 value
        median_gal_indices = {}
        for label, key, color, marker, full_z0_data in channels_info:
            # Mask to find valid galaxies
            valid_mask = (bh_z0 > 0) & (full_z0_data > 0)
            if np.any(valid_mask):
                median_value_z0 = np.median(full_z0_data[valid_mask])
                
                # Find index in the FULL array. 
                # Set invalid galaxies to infinity so they are never chosen.
                diffs = np.abs(full_z0_data - median_value_z0)
                diffs[~valid_mask] = np.inf
                idx = np.argmin(diffs)
                
                median_gal_indices[key] = idx
                print(f"    {label}: median galaxy absolute index = {idx}, z=0 value = {full_z0_data[idx]:.4e}")

        # Step 3: Build trajectories for ALL galaxies across ALL snapshots
        galaxy_trajectories = {key: {} for _, key, _, _, _ in channels_info}

        print("  Reading data for all snapshots...")
        for sn in all_snaps:
            bh = read_hdf(file_list, sn, 'BlackHoleMass') * 1.0e10 / Hubble_h
            if len(bh) == 0:
                continue

            z = all_redshifts[sn] if sn < len(all_redshifts) else None
            if z is None:
                continue

            def safe_read(field):
                arr = read_hdf(file_list, sn, field) * 1.0e10 / Hubble_h
                return arr if len(arr) > 0 else np.zeros(len(bh))

            channel_data = {
                'md': safe_read('MergerDrivenBHaccretionMass'),
                'id': safe_read('InstabilityDrivenBHaccretionMass'),
                'td': safe_read('TorqueDrivenBHaccretionMass'),
                'sm': safe_read('SeedModeBHaccretionMass'),
                'rm': safe_read('RadioModeBHaccretionMass'),
                'bm': safe_read('BHMergerMass'),
            }

            # For each galaxy, record its growth value at this snapshot
            for gal_idx in range(len(bh)):
                for key in galaxy_trajectories:
                    if gal_idx not in galaxy_trajectories[key]:
                        galaxy_trajectories[key][gal_idx] = {'z': [], 'growth': []}
                    
                    growth_val = channel_data[key][gal_idx]
                    galaxy_trajectories[key][gal_idx]['z'].append(z)
                    galaxy_trajectories[key][gal_idx]['growth'].append(growth_val)

        # Step 4: Create the plot
        print("  Creating individual trajectory plot...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for ax_idx, (label, key, color, marker, _) in enumerate(channels_info):
            ax = axes[ax_idx]

            # ============================================
            # PLOT ALL GALAXIES IN LIGHT, SEMI-TRANSPARENT
            # ============================================
            for gal_idx in galaxy_trajectories[key]:
                z_vals = np.array(galaxy_trajectories[key][gal_idx]['z'])
                growth_vals = np.array(galaxy_trajectories[key][gal_idx]['growth'])

                # Only plot if there's data > 0 (can't log of zero or negative)
                valid = growth_vals > 0
                if np.any(valid):
                    log_growth = np.log10(growth_vals[valid])
                    z_valid = z_vals[valid]

                    # Sort by redshift so line plots correctly
                    sort_idx = np.argsort(z_valid)
                    ax.plot(z_valid[sort_idx], log_growth[sort_idx],
                            color=color, alpha=0.05, linewidth=0.8)

            # ============================================
            # HIGHLIGHT THE z=0 MEDIAN GALAXY IN BOLD
            # ============================================
            if key in median_gal_indices:
                med_idx = median_gal_indices[key]
                if med_idx in galaxy_trajectories[key]:
                    z_vals = np.array(galaxy_trajectories[key][med_idx]['z'])
                    growth_vals = np.array(galaxy_trajectories[key][med_idx]['growth'])

                    valid = growth_vals > 0
                    if np.any(valid):
                        log_growth = np.log10(growth_vals[valid])
                        z_valid = z_vals[valid]

                        sort_idx = np.argsort(z_valid)
                        ax.plot(z_valid[sort_idx], log_growth[sort_idx],
                                color=color, linewidth=2.5, marker=marker,
                                markersize=6, label='z=0 Median Galaxy', zorder=100)

            ax.set_xlabel('Redshift', fontsize=11)
            ax.set_ylabel(r'$\log_{10}(\mathrm{Growth\ Mass}\ [M_\odot])$', fontsize=11)
            ax.set_xlim(-0.2, 7.0)
            ax.set_title(label, fontsize=12)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Individual Galaxy BH Growth Trajectories\n(Light: All Galaxies | Bold: z=0 Median Galaxy)', 
                     fontsize=14, y=1.00)
        plt.tight_layout()
        plt.savefig(os.path.join(OutputDir, f'bh_growth_individual_trajectories{OutputFormat}'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: bh_growth_individual_trajectories{OutputFormat}")

        # ================= HISTOGRAMS =================
        print(f"\n  Creating validation histograms for snapshot {validation_snap}...")

        bh_val = read_hdf(file_list, validation_snap, 'BlackHoleMass') * 1.0e10 / Hubble_h
        mask_val = bh_val > 0

        def safe_val(field):
            arr = read_hdf(file_list, validation_snap, field) * 1.0e10 / Hubble_h
            return arr if len(arr) > 0 else np.zeros(len(bh_val))

        md_val = safe_val('MergerDrivenBHaccretionMass')
        id_val = safe_val('InstabilityDrivenBHaccretionMass')
        td_val = safe_val('TorqueDrivenBHaccretionMass')
        sm_val = safe_val('SeedModeBHaccretionMass')
        rm_val = safe_val('RadioModeBHaccretionMass')
        bm_val = safe_val('BHMergerMass')

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        val_z = all_redshifts[validation_snap] if validation_snap < len(all_redshifts) else None

        hist_data = [
            ('Total BH Mass', bh_val[mask_val], 'black'),
            ('Merger-driven', md_val[mask_val], '#2196F3'),
            ('Instability-driven', id_val[mask_val], '#FF9800'),
            ('Torque-driven', td_val[mask_val], '#795548'),
            ('Seed-mode', sm_val[mask_val], '#FF5722'),
            ('Radio mode', rm_val[mask_val], '#9C27B0'),
            ('BH-BH mergers', bm_val[mask_val], '#4CAF50'),
        ]

        for i, (label, data, color) in enumerate(hist_data):
            ax = axes[i]
            d = data[data > 0]

            if len(d):
                ax.hist(np.log10(d), bins=30, color=color, alpha=0.7, edgecolor='black')
                ax.set_title(label, fontsize=11)
                ax.set_xlabel(r'$\log_{10}(M_\odot)$', fontsize=10)
                ax.set_ylabel('Count', fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)

        fig.delaxes(axes[7])
        if val_z is not None:
            fig.suptitle(f'Snapshot {validation_snap} (z={val_z:.3f}) - Growth Channel Distributions', fontsize=13)
        else:
            fig.suptitle(f'Snapshot {validation_snap} - Growth Channel Distributions', fontsize=13)

        plt.tight_layout()
        plt.savefig(os.path.join(OutputDir, f'validation_histograms_snap{validation_snap}{OutputFormat}'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: validation_histograms_snap{validation_snap}{OutputFormat}")

    print("\nDone.")


if __name__ == '__main__':
    main()