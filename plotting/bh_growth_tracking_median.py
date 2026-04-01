#!/usr/bin/env python3
"""
Black hole growth tracking: validates that the three accretion channels
(quasar mode, radio mode, BH-BH mergers) sum to the total BlackHoleMass,
and plots the MEDIAN relative contributions across galaxies.
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

    # -- Plot median channel contributions across all snapshots --
    if 'snapshot_redshifts' in sim_params:
        print("  Computing median BH growth channels across all snapshots...")

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

        # --- Storage ---
        results = []

        for sn in all_snaps:
            bh = read_hdf(file_list, sn, 'BlackHoleMass') * 1.0e10 / Hubble_h
            if len(bh) == 0:
                continue

            mask = bh > 0
            if not np.any(mask):
                continue

            z = all_redshifts[sn] if sn < len(all_redshifts) else None
            if z is None:
                continue

            def safe_read(field):
                arr = read_hdf(file_list, sn, field) * 1.0e10 / Hubble_h
                return arr if len(arr) > 0 else np.zeros(len(bh))

            md = safe_read('MergerDrivenBHaccretionMass')
            id_ = safe_read('InstabilityDrivenBHaccretionMass')
            td = safe_read('TorqueDrivenBHaccretionMass')
            sm = safe_read('SeedModeBHaccretionMass')
            rm = safe_read('RadioModeBHaccretionMass')
            bm = safe_read('BHMergerMass')

            def pct(x):
                valid = x[x > 0]
                return np.percentile(valid, [16, 50, 84]) if len(valid) else [np.nan]*3

            results.append({
                'snap': sn,
                'z': z,
                'age': redshift_to_age_gyr(z),
                'bh': pct(bh[mask]),
                'md': pct(md[mask]),
                'id': pct(id_[mask]),
                'td': pct(td[mask]),
                'sm': pct(sm[mask]),
                'rm': pct(rm[mask]),
                'bm': pct(bm[mask]),
            })

            # --- Validation print ---
            if sn == validation_snap:
                print(f"\n{'='*60}")
                print(f"VALIDATION: Snapshot {sn} (z={z:.4f})")
                print(f"{'='*60}")
                print(f"Galaxies with BH > 0: {np.sum(mask)}")
                print(f"Median BH mass: {results[-1]['bh'][1]:.4e}")

        if len(results) < 2:
            print("Not enough snapshots to plot.")
        else:
            # --- Convert to arrays ---
            results = sorted(results, key=lambda x: x['age'])

            snap_z = np.array([r['z'] for r in results])

            def arr(key, i): return np.array([r[key][i] for r in results])

            # --- Plot ---
            fig, ax = plt.subplots(figsize=(10, 7))

            channels = [
                ('Merger-driven', 'md', '#2196F3', 'o'),
                ('Instability-driven', 'id', '#FF9800', 'D'),
                #('Torque-driven', 'td', '#795548', 'P'),
                #('Seed-mode', 'sm', '#FF5722', 'X'),
                ('Radio mode', 'rm', '#9C27B0', 's'),
                ('BH-BH mergers', 'bm', '#4CAF50', '^'),
            ]

            for label, key, color, marker in channels:
                p16, p50, p84 = arr(key, 0), arr(key, 1), arr(key, 2)
                valid = ~np.isnan(p50)

                # Take log10 safely
                log_p16 = np.log10(p16[valid])
                log_p50 = np.log10(p50[valid])
                log_p84 = np.log10(p84[valid])

                ax.plot(snap_z[valid], log_p50,
                        marker=marker, color=color, label=label,
                        linewidth=1.5, markersize=5)

                ax.fill_between(snap_z[valid], log_p16, log_p84,
                                color=color, alpha=0.25)

            # --- Total BH ---
            #p16, p50, p84 = arr('bh', 0), arr('bh', 1), arr('bh', 2)
            #valid = ~np.isnan(p50)

            #ax.plot(snap_z[valid], p50[valid], 'k-o',
            #        linewidth=2, markersize=4, label='Total BH Mass')

            #ax.fill_between(snap_z[valid], p16[valid], p84[valid],
            #                color='black', alpha=0.15)

            #ax.set_yscale('log')
            ax.set_xlabel('Redshift')
            ax.set_ylabel(r'$\log_{10}(\mathrm{BH\ Mass}\ [M_\odot])$')
            ax.set_xlim( snap_z.min(), 7.0)
            ax.set_title('Median BH Growth Channels vs Redshift')
            ax.legend(fontsize=11)
            #ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(OutputDir, f'bh_growth_median_vs_redshift{OutputFormat}'))
            plt.close()

            print(f"  Saved: bh_growth_median_vs_redshift{OutputFormat}")

            # ================= HISTOGRAMS =================
            print(f"\n  Creating validation histograms for snapshot {validation_snap}...")

            val_entry = next((r for r in results if r['snap'] == validation_snap), None)

            if val_entry is None:
                print(f"WARNING: Snapshot {validation_snap} not available after filtering.")
            else:
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

                hist_data = [
                    ('Total BH Mass', bh_val[mask_val], val_entry['bh'][1], 'black'),
                    ('Merger-driven', md_val[mask_val], val_entry['md'][1], '#2196F3'),
                    ('Instability-driven', id_val[mask_val], val_entry['id'][1], '#FF9800'),
                    ('Torque-driven', td_val[mask_val], val_entry['td'][1], '#795548'),
                    ('Seed-mode', sm_val[mask_val], val_entry['sm'][1], '#FF5722'),
                    ('Radio mode', rm_val[mask_val], val_entry['rm'][1], '#9C27B0'),
                    ('BH-BH mergers', bm_val[mask_val], val_entry['bm'][1], '#4CAF50'),
                ]

                for i, (label, data, median, color) in enumerate(hist_data):
                    ax = axes[i]
                    d = data[data > 0]

                    if len(d):
                        ax.hist(np.log10(d), bins=30, color=color, alpha=0.7)
                        if not np.isnan(median):
                            ax.axvline(np.log10(median), color='red', linestyle='--')

                        ax.set_title(label)
                        ax.set_xlabel(r'$\log_{10}(M)$')
                        ax.grid(True, alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, 'No data', ha='center')

                fig.delaxes(axes[7])
                fig.suptitle(f'Snapshot {validation_snap} (z={val_entry["z"]:.3f})')

                plt.tight_layout()
                plt.savefig(os.path.join(OutputDir, f'validation_histograms_snap{validation_snap}{OutputFormat}'))
                plt.close()

                print(f"  Saved: validation_histograms_snap{validation_snap}{OutputFormat}")

    print("\nDone.")


if __name__ == '__main__':
    main()