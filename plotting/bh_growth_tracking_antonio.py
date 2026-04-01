#!/usr/bin/env python3
"""
Black hole growth tracking: validates that the three accretion channels
(quasar mode, radio mode, BH-BH mergers) sum to the total BlackHoleMass,
and plots the relative contributions.
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
            snap_key = f'Snap_{snap}' if isinstance(snap, int) else snap
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
        if 'snapshot_redshifts' in f['Header']:
            params['snapshot_redshifts'] = np.array(f['Header/snapshot_redshifts'])
    return params


def main():
    parser = argparse.ArgumentParser(description='Black hole growth tracking validation and plots')
    parser.add_argument('-i', '--input-pattern', type=str,
                        default='./output/millennium/model_*.hdf5',
                        help='Path pattern to model HDF5 files')
    parser.add_argument('-s', '--snapshot', type=int, default=None,
                        help='Snapshot number (default: latest)')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory for plots')
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
    # BHMergerMass is a diagnostic (how much BH mass came via coalescence) but is NOT
    # part of the growth sum — mergers don't create new mass, they transfer it between
    # galaxies, and the satellite's growth history is already carried over in the
    # QuasarMode and RadioMode channel totals.
    # BHSeedMass is the initial seed mass placed when BHSeedingOn is enabled.
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

        # Sub-channel consistency: MergerDriven + InstabilityDriven == QuasarMode
        print(f"\n{'='*60}")
        print("SUB-CHANNEL CONSISTENCY CHECK")
        print(f"{'='*60}")
        print("Checking: MergerDriven + InstabilityDriven + TorqueDriven + SeedMode = QuasarModeBHaccretionMass")

        quasar_sub = MergerDriven[bh_mask] + InstabilityDriven[bh_mask] + TorqueDriven[bh_mask] + SeedModeAccretion[bh_mask]
        quasar_tot = QuasarMode[bh_mask]
        sub_residual = quasar_sub - quasar_tot

        # Fractional residual only for galaxies with quasar mode accretion
        has_quasar = quasar_tot > 0
        if np.sum(has_quasar) > 0:
            sub_frac_res = sub_residual[has_quasar] / quasar_tot[has_quasar]
            print(f"\nGalaxies with QuasarMode > 0: {np.sum(has_quasar)}")
            print(f"  Absolute residual sum: {np.sum(np.abs(sub_residual[has_quasar])):.6e} M_sun")
            print(f"  Per-galaxy fractional residual (sub-channels - Quasar) / Quasar:")
            print(f"    Median: {np.median(sub_frac_res):.6f}")
            print(f"    Max:    {np.max(np.abs(sub_frac_res)):.6f}")
            print(f"    99th percentile: {np.percentile(np.abs(sub_frac_res), 99):.6f}")

            sub_bad = np.abs(sub_frac_res) > 0.01
            if np.sum(sub_bad) > 0:
                print(f"\n  PASS/FAIL: WARNING — {np.sum(sub_bad)} galaxies have >1% sub-channel residual")
            else:
                print(f"\n  PASS: All sub-channels sum to QuasarMode for all galaxies (<1% residual)")
        else:
            print(f"\n  No galaxies with QuasarMode accretion — skipping sub-channel check")

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


    # -- Plots 5 & 6: Channel fractions and rates as a function of redshift --
    if 'snapshot_redshifts' in sim_params:
        print("  Computing BH growth channels across all snapshots...")
        all_snaps = sim_params['available_snapshots']
        all_redshifts = sim_params['snapshot_redshifts']

        # Read cosmology for time calculations
        with h5py.File(file_list[0], 'r') as f:
            omega_m = float(f['Header/Simulation'].attrs['omega_matter'])
            omega_l = float(f['Header/Simulation'].attrs['omega_lambda'])

        def redshift_to_age_gyr(z, H0=Hubble_h*100, Om=omega_m, Ol=omega_l):
            """Convert redshift to age of universe in Gyr using numerical integration."""
            from scipy.integrate import quad
            H0_per_gyr = H0 * 1.0222e-3  # km/s/Mpc -> 1/Gyr
            integrand = lambda a: 1.0 / (a * np.sqrt(Om / a**3 + Ol))
            age, _ = quad(integrand, 0, 1.0 / (1.0 + z))
            return age / H0_per_gyr

        snap_z = []
        snap_age = []
        snap_md_total = []
        snap_id_total = []
        snap_td_total = []
        snap_sm_total = []
        snap_rm_total = []
        snap_bm_total = []
        snap_total_bh = []

        for sn in all_snaps:
            bh = read_hdf(file_list, sn, 'BlackHoleMass') * 1.0e10 / Hubble_h
            md = read_hdf(file_list, sn, 'MergerDrivenBHaccretionMass') * 1.0e10 / Hubble_h
            id_ = read_hdf(file_list, sn, 'InstabilityDrivenBHaccretionMass') * 1.0e10 / Hubble_h
            td = read_hdf(file_list, sn, 'TorqueDrivenBHaccretionMass') * 1.0e10 / Hubble_h
            sm = read_hdf(file_list, sn, 'SeedModeBHaccretionMass') * 1.0e10 / Hubble_h
            rm = read_hdf(file_list, sn, 'RadioModeBHaccretionMass') * 1.0e10 / Hubble_h
            bm = read_hdf(file_list, sn, 'BHMergerMass') * 1.0e10 / Hubble_h

            total = bh.sum()
            if total <= 0 or len(bh) == 0:
                continue

            z = all_redshifts[sn] if sn < len(all_redshifts) else None
            if z is None:
                continue

            snap_z.append(z)
            snap_age.append(redshift_to_age_gyr(z))
            snap_md_total.append(md.sum())
            snap_id_total.append(id_.sum())
            snap_td_total.append(td.sum() if len(td) > 0 else 0.0)
            snap_sm_total.append(sm.sum() if len(sm) > 0 else 0.0)
            snap_rm_total.append(rm.sum())
            snap_bm_total.append(bm.sum())
            snap_total_bh.append(total)

        snap_z = np.array(snap_z)
        snap_age = np.array(snap_age)
        snap_md_total = np.array(snap_md_total)
        snap_id_total = np.array(snap_id_total)
        snap_td_total = np.array(snap_td_total)
        snap_sm_total = np.array(snap_sm_total)
        snap_rm_total = np.array(snap_rm_total)
        snap_bm_total = np.array(snap_bm_total)
        snap_total_bh = np.array(snap_total_bh)

        if len(snap_z) > 1:
            # Sort by time (increasing age = decreasing redshift)
            order = np.argsort(snap_age)
            snap_z = snap_z[order]
            snap_age = snap_age[order]
            snap_md_total = snap_md_total[order]
            snap_id_total = snap_id_total[order]
            snap_td_total = snap_td_total[order]
            snap_sm_total = snap_sm_total[order]
            snap_rm_total = snap_rm_total[order]
            snap_bm_total = snap_bm_total[order]
            snap_total_bh = snap_total_bh[order]

            # Compute fractions
            snap_md_frac = snap_md_total #/ snap_total_bh
            snap_id_frac = snap_id_total #/ snap_total_bh
            snap_td_frac = snap_td_total #/ snap_total_bh
            snap_sm_frac = snap_sm_total #/ snap_total_bh
            snap_rm_frac = snap_rm_total #/ snap_total_bh
            snap_bm_frac = snap_bm_total #/ snap_total_bh

            # -- Plot 5: Channel fractions vs redshift --
            # -- Plot: Channel fractions vs redshift with total mass overlay --
            fig, ax1 = plt.subplots(figsize=(8.34, 6))

            # --- Fraction lines ---
            for frac, marker, label, colour in zip(
                    [snap_md_frac, snap_id_frac, snap_td_frac, snap_sm_frac, snap_rm_frac, snap_bm_frac],
                    ['o', 'D', 'P', 'X', 's', '^'],
                    ['Merger-driven', 'Instability-driven', 'Torque-driven', 'Seed-mode', 'Radio mode', 'BH-BH mergers'],
                    ['#2196F3',  '#FF9800', '#795548', '#FF5722', '#9C27B0', '#4CAF50']):
                mask = frac > 0
                if np.any(mask):
                    ax1.plot(snap_z[mask], np.log10(frac[mask]), marker=marker, linestyle='-',
                            color=colour, label=label, markersize=5, linewidth=1.5)

            ax1.plot(snap_z, np.log10(snap_total_bh), 'k--o',markersize=4, linewidth=1.5, label='Total BH Mass')

            ax1.set_ylabel('Total mass contributed by channel [$M_\odot$]')
            ax1.set_xlabel('Redshift')
            ax1.set_title('BH Growth Channels vs Redshift')
            #ax1.set_yscale('log')
            #ax1.set_ylim(1e-7, 2.0)
            #ax1.invert_xaxis()

            # --- Combined legend ---
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            ax1.legend(lines_1, labels_1, loc='best')

            plt.tight_layout()
            plt.savefig(os.path.join(OutputDir, f'bh_growth_channels_vs_redshift_antonio{OutputFormat}'))
            plt.close()

            print(f"  Saved: bh_growth_channels_vs_redshift_antonio{OutputFormat}")

    print("\nDone.")


if __name__ == '__main__':
    main()
