#!/usr/bin/env python3
"""
Black hole Eddington limit analysis with:
1. Time-series: Fraction of super-Eddington BHs vs redshift
2. Distribution: Histogram of accretion rate ratios (where rate > Eddington limit)
3. Summary statistics on dt parameter
4. Accretion rate ratio vs dt scatter plot
5. NEW: Accretion rate function  log10(dN/d log10 lambda) vs log10(lambda)
        Split by Radio Mode (0) and Quasar Mode (1)
"""
import argparse
import glob
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path

# ============================================================================
# MATPLOTLIB STYLE CONFIGURATION (Matching Original Style)
# ============================================================================
plt.rcParams['figure.figsize'] = (8.34, 6.25)
plt.rcParams['figure.dpi'] = 140
plt.rcParams['figure.autolayout'] = True

# Fonts - Cleaned up to avoid environment errors
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 20.0

# Axis and Ticks
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.size'] = 7.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['xtick.minor.size'] = 5.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['xtick.major.pad'] = 9

plt.rcParams['ytick.major.size'] = 7.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['ytick.minor.size'] = 5.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.labelsize'] = 16

plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.fontsize'] = 14

# ============================================================================
# CONFIGURATION
# ============================================================================
MIN_STELLAR_MASS_LOG = 8.5
MIN_HALO_MASS_LOG = 11
MIN_Z0_BH_MASS = 1e4

# Physical constants
SEC_PER_YEAR = 365.25 * 24 * 3600
UNIT_TIME_IN_S = 3.086e19  # 1 Gyr in seconds (Millennium default)

# Millennium simulation box side length in Mpc/h
MILLENNIUM_BOX_MPC_H = 62.5 #500.0

# ============================================================================
# REDSHIFT MAPPING - MILLENNIUM SIMULATION
# ============================================================================
MILLENNIUM_SNAP_TO_Z = {
    0: 127.0, 1: 65.74, 2: 40.0, 3: 26.66, 4: 19.36, 5: 14.78, 6: 11.66,
    7: 9.44, 8: 7.64, 9: 6.44, 10: 5.48, 11: 4.73, 12: 4.19, 13: 3.72,
    14: 3.33, 15: 3.0, 16: 2.73, 17: 2.48, 18: 2.27, 19: 2.07, 20: 1.90,
    21: 1.75, 22: 1.61, 23: 1.48, 24: 1.37, 25: 1.27, 26: 1.18, 27: 1.10,
    28: 1.02, 29: 0.96, 30: 0.90, 31: 0.85, 32: 0.81, 33: 0.77, 34: 0.73,
    35: 0.70, 36: 0.67, 37: 0.63, 38: 0.60, 39: 0.57, 40: 0.54, 41: 0.51,
    42: 0.49, 43: 0.46, 44: 0.43, 45: 0.41, 46: 0.39, 47: 0.37, 48: 0.36,
    49: 0.34, 50: 0.32, 51: 0.31, 52: 0.29, 53: 0.28, 54: 0.27, 55: 0.26,
    56: 0.25, 57: 0.24, 58: 0.23, 59: 0.21, 60: 0.20, 61: 0.18, 62: 0.0
}

# ============================================================================
# HELPERS
# ============================================================================
def get_redshift_from_snapshot(snap_num, snap_to_z_dict=None):
    """Map snapshot number to redshift."""
    if snap_to_z_dict is None:
        snap_to_z_dict = MILLENNIUM_SNAP_TO_Z

    if snap_num in snap_to_z_dict:
        return snap_to_z_dict[snap_num]
    else:
        sorted_snaps = sorted(snap_to_z_dict.keys())
        if snap_num < sorted_snaps[0]:
            return snap_to_z_dict[sorted_snaps[0]]
        if snap_num > sorted_snaps[-1]:
            return snap_to_z_dict[sorted_snaps[-1]]

        for i in range(len(sorted_snaps) - 1):
            if sorted_snaps[i] <= snap_num <= sorted_snaps[i+1]:
                s1, s2 = sorted_snaps[i], sorted_snaps[i+1]
                z1, z2 = snap_to_z_dict[s1], snap_to_z_dict[s2]
                z = z1 + (snap_num - s1) * (z2 - z1) / (s2 - s1)
                return z
    return 0.0

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
                    if c in hf[snap_key]: return c
    return None

def read_data(file_list, snap_num, id_field, h_h):
    """Reads BH growth diagnostics as time-series arrays (Ngal x MAXSNAPS)."""
    all_ids = []
    all_bh_mass = []
    all_stellar_mass = []
    all_mvir = []
    all_bh_max_accretion_history = []  
    all_bh_eddington_rate_limit = []   
    all_bh_mass_at_accretion = []      
    all_dt = []                         
    all_bh_accretion_type = []

    seen_ids = set()

    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f"Snap_{snap_num}"
            if snap_key not in hf: continue
            grp = hf[snap_key]

            gids = grp[id_field][:]
            mask = np.array([gid not in seen_ids for gid in gids])
            for gid in gids[mask]: seen_ids.add(gid)
            if not np.any(mask): continue

            conv = 1e10 / h_h

            m_bh = grp['BlackHoleMass'][:][mask] * conv
            m_stellar = grp['StellarMass'][:][mask] * conv
            m_mvir = grp['Mvir'][:][mask] * conv

            # ----------------------------------------------------------------
            # BHMaxaccretionRate
            # ----------------------------------------------------------------
            if 'BHMaxaccretionRate' in grp:
                bh_max_accr_raw = grp['BHMaxaccretionRate'][:][mask] * conv
            else:
                bh_max_accr_raw = np.zeros((len(m_bh), 63))

            if bh_max_accr_raw.ndim == 1:
                ngal = len(m_bh)
                if len(bh_max_accr_raw) == ngal:
                    bh_max_accr_hist = bh_max_accr_raw.reshape(-1, 1)
                else:
                    maxsnaps = len(bh_max_accr_raw) // ngal
                    bh_max_accr_hist = bh_max_accr_raw.reshape(ngal, maxsnaps)
            else:
                bh_max_accr_hist = bh_max_accr_raw

            # ----------------------------------------------------------------
            # BHEddingtonRateLimit
            # ----------------------------------------------------------------
            if 'BHEddingtonRateLimit' in grp:
                bh_eddington_raw = grp['BHEddingtonRateLimit'][:][mask] * conv
            else:
                bh_eddington_raw = np.zeros_like(bh_max_accr_hist)

            if bh_eddington_raw.ndim == 1:
                ngal = len(m_bh)
                if len(bh_eddington_raw) == ngal:
                    bh_eddington_hist = bh_eddington_raw.reshape(-1, 1)
                else:
                    maxsnaps = len(bh_eddington_raw) // ngal
                    bh_eddington_hist = bh_eddington_raw.reshape(ngal, maxsnaps)
            else:
                bh_eddington_hist = bh_eddington_raw

            # ----------------------------------------------------------------
            # BHMassatAccretion
            # ----------------------------------------------------------------
            if 'BHMassatAccretion' in grp:
                bh_mass_at_accretion_raw = grp['BHMassatAccretion'][:][mask] * conv
            else:
                bh_mass_at_accretion_raw = np.zeros_like(bh_max_accr_hist)

            if bh_mass_at_accretion_raw.ndim == 1:
                ngal = len(m_bh)
                if len(bh_mass_at_accretion_raw) == ngal:
                    bh_mass_at_accretion_hist = bh_mass_at_accretion_raw.reshape(-1, 1)
                else:
                    maxsnaps = len(bh_mass_at_accretion_raw) // ngal
                    bh_mass_at_accretion_hist = bh_mass_at_accretion_raw.reshape(ngal, maxsnaps)
            else:
                bh_mass_at_accretion_hist = bh_mass_at_accretion_raw

            # ----------------------------------------------------------------
            # dt
            # ----------------------------------------------------------------
            if 'dt' in grp:
                dt_raw = grp['dt'][:]
                if dt_raw.ndim == 1:
                    ngal = len(m_bh)
                    if len(dt_raw) == ngal:
                        dt_hist = dt_raw[mask].reshape(-1, 1)
                    else:
                        maxsnaps = len(dt_raw) // ngal
                        dt_hist = dt_raw.reshape(ngal, maxsnaps)[mask]
                else:
                    dt_hist = dt_raw[mask]
            else:
                dt_hist = np.zeros_like(bh_max_accr_hist)

            # ----------------------------------------------------------------
            # NEW: BHAccretionType
            # ----------------------------------------------------------------
            if 'BHAccretionType' in grp:
                bh_acc_type_raw = grp['BHAccretionType'][:]
                if bh_acc_type_raw.ndim == 1:
                    ngal = len(m_bh)
                    if len(bh_acc_type_raw) == ngal:
                        bh_acc_type_hist = bh_acc_type_raw[mask].reshape(-1, 1)
                    else:
                        maxsnaps = len(bh_acc_type_raw) // ngal
                        bh_acc_type_hist = bh_acc_type_raw.reshape(ngal, maxsnaps)[mask]
                else:
                    bh_acc_type_hist = bh_acc_type_raw[mask]
            else:
                bh_acc_type_hist = np.zeros_like(bh_max_accr_hist) - 1

            # ----------------------------------------------------------------
            # Align shapes
            # ----------------------------------------------------------------
            if bh_max_accr_hist.shape != bh_eddington_hist.shape:
                max_shape = (bh_max_accr_hist.shape[0], max(bh_max_accr_hist.shape[1], bh_eddington_hist.shape[1]))
                bh_max_accr_padded = np.zeros(max_shape)
                bh_eddington_padded = np.zeros(max_shape)
                bh_max_accr_padded[:, :bh_max_accr_hist.shape[1]] = bh_max_accr_hist
                bh_eddington_padded[:, :bh_eddington_hist.shape[1]] = bh_eddington_hist
                bh_max_accr_hist = bh_max_accr_padded
                bh_eddington_hist = bh_eddington_padded

            if bh_mass_at_accretion_hist.shape != bh_max_accr_hist.shape:
                max_shape = (bh_max_accr_hist.shape[0], max(bh_max_accr_hist.shape[1], bh_mass_at_accretion_hist.shape[1]))
                bh_mass_at_accretion_padded = np.zeros(max_shape)
                bh_mass_at_accretion_padded[:, :bh_mass_at_accretion_hist.shape[1]] = bh_mass_at_accretion_hist
                bh_mass_at_accretion_hist = bh_mass_at_accretion_padded

            if dt_hist.shape != bh_max_accr_hist.shape:
                max_shape = bh_max_accr_hist.shape
                dt_padded = np.zeros(max_shape)
                dt_padded[:, :dt_hist.shape[1]] = dt_hist
                dt_hist = dt_padded
                
            if bh_acc_type_hist.shape != bh_max_accr_hist.shape:
                max_shape = bh_max_accr_hist.shape
                bh_acc_type_padded = np.zeros(max_shape) - 1
                bh_acc_type_padded[:, :bh_acc_type_hist.shape[1]] = bh_acc_type_hist
                bh_acc_type_hist = bh_acc_type_padded

            all_ids.append(gids[mask])
            all_bh_mass.append(m_bh)
            all_stellar_mass.append(m_stellar)
            all_mvir.append(m_mvir)
            all_bh_max_accretion_history.append(bh_max_accr_hist)
            all_bh_eddington_rate_limit.append(bh_eddington_hist)
            all_bh_mass_at_accretion.append(bh_mass_at_accretion_hist)
            all_dt.append(dt_hist)
            all_bh_accretion_type.append(bh_acc_type_hist)

    return (np.concatenate(all_ids), np.concatenate(all_bh_mass),
            np.concatenate(all_stellar_mass), np.concatenate(all_mvir),
            np.concatenate(all_bh_max_accretion_history),
            np.concatenate(all_bh_eddington_rate_limit),
            np.concatenate(all_bh_mass_at_accretion),
            np.concatenate(all_dt),
            np.concatenate(all_bh_accretion_type))

def create_dt_summary_statistics(dt_data, plot_mask, unit_time_in_s=UNIT_TIME_IN_S):
    """Generate summary statistics for dt parameter across snapshots."""

    if dt_data.ndim == 2:
        dt_filtered = dt_data[plot_mask, :]
    else:
        dt_filtered = dt_data[plot_mask]

    if dt_filtered.ndim == 1:
        dt_filtered = dt_filtered.reshape(-1, 1)

    dt_flat = dt_filtered.flatten()
    dt_nonzero = dt_flat[dt_flat > 0]
    dt_years = dt_nonzero * unit_time_in_s / SEC_PER_YEAR

    if len(dt_years) > 0:
        print("\n" + "=" * 70)
        print("TIME-STEP (dt) SUMMARY STATISTICS")
        print("=" * 70)
        print(f"Number of active time-steps: {len(dt_years):,}")
        print(f"Number of zero entries (inactive): {np.sum(dt_flat == 0):,}")
        print(f"Fraction active: {len(dt_years) / len(dt_flat):.2%}")
        print(f"\ndt values in years (non-zero only):")
        print(f"  Min:    {np.min(dt_years):.6e} yr")
        print(f"  Max:    {np.max(dt_years):.6e} yr")
        print(f"  Mean:   {np.mean(dt_years):.6e} yr")
        print(f"  Median: {np.median(dt_years):.6e} yr")
        print(f"  Std:    {np.std(dt_years):.6e} yr")
        print(f"  P25:    {np.percentile(dt_years, 25):.6e} yr")
        print(f"  P75:    {np.percentile(dt_years, 75):.6e} yr")
        print("=" * 70)

        return dt_years, unit_time_in_s
    else:
        print("\nWARNING: No valid dt data found.")
        return None, None


def create_eddington_redshift_plot(
    bh_max_accr_history,
    bh_eddington_rate_limit,
    plot_mask,
    snap_to_z_dict,
    total_galaxies,
    output_file,
    eddington_threshold=0.0,
    n_bins_z=20
):
    """Plot: Fraction of super-Eddington BHs vs redshift (time-series)"""

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.minorticks_on()

    bh_hist_filtered = bh_max_accr_history[plot_mask]
    bh_edd_filtered = bh_eddington_rate_limit[plot_mask]

    if bh_hist_filtered.ndim == 1:
        bh_hist_filtered = bh_hist_filtered.reshape(-1, 1)
    if bh_edd_filtered.ndim == 1:
        bh_edd_filtered = bh_edd_filtered.reshape(-1, 1)

    if bh_hist_filtered.shape != bh_edd_filtered.shape:
        max_shape = (bh_hist_filtered.shape[0], max(bh_hist_filtered.shape[1], bh_edd_filtered.shape[1]))
        bh_hist_padded = np.zeros(max_shape)
        bh_edd_padded = np.zeros(max_shape)
        bh_hist_padded[:, :bh_hist_filtered.shape[1]] = bh_hist_filtered
        bh_edd_padded[:, :bh_edd_filtered.shape[1]] = bh_edd_filtered
        bh_hist_filtered = bh_hist_padded
        bh_edd_filtered = bh_edd_padded

    ngal = bh_hist_filtered.shape[0]
    maxsnaps = bh_hist_filtered.shape[1]

    print(f"Processing {ngal} galaxies with {maxsnaps} snapshots each")
    print(
        "Counting super-Eddington attempts as: "
        "BHMaxaccretionRate[snap] > BHEddingtonRateLimit[snap]"
    )

    snap_counts_edd = np.zeros(maxsnaps)

    for snap_idx in range(maxsnaps):
        edd_at_snap = bh_hist_filtered[:, snap_idx] > bh_edd_filtered[:, snap_idx]
        snap_counts_edd[snap_idx] = np.sum(edd_at_snap)

    redshifts = np.array([
        get_redshift_from_snapshot(s, snap_to_z_dict)
        for s in range(maxsnaps)
    ])

    valid_snap_mask = (redshifts >= 0.0) & (redshifts <= 7.0)
    redshifts = redshifts[valid_snap_mask]
    snap_counts_edd = snap_counts_edd[valid_snap_mask]

    z_bins = np.linspace(redshifts.min(), redshifts.max(), n_bins_z + 1)
    z_bin_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

    binned_counts_edd = np.zeros(n_bins_z)

    for i in range(n_bins_z):
        snaps_in_bin = np.where(
            (redshifts >= z_bins[i]) &
            (redshifts < z_bins[i + 1])
        )[0]
        if len(snaps_in_bin) > 0:
            binned_counts_edd[i] = np.sum(snap_counts_edd[snaps_in_bin])

    fraction = binned_counts_edd / total_galaxies

    ax.step(
        z_bin_centers, fraction,
        where='mid', linewidth=2.5, color='#D32F2F',
        marker='o', markersize=7, label='Super-Eddington BHs'
    )
    ax.fill_between(z_bin_centers, fraction, step='mid', alpha=0.2, color='#D32F2F')

    ax.set_xlabel('Redshift (z)', fontsize=14)
    ax.set_ylabel('Fraction of BHs', fontsize=14)
    ax.set_xlim(z_bins[0], z_bins[-1])
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')

    print(f"✓ Eddington vs redshift plot saved to: {output_file}")
    print(f"  Total super-Eddington snapshot occurrences: "
          f"{int(np.sum(snap_counts_edd))}")

    plt.close()


def create_eddington_ratio_histogram(
    bh_max_accr_history,
    bh_eddington_rate_limit,
    plot_mask,
    output_file
):
    """Plot: Histogram of accretion rate / Eddington limit ratios"""

    from matplotlib.ticker import ScalarFormatter

    bh_max_accr = bh_max_accr_history[plot_mask]
    bh_eddington = bh_eddington_rate_limit[plot_mask]

    if bh_max_accr.ndim == 1:
        bh_max_accr = bh_max_accr.reshape(-1, 1)
    if bh_eddington.ndim == 1:
        bh_eddington = bh_eddington.reshape(-1, 1)

    max_accr_flat = bh_max_accr.flatten()
    eddington_flat = bh_eddington.flatten()

    exceeded_mask = max_accr_flat > eddington_flat
    max_accr_exceeded = max_accr_flat[exceeded_mask]
    eddington_exceeded = eddington_flat[exceeded_mask]

    if len(max_accr_exceeded) == 0:
        print("WARNING: No cases exceeding Eddington limit found.")
        return

    valid_ratio_mask = (eddington_exceeded > 0) & np.isfinite(eddington_exceeded)
    max_accr_valid = max_accr_exceeded[valid_ratio_mask]
    eddington_valid = eddington_exceeded[valid_ratio_mask]

    if len(max_accr_valid) == 0:
        print("WARNING: No valid Eddington ratios found.")
        return

    ratios = max_accr_valid / eddington_valid
    ratios = ratios[np.isfinite(ratios)]
    ratios = ratios[ratios > 0]

    if len(ratios) == 0:
        print("WARNING: No finite positive ratios.")
        return

    mean_ratio = np.mean(ratios)
    median_ratio = np.median(ratios)
    max_ratio = np.max(ratios)
    min_ratio = np.min(ratios)
    std_ratio = np.std(ratios)

    std_filter_mask = (ratios >= mean_ratio - std_ratio) & (ratios <= mean_ratio + std_ratio)
    ratios_filtered = ratios[std_filter_mask]
    ratios_filtered = ratios_filtered[ratios_filtered > 0]

    if len(ratios_filtered) == 0:
        print("WARNING: No ratios survived filtering.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    n_bins = int(np.sqrt(len(ratios_filtered)))
    n_bins = max(10, min(n_bins, 50))

    xmin = np.min(ratios_filtered)
    xmax = np.max(ratios_filtered)

    bins = np.logspace(np.log10(xmin), np.log10(xmax), n_bins)

    counts, bins, patches = ax.hist(
        ratios_filtered, bins=bins,
        edgecolor='black', alpha=0.7, color='#1976D2', linewidth=1.2
    )

    ax.axvline(mean_ratio, color='#D32F2F', linestyle='--', linewidth=2.5,
               label=f'Mean = {mean_ratio:.2f}')
    ax.axvline(median_ratio, color='#F57C00', linestyle='--', linewidth=2.5,
               label=f'Median = {median_ratio:.2f}')

    ax.set_xlabel('Accretion Rate / Eddington Limit', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='x')
    ax.set_xticks([1, 10, 100])
    ax.set_xlim(xmin, xmax)
    ax.legend(loc='upper center', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
    ax.minorticks_on()

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')

    print(f"✓ Eddington ratio histogram saved to: {output_file}")
    print("\n" + "=" * 70)
    print("EDDINGTON ACCRETION RATE RATIO ANALYSIS")
    print("=" * 70)
    print(f"Number of cases exceeding Eddington limit: {len(ratios):,}")
    print(f"Cases plotted (within mean ± 1σ): {len(ratios_filtered):,}")
    print(f"Mean ratio: {mean_ratio:.6f}")
    print(f"Median ratio: {median_ratio:.6f}")
    print(f"Max ratio: {max_ratio:.6f}")
    print(f"Min ratio: {min_ratio:.6f}")
    print(f"Std deviation: {std_ratio:.6f}")
    print("=" * 70)

    plt.close()


def create_eddington_ratio_vs_dt_plot(
    bh_max_accr_history,
    bh_eddington_rate_limit,
    dt_data,
    plot_mask,
    output_file,
    unit_time_in_s=UNIT_TIME_IN_S
):
    """Plot: Accretion rate / Eddington limit ratio vs dt (time-step)"""

    from matplotlib.ticker import ScalarFormatter

    bh_max_accr = bh_max_accr_history[plot_mask]
    bh_eddington = bh_eddington_rate_limit[plot_mask]
    dt = dt_data[plot_mask]

    if bh_max_accr.ndim == 1:
        bh_max_accr = bh_max_accr.reshape(-1, 1)
    if bh_eddington.ndim == 1:
        bh_eddington = bh_eddington.reshape(-1, 1)
    if dt.ndim == 1:
        dt = dt.reshape(-1, 1)

    max_accr_flat = bh_max_accr.flatten()
    eddington_flat = bh_eddington.flatten()
    dt_flat = dt.flatten()

    exceeded_mask = (max_accr_flat > eddington_flat) & (eddington_flat > 0) & (dt_flat > 0)

    max_accr_exceeded = max_accr_flat[exceeded_mask]
    eddington_exceeded = eddington_flat[exceeded_mask]
    dt_exceeded = dt_flat[exceeded_mask]

    if len(max_accr_exceeded) == 0:
        print("WARNING: No cases exceeding Eddington limit found for ratio vs dt plot.")
        return

    ratios = max_accr_exceeded / eddington_exceeded
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]

    dt_years = dt_exceeded[:len(ratios)] * unit_time_in_s / SEC_PER_YEAR
    dt_million_years = dt_years / 1e6

    if len(ratios) == 0:
        print("WARNING: No valid ratios for dt plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    log_ratios = np.log10(ratios)

    ax.scatter(dt_million_years, log_ratios, alpha=0.5, s=20,
               color='#1976D2', edgecolors='none')

    dt_sorted_idx = np.argsort(dt_million_years)
    dt_sorted = dt_million_years[dt_sorted_idx]
    log_ratios_sorted = log_ratios[dt_sorted_idx]

    n_bins_dt = min(20, len(dt_sorted) // 10)
    if n_bins_dt > 2:
        dt_bin_edges = np.percentile(dt_sorted, np.linspace(0, 100, n_bins_dt + 1))
        dt_bin_centers = []
        log_ratio_medians = []

        for i in range(n_bins_dt):
            if i == n_bins_dt - 1:
                mask_bin = (dt_sorted >= dt_bin_edges[i]) & (dt_sorted <= dt_bin_edges[i+1])
            else:
                mask_bin = (dt_sorted >= dt_bin_edges[i]) & (dt_sorted < dt_bin_edges[i+1])
            if np.sum(mask_bin) > 0:
                dt_bin_centers.append(np.median(dt_sorted[mask_bin]))
                log_ratio_medians.append(np.median(log_ratios_sorted[mask_bin]))

        if len(dt_bin_centers) > 1:
            ax.plot(dt_bin_centers, log_ratio_medians, color='#D32F2F', linewidth=2.5,
                    marker='o', markersize=8, label='Binned median', zorder=10)

    ax.set_xlabel('Time-step dt (million years)', fontsize=14)
    ax.set_ylabel('log$_{10}$(Accretion Rate / Eddington Limit)', fontsize=14)
    ax.set_xscale('linear')
    ax.set_ylim(0, 7)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')

    print(f"✓ Eddington ratio vs dt plot saved to: {output_file}")
    print("\n" + "=" * 70)
    print("EDDINGTON RATIO vs TIME-STEP (dt) ANALYSIS")
    print("=" * 70)
    print(f"Number of data points: {len(ratios):,}")
    print(f"\nlog₁₀(Ratio) statistics:")
    print(f"  Mean: {np.mean(log_ratios):.6f}")
    print(f"  Median: {np.median(log_ratios):.6f}")
    print(f"  Std Dev: {np.std(log_ratios):.6f}")
    print(f"\ndt statistics (million years):")
    print(f"  Min: {np.min(dt_million_years):.6e} Myr")
    print(f"  Max: {np.max(dt_million_years):.6e} Myr")
    print(f"  Mean: {np.mean(dt_million_years):.6e} Myr")
    print(f"  Median: {np.median(dt_million_years):.6e} Myr")
    print(f"\nCorrelation Analysis (linear dt vs log₁₀(ratio)):")
    corr_coeff = np.corrcoef(dt_million_years, log_ratios)[0, 1]
    print(f"  Pearson correlation: {corr_coeff:.4f}")
    coeffs = np.polyfit(dt_million_years, log_ratios, 1)
    print(f"  Linear fit: log₁₀(ratio) = {coeffs[0]:.6f} * dt + {coeffs[1]:.4f}")
    print(f"  Slope: {coeffs[0]:.6f} per million years")
    print("=" * 70)

    plt.close()


# ============================================================================
# NEW: ACCRETION RATE FUNCTION
# ============================================================================
def create_accretion_rate_function(
    bh_max_accr_history,
    bh_eddington_rate_limit,
    bh_accretion_type_history,
    plot_mask,
    output_file,
    sim_volume_mpc3=None,
    n_bins=40,
    lambda_min=1e-4,
    lambda_max=1e4,
    label=None,
):
    """
    Plot the accretion rate function (analogous to a BH mass function):
        log10(dN / d log10 lambda) vs log10(lambda)
    Split by Radio Mode (0) and Quasar Mode (1).
    """

    # ------------------------------------------------------------------
    # 1. Select and flatten
    # ------------------------------------------------------------------
    accr = bh_max_accr_history[plot_mask]
    edd  = bh_eddington_rate_limit[plot_mask]
    acc_type = bh_accretion_type_history[plot_mask]

    if accr.ndim == 1:
        accr = accr.reshape(-1, 1)
    if edd.ndim == 1:
        edd = edd.reshape(-1, 1)
    if acc_type.ndim == 1:
        acc_type = acc_type.reshape(-1, 1)

    accr_flat = accr.flatten()
    edd_flat  = edd.flatten()
    acc_type_flat = acc_type.flatten()

    # ------------------------------------------------------------------
    # 2. Validity mask
    # ------------------------------------------------------------------
    valid = (
        (accr_flat > 0)
        & (edd_flat  > 0)
        & np.isfinite(accr_flat)
        & np.isfinite(edd_flat)
    )

    accr_valid = accr_flat[valid]
    edd_valid  = edd_flat[valid]
    type_valid = acc_type_flat[valid]

    if len(accr_valid) == 0:
        print("WARNING: No valid accretion events found. Skipping accretion rate function plot.")
        return

    # ------------------------------------------------------------------
    # 3. lambda and log10(lambda)
    # ------------------------------------------------------------------
    lam     = accr_valid / edd_valid
    log_lam = np.log10(lam)

    n_total   = len(log_lam)
    
    # Isolate sub-populations
    mask_radio  = (type_valid == 0)
    mask_quasar = (type_valid == 1)
    
    n_radio = np.sum(mask_radio)
    n_quasar = np.sum(mask_quasar)

    # ------------------------------------------------------------------
    # 4. Bin Setup
    # ------------------------------------------------------------------
    log_lam_min = np.floor(log_lam.min() * 2) / 2
    log_lam_max = np.ceil(log_lam.max()  * 2) / 2
    bins        = np.linspace(log_lam_min, log_lam_max, n_bins + 1)
    bin_width   = bins[1] - bins[0]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    if sim_volume_mpc3 is not None:
        y_label = (r'$\log_{10}\left(\frac{\mathrm{d}N}{\mathrm{d}\log_{10}\lambda}'
                   r'\ /\ \mathrm{Mpc}^{-3}\,h^{3}\right)$')
    else:
        y_label = r'$\log_{10}\left(\frac{\mathrm{d}N}{\mathrm{d}\log_{10}\lambda}\right)$'

    # ------------------------------------------------------------------
    # 5. Process categories and plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.34, 6.25))
    ax.minorticks_on()
    
    categories = [
        {'data': log_lam, 'label': 'Total', 'color': '#1976D2', 'z': 3, 'alpha': 0.10, 'lw': 2.5},          # Blue
        {'data': log_lam[mask_quasar], 'label': 'Quasar Mode (1)', 'color': '#D32F2F', 'z': 4, 'alpha': 0.15, 'lw': 2.0}, # Red
        {'data': log_lam[mask_radio], 'label': 'Radio Mode (0)', 'color': '#388E3C', 'z': 5, 'alpha': 0.15, 'lw': 2.0},   # Green
    ]
    
    plot_data_store = []
    global_min_y = np.inf
    
    # Compute histograms and errors for all lines first to find safe global min Y
    for cat in categories:
        counts, _ = np.histogram(cat['data'], bins=bins)
        
        if sim_volume_mpc3 is not None:
            y = counts / (bin_width * sim_volume_mpc3)
        else:
            y = counts / bin_width
            
        positive = y > 0
        if not np.any(positive):
            plot_data_store.append(None)
            continue
            
        log_y = np.full_like(y, np.nan, dtype=float)
        log_y[positive] = np.log10(y[positive])
        
        global_min_y = min(global_min_y, np.nanmin(log_y[positive]))
        
        # Poisson errors propagated into log space
        sigma_counts = np.sqrt(counts.astype(float))
        sigma_y = sigma_counts / (bin_width * sim_volume_mpc3) if sim_volume_mpc3 else sigma_counts / bin_width
        
        err_up   = np.full_like(y, np.nan, dtype=float)
        err_down = np.full_like(y, np.nan, dtype=float)
        err_up[positive] = np.log10(y[positive] + sigma_y[positive]) - log_y[positive]
        
        idx = np.where(positive)[0]
        lower_safe = (y[positive] - sigma_y[positive]) > 0
        for i, safe in zip(idx, lower_safe):
            if safe:
                err_down[i] = log_y[i] - np.log10(y[i] - sigma_y[i])
            else:
                err_down[i] = log_y[i] - np.log10(0.5 * y[i])
                
        plot_data_store.append({
            'log_y': log_y,
            'err_up': err_up,
            'err_down': err_down,
            'positive': positive
        })

    y_floor = global_min_y - 1.5 if not np.isinf(global_min_y) else -10.0
    all_valid_y = []
    
    # Actually draw the lines
    for cat, pdata in zip(categories, plot_data_store):
        if pdata is None: continue
        
        log_y = pdata['log_y']
        positive = pdata['positive']
        err_up = pdata['err_up']
        err_down = pdata['err_down']
        
        all_valid_y.extend(log_y[positive])
        
        # Draw main line
        ax.step(
            bin_centers, log_y,
            where='mid', linewidth=cat['lw'], color=cat['color'],
            label=cat['label'], zorder=cat['z'],
        )
        
        # Draw fill underneath
        ax.fill_between(
            bin_centers, log_y, y_floor,
            step='mid', alpha=cat['alpha'], color=cat['color'], zorder=cat['z']-1
        )
        
        # Draw errorbars
        err_mask = positive & np.isfinite(log_y)
        ax.errorbar(
            bin_centers[err_mask], log_y[err_mask],
            yerr=[err_down[err_mask], err_up[err_mask]],
            fmt='none', ecolor=cat['color'], elinewidth=1.2, capsize=3, zorder=cat['z']+1,
            alpha=0.8
        )

    ax.axvline(
        x=0.0, color='#000000', linestyle='--', linewidth=1.5, alpha=0.75,
        label=r'Eddington limit ($\lambda = 1$)', zorder=2,
    )

    ax.set_xlabel(
        r'$\log_{10}(\dot{M}_\mathrm{BH,\,max}\ /\ \dot{M}_\mathrm{Edd})$',
        fontsize=18,
    )
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_xlim(log_lam_min, log_lam_max)

    if len(all_valid_y) > 0:
        ax.set_ylim(y_floor + 1.0, np.nanmax(all_valid_y) + 0.8)

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.legend(loc='upper right', fontsize=13)
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.6)

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    plt.close()

    print(f"✓ Accretion rate function saved to: {output_file}")
    print("\n" + "=" * 70)
    print("ACCRETION RATE FUNCTION SUMMARY")
    print("=" * 70)
    print(f"  Valid accretion events total:        {n_total:,}")
    print(f"    - Quasar Mode (Type 1):            {n_quasar:,}")
    print(f"    - Radio Mode (Type 0):             {n_radio:,}")
    print(f"  Median log10(lambda):                {np.median(log_lam):.4f}")
    print(f"  Mean   log10(lambda):                {np.mean(log_lam):.4f}")
    print(f"  log10(lambda) range (full):          [{log_lam.min():.2f}, {log_lam.max():.2f}]")
    if sim_volume_mpc3 is not None:
        print(f"  Simulation volume used:              {sim_volume_mpc3:.4e} Mpc^3 h^-3")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-pattern', default='./output/millennium/model_*.hdf5')
    parser.add_argument('-s', '--snapshot', type=int, default=None)
    parser.add_argument('--no-cuts', action='store_true', help='Skip mass cuts (use all galaxies)')
    parser.add_argument('--eddington-threshold', type=float, default=0.0,
                        help='Threshold for BHMaxaccretionRate to count as Eddington exceedance (default: >0)')
    parser.add_argument('--no-timeseries', action='store_true', help='Skip time-series plot')
    parser.add_argument('--no-ratio', action='store_true', help='Skip ratio histogram')
    parser.add_argument('--no-dt-analysis', action='store_true', help='Skip dt analysis and ratio vs dt plot')
    parser.add_argument('--no-rate-function', action='store_true', help='Skip accretion rate function plot')
    parser.add_argument('--sim-volume', type=float, default=None,
                        help=('Comoving simulation volume in Mpc^3 h^-3 for the accretion rate '
                              'function y-axis. Defaults to the Millennium box '
                              f'({MILLENNIUM_BOX_MPC_H}^3 = {MILLENNIUM_BOX_MPC_H**3:.3e} Mpc^3 h^-3). '
                              'Pass 0 to plot raw counts instead of number density.'))
    parser.add_argument('--no-volume', action='store_true',
                        help='Plot raw dN/d(log lambda) counts instead of number density.')
    args = parser.parse_args()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"No files found for {args.input_pattern}"); sys.exit(1)

    sim    = read_simulation_params(file_list[0])
    h_h    = sim['Hubble_h']
    snap_num = args.snapshot if args.snapshot is not None else sim['latest_snapshot']
    id_field = find_id_field(file_list, snap_num)

    redshift = get_redshift_from_snapshot(snap_num)
    print(f"Snapshot: {snap_num} | Redshift: {redshift:.3f} | Hubble_h: {h_h}")
    print("Reading data and tracking growth channels...")

    # Unpack including new bh_acc_type_hist
    ids, bh_mass, stellar_mass, mvir, bh_max_accr_hist, bh_eddington_hist, bh_mass_at_accretion, dt_data, bh_acc_type_hist = \
        read_data(file_list, snap_num, id_field, h_h)

    # ====================================================================
    # RAW HDF5 INSPECTION
    # ====================================================================
    print("\n" + "=" * 70)
    print("Raw HDF5 File Inspection")
    print("=" * 70)

    try:
        with h5py.File(file_list[0], 'r') as hf:
            snap_key = f"Snap_{snap_num}"
            if snap_key in hf:
                grp = hf[snap_key]
                print(f"\nBH-related fields in {snap_key}:")
                bh_fields = sorted([k for k in grp.keys() if 'BH' in k or 'Black' in k or 'dt' in k])
                for field in bh_fields:
                    data   = grp[field][:]
                    flat   = data.flatten()
                    n_nonzero = np.sum(flat > 0)
                    print(f"  {field:35s} shape={str(data.shape):25s} nonzero={n_nonzero:,}/{len(flat):,}")

                if 'BHMassatAccretion' in grp:
                    print(f"\nBHMassatAccretion Raw Data:")
                    raw_data = grp['BHMassatAccretion'][:]
                    flat     = raw_data.flatten()
                    print(f"  Shape: {raw_data.shape}  Dtype: {raw_data.dtype}  "
                          f"Memory: {raw_data.nbytes / (1024**2):.2f} MB")
                    print(f"  Min: {np.min(flat):.6e}  Max: {np.max(flat):.6e}")
                    print(f"  Zero: {np.sum(flat == 0):,}/{len(flat):,}  "
                          f"Non-zero: {np.sum(flat > 0):,}")
                else:
                    print("\n⚠️  BHMassatAccretion NOT FOUND in raw file!")
    except Exception as e:
        print(f"Error inspecting raw file: {e}")

    print("=" * 70)

    # ====================================================================
    # BHMassatAccretion POST-PROCESSING DIAGNOSTICS
    # ====================================================================
    print("\n" + "=" * 70)
    print("BHMassatAccretion After Processing")
    print("=" * 70)

    if bh_mass_at_accretion.ndim == 2:
        ngal, maxsnaps = bh_mass_at_accretion.shape
        current_snap_idx   = min(snap_num, maxsnaps - 1)
        bh_mass_current    = bh_mass_at_accretion[:, current_snap_idx]
        zero_current       = np.sum(bh_mass_current <= 0.0)
        total_current      = len(bh_mass_current)

        print(f"\nAt Snapshot {snap_num} (column {current_snap_idx}):")
        print(f"  Zero: {zero_current:,}/{total_current:,} ({zero_current/total_current:.2%})")
        if total_current - zero_current > 0:
            v = bh_mass_current[bh_mass_current > 0]
            print(f"  Non-zero — min: {v.min():.3e}  max: {v.max():.3e}  mean: {v.mean():.3e}")

        zero_all  = np.sum(bh_mass_at_accretion <= 0.0)
        total_all = bh_mass_at_accretion.size
        print(f"\nAcross ALL {maxsnaps} snapshots:")
        print(f"  Zero: {zero_all:,}/{total_all:,} ({zero_all/total_all:.2%})")
        if total_all - zero_all > 0:
            v = bh_mass_at_accretion[bh_mass_at_accretion > 0]
            print(f"  Non-zero — min: {v.min():.3e}  max: {v.max():.3e}  "
                  f"mean: {v.mean():.3e}  median: {np.median(v):.3e}")
    else:
        zero_m = np.sum(bh_mass_at_accretion <= 0.0)
        total_m = bh_mass_at_accretion.size
        print(f"\nScalar 1D: zero={zero_m:,}/{total_m:,} ({zero_m/total_m:.2%})")

    print("=" * 70)

    # ====================================================================
    # GALAXY SELECTION
    # ====================================================================
    if args.no_cuts:
        plot_mask = (bh_mass > 0)
    else:
        plot_mask = (
            (bh_mass > MIN_Z0_BH_MASS)
            & (stellar_mass > 10**MIN_STELLAR_MASS_LOG)
            & (mvir > 10**MIN_HALO_MASS_LOG)
        )

    print(f"\nInitial galaxies: {len(ids)}")
    print(f"Final plot count: {np.sum(plot_mask)}")

    output_dir = Path(file_list[0]).parent / 'plots'
    output_dir.mkdir(exist_ok=True)

    # Resolve simulation volume for the rate function.
    if args.no_volume:
        sim_volume = None
    elif args.sim_volume is not None:
        sim_volume = args.sim_volume
    else:
        sim_volume = MILLENNIUM_BOX_MPC_H ** 3   # default: 500^3 (Mpc/h)^3

    if sim_volume is not None:
        print(f"\nAccretion rate function: using volume = {sim_volume:.4e} Mpc^3 h^-3")
    else:
        print("\nAccretion rate function: plotting raw counts (no volume normalisation)")

    # ====================================================================
    # PLOTS
    # ====================================================================
    plot_index = 1

    if not args.no_dt_analysis:
        print(f"\n[{plot_index}/5] Computing dt summary statistics...")
        plot_index += 1
        create_dt_summary_statistics(dt_data, plot_mask, UNIT_TIME_IN_S)

    if not args.no_timeseries:
        print(f"\n[{plot_index}/5] Creating time-series plot...")
        plot_index += 1
        create_eddington_redshift_plot(
            bh_max_accr_hist, bh_eddington_hist, plot_mask,
            MILLENNIUM_SNAP_TO_Z, np.sum(plot_mask),
            output_dir / 'bh_eddington_vs_redshift.png',
            eddington_threshold=args.eddington_threshold,
            n_bins_z=20,
        )

    if not args.no_ratio:
        print(f"\n[{plot_index}/5] Creating ratio histogram...")
        plot_index += 1
        create_eddington_ratio_histogram(
            bh_max_accr_hist, bh_eddington_hist, plot_mask,
            output_dir / 'bh_eddington_ratio_histogram.png',
        )

    if not args.no_dt_analysis:
        print(f"\n[{plot_index}/5] Creating ratio vs dt scatter plot...")
        plot_index += 1
        create_eddington_ratio_vs_dt_plot(
            bh_max_accr_hist, bh_eddington_hist, dt_data, plot_mask,
            output_dir / 'bh_eddington_ratio_vs_dt.png',
            unit_time_in_s=UNIT_TIME_IN_S,
        )

    if not args.no_rate_function:
        print(f"\n[{plot_index}/5] Creating accretion rate function (split by mode)...")
        create_accretion_rate_function(
            bh_max_accr_hist, bh_eddington_hist, bh_acc_type_hist, plot_mask,
            output_dir / 'bh_accretion_rate_function.png',
            sim_volume_mpc3=sim_volume,
            n_bins=40,
            lambda_min=1e-4,
            lambda_max=1e4,
        )

    print("\n✓ All plots completed successfully!")


if __name__ == "__main__":
    main()