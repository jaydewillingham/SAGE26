#!/usr/bin/env python3
"""
Black hole Eddington limit analysis with:
1. Time-series: Fraction of super-Eddington BHs vs redshift
2. Distribution: Histogram of accretion rate ratios (where rate > Eddington limit)
"""
import argparse
import glob
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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
MIN_HALO_MASS_LOG = 10.5
MIN_Z0_BH_MASS = 1e4

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
    """Reads fields including BHMaxaccretionRate (time-series array Ngal x MAXSNAPS) and BHEddingtonRateLimit."""
    all_ids = []
    all_bh_mass = []
    all_stellar_mass = []
    all_mvir = []
    all_bh_seed = []
    all_first_channel = []
    all_birth_snaps = []
    all_birth_vals = []
    all_bh_max_accretion_history = []  # Ngal x MAXSNAPS
    all_bh_eddington_rate_limit = []   # Ngal x MAXSNAPS

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

            # Read BHMaxaccretionRate as time-series (Ngal x MAXSNAPS)
            if 'BHMaxaccretionRate' in grp:
                bh_max_accr_raw = grp['BHMaxaccretionRate'][:][mask] * conv
            else:
                # Fallback: create empty time-series
                bh_max_accr_raw = np.zeros((len(m_bh), 63))  # 63 = MAXSNAPS for Millennium
            
            # Handle shape: if 1D, reshape to 2D
            if bh_max_accr_raw.ndim == 1:
                ngal = len(m_bh)
                if len(bh_max_accr_raw) == ngal:
                    # Scalar case - shouldn't happen for history
                    bh_max_accr_hist = bh_max_accr_raw.reshape(-1, 1)
                else:
                    # Flattened case: reshape to (ngal, maxsnaps)
                    maxsnaps = len(bh_max_accr_raw) // ngal
                    bh_max_accr_hist = bh_max_accr_raw.reshape(ngal, maxsnaps)
            else:
                bh_max_accr_hist = bh_max_accr_raw

            # Read BHEddingtonRateLimit as time-series (Ngal x MAXSNAPS)
            if 'BHEddingtonRateLimit' in grp:
                bh_eddington_raw = grp['BHEddingtonRateLimit'][:][mask] * conv
            else:
                # Fallback: create empty time-series with same shape as max_accr
                bh_eddington_raw = np.zeros_like(bh_max_accr_hist)
            
            # Handle shape: if 1D, reshape to 2D
            if bh_eddington_raw.ndim == 1:
                ngal = len(m_bh)
                if len(bh_eddington_raw) == ngal:
                    # Scalar case
                    bh_eddington_hist = bh_eddington_raw.reshape(-1, 1)
                else:
                    # Flattened case: reshape to (ngal, maxsnaps)
                    maxsnaps = len(bh_eddington_raw) // ngal
                    bh_eddington_hist = bh_eddington_raw.reshape(ngal, maxsnaps)
            else:
                bh_eddington_hist = bh_eddington_raw
            
            # Ensure both have the same shape
            if bh_max_accr_hist.shape != bh_eddington_hist.shape:
                print(f"WARNING: Shape mismatch in {f}")
                print(f"  BHMaxaccretionRate shape: {bh_max_accr_hist.shape}")
                print(f"  BHEddingtonRateLimit shape: {bh_eddington_hist.shape}")
                # Pad to match shapes
                max_shape = (bh_max_accr_hist.shape[0], max(bh_max_accr_hist.shape[1], bh_eddington_hist.shape[1]))
                bh_max_accr_padded = np.zeros(max_shape)
                bh_eddington_padded = np.zeros(max_shape)
                bh_max_accr_padded[:, :bh_max_accr_hist.shape[1]] = bh_max_accr_hist
                bh_eddington_padded[:, :bh_eddington_hist.shape[1]] = bh_eddington_hist
                bh_max_accr_hist = bh_max_accr_padded
                bh_eddington_hist = bh_eddington_padded

            all_ids.append(gids[mask])
            all_bh_mass.append(m_bh)
            all_stellar_mass.append(m_stellar)
            all_mvir.append(m_mvir)
            all_bh_max_accretion_history.append(bh_max_accr_hist)
            all_bh_eddington_rate_limit.append(bh_eddington_hist)

    return (np.concatenate(all_ids), np.concatenate(all_bh_mass), 
            np.concatenate(all_stellar_mass), np.concatenate(all_mvir), 
            np.concatenate(all_bh_max_accretion_history),
            np.concatenate(all_bh_eddington_rate_limit))

def create_eddington_redshift_plot(
    bh_max_accr_history,
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

    # Filter data
    bh_hist_filtered = bh_max_accr_history[plot_mask]

    # Ensure 2D shape
    if bh_hist_filtered.ndim == 1:
        bh_hist_filtered = bh_hist_filtered.reshape(-1, 1)

    ngal = bh_hist_filtered.shape[0]
    maxsnaps = bh_hist_filtered.shape[1]

    print(f"Processing {ngal} galaxies with {maxsnaps} snapshots each")
    print(f"Counting super-Eddington attempts as: "
          f"BHMaxaccretionRate[snap] > {eddington_threshold}")

    # Count BHs with super-Eddington accretion at each snapshot
    snap_counts_edd = np.zeros(maxsnaps)

    for snap_idx in range(maxsnaps):
        # BHMaxaccretionRate > 0 means attempted accretion exceeded Eddington
        edd_at_snap = bh_hist_filtered[:, snap_idx] > eddington_threshold
        snap_counts_edd[snap_idx] = np.sum(edd_at_snap)

    # Convert snapshot indices to redshift
    redshifts = np.array([
        get_redshift_from_snapshot(s, snap_to_z_dict)
        for s in range(maxsnaps)
    ])

    # Restrict analysis to 0 <= z <= 7
    valid_snap_mask = (redshifts >= 0.0) & (redshifts <= 7.0)

    redshifts = redshifts[valid_snap_mask]
    snap_counts_edd = snap_counts_edd[valid_snap_mask]

    # Bin by redshift
    z_bins = np.linspace(redshifts.min(), redshifts.max(), n_bins_z + 1)
    z_bin_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    z_bin_width = z_bins[1] - z_bins[0]

    binned_counts_edd = np.zeros(n_bins_z)

    for i in range(n_bins_z):
        snaps_in_bin = np.where(
            (redshifts >= z_bins[i]) &
            (redshifts < z_bins[i + 1])
        )[0]

        if len(snaps_in_bin) > 0:
            binned_counts_edd[i] = np.sum(
                snap_counts_edd[snaps_in_bin]
            )

    # Fraction per redshift bin
    fraction = binned_counts_edd / total_galaxies

    # Plot
    ax.step(
        z_bin_centers,
        fraction,
        where='mid',
        linewidth=2.5,
        color='#D32F2F',
        marker='o',
        markersize=7,
        label='Super-Eddington BHs'
    )

    ax.fill_between(
        z_bin_centers,
        fraction,
        step='mid',
        alpha=0.2,
        color='#D32F2F'
    )

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
    """Plot: Histogram of accretion rate / Eddington limit ratios (where rate > limit)"""
    
    # Filter data
    bh_max_accr = bh_max_accr_history[plot_mask]
    bh_eddington = bh_eddington_rate_limit[plot_mask]
    
    # Ensure 2D shape
    if bh_max_accr.ndim == 1:
        bh_max_accr = bh_max_accr.reshape(-1, 1)
    if bh_eddington.ndim == 1:
        bh_eddington = bh_eddington.reshape(-1, 1)
    
    # Flatten arrays to get all time-series snapshots
    max_accr_flat = bh_max_accr.flatten()
    eddington_flat = bh_eddington.flatten()
    
    # Filter for cases where BHMaxaccretionRate > 0
    # Per your C code: BHMaxaccretionRate is ONLY written when accretion_rate > edd_rate
    # So this gives us all cases that actually exceeded the Eddington limit
    exceeded_mask = max_accr_flat > 0
    max_accr_exceeded = max_accr_flat[exceeded_mask]
    eddington_exceeded = eddington_flat[exceeded_mask]
    
    if len(max_accr_exceeded) == 0:
        print("WARNING: No cases where accretion exceeded Eddington limit found.")
        return
    
    # Filter out cases where BHEddingtonRateLimit is zero or NaN (to avoid division by zero)
    valid_ratio_mask = (eddington_exceeded > 0) & np.isfinite(eddington_exceeded)
    max_accr_valid = max_accr_exceeded[valid_ratio_mask]
    eddington_valid = eddington_exceeded[valid_ratio_mask]
    
    print(f"\nEddington ratio histogram data:")
    print(f"  Cases where BHMaxaccretionRate was written: {len(max_accr_exceeded):,}")
    print(f"  Cases with valid Eddington limit (>0 and finite): {len(max_accr_valid):,}")
    print(f"  (These are all cases that exceeded Eddington limit)")
    
    if len(max_accr_valid) == 0:
        print("WARNING: No valid ratios found (all Eddington limits are zero or NaN).")
        return
    
    # Calculate ratios for all exceeded cases with valid limits
    ratios = max_accr_valid / eddington_valid
    
    if len(ratios) == 0:
        print("WARNING: No ratios to compute.")
        return
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Determine bins
    n_bins = int(np.sqrt(len(ratios)))
    n_bins = max(10, min(n_bins, 50))
    
    # Calculate statistics BEFORE filtering by std dev
    mean_ratio = np.mean(ratios)
    median_ratio = np.median(ratios)
    max_ratio = np.max(ratios)
    std_ratio = np.std(ratios)
    
    # Filter ratios to only those within mean ± 1 std dev for plotting
    std_filter_mask = (ratios >= mean_ratio - std_ratio) & (ratios <= mean_ratio + std_ratio)
    ratios_filtered = ratios[std_filter_mask]
    
    counts, bins, patches = ax.hist(
        ratios_filtered,
        bins=n_bins,
        edgecolor='black',
        alpha=0.7,
        color='#1976D2',
        linewidth=1.2
    )
    
    # Add vertical lines for statistics
    ax.axvline(mean_ratio, color='#D32F2F', linestyle='--', linewidth=2.5, 
               label=f'Mean = {mean_ratio:.2f}')
    ax.axvline(median_ratio, color='#F57C00', linestyle='--', linewidth=2.5, 
               label=f'Median = {median_ratio:.2f}')
    
    # Labels and formatting
    ax.set_xlabel('Accretion Rate / Eddington Limit', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    
    # Set x-axis limits to mean ± 1 standard deviation
    x_min = mean_ratio - std_ratio
    x_max = mean_ratio + std_ratio
    ax.set_xlim(1.0, x_max)
    
    # Place legend in upper left to avoid overlapping with histogram
    ax.legend(loc='upper center', fontsize=11)
    
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
    ax.minorticks_on()
    
    # Add statistics text box
    stats_text = (
        f'N = {len(ratios):,}\n'
        f'N(plotted) = {len(ratios_filtered):,}\n'
        f'Std Dev = {std_ratio:.2f}'
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=11, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    
    print(f"✓ Eddington ratio histogram saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("EDDINGTON ACCRETION RATE RATIO ANALYSIS")
    print("="*70)
    print(f"Number of cases exceeding Eddington limit: {len(ratios):,}")
    print(f"Cases plotted (within mean ± 1σ): {len(ratios_filtered):,}")
    print(f"Mean ratio (max_accretion / eddington_limit): {mean_ratio:.6f}")
    print(f"Median ratio: {median_ratio:.6f}")
    print(f"Max ratio: {max_ratio:.6f}")
    print(f"Min ratio: {np.min(ratios):.6f}")
    print(f"Std deviation: {std_ratio:.6f}")
    print("="*70)
    
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-pattern', default='./output/millennium/model_*.hdf5')
    parser.add_argument('-s', '--snapshot', type=int, default=None)
    parser.add_argument('--no-cuts', action='store_true', help='Skip mass cuts (use all galaxies)')
    parser.add_argument('--eddington-threshold', type=float, default=0.0,
                       help='Threshold for BHMaxaccretionRate to count as Eddington exceedance (default: >0)')
    parser.add_argument('--no-timeseries', action='store_true', help='Skip time-series plot')
    parser.add_argument('--no-ratio', action='store_true', help='Skip ratio histogram')
    args = parser.parse_args()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"No files found for {args.input_pattern}"); sys.exit(1)

    sim = read_simulation_params(file_list[0])
    h_h = sim['Hubble_h']
    snap_num = args.snapshot if args.snapshot is not None else sim['latest_snapshot']
    id_field = find_id_field(file_list, snap_num)
    
    redshift = get_redshift_from_snapshot(snap_num)

    print(f"Snapshot: {snap_num} | Redshift: {redshift:.3f} | Hubble_h: {h_h}")
    print("Reading data and tracking growth channels...")
    
    ids, bh_mass, stellar_mass, mvir, bh_max_accr_hist, bh_eddington_hist = \
        read_data(file_list, snap_num, id_field, h_h)

    # Filtering
    if args.no_cuts:
        plot_mask = (bh_mass > 0)
    else:
        plot_mask = (bh_mass > MIN_Z0_BH_MASS) & (stellar_mass > 10**MIN_STELLAR_MASS_LOG) & (mvir > 10**MIN_HALO_MASS_LOG)

    print(f"\nInitial galaxies: {len(ids)}")
    print(f"Final plot count: {np.sum(plot_mask)}")

    output_dir = Path(file_list[0]).parent / 'plots'
    output_dir.mkdir(exist_ok=True)

    # ========================================================================
    # EDDINGTON LIMIT ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("Eddington Limit Analysis")
    print("="*70)
    
    total_passed = np.sum(plot_mask)
    
    # Create time-series Eddington plot
    if not args.no_timeseries:
        print("\n[1/2] Creating time-series plot...")
        create_eddington_redshift_plot(
            bh_max_accr_hist,
            plot_mask,
            MILLENNIUM_SNAP_TO_Z,
            total_passed,
            output_dir / 'bh_eddington_vs_redshift.png',
            eddington_threshold=args.eddington_threshold,
            n_bins_z=20
        )
    
    # Create ratio histogram plot
    if not args.no_ratio:
        print("\n[2/2] Creating ratio histogram...")
        create_eddington_ratio_histogram(
            bh_max_accr_hist,
            bh_eddington_hist,
            plot_mask,
            output_dir / 'bh_eddington_ratio_histogram.png'
        )

    print("\n✓ All plots completed successfully!")

if __name__ == "__main__":
    main()