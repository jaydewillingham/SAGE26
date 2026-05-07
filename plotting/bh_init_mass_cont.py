#!/usr/bin/env python3
"""
L-Galaxies BH Seed Population Analysis - Enhanced (FIXED)

Analyzes BH seed masses, splits them into populations based on the 
first growth channel (Merger-driven vs Instability-driven), and
creates both 1D histograms and 2D density plots (seed mass vs redshift).

REDSHIFT CALCULATION:
Redshifts are derived from snapshot numbers using a cosmological 
scale factor mapping. This script includes a built-in redshift lookup 
table for the Millennium simulation (or you can provide your own).
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
MONSTER_SEED_THRESHOLD = 1e9  # M_sun

# ============================================================================
# REDSHIFT MAPPING - MILLENNIUM SIMULATION
# ============================================================================
# This is the default redshift-snapshot mapping for the Millennium simulation
# Snapshot 0 = z=127, Snapshot 62 = z=0 (present day)
# Source: http://www.mpa-garching.mpg.de/millennium/snapshots.php
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
    """
    Map snapshot number to redshift.
    
    If snap_to_z_dict is None, uses the default Millennium mapping.
    You can provide your own dictionary if using a different simulation.
    """
    if snap_to_z_dict is None:
        snap_to_z_dict = MILLENNIUM_SNAP_TO_Z
    
    if snap_num in snap_to_z_dict:
        return snap_to_z_dict[snap_num]
    else:
        # If snapshot not in table, try linear interpolation
        sorted_snaps = sorted(snap_to_z_dict.keys())
        if snap_num < sorted_snaps[0]:
            return snap_to_z_dict[sorted_snaps[0]]
        if snap_num > sorted_snaps[-1]:
            return snap_to_z_dict[sorted_snaps[-1]]
        
        # Find surrounding snapshots for interpolation
        for i in range(len(sorted_snaps) - 1):
            if sorted_snaps[i] <= snap_num <= sorted_snaps[i+1]:
                s1, s2 = sorted_snaps[i], sorted_snaps[i+1]
                z1, z2 = snap_to_z_dict[s1], snap_to_z_dict[s2]
                z = z1 + (snap_num - s1) * (z2 - z1) / (s2 - s1)
                return z
    
    return 0.0  # Fallback

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
    """Reads fields and determines the first growth channel for each galaxy."""
    all_ids = []
    all_bh_mass = []
    all_stellar_mass = []
    all_mvir = []
    all_bh_seed = []
    all_first_channel = [] # 0: MD, 1: ID, 2: Other/None
    all_birth_snaps = []
    all_birth_vals = [] # List of dicts or arrays for MD, ID, RM, BM

    seen_ids = set()

    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f"Snap_{snap_num}"
            if snap_key not in hf: continue
            grp = hf[snap_key]

            # Standard fields
            gids = grp[id_field][:]
            mask = np.array([gid not in seen_ids for gid in gids])
            for gid in gids[mask]: seen_ids.add(gid)
            if not np.any(mask): continue

            # Unit conversion 1e10/h
            conv = 1e10 / h_h
            
            m_bh = grp['BlackHoleMass'][:][mask] * conv
            m_stellar = grp['StellarMass'][:][mask] * conv
            m_mvir = grp['Mvir'][:][mask] * conv
            
            if 'BHSeedMass' in grp:
                m_seed = grp['BHSeedMass'][:][mask] * conv
            else:
                m_seed = np.full_like(m_bh, 1e4) # Fallback 10^4 M_sun

            # Growth histories to determine first channel
            # These are usually (Ngal, MAXSNAPS)
            hist_md = grp['MergerDrivenBHaccretionMass'][:][mask] if 'MergerDrivenBHaccretionMass' in grp else None
            hist_id = grp['InstabilityDrivenBHaccretionMass'][:][mask] if 'InstabilityDrivenBHaccretionMass' in grp else None
            hist_rm = grp['RadioModeBHaccretionMass'][:][mask] if 'RadioModeBHaccretionMass' in grp else None
            hist_bm = grp['BHMergerMass'][:][mask] if 'BHMergerMass' in grp else None

            # Determine channel and birth snap
            channels = []
            birth_snaps = []
            birth_vals = []

            for i in range(len(m_bh)):
                chan = 2 
                b_snap = snap_num
                b_v = [0.0, 0.0, 0.0, 0.0] # MD, ID, RM, BM

                # Check for first nonzero snap in history
                if hist_md is not None and hist_id is not None:
                    # Combine all major growth fields to find birth
                    combined = hist_md[i] + hist_id[i]
                    if hist_rm is not None: combined += hist_rm[i]
                    if hist_bm is not None: combined += hist_bm[i]
                    
                    nonzero = np.where(combined > 0)[0]
                    if len(nonzero) > 0:
                        b_snap = nonzero[0]
                        # Determine primary channel at birth
                        v_md = hist_md[i, b_snap] * conv
                        v_id = hist_id[i, b_snap] * conv
                        v_rm = (hist_rm[i, b_snap] if hist_rm is not None else 0.0) * conv
                        v_bm = (hist_bm[i, b_snap] if hist_bm is not None else 0.0) * conv
                        
                        b_v = [v_md, v_id, v_rm, v_bm]
                        
                        if v_md >= v_id:
                            chan = 0 # MD
                        else:
                            chan = 1 # ID
                    else:
                        # Fallback to current snap values if no history found
                        v_md = (hist_md[i] if hist_md.ndim == 1 else hist_md[i, -1]) * conv
                        v_id = (hist_id[i] if hist_id.ndim == 1 else hist_id[i, -1]) * conv
                        b_v = [v_md, v_id, 0.0, 0.0]

                channels.append(chan)
                birth_snaps.append(b_snap)
                birth_vals.append(b_v)

            all_ids.append(gids[mask])
            all_bh_mass.append(m_bh)
            all_stellar_mass.append(m_stellar)
            all_mvir.append(m_mvir)
            all_bh_seed.append(m_seed)
            all_first_channel.append(np.array(channels))
            all_birth_snaps.append(np.array(birth_snaps))
            all_birth_vals.append(np.array(birth_vals))

    return (np.concatenate(all_ids), np.concatenate(all_bh_mass), 
            np.concatenate(all_stellar_mass), np.concatenate(all_mvir), 
            np.concatenate(all_bh_seed), np.concatenate(all_first_channel),
            np.concatenate(all_birth_snaps), np.concatenate(all_birth_vals))

def create_2d_density_plot_combined(
    x_data_md,
    y_data_md,
    x_data_id,
    y_data_id,
    title,
    xlabel,
    ylabel,
    filename,
    n_bins_x=30,
    n_bins_y=30,
    total_passed=None
):

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(14, 6),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    for ax in [ax1, ax2]:
        ax.minorticks_on()

    # ------------------------------------------------------------------
    # Combined valid data
    # ------------------------------------------------------------------
    x_all = np.concatenate([x_data_md, x_data_id])
    y_all = np.concatenate([y_data_md, y_data_id])

    valid_mask_all = np.isfinite(x_all) & np.isfinite(y_all)

    x_valid_all = x_all[valid_mask_all]
    y_valid_all = y_all[valid_mask_all]

    if len(x_valid_all) <= 1:
        print("⚠ Warning: Not enough valid data for combined plot")
        plt.close()
        return

    # ------------------------------------------------------------------
    # Shared bins
    # ------------------------------------------------------------------
    x_bins = np.linspace(
        x_valid_all.min(),
        x_valid_all.max(),
        n_bins_x + 1
    )

    y_bins = np.linspace(
        y_valid_all.min(),
        y_valid_all.max(),
        n_bins_y + 1
    )

    x_bin_width = (x_bins[-1] - x_bins[0]) / n_bins_x
    y_bin_width = (y_bins[-1] - y_bins[0]) / n_bins_y
    bin_area = x_bin_width * y_bin_width

    if total_passed is None:
        total_passed = len(x_valid_all)

    # ------------------------------------------------------------------
    # Filter MD population
    # ------------------------------------------------------------------
    valid_mask_md = (
        np.isfinite(x_data_md) &
        np.isfinite(y_data_md)
    )

    x_md_valid = x_data_md[valid_mask_md]
    y_md_valid = y_data_md[valid_mask_md]

    # ------------------------------------------------------------------
    # Filter ID population
    # ------------------------------------------------------------------
    valid_mask_id = (
        np.isfinite(x_data_id) &
        np.isfinite(y_data_id)
    )

    x_id_valid = x_data_id[valid_mask_id]
    y_id_valid = y_data_id[valid_mask_id]

    # ------------------------------------------------------------------
    # Compute 2D histograms
    # ------------------------------------------------------------------
    h_md, _, _ = np.histogram2d(
        x_md_valid,
        y_md_valid,
        bins=[x_bins, y_bins]
    )

    h_id, _, _ = np.histogram2d(
        x_id_valid,
        y_id_valid,
        bins=[x_bins, y_bins]
    )

    # ------------------------------------------------------------------
    # Density normalization
    # ------------------------------------------------------------------
    h_md_density = h_md / (total_passed * bin_area)
    h_id_density = h_id / (total_passed * bin_area)

    # ------------------------------------------------------------------
    # Convert to log10(density)
    # ------------------------------------------------------------------
    h_md_plot = np.where(
        h_md_density > 0,
        np.log10(h_md_density),
        np.nan
    )

    h_id_plot = np.where(
        h_id_density > 0,
        np.log10(h_id_density),
        np.nan
    )

    # ------------------------------------------------------------------
    # Shared colour scale
    # ------------------------------------------------------------------
    combined_vals = np.concatenate([
        h_md_plot[np.isfinite(h_md_plot)],
        h_id_plot[np.isfinite(h_id_plot)]
    ])

    if len(combined_vals) > 0:
        vmin = combined_vals.min()
        vmax = combined_vals.max()
    else:
        vmin = -5
        vmax = 0

    extent = [
        x_bins[0],
        x_bins[-1],
        y_bins[0],
        y_bins[-1]
    ]

    # ------------------------------------------------------------------
    # Merger-driven panel
    # ------------------------------------------------------------------
    im1 = ax1.imshow(
        h_md_plot.T,
        extent=extent,
        origin='lower',
        cmap='Blues',
        aspect='auto',
        vmin=vmin,
        vmax=vmax
    )

    cbar1 = fig.colorbar(
        im1,
        ax=ax1,
        pad=0.015,
        shrink=0.92
    )

    # Remove label from left colorbar
    cbar1.set_label("")

    ax1.set_title(
        'Merger-driven',
        fontsize=15,
        pad=8
    )

    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_ylabel(ylabel, fontsize=14)

    # ------------------------------------------------------------------
    # Instability-driven panel
    # ------------------------------------------------------------------
    im2 = ax2.imshow(
        h_id_plot.T,
        extent=extent,
        origin='lower',
        cmap='Oranges',
        aspect='auto',
        vmin=vmin,
        vmax=vmax
    )

    cbar2 = fig.colorbar(
        im2,
        ax=ax2,
        pad=0.015,
        shrink=0.92
    )

    # Add label only to right colorbar
    cbar2.set_label(
        r'$\log_{10}(\mathrm{Density})$',
        fontsize=12
    )

    ax2.set_title(
        'Instability-driven',
        fontsize=15,
        pad=8
    )

    ax2.set_xlabel(xlabel, fontsize=14)

    # Remove duplicated y-axis labels
    ax2.tick_params(labelleft=False)

    # ------------------------------------------------------------------
    # Tighten spacing
    # ------------------------------------------------------------------
    fig.subplots_adjust(
        wspace=0.08,
        left=0.08,
        right=0.96,
        bottom=0.12,
        top=0.90
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    plt.savefig(filename, dpi=140)

    print(f"✓ Combined 2D density plot saved to: {filename}")
    print(
        f"  MD sample: {np.sum(valid_mask_md)} galaxies | "
        f"ID sample: {np.sum(valid_mask_id)} galaxies"
    )

    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-pattern', default='./output/millennium/model_*.hdf5')
    parser.add_argument('-s', '--snapshot', type=int, default=None)
    parser.add_argument('--no-cuts', action='store_true')
    args = parser.parse_args()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"No files found for {args.input_pattern}"); sys.exit(1)

    sim = read_simulation_params(file_list[0])
    h_h = sim['Hubble_h']
    snap_num = args.snapshot if args.snapshot is not None else sim['latest_snapshot']
    id_field = find_id_field(file_list, snap_num)
    
    # Get redshift for this snapshot
    redshift = get_redshift_from_snapshot(snap_num)

    print(f"Snapshot: {snap_num} | Redshift: {redshift:.3f} | Hubble_h: {h_h}")
    print("Reading data and tracking growth channels...")
    
    ids, bh_mass, stellar_mass, mvir, bh_seed, channels, b_snaps, b_vals = read_data(file_list, snap_num, id_field, h_h)

    # Filtering
    if args.no_cuts:
        mask = (bh_mass > 0)
    else:
        mask = (bh_mass > MIN_Z0_BH_MASS) & (stellar_mass > 10**MIN_STELLAR_MASS_LOG) & (mvir > 10**MIN_HALO_MASS_LOG)

    # Monster Seed Detection
    monster_mask = (bh_seed > MONSTER_SEED_THRESHOLD)
    
    # Final plot mask
    plot_mask = mask & (~monster_mask)

    print(f"\nInitial galaxies: {len(ids)}")
    print(f"Passed cuts: {np.sum(mask)}")
    print(f"Monster seeds skipped: {np.sum(mask & monster_mask)}")
    print(f"Final plot count: {np.sum(plot_mask)}")

    # Print first 5 detailed table
    print("\nDetailed breakdown of first 5 Galaxy IDs in sample:")
    header = f"{'GalaxyID':<15} | {'Seed Mass':<12} | {'Snap':<4} | {'MD':<10} | {'ID':<10} | {'RM':<10} | {'BM':<10}"
    print(header)
    print("-" * len(header))
    
    plot_indices = np.where(plot_mask)[0]
    for idx in plot_indices[:5]:
        gid = ids[idx]
        seed = bh_seed[idx]
        bsnap = b_snaps[idx]
        bv = b_vals[idx]
        print(f"{int(gid):<15} | {seed:1.2e} | {bsnap:<4} | {bv[0]:1.2e} | {bv[1]:1.2e} | {bv[2]:1.2e} | {bv[3]:1.2e}")

    if np.any(monster_mask & mask):
        print("\nTop 5 Monster Seeds (>10^9 M_sun) detected:")
        monster_indices = np.where(mask & monster_mask)[0]
        for idx in monster_indices[:5]:
            print(f" ID: {ids[idx]:<15} | Seed: {bh_seed[idx]:1.2e} | Total BH: {bh_mass[idx]:1.2e}")

    # ========================================================================
    # PLOTTING - 1D HISTOGRAM (Original)
    # ========================================================================
    fig, ax = plt.subplots()
    ax.minorticks_on()
    
    total_passed = np.sum(plot_mask)
    
    # Split into populations for histogram
    def get_log_seeds(raw_seeds):
        # Filter out 0 or negative values to avoid -inf in log10
        valid = raw_seeds[raw_seeds > 0]
        return np.log10(valid)

    log_seed_md = get_log_seeds(bh_seed[plot_mask & (channels == 0)])
    log_seed_id = get_log_seeds(bh_seed[plot_mask & (channels == 1)])
    log_seed_other = get_log_seeds(bh_seed[plot_mask & (channels == 2)])
    all_log_seeds = get_log_seeds(bh_seed[plot_mask])
    
    if len(all_log_seeds) > 0:
        bin_range = (all_log_seeds.min(), all_log_seeds.max())
        bins = np.linspace(bin_range[0], bin_range[1], 50)
        bin_width = bins[1] - bins[0]

        # CALCULATE WEIGHTS FOR RELATIVE DENSITY
        w_md = np.ones_like(log_seed_md) / (total_passed * bin_width)
        w_id = np.ones_like(log_seed_id) / (total_passed * bin_width)
        w_other = np.ones_like(log_seed_other) / (total_passed * bin_width)

        ax.hist(log_seed_md, bins=bins, weights=w_md, alpha=0.6, 
                label='Merger-driven', color='#2196F3', edgecolor='black', linewidth=0.5)
        ax.hist(log_seed_id, bins=bins, weights=w_id, alpha=0.6, 
                label='Instability-driven', color='#FF9800', edgecolor='black', linewidth=0.5)
        
        if len(log_seed_other) > 0:
            ax.hist(log_seed_other, bins=bins, weights=w_other, alpha=0.4, 
                    label='Other/No Growth', color='#9e9e9e', histtype='step', linewidth=1.5)

        ax.set_xlabel(r'$\log_{10}(M_{\rm BH} [M_{\odot}])$', fontsize=14)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend()
    else:
        print("\nWarning: No valid seeds for histogram plotting.")

    output_dir = Path(file_list[0]).parent / 'plots'
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / 'bh_seed_population_hist.png'
    
    plt.savefig(out_file, bbox_inches='tight')
    print(f"\n✓ 1D Histogram saved to: {out_file}")
    plt.close()

    # ========================================================================
    # PLOTTING - 2D DENSITY PLOTS (NEW: Seed Mass vs Redshift) 
    # ========================================================================
    print("\n" + "="*70)
    print("Creating 2D density plot: Seed Mass vs Redshift (Combined)")
    print("="*70)
    
    # IMPORTANT: Filter out zero or negative seed masses to avoid log10(-inf)
    valid_seed_mask = bh_seed[plot_mask] > 0
    
    if np.sum(valid_seed_mask) > 1:
        # Apply the valid seed mask to get only positive seeds
        birth_redshifts = np.array([get_redshift_from_snapshot(b_snap) 
                                    for b_snap in b_snaps[plot_mask][valid_seed_mask]])
        seed_masses_log = np.log10(bh_seed[plot_mask][valid_seed_mask])
        channels_filtered = channels[plot_mask][valid_seed_mask]
        
        print(f"Seeds with valid masses (>0): {np.sum(valid_seed_mask)} / {np.sum(plot_mask)}")
        if np.sum(valid_seed_mask) < np.sum(plot_mask):
            print(f"  Excluded {np.sum(~valid_seed_mask)} seeds with mass <= 0")
        
        # Filter by population
        mask_md = (channels_filtered == 0)
        mask_id = (channels_filtered == 1)
        
        n_md = np.sum(mask_md)
        n_id = np.sum(mask_id)
        
        print(f"  Merger-driven: {n_md} galaxies")
        print(f"  Instability-driven: {n_id} galaxies")
        
        # Create combined 2D density plot if both populations have data
        if n_md > 0 and n_id > 0:
            create_2d_density_plot_combined(
                seed_masses_log[mask_md],
                birth_redshifts[mask_md],
                seed_masses_log[mask_id],
                birth_redshifts[mask_id],
                'BH Seed Formation: Redshift vs Seed Mass',
                r'$\log_{10}(M_{\rm seed} [M_{\odot}])$',
                'Birth Redshift',
                output_dir / 'bh_seed_2d_density_combined.png',
                n_bins_x=25, n_bins_y=25,
                total_passed=total_passed
            )
        else:
            print(f"  ⚠ Not enough data for combined plot")
    else:
        print(f"⚠ Warning: Not enough valid seeds for 2D density plots")
    
    print("\n✓ All plots completed successfully!")

if __name__ == "__main__":
    main()