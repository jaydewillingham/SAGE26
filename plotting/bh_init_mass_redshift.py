#!/usr/bin/env python3
"""
Analyzes BH seed masses across multiple snapshots, creating a 3x3 grid of plots
showing how the seed mass distribution evolves with different latest_snapshot values.
Also creates individual 2D density plots (seed mass vs redshift) for each snapshot.

Automatically handles populations with only Merger-driven or Instability-driven growth.
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
MIN_STELLAR_MASS_LOG = 8.0
MIN_HALO_MASS_LOG = 10.0
MIN_Z0_BH_MASS = 1e1
MONSTER_SEED_THRESHOLD = 1e14  # M_sun

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
    """Reads fields and determines the first growth channel for each galaxy."""
    all_ids = []
    all_bh_mass = []
    all_stellar_mass = []
    all_mvir = []
    all_bh_seed = []
    all_first_channel = []
    all_birth_snaps = []
    all_birth_vals = []

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
            
            if 'BHSeedMass' in grp:
                m_seed = grp['BHSeedMass'][:][mask] * conv
            else:
                m_seed = np.full_like(m_bh, 1e4)

            hist_md = grp['MergerDrivenBHaccretionMass'][:][mask] if 'MergerDrivenBHaccretionMass' in grp else None
            hist_id = grp['InstabilityDrivenBHaccretionMass'][:][mask] if 'InstabilityDrivenBHaccretionMass' in grp else None
            hist_rm = grp['RadioModeBHaccretionMass'][:][mask] if 'RadioModeBHaccretionMass' in grp else None
            hist_bm = grp['BHMergerMass'][:][mask] if 'BHMergerMass' in grp else None

            channels = []
            birth_snaps = []
            birth_vals = []

            for i in range(len(m_bh)):
                chan = 2 
                b_snap = snap_num
                b_v = [0.0, 0.0, 0.0, 0.0]

                if hist_md is not None and hist_id is not None:
                    combined = hist_md[i] + hist_id[i]
                    if hist_rm is not None: combined += hist_rm[i]
                    if hist_bm is not None: combined += hist_bm[i]
                    
                    nonzero = np.where(combined > 0)[0]
                    if len(nonzero) > 0:
                        b_snap = int(nonzero[0])
                        v_md = hist_md[i, b_snap] * conv
                        v_id = hist_id[i, b_snap] * conv
                        v_rm = (hist_rm[i, b_snap] if hist_rm is not None else 0.0) * conv
                        v_bm = (hist_bm[i, b_snap] if hist_bm is not None else 0.0) * conv
                        
                        b_v = [v_md, v_id, v_rm, v_bm]
                        
                        if v_md >= v_id:
                            chan = 0
                        else:
                            chan = 1
                    else:
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

def create_2d_density_plot_single_or_combined(
    x_data_md,
    y_data_md,
    x_data_id,
    y_data_id,
    title,
    xlabel,
    ylabel,
    filename,
    n_bins_x=50,
    n_bins_y=50,
    total_passed=None
):
    """
    Create 2D density plot. If only one population has data, show single panel.
    If both have data, show side-by-side comparison.
    """
    
    # Filter valid data
    valid_mask_md = np.isfinite(x_data_md) & np.isfinite(y_data_md)
    valid_mask_id = np.isfinite(x_data_id) & np.isfinite(y_data_id)
    
    x_md_valid = x_data_md[valid_mask_md]
    y_md_valid = y_data_md[valid_mask_md]
    x_id_valid = x_data_id[valid_mask_id]
    y_id_valid = y_data_id[valid_mask_id]
    
    n_md = len(x_md_valid)
    n_id = len(x_id_valid)
    
    # Determine whether we have both populations or just one
    has_md = n_md > 0
    has_id = n_id > 0
    
    if not (has_md or has_id):
        print("⚠ Warning: No valid data for 2D plot")
        return
    
    # ------------------------------------------------------------------
    # Combine data for shared bins
    # ------------------------------------------------------------------
    if has_md and has_id:
        x_all = np.concatenate([x_md_valid, x_id_valid])
        y_all = np.concatenate([y_md_valid, y_id_valid])
    elif has_md:
        x_all = x_md_valid
        y_all = y_md_valid
    else:
        x_all = x_id_valid
        y_all = y_id_valid
    
    if len(x_all) <= 1:
        print("⚠ Warning: Not enough valid data for 2D plot")
        return
    
    # ------------------------------------------------------------------
    # Create bins
    # ------------------------------------------------------------------
    x_bins = np.linspace(x_all.min(), x_all.max(), n_bins_x + 1)
    y_bins = np.linspace(y_all.min(), y_all.max(), n_bins_y + 1)
    
    x_bin_width = (x_bins[-1] - x_bins[0]) / n_bins_x
    y_bin_width = (y_bins[-1] - y_bins[0]) / n_bins_y
    bin_area = x_bin_width * y_bin_width
    
    if total_passed is None:
        total_passed = len(x_all)
    
    # ------------------------------------------------------------------
    # Compute histograms
    # ------------------------------------------------------------------
    extent = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]
    
    if has_md:
        h_md, _, _ = np.histogram2d(x_md_valid, y_md_valid, bins=[x_bins, y_bins])
        h_md_density = h_md / (total_passed * bin_area)
        h_md_plot = np.where(h_md_density > 0, np.log10(h_md_density), np.nan)
    
    if has_id:
        h_id, _, _ = np.histogram2d(x_id_valid, y_id_valid, bins=[x_bins, y_bins])
        h_id_density = h_id / (total_passed * bin_area)
        h_id_plot = np.where(h_id_density > 0, np.log10(h_id_density), np.nan)
    
    # ------------------------------------------------------------------
    # Get shared color scale
    # ------------------------------------------------------------------
    vals_list = []
    if has_md:
        vals_list.append(h_md_plot[np.isfinite(h_md_plot)])
    if has_id:
        vals_list.append(h_id_plot[np.isfinite(h_id_plot)])
    
    combined_vals = np.concatenate(vals_list) if vals_list else np.array([])
    
    if len(combined_vals) > 0:
        vmin = combined_vals.min()
        vmax = combined_vals.max()
    else:
        vmin = -5
        vmax = 0
    
    # ------------------------------------------------------------------
    # Create figure based on population availability
    # ------------------------------------------------------------------
    if has_md and has_id:
        # Side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(14, 6),
            sharex=True,
            sharey=True,
            constrained_layout=True
        )
        
        for ax in [ax1, ax2]:
            ax.minorticks_on()
        
        # Merger-driven panel
        im1 = ax1.imshow(
            h_md_plot.T,
            extent=extent,
            origin='lower',
            cmap='Blues',
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )
        
        cbar1 = fig.colorbar(im1, ax=ax1, pad=0.015, shrink=0.92)
        cbar1.set_label("")
        ax1.set_title('Merger-driven', fontsize=15, pad=8)
        ax1.set_xlabel(xlabel, fontsize=14)
        ax1.set_ylabel(ylabel, fontsize=14)
        
        # Instability-driven panel
        im2 = ax2.imshow(
            h_id_plot.T,
            extent=extent,
            origin='lower',
            cmap='Oranges',
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )
        
        cbar2 = fig.colorbar(im2, ax=ax2, pad=0.015, shrink=0.92)
        cbar2.set_label(r'$\log_{10}(\mathrm{Density})$', fontsize=12)
        ax2.set_title('Instability-driven', fontsize=15, pad=8)
        ax2.set_xlabel(xlabel, fontsize=14)
        ax2.tick_params(labelleft=False)
        
        fig.subplots_adjust(wspace=0.08, left=0.08, right=0.96, bottom=0.12, top=0.90)
        
        print(f"  MD sample: {n_md} galaxies | ID sample: {n_id} galaxies")
    
    elif has_md:
        # Single merger-driven panel
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.minorticks_on()
        
        im = ax.imshow(
            h_md_plot.T,
            extent=extent,
            origin='lower',
            cmap='Blues',
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )
        
        cbar = fig.colorbar(im, ax=ax, pad=0.015)
        cbar.set_label(r'$\log_{10}(\mathrm{Density})$', fontsize=12)
        ax.set_title('Merger-driven (only)', fontsize=15, pad=8)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        
        print(f"  MD sample: {n_md} galaxies (ID sample: {n_id})")
    
    else:
        # Single instability-driven panel
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.minorticks_on()
        
        im = ax.imshow(
            h_id_plot.T,
            extent=extent,
            origin='lower',
            cmap='Oranges',
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )
        
        cbar = fig.colorbar(im, ax=ax, pad=0.015)
        cbar.set_label(r'$\log_{10}(\mathrm{Density})$', fontsize=12)
        ax.set_title('Instability-driven (only)', fontsize=15, pad=8)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        
        print(f"  ID sample: {n_id} galaxies (MD sample: {n_md})")
    
    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    plt.savefig(filename, dpi=140)
    print(f"✓ 2D density plot saved to: {filename}")
    plt.close()

def create_snapshot_grid(file_list, snap_numbers, id_field, h_h):
    """
    Create a 3x3 grid of 1D histograms showing seed mass distribution 
    across different latest_snapshot values.
    """
    
    n_snaps = len(snap_numbers)
    if n_snaps == 0:
        print("No snapshots to plot")
        return
    
    # Calculate grid dimensions (prefer 3x3, but adapt if needed)
    n_cols = 3
    n_rows = (n_snaps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows), constrained_layout=True)
    
    # Flatten axes for easier iteration
    if n_snaps == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    all_bins = None
    all_seed_data = []
    
    # First pass: collect all data and determine shared bins
    print("\nFirst pass: collecting data from all snapshots...")
    for snap_num in snap_numbers:
        try:
            ids, bh_mass, stellar_mass, mvir, bh_seed, channels, b_snaps, b_vals = read_data(
                file_list, snap_num, id_field, h_h
            )
            
            # Apply cuts
            mask = (bh_mass > MIN_Z0_BH_MASS) & (stellar_mass > 10**MIN_STELLAR_MASS_LOG) & (mvir > 10**MIN_HALO_MASS_LOG)
            monster_mask = (bh_seed > MONSTER_SEED_THRESHOLD)
            plot_mask = mask & (~monster_mask)
            
            # Get valid seeds
            valid_seed_mask = bh_seed[plot_mask] > 0
            if np.sum(valid_seed_mask) > 0:
                log_seeds = np.log10(bh_seed[plot_mask][valid_seed_mask])
                all_seed_data.append(log_seeds)
            else:
                all_seed_data.append(np.array([]))
        except Exception as e:
            print(f"  ⚠ Error reading snapshot {snap_num}: {e}")
            all_seed_data.append(np.array([]))
    
    # Determine shared bins from all data
    if all_seed_data and any(len(data) > 0 for data in all_seed_data):
        all_combined = np.concatenate([data for data in all_seed_data if len(data) > 0])
        if len(all_combined) > 0:
            bin_range = (all_combined.min(), all_combined.max())
            all_bins = np.linspace(bin_range[0], bin_range[1], 50)
    
    if all_bins is None:
        print("No valid data for grid plot")
        return
    
    # Second pass: plot histograms
    print("\nSecond pass: creating histogram panels...")
    for idx, snap_num in enumerate(snap_numbers):
        ax = axes[idx]
        redshift = get_redshift_from_snapshot(snap_num)
        
        if len(all_seed_data[idx]) > 0:
            log_seeds = all_seed_data[idx]
            bin_width = all_bins[1] - all_bins[0]
            total_in_snap = len(log_seeds)
            
            # Plot with density normalization
            weights = np.ones_like(log_seeds) / (total_in_snap * bin_width)
            ax.hist(log_seeds, bins=all_bins, weights=weights, alpha=0.7, 
                   color='#2196F3', edgecolor='black', linewidth=0.5)
            
            ax.minorticks_on()
            ax.set_title(f'Snapshot {snap_num} (z = {redshift:.2f})', fontsize=13, fontweight='bold')
            ax.set_xlabel(r'$\log_{10}(M_{\rm seed} [M_{\odot}])$', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add sample count
            ax.text(0.98, 0.97, f'N = {total_in_snap}', 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, f'No data\nSnapshot {snap_num}', 
                   transform=ax.transAxes, fontsize=12,
                   verticalalignment='center', horizontalalignment='center')
            ax.set_title(f'Snapshot {snap_num} (z = {redshift:.2f})', fontsize=13, fontweight='bold')
    
    # Hide extra subplots if not using all panels
    for idx in range(n_snaps, len(axes)):
        axes[idx].set_visible(False)
    
    output_dir = Path(file_list[0]).parent / 'plots'
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / 'bh_seed_snapshot_grid.png'
    
    plt.savefig(out_file, dpi=140, bbox_inches='tight')
    print(f"\n✓ Snapshot grid saved to: {out_file}")
    plt.close()

def create_2d_grid_merger_only(file_list, snap_numbers, id_field, h_h, output_dir):
    """
    Create a 3x3 grid of 2D density plots showing only merger-driven populations.
    X-axis: seed mass, Y-axis: birth redshift
    """
    
    n_snaps = len(snap_numbers)
    if n_snaps == 0:
        print("No snapshots to plot")
        return
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_snaps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows), constrained_layout=True)
    
    # Flatten axes for easier iteration
    if n_snaps == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    # First pass: collect all merger-driven data to determine shared bins
    print("\nFirst pass: collecting merger-driven data from all snapshots...")
    all_x_data = []
    all_y_data = []
    snap_data = []  # Store (x, y, n_md) for each snapshot
    
    for snap_num in snap_numbers:
        try:
            redshift = get_redshift_from_snapshot(snap_num)
            
            ids, bh_mass, stellar_mass, mvir, bh_seed, channels, b_snaps, b_vals = read_data(
                file_list, snap_num, id_field, h_h
            )
            
            # Apply cuts
            mask = (bh_mass > MIN_Z0_BH_MASS) & (stellar_mass > 10**MIN_STELLAR_MASS_LOG) & (mvir > 10**MIN_HALO_MASS_LOG)
            monster_mask = (bh_seed > MONSTER_SEED_THRESHOLD)
            plot_mask = mask & (~monster_mask)
            
            # Filter valid seeds
            valid_seed_mask = bh_seed[plot_mask] > 0
            
            if np.sum(valid_seed_mask) > 0:
                # Get only merger-driven population
                channels_filtered = channels[plot_mask][valid_seed_mask]
                mask_md = (channels_filtered == 0)
                
                if np.sum(mask_md) > 0:
                    seed_masses_log = np.log10(bh_seed[plot_mask][valid_seed_mask][mask_md])
                    birth_redshifts = np.array([get_redshift_from_snapshot(b_snap) 
                                               for b_snap in b_snaps[plot_mask][valid_seed_mask][mask_md]])
                    
                    all_x_data.append(seed_masses_log)
                    all_y_data.append(birth_redshifts)
                    snap_data.append((seed_masses_log, birth_redshifts, np.sum(mask_md)))
                else:
                    snap_data.append((np.array([]), np.array([]), 0))
            else:
                snap_data.append((np.array([]), np.array([]), 0))
                
        except Exception as e:
            print(f"  ⚠ Error reading snapshot {snap_num}: {e}")
            snap_data.append((np.array([]), np.array([]), 0))
    
    # Determine shared bins from all data
    if all_x_data and all_y_data:
        all_x = np.concatenate(all_x_data)
        all_y = np.concatenate(all_y_data)
        
        x_bins = np.linspace(all_x.min(), all_x.max(), 20)
        y_bins = np.linspace(all_y.min(), all_y.max(), 20)
    else:
        print("  ⚠ No merger-driven data found")
        return
    
    x_bin_width = (x_bins[-1] - x_bins[0]) / (len(x_bins) - 1)
    y_bin_width = (y_bins[-1] - y_bins[0]) / (len(y_bins) - 1)
    bin_area = x_bin_width * y_bin_width
    
    extent = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]
    total_galaxies = sum(n for _, _, n in snap_data)
    
    # Compute all histograms to get shared color scale
    print("\nComputing histograms for color scale...")
    all_h_plots = []
    
    for idx, snap_num in enumerate(snap_numbers):
        x_md, y_md, n_md = snap_data[idx]
        
        if len(x_md) > 0:
            h, _, _ = np.histogram2d(x_md, y_md, bins=[x_bins, y_bins])
            h_density = h / (total_galaxies * bin_area)
            h_plot = np.where(h_density > 0, np.log10(h_density), np.nan)
            all_h_plots.append(h_plot)
        else:
            all_h_plots.append(np.full((len(x_bins)-1, len(y_bins)-1), np.nan))
    
    # Get shared color scale
    combined_vals = np.concatenate([h[np.isfinite(h)].flatten() for h in all_h_plots])
    if len(combined_vals) > 0:
        vmin = combined_vals.min()
        vmax = combined_vals.max()
    else:
        vmin = -5
        vmax = 0
    
    # Second pass: plot histograms
    print("\nSecond pass: creating 2D histogram panels...")
    for idx, snap_num in enumerate(snap_numbers):
        ax = axes[idx]
        redshift = get_redshift_from_snapshot(snap_num)
        x_md, y_md, n_md = snap_data[idx]
        
        if len(x_md) > 0:
            # Plot the 2D histogram
            im = ax.imshow(
                all_h_plots[idx].T,
                extent=extent,
                origin='lower',
                cmap='Blues',
                aspect='auto',
                vmin=vmin,
                vmax=vmax
            )
            
            ax.minorticks_on()
            ax.set_title(f'Snapshot {snap_num} (z = {redshift:.2f})', 
                        fontsize=13, fontweight='bold')
            ax.set_xlabel(r'$\log_{10}(M_{\rm seed} [M_{\odot}])$', fontsize=11)
            ax.set_ylabel('Birth Redshift', fontsize=11)
            
            # Add sample count
            ax.text(0.98, 0.97, f'N = {n_md}', 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, f'No merger-driven\nSnapshot {snap_num}', 
                   transform=ax.transAxes, fontsize=12,
                   verticalalignment='center', horizontalalignment='center')
            ax.set_title(f'Snapshot {snap_num} (z = {redshift:.2f})', 
                        fontsize=13, fontweight='bold')
            ax.set_xlabel(r'$\log_{10}(M_{\rm seed} [M_{\odot}])$', fontsize=11)
            ax.set_ylabel('Birth Redshift', fontsize=11)
    
    # Add colorbar at the end
    cbar = fig.colorbar(plt.cm.ScalarMappable(
        cmap='Blues', 
        norm=plt.Normalize(vmin=vmin, vmax=vmax)
    ), ax=axes, orientation='horizontal', pad=0.01, aspect=50)
    cbar.set_label(r'$\log_{10}(\mathrm{Density})$', fontsize=12)
    
    # Hide extra subplots if not using all panels
    for idx in range(n_snaps, len(axes)):
        axes[idx].set_visible(False)
    
    out_file = output_dir / 'bh_seed_2d_merger_grid.png'
    plt.savefig(out_file, dpi=140, bbox_inches='tight')
    print(f"\n✓ 2D merger-driven grid saved to: {out_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-pattern', default='./output/millennium/model_*.hdf5')
    parser.add_argument('-s', '--snapshots', type=int, nargs='+', 
                       default=[4, 8, 12, 16, 20, 30, 40, 50, 62],
                       help='Snapshot numbers to analyze (default: 9 snapshots for 3x3 grid)')
    parser.add_argument('--no-cuts', action='store_true')
    args = parser.parse_args()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"No files found for {args.input_pattern}"); sys.exit(1)

    sim = read_simulation_params(file_list[0])
    h_h = sim['Hubble_h']
    id_field = find_id_field(file_list, args.snapshots[0])
    
    output_dir = Path(file_list[0]).parent / 'plots'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Hubble_h: {h_h}")
    print(f"Analyzing {len(args.snapshots)} snapshots: {args.snapshots}")
    print("="*70)
    
    # Create the 3x3 grid
    print("\nCreating snapshot grid (3x3 of 1D histograms)...")
    create_snapshot_grid(file_list, args.snapshots, id_field, h_h)
    
    # Create 3x3 grid of 2D merger-driven density plots
    print("\n" + "="*70)
    print("Creating 3x3 grid of 2D merger-driven density plots...")
    print("="*70)
    
    create_2d_grid_merger_only(file_list, args.snapshots, id_field, h_h, output_dir)
    
    print("\n✓ All analyses completed successfully!")

if __name__ == "__main__":
    main()