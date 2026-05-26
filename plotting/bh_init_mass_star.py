#!/usr/bin/env python3
"""
Analyzes BH seed masses at z=0 (snapshot 62), creating a 9x9 grid of plots
showing how the seed mass distribution varies across different stellar mass bins.

Each panel shows the seed mass distribution for galaxies in a specific stellar mass range.
"""

import argparse
import glob
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# MATPLOTLIB STYLE CONFIGURATION
# ============================================================================
plt.rcParams['figure.figsize'] = (8.34, 6.25)
plt.rcParams['figure.dpi'] = 140
plt.rcParams['figure.autolayout'] = True

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 20.0

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
MIN_HALO_MASS_LOG = 11.0
MIN_Z0_BH_MASS = 1e1
MONSTER_SEED_THRESHOLD = 1e14

# Stellar mass bins in log10(M_sun)
STELLAR_MASS_BINS = [
    (4.0, 5.0),
    (5.0, 6.0),
    (6.0, 7.0),
    (7.0, 8.0),
    (8.0, 9.0),
    (9.0, 10.0),
    (10.0, 11.0),
    (11.0, 12.0),
    (12.0, 13.0),
]

# ============================================================================
# REDSHIFT MAPPING
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
    """Read simulation parameters from file."""
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
    """Find the galaxy ID field name."""
    candidates = ['GalaxyIndex', 'GalaxyID', 'ID', 'galaxy_id', 'id']
    for f in file_list:
        try:
            with h5py.File(f, 'r') as hf:
                snap_key = f"Snap_{snap_num}"
                if snap_key in hf:
                    for c in candidates:
                        if c in hf[snap_key]: 
                            return c
        except:
            continue
    return None

def read_snapshot_data(file_list, snap_num, id_field, h_h):
    """Read all galaxy data for a specific snapshot."""
    ids = []
    bh_mass = []
    stellar_mass = []
    mvir = []
    bh_seed = []
    
    seen_ids = set()
    
    for f in file_list:
        try:
            with h5py.File(f, 'r') as hf:
                snap_key = f"Snap_{snap_num}"
                if snap_key not in hf: 
                    continue
                    
                grp = hf[snap_key]
                
                # Read IDs and filter for duplicates
                gids = grp[id_field][:]
                mask = np.array([gid not in seen_ids for gid in gids])
                for gid in gids[mask]: 
                    seen_ids.add(gid)
                
                if not np.any(mask): 
                    continue
                
                conv = 1e10 / h_h
                
                ids.append(gids[mask])
                bh_mass.append(grp['BlackHoleMass'][:][mask] * conv)
                stellar_mass.append(grp['StellarMass'][:][mask] * conv)
                mvir.append(grp['Mvir'][:][mask] * conv)
                
                if 'BHSeedMass' in grp:
                    bh_seed.append(grp['BHSeedMass'][:][mask] * conv)
                else:
                    bh_seed.append(np.full_like(bh_mass[-1], 1e4))
        
        except Exception as e:
            print(f"  Warning reading {f} snapshot {snap_num}: {e}")
            continue
    
    # Concatenate all data
    if ids:
        return (np.concatenate(ids), np.concatenate(bh_mass), 
                np.concatenate(stellar_mass), np.concatenate(mvir), 
                np.concatenate(bh_seed))
    else:
        print(f"  No data found for snapshot {snap_num}")
        return None

def create_stellar_mass_grid(file_list, snap_num, id_field, h_h, output_dir, 
                             stellar_mass_bins=None, no_cuts=False):
    """
    Create a 9x9 grid showing seed mass distribution across stellar mass bins.
    """
    
    if stellar_mass_bins is None:
        stellar_mass_bins = STELLAR_MASS_BINS
    
    n_bins = len(stellar_mass_bins)
    
    # Determine grid dimensions
    n_cols = 3
    n_rows = (n_bins + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows), constrained_layout=True)
    
    # Flatten axes for easier iteration
    if n_bins == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    # Read all z=0 data
    print("\nReading z=0 data...")
    snap_data = read_snapshot_data(file_list, snap_num, id_field, h_h)
    
    if snap_data is None:
        print("Failed to read snapshot data")
        return
    
    ids, bh_mass, stellar_mass, mvir, bh_seed = snap_data
    
    print(f"Total galaxies at z=0: {len(ids)}")
    
    # Apply basic cuts to all data
    if not no_cuts:
        mask_all = (bh_mass > MIN_Z0_BH_MASS) & (mvir > 10**MIN_HALO_MASS_LOG)
        monster_mask = (bh_seed > MONSTER_SEED_THRESHOLD)
        mask_all = mask_all & (~monster_mask)
    else:
        mask_all = (bh_mass > 0)
    
    bh_mass = bh_mass[mask_all]
    stellar_mass = stellar_mass[mask_all]
    mvir = mvir[mask_all]
    bh_seed = bh_seed[mask_all]
    
    print(f"Galaxies passed cuts: {len(bh_mass)}")
    
    # First pass: collect seed mass data for all bins to determine shared bins
    print("\nFirst pass: collecting seed mass data...")
    all_bin_data = []
    bin_stats = []
    
    for bin_low, bin_high in stellar_mass_bins:
        m_low = 10**bin_low
        m_high = 10**bin_high
        
        mask = (stellar_mass >= m_low) & (stellar_mass < m_high)
        n_in_bin = np.sum(mask)
        
        # Get valid seeds
        valid_mask = bh_seed[mask] > 0
        if np.sum(valid_mask) > 0:
            log_seeds = np.log10(bh_seed[mask][valid_mask])
            all_bin_data.append(log_seeds)
            bin_stats.append((n_in_bin, np.sum(valid_mask)))
        else:
            all_bin_data.append(np.array([]))
            bin_stats.append((n_in_bin, 0))
    
    # Determine shared bins from all data
    if any(len(data) > 0 for data in all_bin_data):
        all_combined = np.concatenate([data for data in all_bin_data if len(data) > 0])
        if len(all_combined) > 0:
            bin_range = (all_combined.min(), all_combined.max())
            all_bins = np.linspace(bin_range[0], bin_range[1], 50)
        else:
            print("No valid seed data found")
            return
    else:
        print("No valid seed data found")
        return
    
    # Second pass: create histogram panels
    print("\nSecond pass: creating histogram panels...")
    for idx, (bin_low, bin_high) in enumerate(stellar_mass_bins):
        ax = axes[idx]
        m_low = 10**bin_low
        m_high = 10**bin_high
        
        n_total, n_valid = bin_stats[idx]
        
        if len(all_bin_data[idx]) > 0:
            log_seeds = all_bin_data[idx]
            bin_width = all_bins[1] - all_bins[0]
            
            # Normalize by total sample passed cuts (not just this bin)
            weights = np.ones_like(log_seeds) / (len(bh_mass) * bin_width)
            
            ax.hist(log_seeds, bins=all_bins, weights=weights, alpha=0.7, 
                   color='#2196F3', edgecolor='black', linewidth=0.5)
            
            ax.minorticks_on()
            ax.set_title(f'$\log_{{10}}(M_{{\\star}})$ = {bin_low:.1f}–{bin_high:.1f}', 
                        fontsize=13, fontweight='bold')
            ax.set_xlabel(r'$\log_{10}(M_{\rm seed} [M_{\odot}])$', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add sample count
            info_text = f'N = {n_valid}\n({n_total} total)'
            ax.text(0.98, 0.97, info_text, 
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, f'No data\n$\log_{{10}}(M_{{\\star}})$ = {bin_low:.1f}–{bin_high:.1f}', 
                   transform=ax.transAxes, fontsize=12,
                   verticalalignment='center', horizontalalignment='center')
            ax.set_title(f'$\log_{{10}}(M_{{\\star}})$ = {bin_low:.1f}–{bin_high:.1f}', 
                        fontsize=13, fontweight='bold')
            ax.set_xlabel(r'$\log_{10}(M_{\rm seed} [M_{\odot}])$', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
    
    # Hide extra subplots if not using all panels
    for idx in range(n_bins, len(axes)):
        axes[idx].set_visible(False)
    
    # Save figure
    redshift = get_redshift_from_snapshot(snap_num)
    out_file = output_dir / f'bh_seed_stellar_mass_grid_z{redshift:.1f}.png'
    
    plt.savefig(out_file, dpi=140, bbox_inches='tight')
    print(f"\n✓ Stellar mass grid saved to: {out_file}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"{'Stellar Mass Bin':<25} {'Total':<10} {'With Seeds':<12} {'Median log(M_seed)':<20}")
    print("-"*70)
    
    for idx, (bin_low, bin_high) in enumerate(stellar_mass_bins):
        m_low = 10**bin_low
        m_high = 10**bin_high
        
        mask = (stellar_mass >= m_low) & (stellar_mass < m_high)
        n_total = np.sum(mask)
        
        valid_mask = bh_seed[mask] > 0
        n_valid = np.sum(valid_mask)
        
        if n_valid > 0:
            log_seeds = np.log10(bh_seed[mask][valid_mask])
            median_seed = np.median(log_seeds)
            print(f"{bin_low:.1f}–{bin_high:.1f}             {n_total:<10} {n_valid:<12} {median_seed:<20.2f}")
        else:
            print(f"{bin_low:.1f}–{bin_high:.1f}             {n_total:<10} {n_valid:<12} {'—':<20}")

def main():
    parser = argparse.ArgumentParser(
        description='Plot BH seed mass distributions across stellar mass bins at z=0'
    )
    parser.add_argument('-i', '--input-pattern', default='./output/millennium/model_*.hdf5')
    parser.add_argument('--stellar-bins', type=float, nargs='+', 
                       default=None,
                       help='Stellar mass bin edges in log10(M_sun). Example: 5 6 7 8 9 10 11 12 13 14')
    parser.add_argument('--no-cuts', action='store_true', help='Skip mass cuts')
    args = parser.parse_args()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"No files found for {args.input_pattern}")
        sys.exit(1)

    sim = read_simulation_params(file_list[0])
    h_h = sim['Hubble_h']
    id_field = find_id_field(file_list, 62)
    
    if id_field is None:
        print("Could not find ID field in data")
        sys.exit(1)
    
    output_dir = Path(file_list[0]).parent / 'plots'
    output_dir.mkdir(exist_ok=True)
    
    # Parse custom stellar mass bins if provided
    if args.stellar_bins is not None:
        if len(args.stellar_bins) < 2:
            print("Error: Need at least 2 bin edges for stellar mass bins")
            sys.exit(1)
        stellar_bins = [(args.stellar_bins[i], args.stellar_bins[i+1]) 
                       for i in range(len(args.stellar_bins)-1)]
    else:
        stellar_bins = STELLAR_MASS_BINS
    
    print(f"Hubble_h: {h_h}")
    print(f"Galaxy ID field: {id_field}")
    print(f"Snapshot: 62 (z=0)")
    print(f"Stellar mass bins: {stellar_bins}")
    print("="*70)
    
    # Create the stellar mass grid
    create_stellar_mass_grid(file_list, 62, id_field, h_h, output_dir, 
                            stellar_mass_bins=stellar_bins, 
                            no_cuts=args.no_cuts)
    
    print("\n✓ Analysis completed!")

if __name__ == "__main__":
    main()