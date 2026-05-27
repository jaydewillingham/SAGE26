#!/usr/bin/env python3
"""
Plot black hole seed masses from simulation output files (HDF5).

This script reads BHSeedMass from output galaxies and creates:
  1. 1D histograms showing seed mass distributions
  2. 2D density plots (seed mass vs redshift/snapshot)
  3. Summary statistics comparing different seeding methods
  
Usage:
    python plot_bh_seeds_from_output.py -i "./output/model_*.hdf5" -s 10
    
where -s is the snapshot number (optional, defaults to last snapshot)
"""

import argparse
import glob
import sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# MATPLOTLIB STYLE
# ============================================================================
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 140
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['legend.frameon'] = False

# ============================================================================
# REDSHIFT MAPPING (MILLENNIUM SIMULATION)
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

def get_redshift_from_snapshot(snap_num, snap_to_z_dict=None):
    """Map snapshot number to redshift."""
    if snap_to_z_dict is None:
        snap_to_z_dict = MILLENNIUM_SNAP_TO_Z
    return snap_to_z_dict.get(snap_num, 0.0)

# ============================================================================
# HDF5 READING FUNCTIONS
# ============================================================================
def read_simulation_params(filepath):
    """Extract simulation parameters from HDF5 file."""
    try:
        with h5py.File(filepath, 'r') as hf:
            if 'Header' in hf:
                header = hf['Header'].attrs
            elif 'Header/Simulation' in hf:
                header = hf['Header/Simulation'].attrs
            else:
                return {'Hubble_h': 0.681, 'latest_snapshot': 62}
            
            h_h = header.get('hubble_h', header.get('HubbleParam', 0.681))
            return {'Hubble_h': h_h, 'latest_snapshot': 62}
    except Exception as e:
        print(f"⚠ Warning reading sim params: {e}")
        return {'Hubble_h': 0.681, 'latest_snapshot': 62}

def list_snapshots(file_list):
    """Find available snapshots in files."""
    snapshots = set()
    for filepath in file_list:
        try:
            with h5py.File(filepath, 'r') as hf:
                for key in hf.keys():
                    if key.startswith('Snap_'):
                        snap_num = int(key.split('_')[1])
                        snapshots.add(snap_num)
        except:
            pass
    return sorted(list(snapshots))

def read_snapshot_data(file_list, snap_num, h_h):
    """
    Read BH seed mass data from a snapshot.
    
    Returns:
        Dictionary with keys: 'seed_mass', 'bh_mass', 'stellar_mass', 'mvir',
                              'galaxy_ids', 'merger_driven_accretion', 'instability_driven_accretion'
    """
    all_seed_mass = []
    all_bh_mass = []
    all_stellar_mass = []
    all_mvir = []
    all_galaxy_ids = []
    all_merger_accretion = []
    all_instability_accretion = []
    
    conversion_factor = 1e10 / h_h  # Convert from 10^10 M_sun/h to M_sun
    
    for filepath in file_list:
        try:
            with h5py.File(filepath, 'r') as hf:
                snap_key = f"Snap_{snap_num}"
                if snap_key not in hf:
                    continue
                
                grp = hf[snap_key]
                
                # Read core fields
                if 'BHSeedMass' in grp:
                    seed_mass = grp['BHSeedMass'][:] * conversion_factor
                    all_seed_mass.append(seed_mass)
                else:
                    print(f"⚠ Warning: BHSeedMass not found in {filepath}:{snap_key}")
                    continue
                
                if 'BlackHoleMass' in grp:
                    bh_mass = grp['BlackHoleMass'][:] * conversion_factor
                    all_bh_mass.append(bh_mass)
                
                if 'StellarMass' in grp:
                    stellar_mass = grp['StellarMass'][:] * conversion_factor
                    all_stellar_mass.append(stellar_mass)
                
                if 'Mvir' in grp:
                    mvir = grp['Mvir'][:] * conversion_factor
                    all_mvir.append(mvir)
                
                # Find galaxy ID field
                id_field = None
                for candidate in ['GalaxyIndex', 'GalaxyID', 'ID', 'galaxy_id']:
                    if candidate in grp:
                        id_field = candidate
                        break
                
                if id_field:
                    gids = grp[id_field][:]
                    all_galaxy_ids.append(gids)
                else:
                    all_galaxy_ids.append(np.arange(len(seed_mass)))
                
                # Growth history fields
                if 'MergerDrivenBHaccretionMass' in grp:
                    md = grp['MergerDrivenBHaccretionMass'][:] * conversion_factor
                    all_merger_accretion.append(md)
                
                if 'InstabilityDrivenBHaccretionMass' in grp:
                    id_field_accretion = grp['InstabilityDrivenBHaccretionMass'][:] * conversion_factor
                    all_instability_accretion.append(id_field_accretion)
        
        except Exception as e:
            print(f"⚠ Error reading {filepath}: {e}")
            continue
    
    # Concatenate all data
    if not all_seed_mass:
        print(f"✗ No seed mass data found for snapshot {snap_num}")
        return None
    
    data = {
        'seed_mass': np.concatenate(all_seed_mass),
        'bh_mass': np.concatenate(all_bh_mass) if all_bh_mass else np.array([]),
        'stellar_mass': np.concatenate(all_stellar_mass) if all_stellar_mass else np.array([]),
        'mvir': np.concatenate(all_mvir) if all_mvir else np.array([]),
        'galaxy_ids': np.concatenate(all_galaxy_ids) if all_galaxy_ids else np.array([]),
        'merger_accretion': np.concatenate(all_merger_accretion) if all_merger_accretion else None,
        'instability_accretion': np.concatenate(all_instability_accretion) if all_instability_accretion else None,
    }
    
    return data

def classify_seeding_method(seed_mass, threshold_for_heavy=1e4):
    """
    Classify seed masses into seeding methods.
    
    Method 1: Light seeds (power-law, 30-100 M_sun)
    Method 2: Heavy seeds (~10^5 M_sun)
    Method 3: Unknown/other
    """
    classification = np.zeros(len(seed_mass), dtype=int)
    
    for i, m in enumerate(seed_mass):
        if 30 <= m <= 100:
            classification[i] = 1  # Light seeds
        elif m >= threshold_for_heavy:
            classification[i] = 2  # Heavy seeds
        else:
            classification[i] = 0  # Unknown/other
    
    return classification

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def plot_seed_mass_distribution_linear(seed_mass, snap_num, output_dir):
    """Create 1D histogram with linear scale."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Remove zeros and negatives
    valid_mask = seed_mass > 0
    seeds = seed_mass[valid_mask]
    
    if len(seeds) == 0:
        print("⚠ No valid seeds for linear histogram")
        plt.close()
        return
    
    ax.hist(seeds, bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel(r'Seed Mass ($M_\odot$)', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(f'BH Seed Mass Distribution (Snapshot {snap_num}, Linear Scale)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Statistics box
    stats_text = f'Total seeds: {len(seeds)}\n'
    stats_text += f'Min: {seeds.min():.2e} M$_\\odot$\n'
    stats_text += f'Max: {seeds.max():.2e} M$_\\odot$\n'
    stats_text += f'Mean: {seeds.mean():.2e} M$_\\odot$\n'
    stats_text += f'Median: {np.median(seeds):.2e} M$_\\odot$'
    
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_file = output_dir / f'bh_seed_linear_snap{snap_num}.png'
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_seed_mass_distribution_log(seed_mass, snap_num, output_dir):
    """Create 1D histogram with log scale."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Remove zeros and negatives
    valid_mask = seed_mass > 0
    seeds = seed_mass[valid_mask]
    
    if len(seeds) == 0:
        print("⚠ No valid seeds for log histogram")
        plt.close()
        return
    
    # Log bins
    bins = np.logspace(np.log10(seeds.min()), np.log10(seeds.max()), 50)
    ax.hist(seeds, bins=bins, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_xscale('log')
    ax.set_xlabel(r'Seed Mass ($M_\odot$)', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(f'BH Seed Mass Distribution (Snapshot {snap_num}, Log Scale)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # Statistics box
    log_seeds = np.log10(seeds)
    stats_text = f'Total seeds: {len(seeds)}\n'
    stats_text += f'Log(Min): {np.log10(seeds.min()):.2f}\n'
    stats_text += f'Log(Max): {np.log10(seeds.max()):.2f}\n'
    stats_text += f'Log(Mean): {np.log10(seeds.mean()):.2f}\n'
    stats_text += f'Log(Median): {np.log10(np.median(seeds)):.2f}'
    
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    output_file = output_dir / f'bh_seed_log_snap{snap_num}.png'
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_seed_mass_by_method(seed_mass, snap_num, output_dir):
    """Create stacked histogram showing different seeding methods."""
    fig, ax = plt.subplots(figsize=(11, 7))
    
    valid_mask = seed_mass > 0
    seeds = seed_mass[valid_mask]
    
    if len(seeds) == 0:
        print("⚠ No valid seeds for method histogram")
        plt.close()
        return
    
    classification = classify_seeding_method(seeds)
    
    # Separate by method
    light_seeds = seeds[classification == 1]
    heavy_seeds = seeds[classification == 2]
    other_seeds = seeds[classification == 0]
    
    bins = np.logspace(np.log10(seeds.min()), np.log10(seeds.max()), 50)
    
    ax.hist(light_seeds, bins=bins, label=f'Light Seeds (30-100 M☉, n={len(light_seeds)})', 
            color='#2196F3', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.hist(heavy_seeds, bins=bins, label=f'Heavy Seeds (≥10⁴ M☉, n={len(heavy_seeds)})', 
            color='#FF9800', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    if len(other_seeds) > 0:
        ax.hist(other_seeds, bins=bins, label=f'Other Seeds (n={len(other_seeds)})', 
                color='#9E9E9E', alpha=0.5, edgecolor='black', linewidth=0.5, histtype='step')
    
    ax.set_xscale('log')
    ax.set_xlabel(r'Seed Mass ($M_\odot$)', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(f'BH Seed Mass by Method (Snapshot {snap_num})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_file = output_dir / f'bh_seed_by_method_snap{snap_num}.png'
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_seed_vs_current_bh_mass(seed_mass, bh_mass, snap_num, output_dir):
    """Create 2D scatter plot of seed mass vs current BH mass."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    valid_mask = (seed_mass > 0) & (bh_mass > 0)
    seeds = seed_mass[valid_mask]
    current = bh_mass[valid_mask]
    
    if len(seeds) < 2:
        print("⚠ Not enough data for seed vs BH mass plot")
        plt.close()
        return
    
    # 2D histogram for density
    h, xedges, yedges = np.histogram2d(np.log10(seeds), np.log10(current), bins=30)
    
    im = ax.imshow(h.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                   origin='lower', aspect='auto', cmap='YlOrRd', norm=LogNorm(vmin=1))
    
    ax.plot([np.log10(seeds.min()), np.log10(seeds.max())], 
            [np.log10(seeds.min()), np.log10(seeds.max())], 
            'k--', alpha=0.5, linewidth=1.5, label='1:1 ratio')
    
    ax.set_xlabel(r'$\log_{10}(M_{\rm seed}) [M_\odot]$', fontsize=13)
    ax.set_ylabel(r'$\log_{10}(M_{\rm BH}) [M_\odot]$', fontsize=13)
    ax.set_title(f'BH Growth: Seed vs Current Mass (Snapshot {snap_num})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=11)
    
    plt.tight_layout()
    output_file = output_dir / f'bh_seed_vs_current_snap{snap_num}.png'
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def print_summary_statistics(seed_mass, snap_num):
    """Print summary statistics to console."""
    valid_mask = seed_mass > 0
    seeds = seed_mass[valid_mask]
    
    print("\n" + "="*70)
    print(f"SEED MASS SUMMARY STATISTICS (Snapshot {snap_num})")
    print("="*70)
    print(f"Total galaxies with seeds: {len(seeds)}")
    print(f"Total galaxies read: {len(seed_mass)}")
    print(f"Galaxies with zero/negative seeds: {np.sum(~valid_mask)}")
    print(f"\nSeed Mass Statistics:")
    print(f"  Min:        {seeds.min():.4e} M☉")
    print(f"  Max:        {seeds.max():.4e} M☉")
    print(f"  Mean:       {seeds.mean():.4e} M☉")
    print(f"  Median:     {np.median(seeds):.4e} M☉")
    print(f"  Std Dev:    {seeds.std():.4e} M☉")
    print(f"  25th %ile:  {np.percentile(seeds, 25):.4e} M☉")
    print(f"  75th %ile:  {np.percentile(seeds, 75):.4e} M☉")
    
    classification = classify_seeding_method(seeds)
    print(f"\nSeeding Method Classification:")
    print(f"  Light seeds (30-100 M☉):     {np.sum(classification == 1):6d} ({100*np.sum(classification == 1)/len(seeds):5.1f}%)")
    print(f"  Heavy seeds (≥10⁴ M☉):      {np.sum(classification == 2):6d} ({100*np.sum(classification == 2)/len(seeds):5.1f}%)")
    print(f"  Other/Intermediate:          {np.sum(classification == 0):6d} ({100*np.sum(classification == 0)/len(seeds):5.1f}%)")
    print("="*70 + "\n")

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Plot BH seed masses from simulation output')
    parser.add_argument('-i', '--input-pattern', default='./output/millennium/model_*.hdf5',
                        help='Glob pattern for input HDF5 files')
    parser.add_argument('-s', '--snapshot', type=int, default=None,
                        help='Snapshot number to analyze (default: last available)')
    parser.add_argument('-o', '--output-dir', default='./plots',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    # Find files
    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"✗ No files matching pattern: {args.input_pattern}")
        sys.exit(1)
    
    print(f"Found {len(file_list)} files")
    
    # Read simulation parameters
    sim_params = read_simulation_params(file_list[0])
    h_h = sim_params['Hubble_h']
    
    # Find available snapshots
    available_snaps = list_snapshots(file_list)
    if not available_snaps:
        print("✗ No snapshots found in files")
        sys.exit(1)
    
    print(f"Available snapshots: {available_snaps}")
    
    # Choose snapshot
    if args.snapshot is None:
        snap_num = available_snaps[-1]
    else:
        snap_num = args.snapshot
    
    if snap_num not in available_snaps:
        print(f"✗ Snapshot {snap_num} not available")
        sys.exit(1)
    
    redshift = get_redshift_from_snapshot(snap_num)
    print(f"\nAnalyzing Snapshot {snap_num} (z={redshift:.3f}) | Hubble_h={h_h:.4f}")
    
    # Read data
    print("Reading snapshot data...")
    data = read_snapshot_data(file_list, snap_num, h_h)
    
    if data is None:
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print statistics
    print_summary_statistics(data['seed_mass'], snap_num)
    
    # Create plots
    print("Creating plots...")
    plot_seed_mass_distribution_linear(data['seed_mass'], snap_num, output_dir)
    plot_seed_mass_distribution_log(data['seed_mass'], snap_num, output_dir)
    plot_seed_mass_by_method(data['seed_mass'], snap_num, output_dir)
    
    if len(data['bh_mass']) > 0:
        plot_seed_vs_current_bh_mass(data['seed_mass'], data['bh_mass'], snap_num, output_dir)
    
    print(f"\n✓ All plots saved to: {output_dir}")

if __name__ == '__main__':
    main()