#!/usr/bin/env python3
"""
Black hole growth channel cumulative distributions.

Generates cumulative distribution plots for each growth channel and 
compares two simulations side-by-side, including a difference plot.

** OPTIMIZED VERSION **
- Implements sequential array loading/filtering to minimize RAM footprint.
- Caps CDF plot points to prevent massive PDF file sizes and rendering freezes.
- Uses linear interpolation to accurately calculate the difference between CDFs.
"""
import numpy as np
import h5py
import glob
import os
import sys
import argparse
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 96
plt.rcParams["font.size"] = 12
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


def compute_cumulative_distribution(data, max_points=2000):
    """
    Compute cumulative distribution for data.
    Limits the output to `max_points` to prevent huge vector graphic files.
    """
    data_sorted = np.sort(data)
    n = len(data_sorted)
    
    if n > max_points:
        # Downsample evenly across the sorted array to perfectly preserve the shape
        indices = np.linspace(0, n - 1, max_points, dtype=int)
        x_sorted = data_sorted[indices]
        cdf = (indices + 1) / n
    else:
        x_sorted = data_sorted
        cdf = np.arange(1, n + 1) / n
        
    return x_sorted, cdf


def load_filtered_dataset(file_list, snap_num, hubble_h):
    """
    Memory-efficient loading. Reads, masks, and stores fields one by one 
    to prevent holding multiple full-size raw arrays in memory.
    """
    # Read BlackHoleMass to establish the mask
    bh_full = read_hdf(file_list, snap_num, 'BlackHoleMass') * 1.0e10 / hubble_h
    if len(bh_full) == 0:
        return {}
        
    mask = bh_full > 0
    n_valid = np.sum(mask)
    
    # Store filtered data and immediately free the full array
    data = {'BlackHoleMass': bh_full[mask]}
    del bh_full 

    fields_to_load = [
        'MergerDrivenBHaccretionMass',
        'InstabilityDrivenBHaccretionMass',
        'TorqueDrivenBHaccretionMass',
        'SeedModeBHaccretionMass',
        'RadioModeBHaccretionMass',
        'BHMergerMass'
    ]

    for field in fields_to_load:
        arr = read_hdf(file_list, snap_num, field)
        if len(arr) > 0:
            data[field] = (arr * 1.0e10 / hubble_h)[mask]
        else:
            data[field] = np.zeros(n_valid)
        del arr  # Free raw array memory immediately
        
    return data


def main():
    parser = argparse.ArgumentParser(description='BH growth channel cumulative distributions')
    parser.add_argument('-i1', '--input-pattern-1', type=str,
                        default='./output/millennium_insitu/model_*.hdf5',
                        help='Path pattern to first simulation (e.g., in-situ)')
    parser.add_argument('-i2', '--input-pattern-2', type=str,
                        default='./output/millennium/model_*.hdf5',
                        help='Path pattern to second simulation (e.g., standard)')
    parser.add_argument('-s', '--snapshot', type=int, default=None,
                        help='Snapshot number (default: latest)')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory for plots')
    args = parser.parse_args()

    # Load first simulation config
    print("Loading first simulation (millennium_insitu)...")
    file_list_1 = sorted(glob.glob(args.input_pattern_1))
    if not file_list_1:
        print(f"Error: No files found matching: {args.input_pattern_1}")
        sys.exit(1)
    print(f"  Found {len(file_list_1)} model files")

    sim_params_1 = read_simulation_params(file_list_1[0])
    Hubble_h_1 = sim_params_1['Hubble_h']
    snap_num_1 = args.snapshot if args.snapshot is not None else sim_params_1['latest_snapshot']
    print(f"  Using snapshot: {snap_num_1}")

    # Load second simulation config
    print("\nLoading second simulation (millennium)...")
    file_list_2 = sorted(glob.glob(args.input_pattern_2))
    if not file_list_2:
        print(f"Error: No files found matching: {args.input_pattern_2}")
        sys.exit(1)
    print(f"  Found {len(file_list_2)} model files")

    sim_params_2 = read_simulation_params(file_list_2[0])
    Hubble_h_2 = sim_params_2['Hubble_h']
    snap_num_2 = args.snapshot if args.snapshot is not None else sim_params_2['latest_snapshot']
    print(f"  Using snapshot: {snap_num_2}")

    if args.output_dir:
        OutputDir = args.output_dir
    else:
        OutputDir = './plots'
    os.makedirs(OutputDir, exist_ok=True)

    channels = [
        ('Total BH Mass', 'BlackHoleMass', 'black'),
        ('Merger-driven', 'MergerDrivenBHaccretionMass', '#2196F3'),
        ('Instability-driven', 'InstabilityDrivenBHaccretionMass', '#FF9800'),
        ('Torque-driven', 'TorqueDrivenBHaccretionMass', '#795548'),
        ('Seed-mode', 'SeedModeBHaccretionMass', '#FF5722'),
        ('Radio mode', 'RadioModeBHaccretionMass', '#9C27B0'),
        ('BH-BH mergers', 'BHMergerMass', '#4CAF50'),
    ]

    # ================= READ DATA FROM BOTH SIMULATIONS =================
    print("\nReading black hole data (Memory-Optimized)...")
    
    data_1 = load_filtered_dataset(file_list_1, snap_num_1, Hubble_h_1)
    if not data_1:
        print("No galaxies found in simulation 1!")
        sys.exit(1)
        
    data_2 = load_filtered_dataset(file_list_2, snap_num_2, Hubble_h_2)
    if not data_2:
        print("No galaxies found in simulation 2!")
        sys.exit(1)

    # ================= PLOT 1: INDIVIDUAL CUMULATIVE DISTRIBUTIONS =================
    print(f"\nGenerating cumulative distribution plots...")
    
    # Plot for Simulation 1
    fig1, axes1 = plt.subplots(2, 4, figsize=(18, 10))
    axes1 = axes1.flatten()

    for i, (label, field, color) in enumerate(channels):
        ax = axes1[i]
        d = data_1[field]
        d = d[d > 0]
        
        if len(d) > 0:
            x_sorted, cdf = compute_cumulative_distribution(d)
            ax.plot(np.log10(x_sorted), cdf, color=color, linewidth=2.5, label=label)
            
            median_idx = np.argmin(np.abs(cdf - 0.5))
            median_val = x_sorted[median_idx]
            
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=np.log10(median_val), color=color, linestyle='--', alpha=0.5)
            
            ax.set_title(label, fontsize=11, fontweight='bold')
            ax.set_xlabel(r'$\log_{10}(M_\odot)$', fontsize=10)
            ax.set_ylabel('Cumulative Fraction', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)

    fig1.delaxes(axes1[7])
    z_val_1 = sim_params_1['snapshot_redshifts'][snap_num_1] if snap_num_1 < len(sim_params_1['snapshot_redshifts']) else None
    if z_val_1 is not None:
        fig1.suptitle(f'Millennium In-Situ: Cumulative Distributions (Snapshot {snap_num_1}, z={z_val_1:.3f})', fontsize=13)
    else:
        fig1.suptitle(f'Millennium In-Situ: Cumulative Distributions (Snapshot {snap_num_1})', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OutputDir, f'cumulative_distributions_millennium_insitu{OutputFormat}'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: cumulative_distributions_millennium_insitu{OutputFormat}")

    # Plot for Simulation 2
    fig2, axes2 = plt.subplots(2, 4, figsize=(18, 10))
    axes2 = axes2.flatten()

    for i, (label, field, color) in enumerate(channels):
        ax = axes2[i]
        d = data_2[field]
        d = d[d > 0]
        
        if len(d) > 0:
            x_sorted, cdf = compute_cumulative_distribution(d)
            ax.plot(np.log10(x_sorted), cdf, color=color, linewidth=2.5, label=label)
            
            median_idx = np.argmin(np.abs(cdf - 0.5))
            median_val = x_sorted[median_idx]
            
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=np.log10(median_val), color=color, linestyle='--', alpha=0.5)
            
            ax.set_title(label, fontsize=11, fontweight='bold')
            ax.set_xlabel(r'$\log_{10}(M_\odot)$', fontsize=10)
            ax.set_ylabel('Cumulative Fraction', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)

    fig2.delaxes(axes2[7])
    z_val_2 = sim_params_2['snapshot_redshifts'][snap_num_2] if snap_num_2 < len(sim_params_2['snapshot_redshifts']) else None
    if z_val_2 is not None:
        fig2.suptitle(f'Millennium Standard: Cumulative Distributions (Snapshot {snap_num_2}, z={z_val_2:.3f})', fontsize=13)
    else:
        fig2.suptitle(f'Millennium Standard: Cumulative Distributions (Snapshot {snap_num_2})', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OutputDir, f'cumulative_distributions_millennium_standard{OutputFormat}'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: cumulative_distributions_millennium_standard{OutputFormat}")

    # ================= PLOT 3: COMPARISON PLOT =================
    print(f"\nGenerating comparison plot...")
    
    fig3, axes3 = plt.subplots(2, 4, figsize=(18, 10))
    axes3 = axes3.flatten()

    for i, (label, field, color) in enumerate(channels):
        ax = axes3[i]
        
        # Simulation 1
        d1 = data_1[field]
        d1 = d1[d1 > 0]
        
        if len(d1) > 0:
            x1_sorted, cdf1 = compute_cumulative_distribution(d1)
            ax.plot(np.log10(x1_sorted), cdf1, color=color, linewidth=2.5, 
                   label='In-Situ', linestyle='-')
        
        # Simulation 2
        d2 = data_2[field]
        d2 = d2[d2 > 0]
        
        if len(d2) > 0:
            x2_sorted, cdf2 = compute_cumulative_distribution(d2)
            ax.plot(np.log10(x2_sorted), cdf2, color=color, linewidth=2.5, 
                   label='Standard', linestyle='--', alpha=0.8)
        
        # Layout
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel(r'$\log_{10}(M_\odot)$', fontsize=10)
        ax.set_ylabel('Cumulative Fraction', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10, loc='best')

    fig3.delaxes(axes3[7])
    fig3.suptitle(f'Comparison: Millennium In-Situ vs Standard\n(Snapshot In-Situ={snap_num_1}, Snapshot Standard={snap_num_2})', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OutputDir, f'cumulative_distributions_comparison{OutputFormat}'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: cumulative_distributions_comparison{OutputFormat}")

    # ================= PLOT 4: DIFFERENCE PLOT =================
    print(f"\nGenerating difference plot...")
    
    fig4, axes4 = plt.subplots(2, 4, figsize=(18, 10))
    axes4 = axes4.flatten()

    for i, (label, field, color) in enumerate(channels):
        ax = axes4[i]
        
        d1 = data_1[field]
        d1 = d1[d1 > 0]
        
        d2 = data_2[field]
        d2 = d2[d2 > 0]
        
        if len(d1) > 0 and len(d2) > 0:
            x1_sorted, cdf1 = compute_cumulative_distribution(d1)
            x2_sorted, cdf2 = compute_cumulative_distribution(d2)
            
            log_x1 = np.log10(x1_sorted)
            log_x2 = np.log10(x2_sorted)
            
            # Create a common grid in log space
            min_val = min(log_x1.min(), log_x2.min())
            max_val = max(log_x1.max(), log_x2.max())
            common_x = np.linspace(min_val, max_val, 2000)
            
            # Interpolate CDFs onto the common grid
            # If mass is below the minimum, CDF is 0. If above maximum, CDF is 1.
            cdf1_interp = np.interp(common_x, log_x1, cdf1, left=0.0, right=1.0)
            cdf2_interp = np.interp(common_x, log_x2, cdf2, left=0.0, right=1.0)
            
            # Calculate difference (In-Situ minus Standard)
            diff = cdf1_interp - cdf2_interp
            
            ax.plot(common_x, diff, color=color, linewidth=2.5)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
            
            ax.set_title(label, fontsize=11, fontweight='bold')
            ax.set_xlabel(r'$\log_{10}(M_\odot)$', fontsize=10)
            ax.set_ylabel(r'$\Delta$ CDF (In-Situ $-$ Standard)', fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)

    fig4.delaxes(axes4[7])
    fig4.suptitle(f'Difference in Cumulative Distributions (In-Situ $-$ Standard)\n(Snapshot In-Situ={snap_num_1}, Snapshot Standard={snap_num_2})', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OutputDir, f'cumulative_distributions_difference{OutputFormat}'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: cumulative_distributions_difference{OutputFormat}")

    print(f"\nAll plots saved to {OutputDir}")
    print("Done.")


if __name__ == '__main__':
    main()