#!/usr/bin/env python3
"""
Black hole growth tracking: MAIN PROGENITOR TRACKING
Tracks the median black hole mass/growth of the "largest progenitor galaxy" 
along the main branch of the merger tree, exactly as done in standard literature.

python bh_growth_progenitors.py \
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

plt.rcParams["figure.figsize"] = (14, 10)
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

def get_tree_keys(file_list, snap):
    """Dynamically find the names of the ID and Progenitor fields in the HDF5 file."""
    with h5py.File(file_list[0], 'r') as hf:
        snap_key = f"Snap_{int(snap)}"
        if snap_key not in hf:
            return None, None
        keys = list(hf[snap_key].keys())
        
        gal_key = next((k for k in keys if k in ['GalaxyID', 'GalID', 'ID']), None)
        prog_key = next((k for k in keys if k in ['FirstProgID', 'FirstProgenitor', 'FirstProg', 'MainProgenitor']), None)
            
    return gal_key, prog_key

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

def main():
    parser = argparse.ArgumentParser(description='Main Progenitor Black Hole Growth Tracking')
    parser.add_argument('-i', '--input-pattern', type=str, default='./output/millennium/model_*.hdf5')
    parser.add_argument('-s', '--snapshot', type=int, default=None)
    parser.add_argument('-o', '--output-dir', type=str, default=None)
    parser.add_argument('--validation-snap', type=int, default=None)
    args = parser.parse_args()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"Error: No files found matching: {args.input_pattern}")
        sys.exit(1)

    sim_params = read_simulation_params(file_list[0])
    Hubble_h = sim_params['Hubble_h']
    all_snaps = np.array(sim_params['available_snapshots'])
    all_redshifts = sim_params['snapshot_redshifts']

    OutputDir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(os.path.abspath(file_list[0])), 'plots')
    os.makedirs(OutputDir, exist_ok=True)

    # Identify the z=0 snapshot (lowest redshift)
    z0_snap = all_snaps[np.argmin(all_redshifts[all_snaps])]
    print(f"z=0 reference snapshot: {z0_snap} (z = {all_redshifts[z0_snap]:.6f})")

    # Check for Tree Identifiers
    gal_key, prog_key = get_tree_keys(file_list, z0_snap)
    use_trees = True
    if not gal_key or not prog_key:
        print("\nWARNING: Could not find 'GalaxyID' or 'FirstProgenitor' fields in HDF5!")
        print("Falling back to Cross-Sectional Median (calculating median of all active galaxies at each redshift).")
        use_trees = False
    else:
        print(f"\nUsing Merger Trees: ID='{gal_key}', Progenitor='{prog_key}'")

    # Sort snapshots DESCENDING (start at z=0, go back in time)
    snaps_backwards = sorted(all_snaps, reverse=True)

    channels_info = [
        ('Merger-driven', 'md', 'MergerDrivenBHaccretionMass', '#2196F3'),
        ('Instability-driven', 'id', 'InstabilityDrivenBHaccretionMass', '#FF9800'),
        ('Radio mode', 'rm', 'RadioModeBHaccretionMass', '#9C27B0'),
        ('BH-BH mergers', 'bm', 'BHMergerMass', '#4CAF50'),
    ]

    # Data structure to hold our results
    results = {key: {'z': [], 'med': [], 'p16': [], 'p84': []} for _, key, _, _ in channels_info}
    results['total'] = {'z': [], 'med': [], 'p16': [], 'p84': []}

    target_progenitors = None

    print("\nTracing Main Progenitors backward through time...")
    for sn in snaps_backwards:
        z = all_redshifts[sn]
        
        bh = read_hdf(file_list, sn, 'BlackHoleMass') * 1.0e10 / Hubble_h
        if len(bh) == 0:
            continue

        channel_data = {
            'md': read_hdf(file_list, sn, 'MergerDrivenBHaccretionMass') * 1.0e10 / Hubble_h,
            'id': read_hdf(file_list, sn, 'InstabilityDrivenBHaccretionMass') * 1.0e10 / Hubble_h,
            'rm': read_hdf(file_list, sn, 'RadioModeBHaccretionMass') * 1.0e10 / Hubble_h,
            'bm': read_hdf(file_list, sn, 'BHMergerMass') * 1.0e10 / Hubble_h,
        }

        if use_trees:
            g_ids = read_hdf(file_list, sn, gal_key)
            p_ids = read_hdf(file_list, sn, prog_key)

            if target_progenitors is None:
                # We are at z=0. Select all valid galaxies as our starting population
                valid_mask = bh > 0
                target_progenitors = g_ids[valid_mask]

            # Find the rows in THIS snapshot that match our target list
            _, _, idx_gals = np.intersect1d(target_progenitors, g_ids, return_indices=True)
            
            bh_current = bh[idx_gals]
            for key in channel_data:
                if len(channel_data[key]) > 0:
                    channel_data[key] = channel_data[key][idx_gals]
            
            # Update target list for the NEXT snapshot backwards in time
            target_progenitors = p_ids[idx_gals]
            # Remove invalid pointers (-1, 0, etc.)
            target_progenitors = target_progenitors[target_progenitors > 0]
        else:
            # Fallback: Just use all valid galaxies at this snapshot
            bh_current = bh

        # --- Calculate Medians for the Matched Galaxies ---
        valid_bh = bh_current[bh_current > 0]
        if len(valid_bh) > 0:
            results['total']['z'].append(z)
            results['total']['med'].append(np.median(valid_bh))
            results['total']['p16'].append(np.percentile(valid_bh, 16))
            results['total']['p84'].append(np.percentile(valid_bh, 84))

        for _, key, _, _ in channels_info:
            data = channel_data[key]
            if len(data) > 0:
                valid_data = data[data > 0]  # ignore zeroes for log-space plotting
                if len(valid_data) > 0:
                    results[key]['z'].append(z)
                    results[key]['med'].append(np.median(valid_data))
                    results[key]['p16'].append(np.percentile(valid_data, 16))
                    results[key]['p84'].append(np.percentile(valid_data, 84))

    # ===================== PLOT RESULTS =====================
    print("\nGenerating Plots...")
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()

    for ax_idx, (label, key, _, color) in enumerate(channels_info):
        ax = axes[ax_idx]
        
        if len(results[key]['med']) > 0:
            # Convert to numpy arrays and log10
            z_vals = np.array(results[key]['z'])
            med_vals = np.log10(np.array(results[key]['med']))
            p16_vals = np.log10(np.array(results[key]['p16']))
            p84_vals = np.log10(np.array(results[key]['p84']))

            # Sort by redshift so it plots a continuous line left-to-right
            sort_idx = np.argsort(z_vals)
            z_vals, med_vals, p16_vals, p84_vals = z_vals[sort_idx], med_vals[sort_idx], p16_vals[sort_idx], p84_vals[sort_idx]

            # Plot Median and Shaded Region
            ax.plot(z_vals, med_vals, color=color, linewidth=2.5, label='Median Main Progenitor')
            ax.fill_between(z_vals, p16_vals, p84_vals, color=color, alpha=0.2, label='16th-84th Percentile')

        ax.set_xlabel('Redshift')
        ax.set_ylabel(r'$\log_{10}(\mathrm{Mass}\ [M_\odot])$')
        ax.set_xlim(-0.2, 7.0)
        ax.set_title(label)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Median Black Hole Growth Tracking\n(Following the Largest Progenitor Galaxies)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OutputDir, f'bh_main_progenitor_tracking{OutputFormat}'))
    plt.close()
    
    print(f"Saved plot to: {os.path.join(OutputDir, f'bh_main_progenitor_tracking{OutputFormat}')}")
    print("Done.")

if __name__ == '__main__':
    main()