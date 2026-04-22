#!/usr/bin/env python3
"""
Black Hole Mass Tracking for Specific Galaxies

This script extracts BH mass and its components (Quasar Mode, Merger-Driven,
Instability-Driven, Radio Mode, BH-BH Mergers, and Seed Mass) for 20 specific
galaxy IDs across all snapshots.

Outputs: CSV files for comparison between models
"""
import numpy as np
import h5py
import glob
import os
import sys
import argparse
import pandas as pd

POSSIBLE_ID_FIELDS = ['GalaxyIndex', 'ID', 'galaxy_id', 'id', 'GalID']

def read_hdf(file_list, snap, field):
    """Read a field from HDF5 files for a given snapshot"""
    data = []
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f"Snap_{int(snap)}"
            if snap_key in hf and field in hf[snap_key]:
                data.append(hf[snap_key][field][:])
    return np.concatenate(data) if data else np.array([])

def find_id_field(file_list, snap):
    """Find the correct ID field name in the HDF5 files"""
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f"Snap_{int(snap)}"
            if snap_key in hf:
                for field_name in POSSIBLE_ID_FIELDS:
                    if field_name in hf[snap_key]:
                        return field_name
    return None

def read_simulation_params(filepath):
    """Extract simulation parameters from HDF5 file"""
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

def lookup_ids_vectorized(gal_ids_target, gal_ids_snapshot):
    """Vectorized lookup of galaxy IDs in a snapshot"""
    sort_idx = np.argsort(gal_ids_snapshot)
    gal_ids_sorted = gal_ids_snapshot[sort_idx]
    indices = np.full(len(gal_ids_target), -1, dtype=int)
    for i, target_id in enumerate(gal_ids_target):
        pos = np.searchsorted(gal_ids_sorted, target_id)
        if pos < len(gal_ids_sorted) and gal_ids_sorted[pos] == target_id:
            indices[i] = sort_idx[pos]
    return indices

def select_galaxy_ids(file_list, snap, id_field, num_galaxies=20):
    """Select the first N galaxy IDs with active black holes at z=0"""
    bh = read_hdf(file_list, snap, 'BlackHoleMass')
    gal_id = read_hdf(file_list, snap, id_field)
    
    # Find galaxies with active BHs
    active_bh_mask = bh > 0
    active_ids = gal_id[active_bh_mask]
    
    # Return first N IDs
    return active_ids[:min(num_galaxies, len(active_ids))]

def extract_bh_data_for_galaxies(file_list, target_gal_ids, all_snaps, all_redshifts, id_field, hubble_h):
    """Extract BH mass components for target galaxy IDs across all snapshots"""
    results = []
    
    for snap in all_snaps:
        bh = read_hdf(file_list, snap, 'BlackHoleMass') * 1.0e10 / hubble_h
        if len(bh) == 0:
            continue
        
        z = all_redshifts[snap] if snap < len(all_redshifts) else None
        if z is None:
            continue
        
        gal_id_sn = read_hdf(file_list, snap, id_field) if id_field else np.arange(len(bh))
        
        # Look up indices for target galaxy IDs
        indices = lookup_ids_vectorized(target_gal_ids, gal_id_sn)
        
        # Read all BH components
        qm = read_hdf(file_list, snap, 'QuasarModeBHaccretionMass') * 1.0e10 / hubble_h
        md = read_hdf(file_list, snap, 'MergerDrivenBHaccretionMass') * 1.0e10 / hubble_h
        id_ = read_hdf(file_list, snap, 'InstabilityDrivenBHaccretionMass') * 1.0e10 / hubble_h
        rm = read_hdf(file_list, snap, 'RadioModeBHaccretionMass') * 1.0e10 / hubble_h
        bm = read_hdf(file_list, snap, 'BHMergerMass') * 1.0e10 / hubble_h
        sd = read_hdf(file_list, snap, 'BHSeedMass') * 1.0e10 / hubble_h
        
        # Extract data for each target galaxy
        for gal_idx, idx in enumerate(indices):
            if idx >= 0:  # Galaxy found in this snapshot
                row = {
                    'Snapshot': snap,
                    'Redshift': z,
                    'GalaxyID': target_gal_ids[gal_idx],
                    'GalaxyIndex': gal_idx + 1,  # 1-indexed
                    'BH_Total_Mass': bh[idx],
                    'QuasarMode_Mass': qm[idx] if len(qm) > idx else 0.0,
                    'MergerDriven_Mass': md[idx] if len(md) > idx else 0.0,
                    'InstabilityDriven_Mass': id_[idx] if len(id_) > idx else 0.0,
                    'RadioMode_Mass': rm[idx] if len(rm) > idx else 0.0,
                    'BHMerger_Mass': bm[idx] if len(bm) > idx else 0.0,
                    'Seed_Mass': sd[idx] if len(sd) > idx else 0.0,
                }
                results.append(row)
    
    return results

def save_galaxy_individual_csvs(df, output_dir, model_name):
    """Save individual CSV file for each galaxy"""
    # Group by GalaxyID
    galaxies = df['GalaxyID'].unique()
    
    for gal_id in galaxies:
        gal_data = df[df['GalaxyID'] == gal_id]
        filename = os.path.join(output_dir, f'{model_name}_galaxy_{gal_id}.csv')
        gal_data.to_csv(filename, index=False)
        print(f"  Saved: {filename} ({len(gal_data)} snapshots)")

def process_model(input_pattern, num_galaxies=20):
    """Process a single model and return extracted data"""
    print(f"\n{'='*70}")
    print(f"PROCESSING MODEL: {input_pattern}")
    print(f"{'='*70}")
    
    file_list = sorted(glob.glob(input_pattern))
    if not file_list:
        print(f"Error: No files found matching: {input_pattern}")
        return None, None
    
    print(f"Found {len(file_list)} files")
    
    sim_params = read_simulation_params(file_list[0])
    hubble_h = sim_params['Hubble_h']
    all_snaps = np.array(sim_params['available_snapshots'])
    all_redshifts = sim_params['snapshot_redshifts']
    latest_snap = sim_params['latest_snapshot']
    
    # Find ID field
    id_field = find_id_field(file_list, latest_snap)
    print(f"ID Field: {id_field}")
    
    # Select galaxy IDs at z=0
    print(f"\nSelecting {num_galaxies} galaxy IDs with active black holes at z=0...")
    target_gal_ids = select_galaxy_ids(file_list, latest_snap, id_field, num_galaxies)
    print(f"Selected {len(target_gal_ids)} galaxy IDs: {target_gal_ids[:10]}..." if len(target_gal_ids) > 10 else f"Selected galaxy IDs: {target_gal_ids}")
    
    # Extract BH data for these galaxies across all snapshots
    print(f"\nExtracting BH mass data across {len(all_snaps)} snapshots...")
    results = extract_bh_data_for_galaxies(file_list, target_gal_ids, all_snaps, all_redshifts, id_field, hubble_h)
    print(f"Extracted {len(results)} data points")
    
    return pd.DataFrame(results), target_gal_ids

def main():
    parser = argparse.ArgumentParser(
        description='Extract BH mass contributions for specific galaxy IDs'
    )
    parser.add_argument(
        '-n', '--num-galaxies',
        type=int,
        default=20,
        help='Number of galaxies to track (default: 20)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='./plots',
        help='Output directory for CSV files'
    )
    parser.add_argument(
        '--model1-pattern',
        type=str,
        default='./output/millennium/model_*.hdf5',
        help='Pattern for model 1 files'
    )
    parser.add_argument(
        '--model2-pattern',
        type=str,
        default='./output/millennium_insitu/model_*.hdf5',
        help='Pattern for model 2 files'
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_subdir = os.path.join(args.output_dir, 'galaxy_bh_tracking_14')
    os.makedirs(output_subdir, exist_ok=True)
    
    print(f"\nOutput directory: {output_subdir}")
    
    # Process model 1
    print("\n" + "="*70)
    print("MODEL 1: Standard Model")
    print("="*70)
    df1, ids1 = process_model(args.model1_pattern, args.num_galaxies)
    
    if df1 is None:
        print("Failed to process model 1. Exiting.")
        sys.exit(1)
    
    # Process model 2
    print("\n" + "="*70)
    print("MODEL 2: In-Situ Model")
    print("="*70)
    df2, ids2 = process_model(args.model2_pattern, args.num_galaxies)
    
    if df2 is None:
        print("Failed to process model 2. Exiting.")
        sys.exit(1)
    
    # Save consolidated model results
    output_file1 = os.path.join(output_subdir, 'model1_galaxy_bh_tracking.csv')
    output_file2 = os.path.join(output_subdir, 'model2_galaxy_bh_tracking.csv')
    
    df1.to_csv(output_file1, index=False)
    df2.to_csv(output_file2, index=False)
    
    print(f"\n{'='*70}")
    print("Consolidated Output Files Created:")
    print(f"{'='*70}")
    print(f"Model 1 (Standard):  {output_file1}")
    print(f"Model 2 (In-Situ):   {output_file2}")
    
    # NEW: Save individual galaxy CSV files for easy comparison
    print(f"\n{'='*70}")
    print("Individual Galaxy Files (Model 1):")
    print(f"{'='*70}")
    model1_gal_dir = os.path.join(output_subdir, 'model1_individual')
    os.makedirs(model1_gal_dir, exist_ok=True)
    save_galaxy_individual_csvs(df1, model1_gal_dir, 'model1')
    
    print(f"\n{'='*70}")
    print("Individual Galaxy Files (Model 2):")
    print(f"{'='*70}")
    model2_gal_dir = os.path.join(output_subdir, 'model2_individual')
    os.makedirs(model2_gal_dir, exist_ok=True)
    save_galaxy_individual_csvs(df2, model2_gal_dir, 'model2')
    
    # Create a summary comparison
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}")
    print(f"\nModel 1 (Standard):")
    print(f"  - Snapshots: {df1['Snapshot'].min()} to {df1['Snapshot'].max()}")
    print(f"  - Unique galaxies: {df1['GalaxyID'].nunique()}")
    print(f"  - Total data points: {len(df1)}")
    print(f"  - Mean BH Mass: {df1['BH_Total_Mass'].mean():.3e} M_sun")
    print(f"  - Max BH Mass: {df1['BH_Total_Mass'].max():.3e} M_sun")
    
    print(f"\nModel 2 (In-Situ):")
    print(f"  - Snapshots: {df2['Snapshot'].min()} to {df2['Snapshot'].max()}")
    print(f"  - Unique galaxies: {df2['GalaxyID'].nunique()}")
    print(f"  - Total data points: {len(df2)}")
    print(f"  - Mean BH Mass: {df2['BH_Total_Mass'].mean():.3e} M_sun")
    print(f"  - Max BH Mass: {df2['BH_Total_Mass'].max():.3e} M_sun")
    
    # Generate per-snapshot comparison
    print(f"\n{'='*70}")
    print("Per-Snapshot Comparison (Mean BH Mass across selected galaxies)")
    print(f"{'='*70}")
    
    snap_comp = pd.DataFrame({
        'Snapshot': df1.groupby('Snapshot')['BH_Total_Mass'].mean().index,
        'Model1_Mean_BH': df1.groupby('Snapshot')['BH_Total_Mass'].mean().values,
        'Model2_Mean_BH': df2.groupby('Snapshot')['BH_Total_Mass'].mean().values,
    })
    snap_comp['Ratio(M2/M1)'] = snap_comp['Model2_Mean_BH'] / snap_comp['Model1_Mean_BH']
    
    snap_output = os.path.join(output_subdir, 'snapshot_comparison.csv')
    snap_comp.to_csv(snap_output, index=False)
    print(snap_comp.to_string())
    print(f"\nComparison saved to: {snap_output}")
    
    # Generate component breakdown comparison at z=0
    print(f"\n{'='*70}")
    print("BH Mass Component Breakdown at z=0")
    print(f"{'='*70}")
    
    z0_idx1 = df1['Redshift'].idxmin()
    z0_idx2 = df2['Redshift'].idxmin()
    
    z0_snap1 = df1.loc[z0_idx1, 'Snapshot']
    z0_snap2 = df2.loc[z0_idx2, 'Snapshot']
    
    df1_z0 = df1[df1['Snapshot'] == z0_snap1]
    df2_z0 = df2[df2['Snapshot'] == z0_snap2]
    
    components = ['QuasarMode_Mass', 'MergerDriven_Mass', 'InstabilityDriven_Mass', 
                  'RadioMode_Mass', 'BHMerger_Mass', 'Seed_Mass']
    
    z0_comp = pd.DataFrame({
        'Component': ['Quasar Mode', 'Merger-Driven', 'Instability-Driven', 
                      'Radio Mode', 'BH-BH Mergers', 'Seed'],
        'Model1_Mean': [df1_z0[c].mean() for c in components],
        'Model2_Mean': [df2_z0[c].mean() for c in components],
    })
    z0_comp['Difference'] = z0_comp['Model2_Mean'] - z0_comp['Model1_Mean']
    z0_comp['Pct_Difference'] = (z0_comp['Difference'] / z0_comp['Model1_Mean'].abs() * 100).round(2)
    
    z0_output = os.path.join(output_subdir, 'z0_component_comparison.csv')
    z0_comp.to_csv(z0_output, index=False)
    print(z0_comp.to_string())
    print(f"\nComponent comparison saved to: {z0_output}")
    
    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()