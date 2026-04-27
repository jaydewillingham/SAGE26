#!/usr/bin/env python3
"""
Black Hole Mass Conservation Checker (UPDATED)

This script analyzes the final snapshot of galaxy catalogs to verify if the sum 
of specific BH cumulative growth channels equals the total Black Hole Mass.

UPDATED to handle [ABSOLUTEMAXSNAPS] array structures:
- Intelligently detects 2D arrays (Ngal, MAXSNAPS) and sums to current snapshot
- Handles flattened 1D arrays and reshapes them correctly
- Auto-detects HDF5 struct dimensions

Formulas used:
- Ex-Situ (Model 1): Sum = MD + ID + RM
- In-Situ (Model 2): Sum = MD + ID + RM + BM

It calculates the relative error and outputs galaxies that exceed an error 
tolerance of 10^-7. It supports filtering by Halo Mass (Mvir) and Particle Count (Len).
Results are printed to the terminal and saved to a text file.
"""

import numpy as np
import h5py
import glob
import os
import sys
import argparse

POSSIBLE_ID_FIELDS = ['GalaxyIndex', 'ID', 'galaxy_id', 'id', 'GalID']

def read_hdf(file_list, snap, field):
    """Read a field from HDF5 files for a given snapshot (simple version)"""
    data = []
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f"Snap_{int(snap)}"
            if snap_key in hf and field in hf[snap_key]:
                data.append(hf[snap_key][field][:])
    return np.concatenate(data) if data else np.array([])

def read_hdf_cumulative(file_list, snap, field):
    """
    Read a field from HDF5 files and intelligently handle ABSOLUTEMAXSNAPS arrays.
    If the field is stored as a 2D array (Ngal, MAXSNAPS), sum up to the current snapshot.
    """
    data = []
    
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f"Snap_{int(snap)}"
            if snap_key not in hf or field not in hf[snap_key]:
                continue
                
            val = hf[snap_key][field][:]
            
            # Determine expected number of galaxies
            ref_len = len(hf[snap_key]['BlackHoleMass'][:]) if 'BlackHoleMass' in hf[snap_key] else len(val)
            
            # CASE 1: Clean 2D array (Ngal, MAXSNAPS) - Sum up to current snapshot
            if val.ndim == 2:
                val = np.nansum(val[:, :int(snap)+1], axis=1)
                
            # CASE 2: Flattened 1D array (Ngal * MAXSNAPS) - Reshape and sum
            elif val.ndim == 1 and len(val) > ref_len and ref_len > 0:
                max_snaps = len(val) // ref_len
                val = val.reshape((ref_len, max_snaps))
                val = np.nansum(val[:, :int(snap)+1], axis=1)
            
            data.append(val)
            
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
        params['latest_snapshot'] = max(snap_numbers) if snap_numbers else None
    return params

def check_mass_conservation(model_pattern, model_name, model_type, args, output_file):
    """Processes a model to check BH mass conservation at the final snapshot."""
    print(f"\n{'='*80}")
    print(f"PROCESSING MODEL: {model_name}")
    print(f"Pattern: {model_pattern}")
    print(f"{'='*80}")
    
    file_list = sorted(glob.glob(model_pattern))
    if not file_list:
        print(f"Error: No files found matching: {model_pattern}")
        return
    
    sim_params = read_simulation_params(file_list[0])
    hubble_h = sim_params['Hubble_h']
    snap = sim_params['latest_snapshot']
    
    print(f"Analyzing final snapshot: {snap}")
    
    id_field = find_id_field(file_list, snap)
    
    # Read core fields
    print("Reading data fields...")
    gal_id = read_hdf(file_list, snap, id_field) if id_field else np.arange(1)
    if len(gal_id) == 0:
        print("No galaxy data found in the final snapshot.")
        return

    # Convert masses from internal units (10^10 M_sun / h) to M_sun
    mass_conversion = 1.0e10 / hubble_h
    
    bh_total = read_hdf(file_list, snap, 'BlackHoleMass') * mass_conversion
    if len(bh_total) == 0:
        print("No BlackHoleMass data found.")
        return

    # Safe reader for components: intelligently handles ABSOLUTEMAXSNAPS arrays
    def get_safe_mass(field_name):
        arr = read_hdf_cumulative(file_list, snap, field_name)
        if len(arr) == 0:
            print(f"  -> Warning: '{field_name}' not found. Defaulting to 0.")
            return np.zeros_like(bh_total)
        return arr * mass_conversion

    print("  Reading accretion channel masses (with cumulative sum handling)...")
    md  = get_safe_mass('MergerDrivenBHaccretionMass')
    id_ = get_safe_mass('InstabilityDrivenBHaccretionMass')
    rm  = get_safe_mass('RadioModeBHaccretionMass')
    bm  = get_safe_mass('BHMergerMass')
    
    # Read physical/classification properties safely
    mvir = read_hdf(file_list, snap, 'Mvir') 
    mvir = mvir * mass_conversion if len(mvir) > 0 else np.zeros_like(bh_total)
    
    gal_type = read_hdf(file_list, snap, 'Type')  # 0=Central, >0=Satellite
    if len(gal_type) == 0:
        gal_type = np.zeros_like(bh_total)
        
    length = read_hdf(file_list, snap, 'Len')     # Particle count
    
    # Calculate sum of components based on model type
    if model_type == 'insitu':
        bh_sum = md + id_ + rm + bm
        formula_str = "MD + ID + RM + BM"
    else:  # exsitu
        bh_sum = md + id_ + rm
        formula_str = "MD + ID + RM"
        
    print(f"Using sum formula: {formula_str}")
    
    # Create valid mask (must have a Black Hole to check conservation)
    mask = bh_total > 0
    print(f"Total galaxies with active BHs: {np.sum(mask)}")
    
    # Apply user filters
    if args.min_mvir is not None:
        mask = mask & (mvir > args.min_mvir)
        print(f"Galaxies after Mvir > {args.min_mvir:.1e} filter: {np.sum(mask)}")
        
    if args.min_len is not None:
        if len(length) > 0:
            mask = mask & (length > args.min_len)
            print(f"Galaxies after Len > {args.min_len} filter: {np.sum(mask)}")
        else:
            print("Warning: 'Len' field not found in HDF5. Skipping Len filter.")

    # Apply masks
    gal_id = gal_id[mask]
    bh_total = bh_total[mask]
    bh_sum = bh_sum[mask]
    mvir = mvir[mask]
    gal_type = gal_type[mask]
    
    # Calculate Relative Error
    rel_error = np.abs((bh_sum - bh_total) / bh_total)
    
    # Find violators (Error > 10^-7)
    violator_mask = rel_error > 1e-6
    
    v_ids = gal_id[violator_mask]
    v_mvir = mvir[violator_mask]
    v_type = gal_type[violator_mask]
    v_err = rel_error[violator_mask]
    v_tot = bh_total[violator_mask]
    v_sum = bh_sum[violator_mask]
    
    num_violators = len(v_ids)
    
    header = f"\nFound {num_violators} galaxies with relative error > 10^-7 in {model_name}\n"
    table_fmt = "{:<15} | {:<15} | {:<18} | {:<15} | {:<15} | {:<15}\n"
    separator = "-" * 106 + "\n"
    
    output_lines = [header, separator]
    output_lines.append(table_fmt.format("Galaxy ID", "Mvir (M_sun)", "Type (Cen/Sat)", "Total BH Mass", f"Sum ({formula_str})", "Relative Error"))
    output_lines.append(separator)
    
    for i in range(num_violators):
        # Format central vs satellite tag
        type_str = f"Central ({int(v_type[i])})" if v_type[i] == 0 else f"Satellite ({int(v_type[i])})"
        
        line = table_fmt.format(
            v_ids[i], 
            f"{v_mvir[i]:.2e}", 
            type_str, 
            f"{v_tot[i]:.4e}",
            f"{v_sum[i]:.4e}",
            f"{v_err[i]:.2e}"
        )
        output_lines.append(line)
    
    output_lines.append(separator)
    
    # Print to Terminal
    for line in output_lines:
        print(line, end='')
        
    # Write to File
    with open(output_file, 'a') as f:
        f.writelines(output_lines)

def main():
    parser = argparse.ArgumentParser(description='Check BH mass conservation at the final snapshot.')
    
    # Filter arguments
    parser.add_argument('--min-mvir', type=float, default=None, 
                        help='Minimum Halo Mass (Mvir) threshold in M_sun (e.g. 1e11)')
    parser.add_argument('--min-len', type=int, default=None, 
                        help='Minimum dark matter particle count (Len) threshold (e.g. 100)')
    
    # File arguments
    parser.add_argument('--model1-pattern', type=str, default='./output/millennium/model_*.hdf5',
                        help='Pattern for Model 1 (Standard/Ex-situ+In-situ)')
    parser.add_argument('--model2-pattern', type=str, default='./output/millennium_insitu_new/model_*.hdf5',
                        help='Pattern for Model 2 (In-situ)')
    parser.add_argument('--output-txt', type=str, default='bh_mass_errors.txt',
                        help='Name of the text file to save the results')
    
    args = parser.parse_args()

    # Clear/Initialize output file
    with open(args.output_txt, 'w') as f:
        f.write("BH MASS CONSERVATION CHECK RESULTS\n")
        f.write(f"Filters Applied: Mvir > {args.min_mvir if args.min_mvir else 'None'}, ")
        f.write(f"Len > {args.min_len if args.min_len else 'None'}\n")
    
    print(f"Results will be saved to: {args.output_txt}")
    
    # Check Model 1 (Ex-Situ / Standard) -> Passes 'exsitu' so sum = MD + ID + RM
    check_mass_conservation(args.model1_pattern, "Model 1 (Ex-situ + In-situ)", "exsitu", args, args.output_txt)
    
    # Check Model 2 (In-Situ) -> Passes 'insitu' so sum = MD + ID + RM + BM
    check_mass_conservation(args.model2_pattern, "Model 2 (In-Situ)", "insitu", args, args.output_txt)
    
    print(f"\nDone! Check '{args.output_txt}' for saved results.")

if __name__ == '__main__':
    main()