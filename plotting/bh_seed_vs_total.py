#!/usr/bin/env python3
import argparse
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# ============================================================================
# MATPLOTLIB STYLE 
# ============================================================================
plt.rcParams['figure.figsize'] = (8.34, 8.34)
plt.rcParams['figure.dpi'] = 140
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18.0
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ============================================================================
# CONFIGURATION
# ============================================================================
MIN_STELLAR_MASS_LOG = 8.5
MIN_HALO_MASS_LOG = 10.5
MIN_Z0_BH_MASS = 1e4

# ============================================================================
# HELPERS
# ============================================================================
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

def main():
    parser = argparse.ArgumentParser(description='Compare BH Seed Mass to Total BH Mass')
    parser.add_argument('-i', '--input-pattern', default='./output/millennium/model_*.hdf5',
                        help='Glob pattern for input HDF5 files')
    parser.add_argument('-s', '--snapshot', type=int, default=None,
                        help='Snapshot number (default: latest)')
    parser.add_argument('-o', '--output', default='seed_vs_total_mass.png',
                        help='Output filename for the plot')
    parser.add_argument('--no-bh-cut', action='store_true',
                        help='Disable BH mass cut (MIN_Z0_BH_MASS)')
    parser.add_argument('--no-stellar-cut', action='store_true',
                        help='Disable stellar mass cut (MIN_STELLAR_MASS_LOG)')
    parser.add_argument('--no-halo-cut', action='store_true',
                        help='Disable halo mass cut (MIN_HALO_MASS_LOG)')
    parser.add_argument('--no-cuts', action='store_true',
                        help='Disable all mass cuts')
    
    args = parser.parse_args()
    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"Error: No files found for {args.input_pattern}")
        sys.exit(1)

    sim = read_simulation_params(file_list[0])
    h_h = sim['Hubble_h']
    snap_num = args.snapshot if args.snapshot is not None else sim['latest_snapshot']

    bh_total = []
    bh_seed = []
    
    # Diagnostic lists to track problematic galaxies
    monster_seeds = []

    print(f"Snapshot: {snap_num}")
    print(f"Hubble_h: {h_h}")

    print(f"\nReading snapshot {snap_num}...")
    
    id_field = find_id_field(file_list, snap_num)
    if not id_field:
        print("Warning: ID field not found, using generic indices.")

    total_galaxies = 0
    
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f"Snap_{snap_num}"
            if snap_key not in hf: continue
            grp = hf[snap_key]
            
            if 'BlackHoleMass' not in grp: continue
            
            # Read required fields
            m_bh = grp['BlackHoleMass'][:] * 1e10 / h_h
            m_stellar = grp['StellarMass'][:] * 1e10 / h_h if 'StellarMass' in grp else np.zeros_like(m_bh)
            m_halo = grp['Mvir'][:] * 1e10 / h_h if 'Mvir' in grp else np.zeros_like(m_bh)
            gids = grp[id_field][:] if id_field and id_field in grp else np.arange(len(m_bh))
            
            # Handle SEED mass
            if 'BHSeedMass' in grp:
                m_seed = grp['BHSeedMass'][:] * 1e10 / h_h
            else:
                # Default seed if missing from file
                m_seed = np.full_like(m_bh, 1e4 * 1e10 / h_h)
            
            total_galaxies += len(m_bh)

            # Apply cuts
            if args.no_cuts:
                mask = (m_bh > 1e-10)
            else:
                mask = (m_bh > 1e-10)
                if not args.no_bh_cut:
                    mask &= (m_bh > MIN_Z0_BH_MASS)
                if not args.no_stellar_cut:
                    mask &= (m_stellar > 10**MIN_STELLAR_MASS_LOG)
                if not args.no_halo_cut:
                    mask &= (m_halo > 10**MIN_HALO_MASS_LOG)
            
            # Filter for "Monster Seeds" (> 10^9 M_sun) to report them but exclude from plot
            monster_mask = (m_seed > 1e9)
            
            # Combine everything for the final plot mask
            # We want: passing mass cuts AND NOT being a monster seed
            plot_mask = mask & (~monster_mask)
            
            # Diagnostic logging for monsters
            if np.any(mask & monster_mask):
                valid_indices = np.where(mask & monster_mask)[0]
                for midx in valid_indices:
                    monster_seeds.append({
                        'id': gids[midx],
                        'seed': m_seed[midx],
                        'total': m_bh[midx],
                        'stellar': m_stellar[midx],
                        'halo': m_halo[midx]
                    })

            bh_total.append(m_bh[plot_mask])
            bh_seed.append(m_seed[plot_mask])

    if not bh_total:
        print("No black holes found passing the cuts!")
        sys.exit(1)

    bh_total = np.concatenate(bh_total)
    bh_seed = np.concatenate(bh_seed)

    print(f"Initial galaxy count: {total_galaxies}")
    print(f"Galaxies passing cuts: {len(bh_total)}")
    
    if len(monster_seeds) > 0:
        print(f"\n! FOUND {len(monster_seeds)} MONSTER SEEDS (> 10^9 M_sun) !")
        print(f"{'GalaxyID':<15} | {'Seed Mass':<12} | {'Total BH':<12} | {'M_stellar':<12}")
        print("-" * 60)
        # Sort by seed mass descending
        monster_seeds.sort(key=lambda x: x['seed'], reverse=True)
        for m in monster_seeds[:10]: # Show first 10
            print(f"{m['id']:<15} | {m['seed']:1.2e} | {m['total']:1.2e} | {m['stellar']:1.2e}")
        if len(monster_seeds) > 10:
            print(f"... and {len(monster_seeds) - 10} more.")
    else:
        print("\n✓ No monster seeds (> 10^9 M_sun) detected in Sample.")

    # Plotting
    fig, ax = plt.subplots()
    
    # Scatter plot
    ax.scatter(bh_seed, bh_total, s=20, alpha=0.5, color='#2196F3', label='Galaxies', edgecolors='white', linewidth=0.3)
    
    # Determine bounds for 1:1 line
    valid_all = np.concatenate([bh_seed, bh_total])
    min_val = valid_all.min()
    max_val = valid_all.max()
    
    # 1:1 Line
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='1:1 Line ($M_{\\rm BH} = M_{\\rm seed}$)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlabel(r'$M_{\rm seed} [M_{\odot}]$')
    ax.set_ylabel(r'$M_{\rm BH, total} [M_{\odot}]$')
    #ax.set_title(f'BH Seed vs Total Mass Comparison (Snap {snap_num})')
    
    ax.legend(frameon=True, loc='lower right')
    # Verification check
    # Check if any BH is significantly smaller than its seed (tolerance for precision)
    violations = np.sum(bh_total < (bh_seed * 0.9999))
    if violations > 0:
        print(f"\n! WARNING: {violations} galaxies have BH mass LESS than seed mass!")
        min_ratio = np.min(bh_total / bh_seed)
        print(f"! Worst ratio: {min_ratio:.4f}")
    else:
        print("\n✓ Verification Success: All BH masses are >= seed mass.")

    # Save logic
    output_path = Path(args.output)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Diagnostic plot saved to: {output_path.absolute()}")

if __name__ == "__main__":
    main()
