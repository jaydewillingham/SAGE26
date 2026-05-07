#!/usr/bin/env python3

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

# X axis
plt.rcParams['xtick.major.size'] = 7.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['xtick.minor.size'] = 5.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['xtick.major.pad'] = 9

# Y axis
plt.rcParams['ytick.major.size'] = 7.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['ytick.minor.size'] = 5.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.labelsize'] = 16

# Font sizes and styles
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino']
plt.rcParams['font.size'] = 20.0
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.title_fontsize'] = 16

# Line widths
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.solid_capstyle'] = 'round'

# Legend
plt.rcParams['legend.frameon'] = False

# Color cycle: blue, orange, green, red, purple, black, grey
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', 
    ['#2196F3', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e'])

# ============================================================================
# CONFIGURATION
# ============================================================================

MIN_STELLAR_MASS_LOG = 8.5
MIN_HALO_MASS_LOG = 10.5
MIN_Z0_BH_MASS = 1e4

OutputFormat = '.png'

# ============================================================================
# HELPERS
# ============================================================================

def read_simulation_params(filepath):
    try:
        with h5py.File(filepath, 'r') as hf:
            if 'Header' in hf:
                header = hf['Header'].attrs
                return {
                    'Hubble_h': header.get('HubbleParam', 0.681),
                    'latest_snapshot': 62,
                }
    except:
        pass
    return {'Hubble_h': 0.681, 'latest_snapshot': 62}


def find_id_field(file_list, snap_num):
    candidates = ['GalaxyIndex', 'GalaxyID', 'ID']
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f"Snap_{snap_num}"
            if snap_key in hf:
                for c in candidates:
                    if c in hf[snap_key]:
                        print(f"✓ Found ID field '{c}'")
                        return c
    return None


# ============================================================================
# CORE FIXED READER
# ============================================================================

def read_all_fields(file_list, snap_num, id_field):
    """
    Read all required fields in a consistent, deduplicated way.
    """

    seen_ids = set()

    ids = []
    bh_mass = []
    stellar_mass = []
    mvir = []
    bh_seed = []

    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f"Snap_{snap_num}"
            if snap_key not in hf:
                continue

            grp = hf[snap_key]

            required = ['BlackHoleMass', 'StellarMass', 'Mvir', id_field]
            if not all(field in grp for field in required):
                continue

            file_ids = grp[id_field][:]

            # mask to remove duplicates
            mask = np.array([gid not in seen_ids for gid in file_ids])

            # update seen IDs
            for gid in file_ids[mask]:
                seen_ids.add(gid)

            if mask.sum() == 0:
                continue

            ids.append(file_ids[mask])
            bh_mass.append(grp['BlackHoleMass'][:][mask])
            stellar_mass.append(grp['StellarMass'][:][mask])
            mvir.append(grp['Mvir'][:][mask])

            # handle optional seed mass
            if 'BHSeedMass' in grp:
                bh_seed.append(grp['BHSeedMass'][:][mask])
            else:
                bh_seed.append(np.full(mask.sum(), np.nan))

    # concatenate everything
    ids = np.concatenate(ids)
    bh_mass = np.concatenate(bh_mass)
    stellar_mass = np.concatenate(stellar_mass)
    mvir = np.concatenate(mvir)
    bh_seed = np.concatenate(bh_seed)

    return ids, bh_mass, stellar_mass, mvir, bh_seed


# ============================================================================
# PLOTTING
# ============================================================================

def plot_bh_seed_histogram(bh_seed_masses, output_path):

    valid = bh_seed_masses[bh_seed_masses > 0]

    if len(valid) == 0:
        print("No valid BH seed masses!")
        return

    print(f"\nStats:")
    #print(f"N = {len(valid)}")
    print(f"Min = {valid.min():.3e}")
    print(f"Max = {valid.max():.3e}")
    print(f"Mean = {valid.mean():.3e}")

    log_m = np.log10(valid)

    fig, ax = plt.subplots()
    ax.hist(log_m, bins=50, density=True, alpha=0.75, edgecolor='black', linewidth=0.5)
    ax.set_xlabel(r'$\log_{10}(M_{\rm BH} [M_{\odot}])$')
    ax.set_ylabel('Density')
    #ax.set_title('BH Seed Mass Distribution')
    #ax.grid(True, alpha=0.3, linestyle='--')

    plt.savefig(output_path, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')
    plt.close()

    print(f"✓ Saved to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(description='Analyze BH seed masses from simulation data')
    parser.add_argument('-i', '--input-pattern', default='./output/millennium/model_*.hdf5',
                        help='Glob pattern for input HDF5 files')
    parser.add_argument('-s', '--snapshot', type=int, default=None,
                        help='Snapshot number (default: latest from header)')
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
        print(f"Error: No files found matching pattern '{args.input_pattern}'")
        sys.exit(1)

    sim = read_simulation_params(file_list[0])
    Hubble_h = sim['Hubble_h']
    snap_num = args.snapshot if args.snapshot else sim['latest_snapshot']

    print(f"Snapshot: {snap_num}")
    print(f"Hubble_h: {Hubble_h}")

    id_field = find_id_field(file_list, snap_num)

    print("\nReading data ...")

    ids, bh_mass, stellar_mass, mvir, bh_seed = read_all_fields(
        file_list, snap_num, id_field
    )

    print(f"Initial galaxy count: {len(ids)}")

    # convert units
    bh_mass *= 1e10 / Hubble_h
    stellar_mass *= 1e10 / Hubble_h
    mvir *= 1e10 / Hubble_h
    bh_seed *= 1e10 / Hubble_h

    # handle missing seeds
    missing = np.isnan(bh_seed)
    print(f"Missing BH seeds: {missing.sum()}")

    # assume fixed seed if missing
    bh_seed[missing] = 1e4

    # Apply mass cuts based on arguments
    if args.no_cuts:
        # Skip all cuts
        print("All mass cuts disabled")
        mass_cut = np.ones(len(bh_seed), dtype=bool)
    else:
        # Build mass cut based on individual flags
        mass_cut = np.ones(len(bh_seed), dtype=bool)
        
        if not args.no_bh_cut:
            bh_cut = (bh_mass > MIN_Z0_BH_MASS)
            print(f"  BH mass cut (>{MIN_Z0_BH_MASS:.1e}): {bh_cut.sum()} / {len(bh_mass)} pass")
            mass_cut &= bh_cut
        else:
            print(f"  BH mass cut disabled")
        
        if not args.no_stellar_cut:
            stellar_cut = (stellar_mass > 10**MIN_STELLAR_MASS_LOG)
            print(f"  Stellar mass cut (>{10**MIN_STELLAR_MASS_LOG:.1e}): {stellar_cut.sum()} / {len(stellar_mass)} pass")
            mass_cut &= stellar_cut
        else:
            print(f"  Stellar mass cut disabled")
        
        if not args.no_halo_cut:
            halo_cut = (mvir > 10**MIN_HALO_MASS_LOG)
            print(f"  Halo mass cut (>{10**MIN_HALO_MASS_LOG:.1e}): {halo_cut.sum()} / {len(mvir)} pass")
            mass_cut &= halo_cut
        else:
            print(f"  Halo mass cut disabled")

    bh_seed = bh_seed[mass_cut]

    print(f"\nAfter cuts: {len(bh_seed)} galaxies")

    output_dir = Path(file_list[0]).parent / 'plots'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f'bh_seed_mass_histogram{OutputFormat}'

    plot_bh_seed_histogram(bh_seed, output_file)

    print("\nDone.")


if __name__ == "__main__":
    main()