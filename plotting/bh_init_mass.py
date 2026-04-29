#!/usr/bin/env python3

import argparse
import glob
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
    print(f"N = {len(valid)}")
    print(f"Min = {valid.min():.3e}")
    print(f"Max = {valid.max():.3e}")
    print(f"Mean = {valid.mean():.3e}")

    log_m = np.log10(valid)

    plt.figure(figsize=(10,6))
    plt.hist(log_m, bins=50, density=True, alpha=0.7)
    plt.xlabel(r'$\log_{10}(M_{\rm BH, seed})$')
    plt.ylabel('Density')
    plt.title('BH Seed Mass Distribution')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✓ Saved to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-pattern', default='./output/millennium/model_*.hdf5')
    parser.add_argument('-s', '--snapshot', type=int, default=None)
    args = parser.parse_args()

    print("="*60)
    print("BH Seed Mass Histogram Generator")
    print("="*60)

    file_list = sorted(glob.glob(args.input_pattern))

    if not file_list:
        print("No files found!")
        sys.exit(1)

    print(f"Found {len(file_list)} files")

    sim = read_simulation_params(file_list[0])
    Hubble_h = sim['Hubble_h']
    snap_num = args.snapshot if args.snapshot else sim['latest_snapshot']

    print(f"Snapshot: {snap_num}")
    print(f"Hubble_h: {Hubble_h}")

    id_field = find_id_field(file_list, snap_num)
    if id_field is None:
        print("No ID field found!")
        sys.exit(1)

    print("\nReading data (deduplicated)...")

    ids, bh_mass, stellar_mass, mvir, bh_seed = read_all_fields(
        file_list, snap_num, id_field
    )

    print(f"Final galaxy count: {len(ids)}")

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

    # mass cuts
    mass_cut = (
        (bh_mass > MIN_Z0_BH_MASS) &
        (stellar_mass > 10**MIN_STELLAR_MASS_LOG) &
        (mvir > 10**MIN_HALO_MASS_LOG)
    )

    bh_seed = bh_seed[mass_cut]

    print(f"After cuts: {len(bh_seed)} galaxies")

    output_dir = Path(file_list[0]).parent / 'plots'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f'bh_seed_mass_histogram{OutputFormat}'

    plot_bh_seed_histogram(bh_seed, output_file)

    print("\nDone.")


if __name__ == "__main__":
    main()