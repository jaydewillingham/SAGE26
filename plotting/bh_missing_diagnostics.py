#!/usr/bin/env python3
"""
Progressive diagnostic analysis of BH populations.
Applies cuts one at a time and outputs statistics for each stage.
Helps identify where zero-BH galaxies come from.
"""

import argparse
import glob
import sys
import h5py
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
MIN_STELLAR_MASS_LOG = 6.0
MIN_HALO_MASS_LOG = 11.0
MIN_Z0_BH_MASS = 0
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
                    if c in hf[snap_key]:
                        return c

    return None


def read_data(file_list, snap_num, id_field, h_h):
    """
    Reads BH, stellar mass, halo mass, bulge mass, and seed mass data.
    """

    all_bh_mass = []
    all_stellar_mass = []
    all_mvir = []
    all_bulge_mass = []
    all_bh_seed = []

    seen_ids = set()

    for f in file_list:
        with h5py.File(f, 'r') as hf:

            snap_key = f"Snap_{snap_num}"

            if snap_key not in hf:
                continue

            grp = hf[snap_key]

            id_field_data = grp[id_field][:]

            mask = np.array([gid not in seen_ids for gid in id_field_data])

            for gid in id_field_data[mask]:
                seen_ids.add(gid)

            if not np.any(mask):
                continue

            conv = 1e10 / h_h

            m_bh = grp['BlackHoleMass'][:][mask] * conv
            m_stellar = grp['StellarMass'][:][mask] * conv
            m_mvir = grp['Mvir'][:][mask] * conv

            if 'BulgeMass' in grp:
                m_bulge = grp['BulgeMass'][:][mask] * conv
            else:
                m_bulge = np.full_like(m_bh, np.nan)

            if 'BHSeedMass' in grp:
                m_seed = grp['BHSeedMass'][:][mask] * conv
            else:
                m_seed = np.full_like(m_bh, 1e4)

            all_bh_mass.append(m_bh)
            all_stellar_mass.append(m_stellar)
            all_mvir.append(m_mvir)
            all_bulge_mass.append(m_bulge)
            all_bh_seed.append(m_seed)

    return (
        np.concatenate(all_bh_mass),
        np.concatenate(all_stellar_mass),
        np.concatenate(all_mvir),
        np.concatenate(all_bulge_mass),
        np.concatenate(all_bh_seed)
    )


def print_diagnostic_section(
    title,
    bh_mass,
    stellar_mass,
    mvir,
    bulge_mass,
    bh_seed,
    mask=None,
    outfile=None
):
    """
    Print diagnostics for a given population.
    Writes to terminal always.
    Writes to file only if outfile is provided.
    """

    def out(text=""):
        print(text)

        if outfile is not None:
            print(text, file=outfile)

    if mask is None:
        mask = np.ones(len(bh_mass), dtype=bool)

    pop_bh = bh_mass[mask]
    pop_stellar = stellar_mass[mask]
    pop_mvir = mvir[mask]
    pop_bulge = bulge_mass[mask]
    pop_seed = bh_seed[mask]

    n_total = np.sum(mask)

    # =========================================================================
    # BH diagnostics
    # =========================================================================
    bh_finite = np.isfinite(pop_bh)
    bh_positive = bh_finite & (pop_bh > 0)
    bh_zero = bh_finite & (pop_bh == 0)
    bh_nonfinite = ~bh_finite

    n_bh_finite = np.sum(bh_finite)
    n_bh_positive = np.sum(bh_positive)
    n_bh_zero = np.sum(bh_zero)
    n_bh_nonfinite = np.sum(bh_nonfinite)

    # =========================================================================
    # Zero-BH galaxy analysis
    # =========================================================================
    if n_bh_zero > 0:

        stellar_of_zero_bh = pop_stellar[bh_zero]
        bulge_of_zero_bh = pop_bulge[bh_zero]

        stellar_median = np.nanmedian(stellar_of_zero_bh)
        stellar_mean = np.nanmean(stellar_of_zero_bh)
        stellar_min = np.nanmin(stellar_of_zero_bh)
        stellar_max = np.nanmax(stellar_of_zero_bh)

        bulge_positive = np.sum(bulge_of_zero_bh > 0)
        bulge_zero = np.sum(bulge_of_zero_bh == 0)
        bulge_nan = np.sum(~np.isfinite(bulge_of_zero_bh))

    else:

        stellar_median = np.nan
        stellar_mean = np.nan
        stellar_min = np.nan
        stellar_max = np.nan

        bulge_positive = 0
        bulge_zero = 0
        bulge_nan = 0

    # =========================================================================
    # PRINT SECTION
    # =========================================================================
    out("\n" + "=" * 80)
    out(f" {title}")
    out("=" * 80)

    out(f"\nTotal galaxies in this population: {n_total:,}")

    out(f"\nBlack Hole Mass Diagnostics:")
    out(f"  Total galaxies: {n_total:,}")
    out(f"  Finite BH mass: {n_bh_finite:,} ({100*n_bh_finite/n_total:.1f}%)")
    out(f"  Positive BH mass: {n_bh_positive:,} ({100*n_bh_positive/n_total:.1f}%)")
    out(f"  ZERO BH mass: {n_bh_zero:,} ({100*n_bh_zero/n_total:.1f}%)")
    out(f"  Non-finite BH mass: {n_bh_nonfinite:,} ({100*n_bh_nonfinite/n_total:.1f}%)")

    if n_bh_zero > 0:

        out(f"\n*** ANALYSIS OF {n_bh_zero:,} ZERO-BH GALAXIES ***")

        out(f"\n  Stellar Mass Properties:")
        out(f"    Median stellar mass: {stellar_median:.3e} M☉")
        out(f"    Mean stellar mass: {stellar_mean:.3e} M☉")
        out(f"    Min stellar mass: {stellar_min:.3e} M☉")
        out(f"    Max stellar mass: {stellar_max:.3e} M☉")

        out(f"\n  Bulge Mass Properties:")
        out(f"    Bulge mass > 0: {bulge_positive:,} ({100*bulge_positive/n_bh_zero:.1f}%)")
        out(f"    Bulge mass = 0: {bulge_zero:,} ({100*bulge_zero/n_bh_zero:.1f}%)")
        out(f"    Bulge mass unavailable: {bulge_nan:,} ({100*bulge_nan/n_bh_zero:.1f}%)")

        in_resolution_limit = np.sum(stellar_of_zero_bh < 1e9)
        above_resolution = np.sum(stellar_of_zero_bh >= 1e9)

        out(f"\n  Stellar Mass Distribution of Zero-BH galaxies:")
        out(f"    Below 10^9 M☉ (resolution limit): {in_resolution_limit:,} ({100*in_resolution_limit/n_bh_zero:.1f}%)")
        out(f"    Above 10^9 M☉ (resolved): {above_resolution:,} ({100*above_resolution/n_bh_zero:.1f}%)")

    # =========================================================================
    # Seed mass statistics
    # =========================================================================
    if np.sum(np.isfinite(pop_seed)) > 0:

        seed_valid = pop_seed[np.isfinite(pop_seed)]

        out(f"\nBH Seed Mass Statistics:")
        out(f"  Mean seed mass: {np.mean(seed_valid):.3e} M☉")
        out(f"  Median seed mass: {np.median(seed_valid):.3e} M☉")
        out(f"  Min seed mass: {np.min(seed_valid):.3e} M☉")
        out(f"  Max seed mass: {np.max(seed_valid):.3e} M☉")


# ============================================================================
# MAIN
# ============================================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        '--input-pattern',
        default='./output/millennium/model_*.hdf5'
    )

    parser.add_argument(
        '-s',
        '--snapshot',
        type=int,
        default=None
    )

    parser.add_argument(
        '-o',
        '--output-file',
        type=str,
        default=None
    )

    args = parser.parse_args()

    # =========================================================================
    # Find files
    # =========================================================================
    file_list = sorted(glob.glob(args.input_pattern))

    if not file_list:
        print(f"No files found for {args.input_pattern}")
        sys.exit(1)

    sim = read_simulation_params(file_list[0])

    h_h = sim['Hubble_h']

    snap_num = (
        args.snapshot
        if args.snapshot is not None
        else sim['latest_snapshot']
    )

    id_field = find_id_field(file_list, snap_num)

    print(f"Snapshot: {snap_num} | Hubble_h: {h_h}")
    print("Reading data...")

    bh_mass, stellar_mass, mvir, bulge_mass, bh_seed = read_data(
        file_list,
        snap_num,
        id_field,
        h_h
    )

    print(f"Total galaxies loaded: {len(bh_mass):,}")

    # =========================================================================
    # Output file
    # =========================================================================
    outfile = None

    if args.output_file:

        output_path = Path(args.output_file)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        outfile = open(output_path, 'w')

        print("BH Population Diagnostic Analysis", file=outfile)
        print(f"Snapshot: {snap_num} | Hubble_h: {h_h}", file=outfile)
        print(f"Total galaxies loaded: {len(bh_mass):,}", file=outfile)

    # =========================================================================
    # SECTION 0
    # =========================================================================
    print_diagnostic_section(
        "SECTION 0: ALL GALAXIES (NO CUTS)",
        bh_mass,
        stellar_mass,
        mvir,
        bulge_mass,
        bh_seed,
        outfile=outfile
    )

    # =========================================================================
    # SECTION 1
    # =========================================================================
    stellar_low = 10.0

    mask_stellar_low = stellar_mass > 10**stellar_low

    print_diagnostic_section(
        f"SECTION 1: STELLAR MASS CUT (M_stellar > 10^{stellar_low} M☉)",
        bh_mass,
        stellar_mass,
        mvir,
        bulge_mass,
        bh_seed,
        mask=mask_stellar_low,
        outfile=outfile
    )

    # =========================================================================
    # SECTION 2
    # =========================================================================
    stellar_high = MIN_STELLAR_MASS_LOG

    mask_stellar_high = stellar_mass > 10**stellar_high

    print_diagnostic_section(
        f"SECTION 2: STELLAR MASS CUT (M_stellar > 10^{stellar_high} M☉)",
        bh_mass,
        stellar_mass,
        mvir,
        bulge_mass,
        bh_seed,
        mask=mask_stellar_high,
        outfile=outfile
    )

    # =========================================================================
    # SECTION 3
    # =========================================================================
    mask_stellar_halo = (
        (stellar_mass > 10**MIN_STELLAR_MASS_LOG)
        &
        (mvir > 10**MIN_HALO_MASS_LOG)
    )

    print_diagnostic_section(
        f"SECTION 3: STELLAR + HALO MASS CUTS "
        f"(M_stellar > 10^{MIN_STELLAR_MASS_LOG}, "
        f"M_halo > 10^{MIN_HALO_MASS_LOG})",

        bh_mass,
        stellar_mass,
        mvir,
        bulge_mass,
        bh_seed,

        mask=mask_stellar_halo,
        outfile=outfile
    )

    # =========================================================================
    # SECTION 4
    # =========================================================================
    mask_all_cuts = (
        (bh_mass > MIN_Z0_BH_MASS)
        &
        (stellar_mass > 10**MIN_STELLAR_MASS_LOG)
        &
        (mvir > 10**MIN_HALO_MASS_LOG)
    )

    print_diagnostic_section(
        f"SECTION 4: ALL CUTS "
        f"(M_BH > {MIN_Z0_BH_MASS}, "
        f"M_stellar > 10^{MIN_STELLAR_MASS_LOG}, "
        f"M_halo > 10^{MIN_HALO_MASS_LOG})",

        bh_mass,
        stellar_mass,
        mvir,
        bulge_mass,
        bh_seed,

        mask=mask_all_cuts,
        outfile=outfile
    )

    # =========================================================================
    # SECTION 5 (TERMINAL ONLY)
    # =========================================================================
    monster_mask = bh_seed > MONSTER_SEED_THRESHOLD

    mask_final = mask_all_cuts & (~monster_mask)

    print_diagnostic_section(
        f"SECTION 5: FINAL SAMPLE "
        f"(All cuts + monster seeds removed "
        f"> {MONSTER_SEED_THRESHOLD:.1e})",

        bh_mass,
        stellar_mass,
        mvir,
        bulge_mass,
        bh_seed,

        mask=mask_final
    )

    # =========================================================================
    # Close output file
    # =========================================================================
    if outfile is not None:

        outfile.close()

        print(f"\n✓ Output saved to: {output_path}")

    print("\n" + "=" * 80)
    print("✓ Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()