#!/usr/bin/env python3
import argparse
import glob
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

# ============================================================================
# MATPLOTLIB STYLE CONFIGURATION (Matching Original Style)
# ============================================================================
plt.rcParams['figure.figsize'] = (8.34, 6.25)
plt.rcParams['figure.dpi'] = 140
plt.rcParams['figure.autolayout'] = True

# Fonts - Cleaned up to avoid environment errors
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 20.0

# Axis and Ticks
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
MIN_STELLAR_MASS_LOG = 8.5
MIN_HALO_MASS_LOG = 10.5
MIN_Z0_BH_MASS = 1e4

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

def read_data(file_list, snap_num, id_field, h_h):
    """Reads fields including BHMaxaccretionMass (time-series array Ngal x MAXSNAPS)."""
    all_ids = []
    all_bh_mass = []
    all_stellar_mass = []
    all_mvir = []
    all_bh_seed = []
    all_first_channel = []
    all_birth_snaps = []
    all_birth_vals = []
    all_bh_max_accretion_history = []  # NEW: Track full time-series (Ngal x MAXSNAPS)

    seen_ids = set()

    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f"Snap_{snap_num}"
            if snap_key not in hf: continue
            grp = hf[snap_key]

            gids = grp[id_field][:]
            mask = np.array([gid not in seen_ids for gid in gids])
            for gid in gids[mask]: seen_ids.add(gid)
            if not np.any(mask): continue

            conv = 1e10 / h_h
            
            m_bh = grp['BlackHoleMass'][:][mask] * conv
            m_stellar = grp['StellarMass'][:][mask] * conv
            m_mvir = grp['Mvir'][:][mask] * conv

            # Read BHMaxaccretionMass as time-series (Ngal x MAXSNAPS)
            if 'BHMaxaccretionMass' in grp:
                bh_max_accr_hist = grp['BHMaxaccretionMass'][:][mask] * conv
            else:
                # Fallback: create empty time-series
                bh_max_accr_hist = np.zeros((len(m_bh), 63))  # 63 = MAXSNAPS for Millennium

            all_ids.append(gids[mask])
            all_bh_mass.append(m_bh)
            all_stellar_mass.append(m_stellar)
            all_mvir.append(m_mvir)
            all_bh_max_accretion_history.append(bh_max_accr_hist)

    return (np.concatenate(all_ids), np.concatenate(all_bh_mass), 
            np.concatenate(all_stellar_mass), np.concatenate(all_mvir), 
            np.concatenate(all_bh_max_accretion_history))

def create_eddington_redshift_plot(
    bh_max_accr_history,
    b_snaps,
    plot_mask,
    snap_to_z_dict,
    total_galaxies,
    output_file,
    eddington_threshold=0.0,
    n_bins_z=20
):
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.minorticks_on()
    
    # Extract filtered data
    bh_hist_filtered = bh_max_accr_history[plot_mask]
    b_snaps_filtered = b_snaps[plot_mask]
    
    ngal = len(b_snaps_filtered)
    maxsnaps = bh_hist_filtered.shape[1]
    
    print(f"Processing {ngal} galaxies with {maxsnaps} snapshots each")
    print(f"Counting Eddington exceedances as: BHMaxaccretionMass[snap] > {eddington_threshold}")
    
    # For each snapshot, count how many galaxies exceeded Eddington
    # Only count galaxies that have been born by that snapshot
    snap_counts_edd = np.zeros(maxsnaps)
    snap_counts_total = np.zeros(maxsnaps)
    
    for snap_idx in range(maxsnaps):
        # Which galaxies have been born by this snapshot?
        born_mask = b_snaps_filtered <= snap_idx
        snap_counts_total[snap_idx] = np.sum(born_mask)
        
        if snap_counts_total[snap_idx] > 0:
            # Among born galaxies, how many exceeded Eddington?
            # BHMaxaccretionMass[snap] > 0 means Eddington was exceeded
            edd_at_snap = (bh_hist_filtered[born_mask, snap_idx] > eddington_threshold)
            snap_counts_edd[snap_idx] = np.sum(edd_at_snap)
    
    # Convert snapshot indices to redshifts
    redshifts = np.array([get_redshift_from_snapshot(s, snap_to_z_dict) for s in range(maxsnaps)])
    
    # Bin by redshift
    z_bins = np.linspace(redshifts.min(), redshifts.max(), n_bins_z + 1)
    z_bin_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    z_bin_width = z_bins[1] - z_bins[0]
    
    binned_counts_edd = np.zeros(n_bins_z)
    binned_counts_total = np.zeros(n_bins_z)
    
    for i in range(n_bins_z):
        snaps_in_bin = np.where((redshifts >= z_bins[i]) & (redshifts < z_bins[i+1]))[0]
        if len(snaps_in_bin) > 0:
            binned_counts_edd[i] = np.sum(snap_counts_edd[snaps_in_bin])
            binned_counts_total[i] = np.sum(snap_counts_total[snaps_in_bin])
    
    # Density: count per bin / (total galaxies × bin width)
    density = binned_counts_edd / (total_galaxies * z_bin_width)
    
    # Plot density
    ax.step(z_bin_centers, density, where='mid', linewidth=2.5, 
            color='#D32F2F', label='Eddington Exceedances', marker='o', markersize=7)
    ax.fill_between(z_bin_centers, density, step='mid', alpha=0.2, color='#D32F2F')
    
    # Also show fraction on secondary axis
    ax_twin = ax.twinx()
    fraction = np.where(binned_counts_total > 0, binned_counts_edd / binned_counts_total, 0)
    ax_twin.plot(z_bin_centers, fraction, linewidth=2.0, 
                 color='#1976D2', alpha=0.6, linestyle='--', marker='s', markersize=5,
                 label='Fraction of Galaxies')
    
    ax.set_xlabel('Redshift (z)', fontsize=14)
    ax.set_ylabel(r'Density [Events / (total × $\Delta z$)]', fontsize=12, color='#D32F2F')
    ax_twin.set_ylabel('Fraction of Born Galaxies', fontsize=12, color='#1976D2')
    
    ax.tick_params(axis='y', labelcolor='#D32F2F')
    ax_twin.tick_params(axis='y', labelcolor='#1976D2')
    
    ax.set_xlim(z_bins[0] - 1, z_bins[-1] + 1)
    ax.set_ylim(bottom=0)
    
    # Legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)
    
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    print(f"✓ Eddington vs redshift plot saved to: {output_file}")
    print(f"  Total Eddington exceedance events (across all times): {int(np.sum(snap_counts_edd))}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-pattern', default='./output/millennium/model_*.hdf5')
    parser.add_argument('-s', '--snapshot', type=int, default=None)
    parser.add_argument('--no-cuts', action='store_true', help='Skip mass cuts (use all galaxies)')
    parser.add_argument('--eddington-threshold', type=float, default=0.0,
                       help='Threshold for BHMaxaccretionMass to count as Eddington exceedance (default: >0)')
    args = parser.parse_args()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"No files found for {args.input_pattern}"); sys.exit(1)

    sim = read_simulation_params(file_list[0])
    h_h = sim['Hubble_h']
    snap_num = args.snapshot if args.snapshot is not None else sim['latest_snapshot']
    id_field = find_id_field(file_list, snap_num)
    
    redshift = get_redshift_from_snapshot(snap_num)

    print(f"Snapshot: {snap_num} | Redshift: {redshift:.3f} | Hubble_h: {h_h}")
    print("Reading data and tracking growth channels...")
    
    ids, bh_mass, stellar_mass, mvir, bh_max_accr_hist = \
        read_data(file_list, snap_num, id_field, h_h)

    # Filtering
    if args.no_cuts:
        plot_mask = (bh_mass > 0)
    else:
        plot_mask = (bh_mass > MIN_Z0_BH_MASS) & (stellar_mass > 10**MIN_STELLAR_MASS_LOG) & (mvir > 10**MIN_HALO_MASS_LOG)

    print(f"\nInitial galaxies: {len(ids)}")
    print(f"Final plot count: {np.sum(plot_mask)}")

    output_dir = Path(file_list[0]).parent / 'plots'
    output_dir.mkdir(exist_ok=True)

    # ========================================================================
    # EDDINGTON LIMIT EXCEEDANCE ANALYSIS (TIME-SERIES)
    # ========================================================================
    print("\n" + "="*70)
    print("Eddington Limit Exceedance Analysis (Time-Series)")
    print("="*70)
    
    total_passed = np.sum(plot_mask)
    
    # Create time-series Eddington plot
    # Note: passing zero array for birth snaps since we're not tracking growth channels anymore
    birth_snaps = np.zeros(len(bh_mass), dtype=int)
    create_eddington_redshift_plot(
        bh_max_accr_hist,
        birth_snaps,
        plot_mask,
        MILLENNIUM_SNAP_TO_Z,
        total_passed,
        output_dir / 'bh_eddington_vs_redshift.png',
        eddington_threshold=args.eddington_threshold,
        n_bins_z=20
    )
    
    print("\n✓ All plots completed successfully!")

if __name__ == "__main__":
    main()