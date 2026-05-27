#!/usr/bin/env python3
"""
Enhanced Black hole Eddington limit analysis with deltaBH histograms:

1. Time-series: Fraction of super-Eddington BHs vs redshift
2. Distribution: Histogram of accretion rate ratios (where rate > Eddington limit)
3. Summary statistics on dt parameter
4. Accretion rate ratio vs dt scatter plot
5. NEW: deltaBH histogram - BH mass growth difference with/without SE accretion
6. NEW: deltaBH vs mass bins - Track how SE impact varies with BH mass
"""
import argparse
import glob
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# MATPLOTLIB STYLE CONFIGURATION
# ============================================================================
plt.rcParams['figure.figsize'] = (8.34, 6.25)
plt.rcParams['figure.dpi'] = 140
plt.rcParams['figure.autolayout'] = True

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 20.0

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
MIN_HALO_MASS_LOG = 11
MIN_Z0_BH_MASS = 1e4
MIN_BH_MASS_FOR_DELTABH = 1e5  # Only well-resolved BHs

SEC_PER_YEAR = 365.25 * 24 * 3600
UNIT_TIME_IN_S = 3.086e19  # 1 Gyr in seconds (Millennium default)

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
    """Reads BH growth diagnostics as time-series arrays (Ngal x MAXSNAPS)."""
    all_ids = []
    all_bh_mass = []
    all_stellar_mass = []
    all_mvir = []
    all_bh_max_accretion_history = []
    all_bh_eddington_rate_limit = []
    all_bh_mass_at_accretion = []
    all_dt = []

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

            # Read time-series arrays
            if 'BHMaxaccretionRate' in grp:
                bh_max_accr_raw = grp['BHMaxaccretionRate'][:][mask] * conv
            else:
                bh_max_accr_raw = np.zeros((len(m_bh), 63))
            
            if bh_max_accr_raw.ndim == 1:
                ngal = len(m_bh)
                if len(bh_max_accr_raw) == ngal:
                    bh_max_accr_hist = bh_max_accr_raw.reshape(-1, 1)
                else:
                    maxsnaps = len(bh_max_accr_raw) // ngal
                    bh_max_accr_hist = bh_max_accr_raw.reshape(ngal, maxsnaps)
            else:
                bh_max_accr_hist = bh_max_accr_raw

            if 'BHEddingtonRateLimit' in grp:
                bh_eddington_raw = grp['BHEddingtonRateLimit'][:][mask] * conv
            else:
                bh_eddington_raw = np.zeros_like(bh_max_accr_hist)

            if bh_eddington_raw.ndim == 1:
                ngal = len(m_bh)
                if len(bh_eddington_raw) == ngal:
                    bh_eddington_hist = bh_eddington_raw.reshape(-1, 1)
                else:
                    maxsnaps = len(bh_eddington_raw) // ngal
                    bh_eddington_hist = bh_eddington_raw.reshape(ngal, maxsnaps)
            else:
                bh_eddington_hist = bh_eddington_raw

            if 'BHMassatAccretion' in grp:
                bh_mass_at_accretion_raw = grp['BHMassatAccretion'][:][mask] * conv
            else:
                bh_mass_at_accretion_raw = np.zeros_like(bh_max_accr_hist)
            
            if bh_mass_at_accretion_raw.ndim == 1:
                ngal = len(m_bh)
                if len(bh_mass_at_accretion_raw) == ngal:
                    bh_mass_at_accretion_hist = bh_mass_at_accretion_raw.reshape(-1, 1)
                else:
                    maxsnaps = len(bh_mass_at_accretion_raw) // ngal
                    bh_mass_at_accretion_hist = bh_mass_at_accretion_raw.reshape(ngal, maxsnaps)
            else:
                bh_mass_at_accretion_hist = bh_mass_at_accretion_raw
            
            if 'dt' in grp:
                dt_raw = grp['dt'][:]
                if dt_raw.ndim == 1:
                    ngal = len(m_bh)
                    if len(dt_raw) == ngal:
                        dt_hist = dt_raw[mask].reshape(-1, 1)
                    else:
                        maxsnaps = len(dt_raw) // ngal
                        dt_hist = dt_raw.reshape(ngal, maxsnaps)[mask]
                else:
                    dt_hist = dt_raw[mask]
            else:
                dt_hist = np.zeros_like(bh_max_accr_hist)
            
            # Ensure consistent shapes
            if bh_max_accr_hist.shape != bh_eddington_hist.shape:
                max_shape = (bh_max_accr_hist.shape[0], max(bh_max_accr_hist.shape[1], bh_eddington_hist.shape[1]))
                bh_max_accr_padded = np.zeros(max_shape)
                bh_eddington_padded = np.zeros(max_shape)
                bh_max_accr_padded[:, :bh_max_accr_hist.shape[1]] = bh_max_accr_hist
                bh_eddington_padded[:, :bh_eddington_hist.shape[1]] = bh_eddington_hist
                bh_max_accr_hist = bh_max_accr_padded
                bh_eddington_hist = bh_eddington_padded

            if bh_mass_at_accretion_hist.shape != bh_max_accr_hist.shape:
                max_shape = bh_max_accr_hist.shape
                bh_mass_at_accretion_padded = np.zeros(max_shape)
                bh_mass_at_accretion_padded[:, :bh_mass_at_accretion_hist.shape[1]] = bh_mass_at_accretion_hist
                bh_mass_at_accretion_hist = bh_mass_at_accretion_padded

            if dt_hist.shape != bh_max_accr_hist.shape:
                max_shape = bh_max_accr_hist.shape
                dt_padded = np.zeros(max_shape)
                dt_padded[:, :dt_hist.shape[1]] = dt_hist
                dt_hist = dt_padded

            all_ids.append(gids[mask])
            all_bh_mass.append(m_bh)
            all_stellar_mass.append(m_stellar)
            all_mvir.append(m_mvir)
            all_bh_max_accretion_history.append(bh_max_accr_hist)
            all_bh_eddington_rate_limit.append(bh_eddington_hist)
            all_bh_mass_at_accretion.append(bh_mass_at_accretion_hist)
            all_dt.append(dt_hist)

    return (np.concatenate(all_ids), np.concatenate(all_bh_mass), 
            np.concatenate(all_stellar_mass), np.concatenate(all_mvir), 
            np.concatenate(all_bh_max_accretion_history),
            np.concatenate(all_bh_eddington_rate_limit),
            np.concatenate(all_bh_mass_at_accretion),
            np.concatenate(all_dt))

def compute_deltabh(bh_max_accr_history, bh_eddington_rate_limit, bh_mass_at_accretion):
    """
    Compute deltaBH: the integral of accretion rates over time.
    
    For each galaxy:
    - deltaBH_total = sum over all snapshots of BHMaxaccretionRate * dt
    - deltaBH_se = sum over snapshots where BHMaxaccretionRate > BHEddingtonRateLimit
    - deltaBH_non_se = deltaBH_total - deltaBH_se
    
    Args:
        bh_max_accr_history: (Ngal x MAXSNAPS) array
        bh_eddington_rate_limit: (Ngal x MAXSNAPS) array
        bh_mass_at_accretion: (Ngal x MAXSNAPS) array (time-steps; already includes dt)
    
    Returns:
        deltaBH_se: Super-Eddington growth per galaxy
        deltaBH_non_se: Non-super-Eddington growth per galaxy
        deltaBH_total: Total growth per galaxy
    """
    
    # Ensure 2D
    if bh_max_accr_history.ndim == 1:
        bh_max_accr_history = bh_max_accr_history.reshape(-1, 1)
    if bh_eddington_rate_limit.ndim == 1:
        bh_eddington_rate_limit = bh_eddington_rate_limit.reshape(-1, 1)
    if bh_mass_at_accretion.ndim == 1:
        bh_mass_at_accretion = bh_mass_at_accretion.reshape(-1, 1)
    
    ngal = bh_max_accr_history.shape[0]
    
    deltaBH_se = np.zeros(ngal)
    deltaBH_non_se = np.zeros(ngal)
    deltaBH_total = np.zeros(ngal)
    
    for i in range(ngal):
        max_accr = bh_max_accr_history[i, :]
        eddington = bh_eddington_rate_limit[i, :]
        
        # Total growth (sum of all accretion)
        deltaBH_total[i] = np.sum(max_accr)
        
        # Super-Eddington growth (where max_accr > eddington)
        se_mask = max_accr > eddington
        deltaBH_se[i] = np.sum(max_accr[se_mask])
        
        # Non-super-Eddington growth
        deltaBH_non_se[i] = deltaBH_total[i] - deltaBH_se[i]
    
    return deltaBH_se, deltaBH_non_se, deltaBH_total

def create_deltabh_histogram(deltaBH_se, deltaBH_non_se, deltaBH_total, plot_mask, bh_mass, output_file):
    """Create histogram of deltaBH showing impact of super-Eddington accretion."""
    
    # Filter by resolution threshold
    well_resolved_mask = plot_mask & (bh_mass >= MIN_BH_MASS_FOR_DELTABH)
    
    deltaBH_se_filtered = deltaBH_se[well_resolved_mask]
    deltaBH_non_se_filtered = deltaBH_non_se[well_resolved_mask]
    deltaBH_total_filtered = deltaBH_total[well_resolved_mask]
    
    # Remove NaN/Inf
    valid_mask = (
        np.isfinite(deltaBH_se_filtered) &
        np.isfinite(deltaBH_non_se_filtered) &
        np.isfinite(deltaBH_total_filtered) &
        (deltaBH_total_filtered > 0)
    )
    
    deltaBH_se_valid = deltaBH_se_filtered[valid_mask]
    deltaBH_non_se_valid = deltaBH_non_se_filtered[valid_mask]
    deltaBH_total_valid = deltaBH_total_filtered[valid_mask]
    
    if len(deltaBH_total_valid) < 10:
        print("⚠ Not enough well-resolved BHs for deltaBH histogram")
        return
    
    # Compute fraction from super-Eddington
    frac_se = np.zeros_like(deltaBH_se_valid)
    frac_se[deltaBH_total_valid > 0] = deltaBH_se_valid[deltaBH_total_valid > 0] / deltaBH_total_valid[deltaBH_total_valid > 0]
    frac_se = np.clip(frac_se, 0, 1)  # Ensure 0-1 range
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ====================================================================
    # LEFT: Absolute deltaBH values (stacked bar histogram)
    # ====================================================================
    bins = np.logspace(np.log10(deltaBH_total_valid.min()), 
                       np.log10(deltaBH_total_valid.max()), 40)
    
    # Bin data
    bin_edges = bins
    bin_indices = np.digitize(deltaBH_total_valid, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    
    deltaBH_se_binned = np.zeros(len(bins) - 1)
    deltaBH_non_se_binned = np.zeros(len(bins) - 1)
    
    for i in range(len(bins) - 1):
        mask_in_bin = bin_indices == i
        if np.sum(mask_in_bin) > 0:
            deltaBH_se_binned[i] = np.mean(deltaBH_se_valid[mask_in_bin])
            deltaBH_non_se_binned[i] = np.mean(deltaBH_non_se_valid[mask_in_bin])
    
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    ax1.bar(bin_centers, deltaBH_se_binned, width=0.7*(bins[1]-bins[0]),
            label='Super-Eddington growth', color='#D32F2F', alpha=0.7, edgecolor='black')
    ax1.bar(bin_centers, deltaBH_non_se_binned, width=0.7*(bins[1]-bins[0]),
            bottom=deltaBH_se_binned, label='Non-SE growth', color='#1976D2', alpha=0.7, edgecolor='black')
    
    ax1.set_xscale('log')
    ax1.set_ylabel('Mean ΔM$_{\\mathrm{BH}}$ (M$_\\odot$)', fontsize=12)
    ax1.set_xlabel('Total ΔM$_{\\mathrm{BH}}$ (M$_\\odot$)', fontsize=12)
    ax1.set_title('BH Growth: SE vs Non-SE Accretion', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ====================================================================
    # RIGHT: Fraction from super-Eddington
    # ====================================================================
    ax2.hist(frac_se, bins=50, color='#FF9800', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Fraction from Super-Eddington (ΔM$_{\\mathrm{SE}}$ / ΔM$_{\\mathrm{total}}$)', fontsize=12)
    ax2.set_ylabel('Number of Galaxies', fontsize=12)
    ax2.set_title('SE Contribution to Total Growth', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Statistics box
    stats_text = (
        f'Well-resolved BHs (M > 10$^5$ M$_\\odot$): {len(deltaBH_total_valid):,}\n'
        f'Mean frac(SE): {np.mean(frac_se):.2%}\n'
        f'Median frac(SE): {np.median(frac_se):.2%}\n'
        f'Max frac(SE): {np.max(frac_se):.2%}'
    )
    ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    
    print(f"✓ deltaBH histogram saved to: {output_file}")
    print(f"  Well-resolved BHs analyzed: {len(deltaBH_total_valid):,}")
    print(f"  Mean SE contribution: {np.mean(frac_se):.2%}")
    print(f"  Median SE contribution: {np.median(frac_se):.2%}")
    
    plt.close()

def create_deltabh_mass_bins(deltaBH_se, deltaBH_non_se, deltaBH_total,
                             plot_mask, bh_mass, output_file):
    """
    Create simplified deltaBH comparison plot.

    Single panel:
    - Overplot SE and non-SE growth versus BH mass
    """

    # Filter by resolution threshold
    well_resolved_mask = (
        plot_mask &
        (bh_mass >= MIN_BH_MASS_FOR_DELTABH)
    )

    deltaBH_se_filtered = deltaBH_se[well_resolved_mask]
    deltaBH_non_se_filtered = deltaBH_non_se[well_resolved_mask]
    deltaBH_total_filtered = deltaBH_total[well_resolved_mask]
    bh_mass_filtered = bh_mass[well_resolved_mask]

    # Remove invalid values
    valid_mask = (
        np.isfinite(deltaBH_se_filtered) &
        np.isfinite(deltaBH_non_se_filtered) &
        np.isfinite(deltaBH_total_filtered) &
        np.isfinite(bh_mass_filtered) &
        (deltaBH_total_filtered > 0) &
        (bh_mass_filtered > 0)
    )

    deltaBH_se_valid = deltaBH_se_filtered[valid_mask]
    deltaBH_non_se_valid = deltaBH_non_se_filtered[valid_mask]
    bh_mass_valid = bh_mass_filtered[valid_mask]

    if len(deltaBH_se_valid) < 20:
        print("⚠ Not enough well-resolved BHs for deltaBH analysis")
        return

    # ============================================================
    # Create figure
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 7))

    # ------------------------------------------------------------
    # Super-Eddington growth
    # ------------------------------------------------------------
    ax.scatter(
        bh_mass_valid,
        deltaBH_se_valid,
        alpha=0.5,
        s=18,
        color='#D32F2F',
        label='Super-Eddington'
    )

    # ------------------------------------------------------------
    # Non-SE growth
    # ------------------------------------------------------------
    ax.scatter(
        bh_mass_valid,
        deltaBH_non_se_valid,
        alpha=0.2,
        s=18,
        color='#1976D2',
        label='Non-SE'
    )

    # ============================================================
    # Axes formatting
    # ============================================================
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'BH Mass (M$_\odot$)', fontsize=14)
    ax.set_ylabel(r'$\Delta M_{\rm BH}$ (M$_\odot$)', fontsize=14)

    ax.set_title(
        'BH Growth from SE and Non-SE Accretion',
        fontsize=15,
        fontweight='bold'
    )

    ax.grid(True, alpha=0.1)

    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')

    print(f"✓ deltaBH mass comparison saved to: {output_file}")

    # ============================================================
    # Summary statistics
    # ============================================================
    frac_se = np.zeros_like(deltaBH_se_valid)

    total_growth = deltaBH_se_valid + deltaBH_non_se_valid

    mask = total_growth > 0

    frac_se[mask] = deltaBH_se_valid[mask] / total_growth[mask]

    print("\n" + "="*70)
    print("SIMPLIFIED DELTABH SUMMARY")
    print("="*70)
    print(f"Objects analysed: {len(deltaBH_se_valid):,}")
    print(f"Mean SE fraction: {np.mean(frac_se):.4f}")
    print(f"Median SE fraction: {np.median(frac_se):.4f}")
    print(f"Max SE fraction: {np.max(frac_se):.4f}")
    print("="*70)

    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Enhanced Eddington limit analysis with deltaBH')
    parser.add_argument('-i', '--input-pattern', default='./output/millennium/model_*.hdf5')
    parser.add_argument('-s', '--snapshot', type=int, default=None)
    parser.add_argument('--no-cuts', action='store_true')
    parser.add_argument('--no-deltabh', action='store_true', help='Skip deltaBH analysis')
    args = parser.parse_args()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"No files found for {args.input_pattern}")
        sys.exit(1)

    sim = read_simulation_params(file_list[0])
    h_h = sim['Hubble_h']
    snap_num = args.snapshot if args.snapshot is not None else sim['latest_snapshot']
    id_field = find_id_field(file_list, snap_num)
    
    redshift = get_redshift_from_snapshot(snap_num)

    print(f"Snapshot: {snap_num} | Redshift: {redshift:.3f} | Hubble_h: {h_h}")
    print("Reading data...")
    
    ids, bh_mass, stellar_mass, mvir, bh_max_accr_hist, bh_eddington_hist, bh_mass_at_accretion, dt_data = \
        read_data(file_list, snap_num, id_field, h_h)

    # Filtering
    if args.no_cuts:
        plot_mask = (bh_mass > 0)
    else:
        plot_mask = (bh_mass > MIN_Z0_BH_MASS) & (stellar_mass > 10**MIN_STELLAR_MASS_LOG) & (mvir > 10**MIN_HALO_MASS_LOG)

    print(f"Initial galaxies: {len(ids)}")
    print(f"Final plot count: {np.sum(plot_mask)}")

    output_dir = Path(file_list[0]).parent / 'plots'
    output_dir.mkdir(exist_ok=True)

    # ========================================================================
    # DELTABH ANALYSIS
    # ========================================================================
    if not args.no_deltabh:
        print("\n" + "="*70)
        print("Computing deltaBH (BH mass growth) analysis...")
        print("="*70)
        
        deltaBH_se, deltaBH_non_se, deltaBH_total = compute_deltabh(
            bh_max_accr_hist,
            bh_eddington_hist,
            bh_mass_at_accretion
        )
        
        print("\n[1/2] Creating deltaBH histogram...")
        create_deltabh_histogram(
            deltaBH_se,
            deltaBH_non_se,
            deltaBH_total,
            plot_mask,
            bh_mass,
            output_dir / 'bh_deltabh_histogram.png'
        )
        
        print("\n[2/2] Creating mass-binned deltaBH analysis...")
        create_deltabh_mass_bins(
            deltaBH_se,
            deltaBH_non_se,
            deltaBH_total,
            plot_mask,
            bh_mass,
            output_dir / 'bh_deltabh_mass_bins.png'
        )

    print("\n✓ deltaBH analysis completed successfully!")

if __name__ == "__main__":
    main()