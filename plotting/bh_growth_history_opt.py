#!/usr/bin/env python3
"""
Black hole growth tracking: validates that the three accretion channels
(quasar mode, radio mode, BH-BH mergers) sum to the total BlackHoleMass,
and plots the MEDIAN relative contributions across galaxies.

MODIFIED VERSION:
1. Tracks the growth trajectory of EACH INDIVIDUAL GALAXY across all snapshots.
2. Plots all galaxy trajectories in light, semi-transparent colors.
3. Generates TWO trajectory plots:
   a) Independent tracking: Selects the specific range (e.g., middle 20%) for EACH channel at z=0.
   b) Common Mass tracking: Selects the specific range based on TOTAL BH MASS at z=0, 
      and tracks that exact same subset of galaxies across all channels.

**MEMORY OPTIMIZED FOR SUPERCOMPUTERS**: Bypasses heavy Python dictionaries 
and utilizes pre-allocated NumPy matrices to prevent Out-of-Memory (OOM) crashes.
"""
import numpy as np
import h5py
import glob
import os
import sys
import argparse
import matplotlib.pyplot as plt
import warnings

plt.rcParams["figure.figsize"] = (8.34, 6.25)
plt.rcParams["figure.dpi"] = 96
plt.rcParams["font.size"] = 14
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'

OutputFormat = '.pdf'

# ============================================================
# USER OPTIONS
# ============================================================
# Choose which statistical range of galaxies to track back in time from z=0.
# Valid options:
#   "1-sigma"   -> Middle 68% (16th to 84th percentile)
#   "iqr"       -> Middle 50% (25th to 75th percentile - Interquartile Range)
#   "middle_20" -> Middle 20% (40th to 60th percentile)
#   "middle_10" -> Middle 10% (45th to 55th percentile - Very tight to the median)

TRACKING_RANGE = "middle_10"  # <--- SET YOUR CHOICE HERE

# ============================================================

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


def main():
    parser = argparse.ArgumentParser(description='Black hole growth tracking validation and plots')
    parser.add_argument('-i', '--input-pattern', type=str,
                        default='./output/millennium_insitu/model_*.hdf5',
                        help='Path pattern to model HDF5 files')
    parser.add_argument('-s', '--snapshot', type=int, default=None,
                        help='Snapshot number (default: latest)')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory for plots')
    parser.add_argument('--validation-snap', type=int, default=None,
                        help='Snapshot to create validation histograms for (default: latest)')
    args = parser.parse_args()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"Error: No files found matching: {args.input_pattern}")
        sys.exit(1)

    print(f"Found {len(file_list)} model files.")

    sim_params = read_simulation_params(file_list[0])
    Hubble_h = sim_params['Hubble_h']

    if args.snapshot is not None:
        snap_num = args.snapshot
    else:
        snap_num = sim_params['latest_snapshot']
    print(f"Using snapshot: {snap_num}")

    if args.validation_snap is not None:
        validation_snap = args.validation_snap
    else:
        validation_snap = snap_num

    if 'snapshot_redshifts' in sim_params and snap_num < len(sim_params['snapshot_redshifts']):
        print(f"Redshift: {sim_params['snapshot_redshifts'][snap_num]:.4f}")

    if args.output_dir:
        OutputDir = args.output_dir
    else:
        input_dir = os.path.dirname(os.path.abspath(file_list[0]))
        OutputDir = os.path.join(input_dir, 'plots')
    os.makedirs(OutputDir, exist_ok=True)

    # Read data
    print("Reading black hole data...")
    BlackHoleMass = read_hdf(file_list, snap_num, 'BlackHoleMass') * 1.0e10 / Hubble_h
    QuasarMode = read_hdf(file_list, snap_num, 'QuasarModeBHaccretionMass') * 1.0e10 / Hubble_h
    MergerDriven = read_hdf(file_list, snap_num, 'MergerDrivenBHaccretionMass') * 1.0e10 / Hubble_h
    InstabilityDriven = read_hdf(file_list, snap_num, 'InstabilityDrivenBHaccretionMass') * 1.0e10 / Hubble_h
    TorqueDriven = read_hdf(file_list, snap_num, 'TorqueDrivenBHaccretionMass') * 1.0e10 / Hubble_h
    SeedModeAccretion = read_hdf(file_list, snap_num, 'SeedModeBHaccretionMass') * 1.0e10 / Hubble_h
    RadioMode = read_hdf(file_list, snap_num, 'RadioModeBHaccretionMass') * 1.0e10 / Hubble_h
    BHMerger = read_hdf(file_list, snap_num, 'BHMergerMass') * 1.0e10 / Hubble_h
    BHSeedMass = read_hdf(file_list, snap_num, 'BHSeedMass') * 1.0e10 / Hubble_h

    if len(BlackHoleMass) == 0:
        print("No galaxies found!")
        sys.exit(1)

    bh_mask = BlackHoleMass > 0
    n_bh = np.sum(bh_mask)

    # ===================== PLOTS =====================
    print(f"\nGenerating plots in {OutputDir}...")

    # -- Plot individual galaxy trajectories --
    if 'snapshot_redshifts' in sim_params:
        print("  Computing statistical group growth trajectories across all snapshots...")

        all_snaps = np.array(sim_params['available_snapshots'])
        all_redshifts = sim_params['snapshot_redshifts']

        # Setup the percentile ranges based on user selection
        range_options = {
            "1-sigma": ([16, 50, 84], "1-sigma (16%-84%)"),
            "iqr": ([25, 50, 75], "IQR (25%-75%)"),
            "middle_20": ([40, 50, 60], "Middle 20% (40%-60%)"),
            "middle_10": ([45, 50, 55], "Middle 10% (45%-55%)")
        }

        if TRACKING_RANGE not in range_options:
            chosen_range = "middle_20"
        else:
            chosen_range = TRACKING_RANGE

        pct_vals, range_name = range_options[chosen_range]
        print(f"  Using Tracking Range: {range_name}")

        # Step 1: Identify the z=0 snapshot (lowest redshift)
        z0_snap = all_snaps[np.argmin(all_redshifts[all_snaps])]
        print(f"  z=0 reference snapshot: {z0_snap} (z = {all_redshifts[z0_snap]:.6f})")

        # Step 2: Read z=0 data
        bh_z0 = read_hdf(file_list, z0_snap, 'BlackHoleMass') * 1.0e10 / Hubble_h

        def safe_read_z0(field):
            arr = read_hdf(file_list, z0_snap, field) * 1.0e10 / Hubble_h
            return arr if len(arr) > 0 else np.zeros(len(bh_z0))

        md_z0 = safe_read_z0('MergerDrivenBHaccretionMass')
        id_z0 = safe_read_z0('InstabilityDrivenBHaccretionMass')
        td_z0 = safe_read_z0('TorqueDrivenBHaccretionMass')
        sm_z0 = safe_read_z0('SeedModeBHaccretionMass')
        rm_z0 = safe_read_z0('RadioModeBHaccretionMass')
        bm_z0 = safe_read_z0('BHMergerMass')

        channels_info = [
            ('Merger-driven', 'md', '#2196F3', 'o', md_z0),
            ('Instability-driven', 'id', '#FF9800', 'D', id_z0),
            ('Radio mode', 'rm', '#9C27B0', 's', rm_z0),
            ('BH-BH mergers', 'bm', '#4CAF50', '^', bm_z0),
        ]

        # ---------------------------------------------------------
        # FIND INDEPENDENT GROUPS (per channel)
        # ---------------------------------------------------------
        print(f"\n  Finding INDEPENDENT samples for each channel based on their specific growth at z=0...")
        independent_groups = {}
        for label, key, color, marker, full_z0_data in channels_info:
            valid_mask = (bh_z0 > 0) & (full_z0_data > 0)
            if np.any(valid_mask):
                valid_data = full_z0_data[valid_mask]
                p_lower, p50, p_upper = np.percentile(valid_data, pct_vals)
                in_band = (full_z0_data >= p_lower) & (full_z0_data <= p_upper) & valid_mask
                sample_indices = np.where(in_band)[0]
                independent_groups[key] = sample_indices
                print(f"    {label}: [{p_lower:.2e} - {p_upper:.2e}], tracking {len(sample_indices)} galaxies.")
            else:
                independent_groups[key] = np.array([], dtype=int)

        # ---------------------------------------------------------
        # FIND COMMON GROUP (based entirely on Total BH Mass)
        # ---------------------------------------------------------
        print(f"\n  Finding COMMON sample based on TOTAL Black Hole Mass at z=0...")
        valid_bh_mask = bh_z0 > 0
        if np.any(valid_bh_mask):
            valid_bh_data = bh_z0[valid_bh_mask]
            p_lower_bh, p50_bh, p_upper_bh = np.percentile(valid_bh_data, pct_vals)
            in_band_bh = (bh_z0 >= p_lower_bh) & (bh_z0 <= p_upper_bh) & valid_bh_mask
            common_sample_indices = np.where(in_band_bh)[0]
            print(f"    Total BH Mass: [{p_lower_bh:.2e} - {p_upper_bh:.2e}], tracking {len(common_sample_indices)} galaxies across all channels.")
        else:
            common_sample_indices = np.array([], dtype=int)

        # ---------------------------------------------------------
        # MEMORY OPTIMIZATION: Track ONLY the needed galaxies
        # ---------------------------------------------------------
        # To prevent OOM errors, we randomly select a fixed background sample (max 2000 lines) 
        # instead of rendering millions of background lines.
        num_bg_lines = min(2000, len(bh_z0))
        bg_indices = np.random.choice(len(bh_z0), num_bg_lines, replace=False)

        # Create mapping of what indices we need to extract for each channel
        tracked_indices = {}
        idx_to_col = {}     # Maps absolute galaxy row index to Matrix column index
        history_data = {}   # history_data[key] = 2D Numpy Array (Snapshots x Galaxies)
        
        for _, key, _, _, _ in channels_info:
            # Union of all required indices for this specific channel
            req_indices = np.unique(np.concatenate([
                bg_indices,
                independent_groups.get(key, []),
                common_sample_indices
            ])).astype(int)
            
            tracked_indices[key] = req_indices
            idx_to_col[key] = {gal_idx: col for col, gal_idx in enumerate(req_indices)}
            history_data[key] = np.zeros((len(all_snaps), len(req_indices)))

        history_z = np.zeros(len(all_snaps))

        print("\n  Reading data for all snapshots to build trajectories (Memory Optimized)...")
        for s_idx, sn in enumerate(all_snaps):
            bh = read_hdf(file_list, sn, 'BlackHoleMass') * 1.0e10 / Hubble_h
            if len(bh) == 0: continue

            z = all_redshifts[sn] if sn < len(all_redshifts) else None
            if z is None: continue
            
            history_z[s_idx] = z

            def safe_read(field):
                arr = read_hdf(file_list, sn, field) * 1.0e10 / Hubble_h
                return arr if len(arr) > 0 else np.zeros(len(bh))

            channel_data = {
                'md': safe_read('MergerDrivenBHaccretionMass'),
                'id': safe_read('InstabilityDrivenBHaccretionMass'),
                'td': safe_read('TorqueDrivenBHaccretionMass'),
                'sm': safe_read('SeedModeBHaccretionMass'),
                'rm': safe_read('RadioModeBHaccretionMass'),
                'bm': safe_read('BHMergerMass'),
            }
            # Fast vectorised assignment for tracked galaxies
            for _, key, _, _, _ in channels_info:
                req_idx = tracked_indices[key]
                # Protect against changing array lengths in early snapshots
                valid_mask = req_idx < len(bh)
                valid_idx = req_idx[valid_mask]
                
                history_data[key][s_idx, valid_mask] = channel_data[key][valid_idx]


        # =========================================================
        # PLOT 1: INDEPENDENT TRACKING
        # =========================================================
        print("  Creating INDEPENDENT trajectories plot...")
        fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
        axes1 = axes1.flatten()

        # Suppress log10(0) warnings for clean console output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for ax_idx, (label, key, color, marker, _) in enumerate(channels_info):
                ax = axes1[ax_idx]

                # 1. Background fuzz (Matrix plotting is instantly fast)
                bg_cols = [idx_to_col[key][i] for i in bg_indices if i in idx_to_col[key]]
                if len(bg_cols) > 0:
                    bg_matrix = history_data[key][:, bg_cols]
                    bg_matrix[bg_matrix <= 0] = np.nan
                    
                    # Plot 2D array directly (Matplotlib plots columns as lines)
                    ax.plot(history_z, np.log10(bg_matrix), color=color, alpha=0.05, linewidth=0.8)

                # 2. Bold line (Independent Group Median)
                ind_indices = independent_groups.get(key, [])
                if len(ind_indices) > 0:
                    ind_cols = [idx_to_col[key][i] for i in ind_indices if i in idx_to_col[key]]
                    ind_matrix = history_data[key][:, ind_cols]
                    ind_matrix[ind_matrix <= 0] = np.nan
                    
                    med_growth = np.nanmedian(ind_matrix, axis=1)
                    valid = ~np.isnan(med_growth)
                    
                    if np.any(valid):
                        ax.plot(history_z[valid], np.log10(med_growth[valid]), color=color, linewidth=2.5, marker=marker,
                                markersize=6, label=f'{range_name} Median (N={len(ind_indices)})', zorder=100)

                ax.set_xlabel('Redshift', fontsize=11)
                ax.set_ylabel(r'$\log_{10}(\mathrm{Growth\ Mass}\ [M_\odot])$', fontsize=11)
                ax.set_xlim(-0.2, 7.0)
                ax.set_title(label, fontsize=12)
                ax.legend(fontsize=10, loc='best')
                ax.grid(True, alpha=0.3)

        plt.suptitle(f'BH Growth Trajectories (INDEPENDENT TRACKING)\n(Light: Sample Galaxies | Bold: Median of specific {range_name} for EACH channel at z=0)', fontsize=14, y=1.00)
        plt.tight_layout()
        plt.savefig(os.path.join(OutputDir, f'bh_growth_individual_trajectories_independent{OutputFormat}'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: bh_growth_individual_trajectories_independent{OutputFormat}")

        # =========================================================
        # PLOT 2: COMMON MASS TRACKING
        # =========================================================
        print("  Creating COMMON MASS trajectories plot...")
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
        axes2 = axes2.flatten()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for ax_idx, (label, key, color, marker, _) in enumerate(channels_info):
                ax = axes2[ax_idx]

                # 1. Background fuzz
                bg_cols = [idx_to_col[key][i] for i in bg_indices if i in idx_to_col[key]]
                if len(bg_cols) > 0:
                    bg_matrix = history_data[key][:, bg_cols]
                    bg_matrix[bg_matrix <= 0] = np.nan
                    ax.plot(history_z, np.log10(bg_matrix), color=color, alpha=0.05, linewidth=0.8)

                # 2. Bold line (Common Sample Median)
                if len(common_sample_indices) > 0:
                    com_cols = [idx_to_col[key][i] for i in common_sample_indices if i in idx_to_col[key]]
                    com_matrix = history_data[key][:, com_cols]
                    com_matrix[com_matrix <= 0] = np.nan
                    
                    med_growth = np.nanmedian(com_matrix, axis=1)
                    valid = ~np.isnan(med_growth)
                    
                    if np.any(valid):
                        ax.plot(history_z[valid], np.log10(med_growth[valid]), color=color, linewidth=2.5, marker=marker,
                                markersize=6, label=f'Common {range_name} Median (Total BH Mass)', zorder=100)

                ax.set_xlabel('Redshift', fontsize=11)
                ax.set_ylabel(r'$\log_{10}(\mathrm{Growth\ Mass}\ [M_\odot])$', fontsize=11)
                ax.set_xlim(-0.2, 7.0)
                ax.set_title(label, fontsize=12)
                ax.legend(fontsize=10, loc='best')
                ax.grid(True, alpha=0.3)

        plt.suptitle(f'BH Growth Trajectories (COMMON MASS TRACKING)\n(Light: Sample Galaxies | Bold: Median of galaxies with the {range_name} Total BH Mass at z=0)', fontsize=14, y=1.00)
        plt.tight_layout()
        plt.savefig(os.path.join(OutputDir, f'bh_growth_individual_trajectories_common_mass{OutputFormat}'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: bh_growth_individual_trajectories_common_mass{OutputFormat}")

        # ================= HISTOGRAMS =================
        # print(f"\n  Creating validation histograms for snapshot {validation_snap}...")
        # bh_val = read_hdf(file_list, validation_snap, 'BlackHoleMass') * 1.0e10 / Hubble_h
        # mask_val = bh_val > 0
        #
        # def safe_val(field):
        #     arr = read_hdf(file_list, validation_snap, field) * 1.0e10 / Hubble_h
        #     return arr if len(arr) > 0 else np.zeros(len(bh_val))
        #
        # md_val = safe_val('MergerDrivenBHaccretionMass')
        # id_val = safe_val('InstabilityDrivenBHaccretionMass')
        # td_val = safe_val('TorqueDrivenBHaccretionMass')
        # sm_val = safe_val('SeedModeBHaccretionMass')
        # rm_val = safe_val('RadioModeBHaccretionMass')
        # bm_val = safe_val('BHMergerMass')
        #
        # fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        # axes = axes.flatten()
        # val_z = all_redshifts[validation_snap] if validation_snap < len(all_redshifts) else None
        #
        # hist_data = [
        #     ('Total BH Mass', bh_val[mask_val], 'black'),
        #     ('Merger-driven', md_val[mask_val], '#2196F3'),
        #     ('Instability-driven', id_val[mask_val], '#FF9800'),
        #     ('Torque-driven', td_val[mask_val], '#795548'),
        #     ('Seed-mode', sm_val[mask_val], '#FF5722'),
        #     ('Radio mode', rm_val[mask_val], '#9C27B0'),
        #     ('BH-BH mergers', bm_val[mask_val], '#4CAF50'),
        # ]
        #
        # for i, (label, data, color) in enumerate(hist_data):
        #     ax = axes[i]
        #     d = data[data > 0]
        #     if len(d):
        #         ax.hist(np.log10(d), bins=30, color=color, alpha=0.7, edgecolor='black')
        #         ax.set_title(label, fontsize=11)
        #         ax.set_xlabel(r'$\log_{10}(M_\odot)$', fontsize=10)
        #         ax.set_ylabel('Count', fontsize=10)
        #         ax.grid(True, alpha=0.3, axis='y')
        #     else:
        #         ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
        #
        # fig.delaxes(axes[7])
        # if val_z is not None:
        #     fig.suptitle(f'Snapshot {validation_snap} (z={val_z:.3f}) - Growth Channel Distributions', fontsize=13)
        # else:
        #     fig.suptitle(f'Snapshot {validation_snap} - Growth Channel Distributions', fontsize=13)
        #
        # plt.tight_layout()
        # plt.savefig(os.path.join(OutputDir, f'validation_histograms_snap{validation_snap}{OutputFormat}'), dpi=150, bbox_inches='tight')
        # plt.close()
        # print(f"  Saved: validation_histograms_snap{validation_snap}{OutputFormat}")

    print("\nDone.")

if __name__ == '__main__':
    main()