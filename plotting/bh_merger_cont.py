"""
Plot BH merger contribution (%) vs. z=0 BH mass using galaxy ID tracking.

This script:
1. Reads z=0 BH mass for each galaxy
2. Uses the ID system to track those same galaxies backward in time
3. Accumulates BH mass growth from different accretion channels
4. Plots the percentage contribution from BH-BH mergers vs. initial z=0 BH mass
"""

import argparse
import glob
import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ================= CONFIG =================
MIN_STELLAR_MASS_LOG = 8.5
MIN_HALO_MASS_LOG = 10.5
MIN_Z0_BH_MASS = 1e6  # Minimum z=0 BH mass to include

TRACKING_RANGE = "all"  # Use ID tracking across all snapshots
OutputFormat = '.png'

# ================= HELPER FUNCTIONS =================

def read_simulation_params(filepath):
    """Extract simulation parameters from HDF5 file."""
    with h5py.File(filepath, 'r') as hf:
        if 'Header' in hf:
            header = hf['Header'].attrs
            return {
                'Hubble_h': header.get('HubbleParam', 0.681),
                'latest_snapshot': 62,  # z=0 snapshot (0-indexed, so max is 62)
                'available_snapshots': list(range(63)),  # Snapshots 0-62
                'snapshot_redshifts': header.get('Redshifts', np.arange(63)[::-1] * 0.1),
            }
    return {'Hubble_h': 0.681, 'latest_snapshot': 62, 'available_snapshots': list(range(63))}

def read_hdf(file_list, snap_num, field):
    """Read field from HDF5 files for a given snapshot."""
    data = []
    for f in file_list:
        try:
            with h5py.File(f, 'r') as hf:
                snap_key = f"Snap_{snap_num}"
                if snap_key in hf and field in hf[snap_key]:
                    data.append(hf[snap_key][field][:])
        except Exception as e:
            print(f"Warning: Could not read {field} from {f}: {e}")
            continue
    return np.concatenate(data) if data else np.array([])

def find_id_field(file_list, snap_num):
    """Find the ID field name in HDF5 files."""
    id_candidates = ['GalaxyIndex', 'GalaxyID', 'ID', 'ParticleID', 'GalID']
    
    # Try the requested snapshot first, then nearby snapshots
    snaps_to_try = [snap_num, 0, 63, 32, 16, 48]
    
    for sn in snaps_to_try:
        for f in file_list[:1]:  # Check first file only
            try:
                with h5py.File(f, 'r') as hf:
                    snap_key = f"Snap_{sn}"
                    if snap_key in hf:
                        for candidate in id_candidates:
                            if candidate in hf[snap_key]:
                                print(f"Found ID field '{candidate}' in Snap_{sn}")
                                return candidate
            except:
                pass
    return None

def lookup_ids_vectorized(target_ids, gal_id_array):
    """
    Vectorized lookup: for each target ID, find its index in gal_id_array.
    Returns array of indices (or -1 if not found).
    """
    sort_idx = np.argsort(gal_id_array)
    sorted_ids = gal_id_array[sort_idx]
    insert_pos = np.searchsorted(sorted_ids, target_ids)
    insert_pos = np.clip(insert_pos, 0, len(sorted_ids) - 1)
    
    found = sorted_ids[insert_pos] == target_ids
    result = np.full(len(target_ids), -1, dtype=int)
    result[found] = sort_idx[insert_pos[found]]
    return result

# ================= MAIN SCRIPT =================

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-pattern', type=str, 
                    default='./output/millennium_insitu_new/model_*.hdf5')
parser.add_argument('-s', '--snapshot', type=int, default=None)
parser.add_argument('-o', '--output-dir', type=str, default=None)
args = parser.parse_args()

file_list = sorted(glob.glob(args.input_pattern))
if not file_list:
    print(f"Error: No files found matching: {args.input_pattern}")
    sys.exit(1)

sim_params = read_simulation_params(file_list[0])
Hubble_h = sim_params['Hubble_h']
snap_num = args.snapshot if args.snapshot is not None else sim_params['latest_snapshot']

print(f"Using snapshot number: {snap_num}")
print(f"Hubble_h: {Hubble_h}")

OutputDir = args.output_dir if args.output_dir else os.path.join(
    os.path.dirname(os.path.abspath(file_list[0])), 'plots'
)
os.makedirs(OutputDir, exist_ok=True)

id_field = find_id_field(file_list, snap_num)
if id_field is None:
    print("ERROR: No ID field found!")
    sys.exit(1)

print("\nReading z=0 baseline data...")
BlackHoleMass_z0 = read_hdf(file_list, snap_num, 'BlackHoleMass') * 1.0e10 / Hubble_h
StellarMass_z0 = read_hdf(file_list, snap_num, 'StellarMass') * 1.0e10 / Hubble_h
Mvir_z0 = read_hdf(file_list, snap_num, 'Mvir') * 1.0e10 / Hubble_h

# Flatten in case arrays are 2D
BlackHoleMass_z0 = BlackHoleMass_z0.flatten()
StellarMass_z0 = StellarMass_z0.flatten()
Mvir_z0 = Mvir_z0.flatten()

if len(BlackHoleMass_z0) == 0:
    print("ERROR: No data read at z=0!")
    sys.exit(1)

# Apply mass cuts at z=0
bh_mask = (BlackHoleMass_z0 > MIN_Z0_BH_MASS) & \
          (StellarMass_z0 > 10**MIN_STELLAR_MASS_LOG) & \
          (Mvir_z0 > 10**MIN_HALO_MASS_LOG)

print(f"Galaxies passing mass cuts at z=0: {np.sum(bh_mask)}")

# Read z=0 galaxy IDs
gal_id_z0 = []
for f in file_list:
    with h5py.File(f, 'r') as hf:
        if f"Snap_{snap_num}" in hf and id_field in hf[f"Snap_{snap_num}"]:
            gal_id_z0.append(hf[f"Snap_{snap_num}"][id_field][:])
gal_id_z0 = np.concatenate(gal_id_z0) if gal_id_z0 else None

# Flatten in case it's 2D
if gal_id_z0 is not None:
    gal_id_z0 = gal_id_z0.flatten()

# Extract selected galaxy data
selected_z0_bh_mass = BlackHoleMass_z0[bh_mask]
selected_z0_ids = gal_id_z0[bh_mask] if gal_id_z0 is not None else None

print(f"Selected {len(selected_z0_bh_mass)} galaxies for tracking")

if selected_z0_ids is None or len(selected_z0_ids) == 0:
    print("ERROR: Could not extract galaxy IDs!")
    sys.exit(1)

# Get all snapshots and redshifts
all_snaps = np.array(sim_params['available_snapshots'])
all_redshifts = sim_params['snapshot_redshifts']

print("\nExtracting BH growth across snapshots...")
print(f"Tracking {len(all_snaps)} snapshots from z=0 to high redshift")

# Storage: for each galaxy, accumulate BH mass from each channel
bh_from_mergers = np.zeros(len(selected_z0_ids))
bh_from_merger_driven = np.zeros(len(selected_z0_ids))
bh_from_instability = np.zeros(len(selected_z0_ids))
bh_from_radio = np.zeros(len(selected_z0_ids))

for sn in all_snaps:
    # Read accretion channel masses
    md = read_hdf(file_list, sn, 'MergerDrivenBHaccretionMass') * 1.0e10 / Hubble_h
    id_ = read_hdf(file_list, sn, 'InstabilityDrivenBHaccretionMass') * 1.0e10 / Hubble_h
    rm = read_hdf(file_list, sn, 'RadioModeBHaccretionMass') * 1.0e10 / Hubble_h
    bm = read_hdf(file_list, sn, 'BHMergerMass') * 1.0e10 / Hubble_h
    
    # Flatten in case arrays are 2D
    md = md.flatten()
    id_ = id_.flatten()
    rm = rm.flatten()
    bm = bm.flatten()
    
    if len(md) == 0:
        continue
    
    z = all_redshifts[sn] if sn < len(all_redshifts) else None
    if z is None:
        continue
    
    # Read galaxy IDs at this snapshot
    gal_id_sn = []
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            if f"Snap_{sn}" in hf and id_field in hf[f"Snap_{sn}"]:
                gal_id_sn.append(hf[f"Snap_{sn}"][id_field][:])
    gal_id_sn = np.concatenate(gal_id_sn) if gal_id_sn else None
    
    # Flatten in case it's 2D
    if gal_id_sn is not None:
        gal_id_sn = gal_id_sn.flatten()
    
    if gal_id_sn is None:
        continue
    
    # Find indices of selected galaxies at this snapshot
    valid_idx = lookup_ids_vectorized(selected_z0_ids, gal_id_sn)
    valid_mask = valid_idx >= 0
    
    if np.sum(valid_mask) == 0:
        continue
    
    # Accumulate BH mass from each channel
    # Only update the galaxies that were found at this snapshot
    bh_from_mergers[valid_mask] += bm[valid_idx[valid_mask]]
    bh_from_merger_driven[valid_mask] += md[valid_idx[valid_mask]]
    bh_from_instability[valid_mask] += id_[valid_idx[valid_mask]]
    bh_from_radio[valid_mask] += rm[valid_idx[valid_mask]]
    
    if sn % 10 == 0:
        print(f"  Snap {sn}: z={z:.2f}, matched {np.sum(valid_mask)} galaxies")

# Calculate percentage contributions
total_accreted = bh_from_mergers + bh_from_merger_driven + bh_from_instability + bh_from_radio
valid_total = total_accreted > 0

pct_from_mergers = np.zeros_like(total_accreted)
pct_from_mergers[valid_total] = 100.0 * bh_from_mergers[valid_total] / total_accreted[valid_total]

print(f"\nProcessed {len(selected_z0_ids)} galaxies")
print(f"  Median % from BH mergers: {np.median(pct_from_mergers[valid_total]):.1f}%")
print(f"  Mean % from BH mergers: {np.mean(pct_from_mergers[valid_total]):.1f}%")

# ================= PLOTTING =================

fig, ax = plt.subplots(figsize=(10, 7))

# Only plot galaxies with valid data
plot_mask = (selected_z0_bh_mass > 0) & valid_total
x_data = selected_z0_bh_mass[plot_mask]
y_data = pct_from_mergers[plot_mask]

# Scatter plot
scatter = ax.scatter(x_data, y_data, alpha=0.5, s=20, edgecolors='none')

ax.set_xlabel(r'$M_{\rm BH}\,[M_\odot]$', fontsize=12)
ax.set_ylabel(r'BH-BH Merger Contribution (%)', fontsize=12)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
out_filename = f'bh_merger_contribution_vs_z0mass{OutputFormat}'
plt.savefig(os.path.join(OutputDir, out_filename), bbox_inches='tight', dpi=150)
print(f"\nPlot saved to {out_filename}")

# ================= OPTIONAL: HISTOGRAM =================

fig, ax = plt.subplots(figsize=(8, 5))

# Histogram of merger contribution percentages
ax.hist(y_data, bins=30, alpha=0.7, color='orange', edgecolor='black')
ax.set_xlabel(r'BH-BH Merger Contribution (%)', fontsize=12)
ax.set_ylabel('Number of galaxies', fontsize=12)
ax.set_ylim(0, 20)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_filename_hist = f'bh_merger_contribution_histogram{OutputFormat}'
plt.savefig(os.path.join(OutputDir, out_filename_hist), bbox_inches='tight', dpi=150)
print(f"Histogram saved to {out_filename_hist}")

print("\nDone!")