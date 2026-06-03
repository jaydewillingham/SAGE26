#!/usr/bin/env python3
"""
Accretion Rate Function across Redshift Panels
================================================
Plots log10(dN / d log10 lambda) vs log10(lambda) for multiple snapshots
in a configurable grid layout. Each panel shows Total, Quasar Mode, and
Radio Mode accretion rate functions at a different redshift.

Usage examples
--------------
# Default: 9-panel 3x3 grid at snapshots ~z=0,0.5,1,2,3,4,5,6,7
python bh_accretion_rate_function_redshift_panels.py

# Custom snapshots
python bh_accretion_rate_function_redshift_panels.py -s 62 50 40 30 20 10 5

# Skip volume normalisation (raw counts)
python bh_accretion_rate_function_redshift_panels.py --no-volume

# 4x2 layout
python bh_accretion_rate_function_redshift_panels.py -s 62 50 40 30 20 10 5 3 --ncols 4
"""

import argparse
import glob
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path

# ============================================================================
# MATPLOTLIB STYLE
# ============================================================================
plt.rcParams['figure.dpi'] = 140
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 20.0
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.size'] = 7.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['xtick.minor.size'] = 5.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['xtick.major.pad'] = 6
plt.rcParams['ytick.major.size'] = 7.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['ytick.minor.size'] = 5.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.fontsize'] = 10

# ============================================================================
# CONFIGURATION
# ============================================================================
MIN_STELLAR_MASS_LOG = 8.5
MIN_HALO_MASS_LOG = 11.0
MIN_Z0_BH_MASS = 1e4

# Millennium box side length in Mpc/h  (change for a subbox run)
MILLENNIUM_BOX_MPC_H = 62.5

# Colours for the three lines in every panel
COLOR_TOTAL  = '#1976D2'   # blue
COLOR_QUASAR = '#D32F2F'   # red
COLOR_RADIO  = '#388E3C'   # green

# ============================================================================
# REDSHIFT MAP  (Millennium snapshot → z)
# ============================================================================
MILLENNIUM_SNAP_TO_Z = {
    0: 127.0, 1: 65.74, 2: 40.0,  3: 26.66, 4: 19.36, 5: 14.78, 6: 11.66,
    7: 9.44,  8: 7.64,  9: 6.44, 10: 5.48, 11: 4.73, 12: 4.19, 13: 3.72,
    14: 3.33, 15: 3.0,  16: 2.73, 17: 2.48, 18: 2.27, 19: 2.07, 20: 1.90,
    21: 1.75, 22: 1.61, 23: 1.48, 24: 1.37, 25: 1.27, 26: 1.18, 27: 1.10,
    28: 1.02, 29: 0.96, 30: 0.90, 31: 0.85, 32: 0.81, 33: 0.77, 34: 0.73,
    35: 0.70, 36: 0.67, 37: 0.63, 38: 0.60, 39: 0.57, 40: 0.54, 41: 0.51,
    42: 0.49, 43: 0.46, 44: 0.43, 45: 0.41, 46: 0.39, 47: 0.37, 48: 0.36,
    49: 0.34, 50: 0.32, 51: 0.31, 52: 0.29, 53: 0.28, 54: 0.27, 55: 0.26,
    56: 0.25, 57: 0.24, 58: 0.23, 59: 0.21, 60: 0.20, 61: 0.18, 62: 0.0,
}

# Default snapshots chosen to span z ≈ 0 → 7 in 9 steps
DEFAULT_SNAPSHOTS = [62, 55, 50, 40, 30, 20, 12, 8, 6]

# ============================================================================
# HELPERS
# ============================================================================

def snap_to_z(snap, d=MILLENNIUM_SNAP_TO_Z):
    if snap in d:
        return d[snap]
    snaps = sorted(d)
    if snap < snaps[0]:  return d[snaps[0]]
    if snap > snaps[-1]: return d[snaps[-1]]
    for i in range(len(snaps) - 1):
        if snaps[i] <= snap <= snaps[i+1]:
            z1, z2 = d[snaps[i]], d[snaps[i+1]]
            return z1 + (snap - snaps[i]) * (z2 - z1) / (snaps[i+1] - snaps[i])
    return 0.0


def read_sim_params(filepath):
    try:
        with h5py.File(filepath, 'r') as hf:
            hdr = hf['Header/Simulation'].attrs if 'Header/Simulation' in hf else hf['Header'].attrs
            return float(hdr.get('hubble_h', hdr.get('HubbleParam', 0.681)))
    except Exception:
        return 0.681


def find_id_field(file_list, snap_num):
    candidates = ['GalaxyIndex', 'GalaxyID', 'ID', 'galaxy_id', 'id']
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            grp_key = f'Snap_{snap_num}'
            if grp_key in hf:
                for c in candidates:
                    if c in hf[grp_key]:
                        return c
    return None


def read_snapshot(file_list, snap_num, id_field, h_h):
    """
    Returns (bh_mass, stellar_mass, mvir,
             bh_max_accr_hist,  bh_eddington_hist,
             bh_acc_type_hist)
    for the requested snapshot.  All accretion history arrays have
    shape (Ngal, MAXSNAPS).
    """
    conv = 1e10 / h_h

    out = dict(ids=[], bh=[], sm=[], mv=[],
               accr=[], edd=[], acc_type=[])
    seen = set()

    for f in file_list:
        with h5py.File(f, 'r') as hf:
            key = f'Snap_{snap_num}'
            if key not in hf:
                continue
            grp = hf[key]

            gids = grp[id_field][:]
            mask = np.array([g not in seen for g in gids], dtype=bool)
            for g in gids[mask]:
                seen.add(g)
            if not mask.any():
                continue

            def _hist(field, fallback_shape):
                """Load a (Ngal, MAXSNAPS) history array."""
                if field not in grp:
                    return np.zeros(fallback_shape)
                raw = grp[field][:][mask]
                if field in ('BHMaxaccretionRate', 'BHEddingtonRateLimit',
                             'BHMassatAccretion'):
                    raw = raw * conv
                if raw.ndim == 1:
                    ng = mask.sum()
                    if len(raw) == ng:
                        return raw.reshape(ng, 1)
                    ms = len(raw) // ng
                    return raw.reshape(ng, ms)
                return raw

            ng = int(mask.sum())
            # We only need this to establish fallback shape once
            fallback = (ng, 63)

            accr_h     = _hist('BHMaxaccretionRate',   fallback)
            edd_h      = _hist('BHEddingtonRateLimit',  fallback)

            # BHAccretionType has no unit conversion
            if 'BHAccretionType' in grp:
                raw_t = grp['BHAccretionType'][:][mask]
                if raw_t.ndim == 1:
                    if len(raw_t) == ng:
                        acc_type_h = raw_t.reshape(ng, 1)
                    else:
                        ms = len(raw_t) // ng
                        acc_type_h = raw_t.reshape(ng, ms)
                else:
                    acc_type_h = raw_t
            else:
                acc_type_h = np.full(accr_h.shape, -1, dtype=float)

            # Align shapes
            def _align(a, b):
                if a.shape == b.shape:
                    return a, b
                nc = max(a.shape[1], b.shape[1])
                pa = np.zeros((a.shape[0], nc)); pa[:, :a.shape[1]] = a
                pb = np.zeros((b.shape[0], nc)); pb[:, :b.shape[1]] = b
                return pa, pb

            accr_h, edd_h     = _align(accr_h, edd_h)
            accr_h, acc_type_h = _align(accr_h, acc_type_h)

            out['ids'].append(gids[mask])
            out['bh'].append(grp['BlackHoleMass'][:][mask] * conv)
            out['sm'].append(grp['StellarMass'][:][mask]   * conv)
            out['mv'].append(grp['Mvir'][:][mask]           * conv)
            out['accr'].append(accr_h)
            out['edd'].append(edd_h)
            out['acc_type'].append(acc_type_h)

    if not out['bh']:
        return tuple(np.array([]) for _ in range(6))

    return (
        np.concatenate(out['bh']),
        np.concatenate(out['sm']),
        np.concatenate(out['mv']),
        np.concatenate(out['accr']),
        np.concatenate(out['edd']),
        np.concatenate(out['acc_type']),
    )


# ============================================================================
# PER-PANEL ACCRETION RATE FUNCTION  (returns data dict, does NOT plot)
# ============================================================================

def compute_arf(bh_mass, stellar_mass, mvir,
                accr_hist, edd_hist, acc_type_hist,
                no_cuts=False, sim_volume=None, n_bins=35):
    """
    Compute dN/d log10(lambda) [or number density] vs log10(lambda)
    split into Total / Quasar Mode (type==1) / Radio Mode (type==0).

    Returns a dict with keys:
        bin_centers, bins,
        total, quasar, radio           <- each a dict with keys:
            log_y, err_up, err_down, positive, n
        y_label, y_floor
    or None if no valid data.
    """
    # ---- galaxy selection ----
    if no_cuts:
        plot_mask = bh_mass > 0
    else:
        plot_mask = (
            (bh_mass       > MIN_Z0_BH_MASS)
            & (stellar_mass > 10**MIN_STELLAR_MASS_LOG)
            & (mvir         > 10**MIN_HALO_MASS_LOG)
        )

    accr     = accr_hist[plot_mask].flatten()
    edd      = edd_hist[plot_mask].flatten()
    acc_type = acc_type_hist[plot_mask].flatten()

    valid = (accr > 0) & (edd > 0) & np.isfinite(accr) & np.isfinite(edd)
    if not valid.any():
        return None

    lam      = accr[valid] / edd[valid]
    log_lam  = np.log10(lam)
    typ      = acc_type[valid]

    # ---- bins ----
    lo   = np.floor(log_lam.min() * 2) / 2
    hi   = np.ceil( log_lam.max() * 2) / 2
    bins = np.linspace(lo, hi, n_bins + 1)
    bw   = bins[1] - bins[0]
    bc   = 0.5 * (bins[:-1] + bins[1:])

    norm = sim_volume if sim_volume is not None else 1.0

    def _one(data):
        counts, _ = np.histogram(data, bins=bins)
        y = counts / (bw * norm)
        pos = y > 0
        log_y = np.full(len(y), np.nan)
        log_y[pos] = np.log10(y[pos])

        sig_c  = np.sqrt(counts.astype(float))
        sig_y  = sig_c / (bw * norm)
        eu     = np.full(len(y), np.nan)
        ed     = np.full(len(y), np.nan)
        eu[pos] = np.log10(y[pos] + sig_y[pos]) - log_y[pos]
        for i in np.where(pos)[0]:
            lo_val = y[i] - sig_y[i]
            ed[i]  = (log_y[i] - np.log10(lo_val)
                      if lo_val > 0
                      else log_y[i] - np.log10(0.5 * y[i]))
        return dict(log_y=log_y, err_up=eu, err_down=ed,
                    positive=pos, n=len(data))

    result = dict(
        bin_centers = bc,
        bins        = bins,
        total       = _one(log_lam),
        quasar      = _one(log_lam[typ == 1]),
        radio       = _one(log_lam[typ == 0]),
    )

    if sim_volume is not None:
        result['y_label'] = (r'$\log_{10}(\mathrm{d}N\,/\,'
                             r'\mathrm{d}\log_{10}\lambda'
                             r'\ [\mathrm{Mpc}^{-3}\,h^{3}])$')
    else:
        result['y_label'] = r'$\log_{10}(\mathrm{d}N\,/\,\mathrm{d}\log_{10}\lambda)$'

    # global y floor for this panel
    all_vals = np.concatenate([
        result[k]['log_y'][result[k]['positive']]
        for k in ('total', 'quasar', 'radio')
        if result[k]['positive'].any()
    ])
    result['y_floor'] = float(all_vals.min()) - 1.5 if len(all_vals) else -10.0
    result['n_total'] = int(valid.sum())

    return result


# ============================================================================
# DRAW ONE PANEL
# ============================================================================

def draw_panel(ax, data, show_legend=False, show_xlabel=True, show_ylabel=True):
    """Draw a single ARF panel onto `ax` from a data dict returned by compute_arf."""
    bc     = data['bin_centers']
    lo     = data['bins'][0]
    hi     = data['bins'][-1]
    y_floor = data['y_floor']

    cats = [
        ('total',  'Total',       COLOR_TOTAL,  2.5, 0.10, 3),
        ('quasar', 'Quasar Mode', COLOR_QUASAR, 2.0, 0.15, 4),
        ('radio',  'Radio Mode',  COLOR_RADIO,  2.0, 0.15, 5),
    ]

    for key, label, color, lw, alpha, zo in cats:
        d      = data[key]
        log_y  = d['log_y']
        pos    = d['positive']
        eu     = d['err_up']
        ed     = d['err_down']

        if not pos.any():
            continue

        ax.step(bc, log_y, where='mid', linewidth=lw,
                color=color, label=label, zorder=zo)
        ax.fill_between(bc, log_y, y_floor,
                        step='mid', alpha=alpha, color=color, zorder=zo - 1)
        err_mask = pos & np.isfinite(log_y)
        ax.errorbar(bc[err_mask], log_y[err_mask],
                    yerr=[ed[err_mask], eu[err_mask]],
                    fmt='none', ecolor=color, elinewidth=1.0,
                    capsize=2, zorder=zo + 1, alpha=0.75)

    ax.axvline(0.0, color='k', linestyle='--', linewidth=1.2, alpha=0.6, zorder=2)

    ax.set_xlim(lo, hi)
    all_valid = np.concatenate([
        data[k]['log_y'][data[k]['positive']]
        for k in ('total', 'quasar', 'radio')
        if data[k]['positive'].any()
    ])
    if len(all_valid):
        ax.set_ylim(y_floor + 1.0, all_valid.max() + 0.8)

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.minorticks_on()
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)

    if show_xlabel:
        ax.set_xlabel(
            r'$\log_{10}(\dot{M}_\mathrm{BH,max}\,/\,\dot{M}_\mathrm{Edd})$',
            fontsize=11)
    if show_ylabel:
        ax.set_ylabel(data['y_label'], fontsize=10)
    if show_legend:
        ax.legend(loc='upper right', fontsize=9)


# ============================================================================
# MAIN GRID FIGURE
# ============================================================================

def make_redshift_panel_grid(
    file_list, snap_numbers, id_field, h_h,
    sim_volume, no_cuts, n_bins, ncols, output_path,
    share_axes=True,
):
    n_snaps = len(snap_numbers)
    nrows   = (n_snaps + ncols - 1) // ncols

    panel_w = 4.0
    panel_h = 3.6
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * panel_w, nrows * panel_h),
        sharex=share_axes,
        sharey=False,          # y ranges differ substantially by redshift
        layout='constrained',
    )
    axes_flat = np.array(axes).flatten()
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, hspace=0.05, wspace=0.05)

    # ---- collect data for each snapshot ----
    panel_data  = []
    redshifts   = []

    for snap in snap_numbers:
        z = snap_to_z(snap)
        redshifts.append(z)
        print(f"  Reading snap {snap:3d}  (z = {z:.2f}) …", end=' ', flush=True)

        bh, sm, mv, accr, edd, atype = read_snapshot(
            file_list, snap, id_field, h_h)

        if len(bh) == 0:
            print("no data found")
            panel_data.append(None)
            continue

        d = compute_arf(bh, sm, mv, accr, edd, atype,
                        no_cuts=no_cuts,
                        sim_volume=sim_volume,
                        n_bins=n_bins)
        panel_data.append(d)

        if d is None:
            print("no valid accretion events after cuts")
        else:
            print(f"N_valid = {d['n_total']:,} "
                  f"(quasar={d['quasar']['n']:,}, radio={d['radio']['n']:,})")

    # ---- determine shared x limits (union of all bin ranges) ----
    if share_axes:
        all_lo, all_hi = [], []
        for d in panel_data:
            if d is not None:
                all_lo.append(d['bins'][0])
                all_hi.append(d['bins'][-1])
        x_lo = min(all_lo) if all_lo else -4
        x_hi = max(all_hi) if all_hi else  4

    # ---- draw panels ----
    print("\nDrawing panels …")
    for idx, (snap, z, d) in enumerate(zip(snap_numbers, redshifts, panel_data)):
        ax  = axes_flat[idx]
        row = idx // ncols
        col = idx  % ncols

        # Label position flags
        is_bottom_row  = (row == nrows - 1) or (idx + ncols >= n_snaps)
        is_left_col    = (col == 0)
        show_legend_here = (idx == 0)   # legend only on first panel

        if d is None:
            ax.text(0.5, 0.5, f'No data\nz = {z:.2f}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
        else:
            if share_axes:
                d['bins']        = np.linspace(x_lo, x_hi, n_bins + 1)
                d['bin_centers'] = 0.5 * (d['bins'][:-1] + d['bins'][1:])

            draw_panel(ax, d,
                       show_legend=show_legend_here,
                       show_xlabel=is_bottom_row,
                       show_ylabel=False)  # supylabel() handles the shared y-label

        # Redshift label in upper-left corner
        ax.text(0.04, 0.96, f'z = {z:.2f}',
                transform=ax.transAxes, fontsize=11,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor='white', alpha=0.75, edgecolor='none'))

    # Hide unused panels
    for idx in range(n_snaps, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared y-label — supylabel is constrained_layout-aware and won't squish panels
    y_label_text = (
        r'$\log_{10}(\mathrm{d}N\,/\,\mathrm{d}\log_{10}\lambda'
        + (r'\ [\mathrm{Mpc}^{-3}\,h^{3}])$' if sim_volume else r')$')
    )
    fig.supylabel(y_label_text, fontsize=13)
    # Give the supylabel enough horizontal room (bbox_inches='tight' respects this)
    plt.subplots_adjust(left=0.08)

    plt.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Grid saved to: {output_path}")


# ============================================================================
# OPTIONAL: individual panel outputs
# ============================================================================

def make_individual_panels(
    file_list, snap_numbers, id_field, h_h,
    sim_volume, no_cuts, n_bins, output_dir,
):
    for snap in snap_numbers:
        z = snap_to_z(snap)
        print(f"  Individual panel snap {snap:3d}  (z = {z:.2f}) …", end=' ', flush=True)

        bh, sm, mv, accr, edd, atype = read_snapshot(
            file_list, snap, id_field, h_h)
        if len(bh) == 0:
            print("no data"); continue

        d = compute_arf(bh, sm, mv, accr, edd, atype,
                        no_cuts=no_cuts, sim_volume=sim_volume, n_bins=n_bins)
        if d is None:
            print("no valid events"); continue

        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        draw_panel(ax, d, show_legend=True, show_xlabel=True, show_ylabel=True)
        ax.set_title(f'z = {z:.2f}  (snapshot {snap})', fontsize=13)
        fname = output_dir / f'arf_snap{snap:03d}_z{z:.2f}.png'
        plt.tight_layout()
        plt.savefig(fname, dpi=140, bbox_inches='tight')
        plt.close()
        print(f"saved → {fname.name}  (N={d['n_total']:,})")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Accretion Rate Function across redshift panels')
    parser.add_argument('-i', '--input-pattern',
                        default='./output/millennium/model_*.hdf5')
    parser.add_argument('-s', '--snapshots', type=int, nargs='+',
                        default=DEFAULT_SNAPSHOTS,
                        help='Snapshot numbers to include (default: 9 snapshots)')
    parser.add_argument('--ncols', type=int, default=3,
                        help='Number of columns in the panel grid (default: 3)')
    parser.add_argument('--nbins', type=int, default=35,
                        help='Number of lambda bins per panel (default: 35)')
    parser.add_argument('--no-cuts', action='store_true',
                        help='Disable stellar/halo/BH mass cuts')
    parser.add_argument('--no-volume', action='store_true',
                        help='Plot raw dN/d(log lambda) instead of number density')
    parser.add_argument('--sim-volume', type=float, default=None,
                        help=(f'Box volume in Mpc^3 h^-3 '
                              f'(default: {MILLENNIUM_BOX_MPC_H}^3 = '
                              f'{MILLENNIUM_BOX_MPC_H**3:.3e})'))
    parser.add_argument('--individual', action='store_true',
                        help='Also save individual panel PNGs per snapshot')
    parser.add_argument('--no-share-x', action='store_true',
                        help='Allow each panel to have its own x-axis range')
    args = parser.parse_args()

    # ---- find files ----
    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        sys.exit(f'ERROR: no files matched "{args.input_pattern}"')
    print(f'Found {len(file_list)} file(s).  First: {file_list[0]}')

    # ---- sim params ----
    h_h      = read_sim_params(file_list[0])
    id_field = find_id_field(file_list, args.snapshots[0])
    if id_field is None:
        sys.exit('ERROR: could not determine galaxy ID field from HDF5 files.')
    print(f'Hubble_h = {h_h}  |  ID field = {id_field}')

    # ---- volume ----
    if args.no_volume:
        sim_volume = None
    elif args.sim_volume is not None:
        sim_volume = args.sim_volume
    else:
        sim_volume = MILLENNIUM_BOX_MPC_H ** 3
    print(f'Volume: {sim_volume:.4e} Mpc^3 h^-3' if sim_volume else
          'Volume: raw counts (no normalisation)')

    # ---- output directory ----
    output_dir = Path(file_list[0]).parent / 'plots'
    output_dir.mkdir(exist_ok=True)

    # ---- sort snapshots by descending redshift (high-z first) ----
    snaps = sorted(args.snapshots, key=lambda s: -snap_to_z(s))
    print(f'\nSnapshots ({len(snaps)}):  '
          + '  '.join(f'{s}(z={snap_to_z(s):.1f})' for s in snaps))
    print('=' * 70)

    # ---- grid figure ----
    print('\nBuilding panel grid …')
    grid_path = output_dir / 'arf_redshift_panels.png'
    make_redshift_panel_grid(
        file_list, snaps, id_field, h_h,
        sim_volume=sim_volume,
        no_cuts=args.no_cuts,
        n_bins=args.nbins,
        ncols=args.ncols,
        output_path=grid_path,
        share_axes=not args.no_share_x,
    )

    # ---- optional individual panels ----
    if args.individual:
        print('\nSaving individual panels …')
        make_individual_panels(
            file_list, snaps, id_field, h_h,
            sim_volume=sim_volume,
            no_cuts=args.no_cuts,
            n_bins=args.nbins,
            output_dir=output_dir,
        )

    print('\n✓ Done.')


if __name__ == '__main__':
    main()