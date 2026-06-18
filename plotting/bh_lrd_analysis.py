#!/usr/bin/env python3
"""
bh_lrd_analysis.py
==================
Recreates panel (a) of Figure 1 from Chen & Mo (2026), arXiv:2605.31077:

    log10(Mdot_BH  [M_sun/yr])  vs  log10(M_BH  [M_sun])   at a chosen redshift

HOW THE DATA IS STORED (important!)
-----------------------------------
In the SAGE26 HDF5 output, each `Snap_N` group is a *galaxy catalogue* at that
output time.  The high-N catalogues (e.g. Snap_62 / Snap_63) hold every
surviving galaxy together with that galaxy's FULL black-hole accretion history
in `[Ngal, ABSOLUTEMAXSNAPS]` arrays:

    BHMaxaccretionRate[:, c]    Mdot_BH recorded at snapshot c
    BHEddingtonRateLimit[:, c]  Mdot_Edd at snapshot c
    BHMassatAccretion[:, c]     M_BH at the time of that accretion episode
    BHAccretionType[:, c]       0=Radio, 1=Merger, 2=Disk Instability

So to reproduce the z=5 plane we read the LATE catalogue (most complete) and
slice the history COLUMN closest to z=5 (snapshot 10, z=5.48 in Millennium).
We do NOT read the `Snap_10` group directly — at z=5 the catalogue is nearly
empty and the histories there are blank.

The x-axis uses BHMassatAccretion (M_BH at that epoch), NOT the z=0
BlackHoleMass, so the plot shows the true (M_BH, Mdot_BH) plane at that redshift.

LRD selection criteria (Chen & Mo 2026, §II.2 / Fig. 1):
    Red dots  (full LRD):  Mdot_BH >= 0.1 M_sun/yr  AND  Mdot_BH >= Mdot_Edd
                           AND  f_BH = M_BH/M_star >= 0.03
    Blue dots (partial):   Mdot_BH >= 0.1 M_sun/yr  AND  Mdot_BH >= Mdot_Edd
                           AND  f_BH < 0.03
Reference lines:
    red  solid  -> Mdot_BH = Mdot_Edd
    orange solid-> Mdot_BH = 10 * Mdot_Edd
    red  dashed -> Mdot_BH = 0.1  M_sun/yr   (default BHAR threshold)
    red  dotted -> Mdot_BH = 0.05 M_sun/yr   (alternative threshold)
Contours enclose 68 / 95 / 99.7 % of the plotted BHs.

NOTE ON f_BH:  the per-epoch stellar mass is not stored alongside the accretion
history, so f_BH is computed from the catalogue-level StellarMass.  Pass
--no-fbh to disable the f_BH split (all selected BHs drawn red) if you'd rather
not mix epochs.  See the comment in compute_selection() for details.

Usage
-----
    python3 plotting/bh_lrd_analysis.py
    python3 plotting/bh_lrd_analysis.py -s 10            # z = 5.48 column
    python3 plotting/bh_lrd_analysis.py -s 10 --window 1 # stack snaps 9-11
    python3 plotting/bh_lrd_analysis.py --catalogue Snap_63
    python3 plotting/bh_lrd_analysis.py --bhar-floor 0.05
"""

import argparse
import glob
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# ============================================================================
# MATPLOTLIB STYLE  (matching bh_eddington_analysis.py)
# ============================================================================
plt.rcParams.update({
    'figure.dpi': 140,
    'figure.autolayout': True,
    'font.family': 'serif',
    'font.size': 20.0,
    'axes.linewidth': 1.5,
    'xtick.major.size': 7.5, 'xtick.major.width': 1.5,
    'xtick.minor.size': 5.5, 'xtick.minor.width': 0.5,
    'xtick.direction': 'in', 'xtick.top': True, 'xtick.labelsize': 16,
    'xtick.major.pad': 9,
    'ytick.major.size': 7.5, 'ytick.major.width': 1.5,
    'ytick.minor.size': 5.5, 'ytick.minor.width': 0.5,
    'ytick.direction': 'in', 'ytick.right': True, 'ytick.labelsize': 16,
    'legend.frameon': False, 'legend.fontsize': 12,
})

# ============================================================================
# CONSTANTS & UNIT CONVERSIONS
# ============================================================================
SEC_PER_YEAR   = 365.25 * 24 * 3600    # s / yr
UNIT_TIME_IN_S = 3.086e19              # SAGE Millennium code time unit (~1 Gyr)

# Eddington rate from M_BH (eta = 0.1):  Mdot_Edd = M_BH / T_SALPETER_YR
T_SALPETER_YR  = 4.5e8                 # yr

# ── LRD selection (Chen & Mo 2026) ─────────────────────────────────────────
LRD_BHAR_DEFAULT = 0.1     # M_sun/yr  (red dashed line)
LRD_BHAR_ALT     = 0.05    # M_sun/yr  (red dotted line)
LRD_FBHM_THRESH  = 0.03    # M_BH / M_star >= 3% -> red, else blue

# ── Millennium snapshot -> redshift ────────────────────────────────────────
MILLENNIUM_SNAP_TO_Z = {
    0: 127.0, 1: 65.74, 2: 40.0,  3: 26.66, 4: 19.36, 5: 14.78, 6: 11.66,
    7: 9.44,  8: 7.64,  9: 6.44,  10: 5.48, 11: 4.73, 12: 4.19, 13: 3.72,
    14: 3.33, 15: 3.0,  16: 2.73, 17: 2.48, 18: 2.27, 19: 2.07, 20: 1.90,
    21: 1.75, 22: 1.61, 23: 1.48, 24: 1.37, 25: 1.27, 26: 1.18, 27: 1.10,
    28: 1.02, 29: 0.96, 30: 0.90, 31: 0.85, 32: 0.81, 33: 0.77, 34: 0.73,
    35: 0.70, 36: 0.67, 37: 0.63, 38: 0.60, 39: 0.57, 40: 0.54, 41: 0.51,
    42: 0.49, 43: 0.46, 44: 0.43, 45: 0.41, 46: 0.39, 47: 0.37, 48: 0.36,
    49: 0.34, 50: 0.32, 51: 0.31, 52: 0.29, 53: 0.28, 54: 0.27, 55: 0.26,
    56: 0.25, 57: 0.24, 58: 0.23, 59: 0.21, 60: 0.20, 61: 0.18, 62: 0.0,
}

def snap_to_z(snap):
    return MILLENNIUM_SNAP_TO_Z.get(snap, 0.0)

# ============================================================================
# I/O
# ============================================================================

def read_sim_params(filepath):
    """Extract Hubble_h from HDF5 header."""
    try:
        with h5py.File(filepath, 'r') as hf:
            for grp in ('Header/Simulation', 'Header', 'Parameters'):
                if grp in hf:
                    attrs = hf[grp].attrs
                    for key in ('hubble_h', 'HubbleParam', 'Hubble_h'):
                        if key in attrs:
                            return float(attrs[key])
    except Exception:
        pass
    return 0.73   # Millennium default


def pick_catalogue_group(hf, requested=None):
    """
    Choose which Snap_N catalogue group to read the histories from.
    Default: the highest-numbered group that actually has galaxies
    (the most complete catalogue, carrying full accretion histories).
    """
    snaps = sorted(
        [k for k in hf.keys() if k.startswith('Snap_')],
        key=lambda s: int(s.split('_')[1]),
    )
    if requested is not None:
        if requested in hf:
            return requested
        print(f'  WARNING: requested catalogue "{requested}" not found; '
              f'falling back to auto-select.')
    # auto: walk from the top until we find a populated group
    for s in reversed(snaps):
        try:
            if hf[s]['BlackHoleMass'].shape[0] > 0:
                return s
        except Exception:
            continue
    return snaps[-1]


def _history_column(arr2d, col):
    """Return column `col` of a [Ngal, MAXSNAPS] array, clamped to bounds."""
    c = min(max(col, 0), arr2d.shape[1] - 1)
    return arr2d[:, c]


def read_epoch(file_list, snap_col, h_h, catalogue=None, window=0):
    """
    Read the (M_BH, Mdot_BH) plane at the history column `snap_col`
    (= Millennium snapshot index) from the most complete catalogue group.

    window > 0 stacks columns [snap_col-window, snap_col+window] to fight
    sparsity at very high z (each event still becomes its own point).

    Returns dict of physical-unit arrays:
        bh_mass        [M_sun]   (M_BH at the accretion epoch)
        mdot_msun_yr   [M_sun/yr]
        mdot_edd       [M_sun/yr]
        acc_type       {0,1,2,-1}
        stellar_mass   [M_sun]   (catalogue-level; see f_BH caveat)
        cat_group      str       (which group was read)
    """
    mass_conv = 1e10 / h_h
    rate_conv = mass_conv / (UNIT_TIME_IN_S / SEC_PER_YEAR)

    out_bh, out_mdot, out_edd, out_type, out_star = [], [], [], [], []
    cat_used = None
    cols = list(range(snap_col - window, snap_col + window + 1))

    for fpath in file_list:
        with h5py.File(fpath, 'r') as hf:
            cat = pick_catalogue_group(hf, catalogue)
            cat_used = cat
            grp = hf[cat]

            mdot_h = grp['BHMaxaccretionRate'][:]      # [Ngal, MAXSNAPS]
            edd_h  = grp['BHEddingtonRateLimit'][:]
            mass_h = (grp['BHMassatAccretion'][:]
                      if 'BHMassatAccretion' in grp else None)
            type_h = (grp['BHAccretionType'][:]
                      if 'BHAccretionType' in grp else None)
            star   = grp['StellarMass'][:]              # [Ngal] catalogue-level

            for c in cols:
                if c < 0 or c >= mdot_h.shape[1]:
                    continue
                mdot_c = _history_column(mdot_h, c)
                edd_c  = _history_column(edd_h,  c)

                if mass_h is not None:
                    bh_c = _history_column(mass_h, c)
                else:
                    # fallback: no per-epoch mass -> derive from Eddington rate
                    bh_c = edd_c * (T_SALPETER_YR / (mass_conv) ) * mass_conv
                    bh_c = edd_c * rate_conv * T_SALPETER_YR  # M_sun

                type_c = (_history_column(type_h, c)
                          if type_h is not None
                          else np.full_like(mdot_c, -1.0))

                # only keep galaxies that actually have an event in this column
                ev = mdot_c > 0
                if not np.any(ev):
                    continue

                out_bh.append(bh_c[ev]   * mass_conv)
                out_mdot.append(mdot_c[ev] * rate_conv)
                out_edd.append(edd_c[ev]  * rate_conv)
                out_type.append(type_c[ev])
                out_star.append(star[ev]  * mass_conv)

    if not out_bh:
        return {
            'bh_mass': np.array([]), 'mdot_msun_yr': np.array([]),
            'mdot_edd': np.array([]), 'acc_type': np.array([]),
            'stellar_mass': np.array([]), 'cat_group': cat_used,
        }

    return {
        'bh_mass'      : np.concatenate(out_bh),
        'mdot_msun_yr' : np.concatenate(out_mdot),
        'mdot_edd'     : np.concatenate(out_edd),
        'acc_type'     : np.concatenate(out_type),
        'stellar_mass' : np.concatenate(out_star),
        'cat_group'    : cat_used,
    }

# ============================================================================
# HELPERS
# ============================================================================

def eddington_mdot(log_mbh):
    """log10(Mdot_Edd [M_sun/yr]) from log10(M_BH [M_sun])  (eta=0.1)."""
    return log_mbh - np.log10(T_SALPETER_YR)

# ============================================================================
# MAIN PLOT
# ============================================================================

def plot_panel_a(data, snap_col, output_file,
                 show_lrd=True, use_fbh=True,
                 bhar_floor=LRD_BHAR_DEFAULT, z_override=None):

    bh_mass = data['bh_mass']
    mdot    = data['mdot_msun_yr']
    medd    = data['mdot_edd']
    star    = data['stellar_mass']

    if len(bh_mass) == 0:
        print('  ERROR: no accretion events found in the selected column(s).')
        print('         Try a later column (-s 15), a wider --window, '
              'or a different --catalogue.')
        return

    # quality: positive, finite mass & rate
    valid = (
        (bh_mass > 0) & (mdot > 0) & np.isfinite(bh_mass) & np.isfinite(mdot)
    )
    bh_mass = bh_mass[valid]; mdot = mdot[valid]
    medd    = medd[valid];    star = star[valid]

    log_mbh  = np.log10(bh_mass)
    log_mdot = np.log10(mdot)

    print(f'  Accretion events plotted: {len(log_mbh):,}')

    # ── selection masks ───────────────────────────────────────────────────
    bhar_pass = mdot >= bhar_floor
    edd_pass  = mdot >= np.where(medd > 0, medd, np.inf)

    if use_fbh:
        f_bh     = bh_mass / np.where(star > 0, star, np.inf)
        fbh_pass = f_bh >= LRD_FBHM_THRESH
        lrd_red  = bhar_pass & edd_pass & fbh_pass
        lrd_blue = bhar_pass & edd_pass & ~fbh_pass
    else:
        lrd_red  = bhar_pass & edd_pass
        lrd_blue = np.zeros(len(log_mbh), dtype=bool)

    print(f'  LRD red  (full):    {lrd_red.sum():,}')
    if use_fbh:
        print(f'  LRD blue (f_BH<3%): {lrd_blue.sum():,}')

    # ── figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.minorticks_on()

    x_lo, x_hi = 2.0, 10.5

    # safe y-limits (guard against tiny samples)
    if len(log_mdot) >= 5:
        y_lo = max(-13.5, np.percentile(log_mdot, 0.5) - 0.3)
        y_hi = max(2.5,   np.percentile(log_mdot, 99.5) + 0.5)
    else:
        y_lo, y_hi = log_mdot.min() - 0.5, log_mdot.max() + 0.5
    y_lo = np.floor(y_lo * 2) / 2
    y_hi = np.ceil(y_hi  * 2) / 2

    # ── LRD shaded region ─────────────────────────────────────────────────
    if show_lrd:
        x_fill  = np.array([x_lo, x_hi])
        y_lower = np.maximum(eddington_mdot(x_fill), np.log10(bhar_floor))
        ax.fill_between(x_fill, y_lower, y_hi,
                        color='#D32F2F', alpha=0.08, zorder=0)

    # ── grey background scatter ───────────────────────────────────────────
    bg = ~(lrd_red | lrd_blue) if show_lrd else np.ones(len(log_mbh), dtype=bool)
    x_bg, y_bg = log_mbh[bg], log_mdot[bg]

    N_SCATTER = 30_000
    if len(x_bg) > N_SCATTER:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_bg), N_SCATTER, replace=False)
        x_sc, y_sc = x_bg[idx], y_bg[idx]
    else:
        x_sc, y_sc = x_bg, y_bg
    ax.scatter(x_sc, y_sc, s=4, color='#999999', alpha=0.20,
               linewidths=0, rasterized=True, zorder=1)

    # ── KDE contours (68/95/99.7%) ────────────────────────────────────────
    if len(x_bg) >= 50:
        try:
            N_KDE = 60_000
            if len(x_bg) > N_KDE:
                rng = np.random.default_rng(77)
                idx = rng.choice(len(x_bg), N_KDE, replace=False)
                xk, yk = x_bg[idx], y_bg[idx]
            else:
                xk, yk = x_bg, y_bg

            kde  = gaussian_kde(np.vstack([xk, yk]), bw_method='scott')
            xi   = np.linspace(x_lo, x_hi, 250)
            yi   = np.linspace(y_lo, y_hi, 250)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi   = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

            z_sort = np.sort(Zi.ravel())[::-1]
            z_cum  = np.cumsum(z_sort) / z_sort.sum()
            def lvl(frac):
                return z_sort[min(np.searchsorted(z_cum, frac), len(z_sort) - 1)]
            levels = sorted([lvl(0.683), lvl(0.954), lvl(0.997)])

            ax.contour(Xi, Yi, Zi, levels=levels,
                       colors='#333333',
                       linewidths=[0.9, 1.2, 1.6],
                       linestyles=[':', '--', '-'], zorder=2)
        except Exception as e:
            print(f'  WARNING: KDE contours skipped ({e})')
    else:
        print('  (too few background points for KDE contours)')

    # ── LRD coloured dots ─────────────────────────────────────────────────
    if show_lrd:
        if lrd_blue.sum() > 0:
            ax.scatter(log_mbh[lrd_blue], log_mdot[lrd_blue],
                       s=35, color='#1565C0', edgecolors='white',
                       linewidths=0.3, zorder=5)
        if lrd_red.sum() > 0:
            ax.scatter(log_mbh[lrd_red], log_mdot[lrd_red],
                       s=35, color='#C62828', edgecolors='white',
                       linewidths=0.3, zorder=6)

    # ── reference lines ───────────────────────────────────────────────────
    x_ref = np.linspace(x_lo, x_hi, 400)
    y_edd = eddington_mdot(x_ref)
    ax.plot(x_ref, y_edd, color='#C62828', lw=1.8, ls='-', zorder=4)
    ax.plot(x_ref, y_edd + 1.0, color='#E65100', lw=1.8, ls='-', zorder=4)

    ax.axhline(np.log10(LRD_BHAR_DEFAULT), color='#C62828', lw=1.3,
               ls='--', zorder=3, alpha=0.85)
    ax.axhline(np.log10(LRD_BHAR_ALT), color='#C62828', lw=1.0,
               ls=':', zorder=3, alpha=0.70)

    ax.annotate(
        rf'$\dot{{M}}_{{\rm BH}} = {LRD_BHAR_DEFAULT}\,M_\odot\,\mathrm{{yr}}^{{-1}}$',
        xy=(x_hi - 0.2, np.log10(LRD_BHAR_DEFAULT) + 0.15),
        fontsize=10.5, color='#C62828', ha='right',
    )

    if show_lrd:
        ax.text(x_lo + 3.3, y_hi - 1.45, 'LRD selection',
                fontsize=13, color='#C62828')

    # ── axes & decorations ────────────────────────────────────────────────
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r'$M_{\rm BH}\ [M_\odot]$', fontsize=18)
    ax.set_ylabel(r'$\dot{M}_{\rm BH}\ [M_\odot\,\mathrm{yr}^{-1}]$', fontsize=18)
    ax.set_xticks(np.arange(int(np.ceil(x_lo)), int(np.floor(x_hi)) + 1, 2))

    redshift = z_override if z_override is not None else snap_to_z(snap_col)
    ax.text(0.97, 0.04, rf'$z = {redshift:.1f}$',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=15)
   # ax.text(0.03, 0.97, 'a', transform=ax.transAxes, ha='left', va='top',
    #        fontsize=18, fontweight='bold')

    # ── legend ────────────────────────────────────────────────────────────
    handles = [
        Line2D([0], [0], color='#C62828', lw=1.8,
               label=r'$\dot{M}_{\rm BH} = \dot{M}_{\rm Edd}$'),
        Line2D([0], [0], color='#E65100', lw=1.8,
               label=r'$\dot{M}_{\rm BH} = 10\,\dot{M}_{\rm Edd}$'),
    ]
    if show_lrd:
        handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#C62828',
                   markersize=7,
                   label=(r'LRD ($f_{\rm BH}\geq 3\%$)' if use_fbh else 'LRD')))
        if use_fbh:
            handles.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='#1565C0', markersize=7,
                       label=r'LRD ($f_{\rm BH}<3\%$)'))
    ax.legend(handles=handles, loc='upper left', fontsize=10.5,
              handlelength=1.6, handletextpad=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'✓  Saved  →  {output_file}')

# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description='Recreate panel (a) of Chen & Mo 2026 (arXiv:2605.31077).'
    )
    p.add_argument('-i', '--input-pattern',
                   default='./output/millennium/model_*.hdf5')
    p.add_argument('-s', '--snapshot', type=int, default=10,
                   help='History COLUMN (= Millennium snapshot) to slice. '
                        'Default 10 -> z=5.48 (closest to paper z=5).')
    p.add_argument('--window', type=int, default=0,
                   help='Stack columns [s-window, s+window] to fight sparsity '
                        'at high z (default 0 = single column).')
    p.add_argument('--catalogue', default=None,
                   help='Force a specific Snap_N catalogue group to read '
                        'histories from (default: most complete, auto-selected).')
    p.add_argument('--no-lrd', action='store_true',
                   help='Skip LRD selection overlay.')
    p.add_argument('--no-fbh', action='store_true',
                   help='Disable the f_BH red/blue split (all selected = red). '
                        'Use if mixing the per-epoch M_BH with catalogue-level '
                        'M_star is a concern.')
    p.add_argument('--bhar-floor', type=float, default=LRD_BHAR_DEFAULT,
                   help=f'BHAR floor in M_sun/yr (default {LRD_BHAR_DEFAULT}; '
                        f'paper alternative {LRD_BHAR_ALT}).')
    p.add_argument('--output', default=None)
    p.add_argument('--z', type=float, default=None,
                   help='Override redshift label on the plot.')
    args = p.parse_args()

    files = sorted(glob.glob(args.input_pattern))
    if not files:
        print(f'ERROR: no files matched "{args.input_pattern}"'); sys.exit(1)

    h_h  = read_sim_params(files[0])
    z    = args.z if args.z is not None else snap_to_z(args.snapshot)

    print(f'Files:       {len(files)}')
    print(f'History col: {args.snapshot}  ->  z ~ {z:.3f}'
          + (f'  (+/- {args.window})' if args.window else ''))
    print(f'Hubble_h:    {h_h}')
    print(f'BHAR floor:  {args.bhar_floor} M_sun/yr')
    print('Reading data...')

    data = read_epoch(files, args.snapshot, h_h,
                      catalogue=args.catalogue, window=args.window)
    print(f'Catalogue read: {data["cat_group"]}')

    if args.output:
        out = Path(args.output)
    else:
        d = Path(files[0]).parent / 'plots'
        d.mkdir(exist_ok=True)
        out = d / f'lrd_bh_accretion_scatter_snap{args.snapshot:02d}.png'

    print('Plotting...')
    plot_panel_a(data, args.snapshot, out,
                 show_lrd=(not args.no_lrd),
                 use_fbh=(not args.no_fbh),
                 bhar_floor=args.bhar_floor,
                 z_override=args.z)


if __name__ == '__main__':
    main()