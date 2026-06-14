"""
plot_timescales.py
------------------
Read dt.txt, tdyn_B_end.txt, tdyn_disk.txt, hubbletime.txt, halotime.txt and produce:
  - Figure 1: five-panel histogram (one per timescale)
  - Figure 2: all five overlaid on a single axis
  - Figure 4: bulge vs disk tdyn comparison
  - Figure 3: three-panel histogram of BRadii.txt, BMass_B.txt, BMass_SG.txt
              (BulgeRadius in kpc, BulgeMass and StellarMass+ColdGas in Msun)
  - Figure 5: three-panel histogram of disk properties:
              DRadii.txt    -> DiskScaleRadius (Mpc/h, converted to kpc)
              DMass_S.txt   -> M_disk = StellarMass - BulgeMass
              DMass_SG.txt  -> M_disk = StellarMass + ColdGas - BulgeMass
              If the DMass_* files are absent, the masses are derived
              element-wise from raw dumps:
                  DMass_S  = SMass.txt    - BMass_B.txt
                  DMass_SG = BMass_SG.txt - BMass_B.txt
  - Figure 6: mass function comparison (baryonic, bulge, disk)
              Requires volume to be specified via --volume or read from HDF5

Usage:
    python plot_timescales.py [--dir PATH] [--units {code,myr}] [--log] [--save] [--volume V]

Options:
    --dir PATH          directory containing the .txt files       [default: .]
    --units {code,myr}  units the timescale files are stored in   [default: code]
    --log               use log10 x-axis on timescale plots       [default: off]
    --save              save figures to PNG instead of showing    [default: off]
    --bins N            number of histogram bins                  [default: 80]
    --volume V          simulation volume in (Mpc/h)^3            [default: read from HDF5]

If --units code, timescale values are converted to Myr using
    UnitTime_in_Megayears = 3.08568e24 / 1e5 / 3.155e13  (~978,029 Myr/code unit)
before plotting.

BRadii.txt and DRadii.txt are stored in Mpc/h and converted to kpc on load.
All mass files are in 10^10 Msun/h and converted to Msun on load.
"""

import argparse
import os
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

try:
    import h5py
except ImportError:
    h5py = None

# ── Palatino / mathpazo style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":         "serif",
    "font.serif":          ["Palatino", "Palatino Linotype", "DejaVu Serif"],
    "mathtext.fontset":    "cm",
    "axes.labelsize":      13,
    "axes.titlesize":      13,
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize":     11,
    "legend.framealpha":   0.8,
    "figure.dpi":          150,
})

# ── Unit conversions ──────────────────────────────────────────────────────────
UNIT_LENGTH_CM     = 3.08568e24
UNIT_VELOCITY_CM_S = 1.0e5
UNIT_TIME_S        = UNIT_LENGTH_CM / UNIT_VELOCITY_CM_S
SEC_PER_MYR        = 3.155e13
UNIT_TIME_IN_MYR   = UNIT_TIME_S / SEC_PER_MYR   # ≈ 978 029 Myr/code unit

HUBBLE_h           = 0.73
RADII_TO_KPC       = 1e3 / HUBBLE_h               # Mpc/h → kpc
MASS_TO_MSUN       = 1e10 / HUBBLE_h              # 10^10 Msun/h → Msun

# ── Timescale metadata ────────────────────────────────────────────────────────
COLOURS = {
    "dt":         "#4477AA",   # blue
    "tdyn_B_end": "#EE6677",   # red   — bulge dynamical time
    "tdyn_disk":  "#AA3377",   # purple — disk dynamical time
    "hubbletime": "#228833",   # green
    "halotime":   "#CCBB44",   # yellow
}
LABELS = {
    "dt":         r"$\Delta t$ (snapshot step)",
    "tdyn_B_end": r"$t_{\rm dyn}$ (bulge)",
    "tdyn_disk":  r"$t_{\rm dyn}$ (disk)",
    "hubbletime": r"$0.1 \times t_H(z) = 1/H(z)$",
    "halotime":   r"$t_{\rm halo} = R_{\rm vir}/V_{\rm vir}$",
}
KEYS = ["dt", "tdyn_B_end", "tdyn_disk", "hubbletime", "halotime"]

# ── Bulge property metadata ───────────────────────────────────────────────────
BULGE_COLOURS = {
    "BRadii":   "#AA3377",
    "BMass_B":  "#EE7733",
    "BMass_SG": "#009988",
}
BULGE_LABELS = {
    "BRadii":   r"$r_{\rm bulge}$  (kpc)",
    "BMass_B":  r"$M_{\rm bulge}$  ($M_\odot$)",
    "BMass_SG": r"$M_\star + M_{\rm cold}$  ($M_\odot$)",
}
BULGE_KEYS = ["BRadii", "BMass_B", "BMass_SG"]

# ── Disk property metadata ────────────────────────────────────────────────────
DISK_COLOURS = {
    "DRadii":   "#4477AA",
    "DMass_S":  "#EE6677",
    "DMass_SG": "#228833",
}
DISK_LABELS = {
    "DRadii":   r"$r_{\rm disk}$ (DiskScaleRadius)  (kpc)",
    "DMass_S":  r"$M_{\rm disk} = M_\star - M_{\rm bulge}$  ($M_\odot$)",
    "DMass_SG": r"$M_{\rm disk} = M_\star + M_{\rm cold} - M_{\rm bulge}$  ($M_\odot$)",
}
DISK_KEYS = ["DRadii", "DMass_S", "DMass_SG"]


# ─────────────────────────────────────────────────────────────────────────────
def load_file(directory, key, quiet=False):
    path = os.path.join(directory, f"{key}.txt")
    if not os.path.isfile(path):
        if not quiet:
            print(f"  WARNING: {path} not found – skipping.")
        return None
    data = np.loadtxt(path).ravel()
    data = data[np.isfinite(data) & (data > 0.0)]
    return data


def load_raw(directory, key):
    """Load a dump file WITHOUT filtering (preserves row alignment)."""
    path = os.path.join(directory, f"{key}.txt")
    if not os.path.isfile(path):
        return None
    return np.loadtxt(path).ravel()


def find_hdf5_files(directory, recursive=True):
    """
    Locate SAGE model HDF5 files. Tries, in order:
      1. model_*.hdf5 directly in `directory`
      2. any *.hdf5 directly in `directory`
      3. (if recursive) model_*.hdf5 anywhere beneath `directory`
      4. (if recursive) any *.hdf5 anywhere beneath `directory`
    Returns a sorted list of paths (possibly empty).
    """
    if h5py is None:
        return []
    for pattern, rec in [
        (os.path.join(directory, "model_*.hdf5"), False),
        (os.path.join(directory, "*.hdf5"),       False),
        (os.path.join(directory, "**", "model_*.hdf5"), True),
        (os.path.join(directory, "**", "*.hdf5"),       True),
    ]:
        if rec and not recursive:
            continue
        hits = sorted(glob.glob(pattern, recursive=rec))
        if hits:
            return hits
    return []


def read_volume_from_hdf5(directory):
    """
    Attempt to read simulation volume from HDF5 file in the directory.
    Returns (volume in (Mpc/h)^3, hubble_h) or (None, None) if unable to read.
    """
    hdf5_files = find_hdf5_files(directory)
    if not hdf5_files:
        return None, None
    
    try:
        with h5py.File(hdf5_files[0], 'r') as f:
            # Try to read simulation parameters from header
            if 'Header/Simulation' in f:
                sim = f['Header/Simulation']
                hubble_h = float(sim.attrs.get('hubble_h', 0.73))
                box_size = float(sim.attrs.get('box_size', 500.0))
                volume_frac = float(sim.attrs.get('frac_volume_processed', 1.0))
                volume = (box_size / hubble_h) ** 3 * volume_frac
                return volume, hubble_h
    except Exception as e:
        pass
    
    return None, None


def read_baryonic_by_type_from_hdf5(directory, snapshot=None):
    """
    Read galaxy masses from HDF5 for a single snapshot group, summed across all
    model_*.hdf5 files. All masses returned in Msun.

    Reads StellarMass, ColdGas, BulgeMass (and Type if present) and derives:
        baryonic = StellarMass + ColdGas                 -> 'all'
        bulge    = BulgeMass                             -> 'bulge'
        disk_s   = StellarMass - BulgeMass               -> 'disk_s'
        disk_sg  = StellarMass + ColdGas - BulgeMass     -> 'disk_sg'
    The disk masses are filtered to positive values on the way out.

    If `snapshot` is None, the highest-numbered Snap_* group is used (z=0).
    Returns a dict with keys 'all', 'cen', 'sat', 'bulge', 'disk_s', 'disk_sg'
    (each a Msun array or None), plus 'volume', 'hubble_h', 'snapshot'.
    Returns None on failure.
    """
    if h5py is None:
        return None

    hdf5_files = find_hdf5_files(directory)
    if not hdf5_files:
        return None

    try:
        # Header / volume from the first file
        with h5py.File(hdf5_files[0], 'r') as f:
            if 'Header/Simulation' not in f:
                return None
            sim = f['Header/Simulation']
            hubble_h = float(sim.attrs.get('hubble_h', 0.73))
            box_size = float(sim.attrs.get('box_size', 500.0))

            # Resolve snapshot group name
            snap_groups = [k for k in f.keys() if k.startswith('Snap_')]
            if not snap_groups:
                return None
            if snapshot is None:
                snap_nums = sorted(int(s.split('_')[1]) for s in snap_groups)
                snap_name = f"Snap_{snap_nums[-1]}"
            else:
                snap_name = f"Snap_{snapshot}"

        # Sum VolumeFraction across files and concatenate galaxy arrays
        total_vf = 0.0
        sm_list, cg_list, bm_list, type_list = [], [], [], []
        for path in hdf5_files:
            with h5py.File(path, 'r') as f:
                rt = f.get('Header/Runtime')
                if rt is not None and 'frac_volume_processed' in rt.attrs:
                    total_vf += float(rt.attrs['frac_volume_processed'])
                if snap_name not in f:
                    continue
                grp = f[snap_name]
                if 'StellarMass' not in grp or 'ColdGas' not in grp:
                    continue
                sm_list.append(np.array(grp['StellarMass']))
                cg_list.append(np.array(grp['ColdGas']))
                bm_list.append(np.array(grp['BulgeMass'])
                               if 'BulgeMass' in grp else None)
                if 'Type' in grp:
                    type_list.append(np.array(grp['Type']))

        if not sm_list:
            return None
        if total_vf <= 0.0:
            total_vf = 1.0

        sm = np.concatenate(sm_list)
        cg = np.concatenate(cg_list)
        to_msun = 1.0e10 / hubble_h
        bary_msun = (sm + cg) * to_msun               # M_stars + M_cold

        result = {
            'hubble_h': hubble_h,
            'volume': (box_size / hubble_h) ** 3 * total_vf,
            'all': bary_msun,
            'cen': None,
            'sat': None,
            'bulge': None,
            'disk_s': None,
            'disk_sg': None,
            'snapshot': snap_name,
        }

        # Centrals / satellites split
        if type_list and sum(len(t) for t in type_list) == len(bary_msun):
            gtype = np.concatenate(type_list)
            result['cen'] = bary_msun[gtype == 0]
            result['sat'] = bary_msun[gtype == 1]

        # Bulge / disk decomposition (only if BulgeMass present in every file)
        if all(b is not None for b in bm_list):
            bm = np.concatenate(bm_list)
            if len(bm) == len(sm):
                bulge_msun   = bm * to_msun
                disk_s_code  = sm - bm                 # stellar disk
                disk_sg_code = sm + cg - bm            # disk incl. cold gas
                # keep positive masses (negatives flag a decomposition bug)
                result['bulge']   = bulge_msun[bulge_msun > 0.0]
                result['disk_s']  = (disk_s_code[disk_s_code > 0.0]) * to_msun
                result['disk_sg'] = (disk_sg_code[disk_sg_code > 0.0]) * to_msun
        return result
    except Exception:
        return None


def load_disk_mass(directory, key):
    """
    Load a disk-mass file directly if present; otherwise derive it
    element-wise from raw (unfiltered, row-aligned) dumps.

    The StellarMass entering the disk masses is the same Gal[p].StellarMass
    used in the bulge dumps (BMass_SG = StellarMass + ColdGas), so:
        DMass_SG = BMass_SG - BMass_B                  (exact, no new dumps)
        DMass_S  = SMass - BMass_B                     (requires SMass.txt)
    Returns data in code mass units (10^10 Msun/h), or None.
    """
    direct = load_file(directory, key, quiet=True)
    if direct is not None:
        return direct

    bulge = load_raw(directory, "BMass_B")
    if key == "DMass_S":
        total, src = load_raw(directory, "SMass"), "SMass - BMass_B"
    else:
        total, src = load_raw(directory, "BMass_SG"), "BMass_SG - BMass_B"

    if total is None or bulge is None:
        print(f"  WARNING: {key}.txt not found and cannot derive it – skipping.")
        return None
    if len(total) != len(bulge):
        print(f"  WARNING: cannot derive {key} ({src}): "
              f"row counts differ ({len(total)} vs {len(bulge)}).")
        return None

    data = total - bulge
    n_bad = np.sum(~(np.isfinite(data) & (data > 0.0)))
    data = data[np.isfinite(data) & (data > 0.0)]
    print(f"  NOTE: {key} derived element-wise as {src} "
          f"({n_bad:,} non-positive/NaN rows dropped).")
    return data


def convert_to_myr(data, units):
    if units == "code":
        return data * UNIT_TIME_IN_MYR
    return data


def make_bins(datasets_list, n_bins, log):
    finite = np.concatenate([d for d in datasets_list if d is not None])
    if log:
        lo, hi = np.log10(finite.min()), np.log10(finite.max())
        return np.logspace(lo, hi, n_bins + 1)
    lo, hi = finite.min(), finite.max()
    return np.linspace(lo, hi, n_bins + 1)


def apply_log_formatter(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)


def _mass_function(masses_msun, binwidth, volume):
    """
    Histogram log10(mass) and normalise to phi = N / volume / binwidth.
    Mirrors the SAGE allresults BaryonicMassFunction recipe exactly:
        mi = floor(min) - 2 ; ma = floor(max) + 2 ; NB = (ma-mi)/binwidth
    Returns (bin_centres, phi).
    """
    masses_msun = masses_msun[np.isfinite(masses_msun) & (masses_msun > 0.0)]
    if masses_msun.size == 0:
        return np.array([]), np.array([])
    mass = np.log10(masses_msun)
    mi = np.floor(mass.min()) - 2
    ma = np.floor(mass.max()) + 2
    NB = int((ma - mi) / binwidth)
    counts, binedges = np.histogram(mass, range=(mi, ma), bins=NB)
    centres = binedges[:-1] + 0.5 * binwidth
    phi = counts / volume / binwidth
    return centres, phi


def validate_decomposition(directory, volume=None, hubble_h=HUBBLE_h):
    """
    Row-by-row physical sanity check on the raw (unfiltered, row-aligned) dumps.
    Every galaxy must satisfy   M_bulge <= M_stars <= M_baryonic,   where
        baryonic = BMass_SG  = M_stars + M_cold
        bulge    = BMass_B   = M_bulge
        stars    = SMass                     (only if dumped)
    A violation cannot be a sampling/volume/units artefact — those scale every
    row identically — so it points to a data or physics bug (e.g. the wrong
    mass being written into a dump). Prints counts/fractions + worst offenders,
    and, if SMass is present, the component mass densities (which must sum to
    the baryonic density). Returns True if all checks pass.
    """
    print("--- Decomposition sanity check (per galaxy, from dumps) ---")
    bary  = load_raw(directory, "BMass_SG")   # M_stars + M_cold  (code units)
    bulge = load_raw(directory, "BMass_B")    # M_bulge
    stars = load_raw(directory, "SMass")      # M_stars (may be None)

    if bary is None or bulge is None:
        print("  Need BMass_SG.txt and BMass_B.txt to validate – skipping.")
        return True
    if len(bary) != len(bulge):
        print(f"  Row counts differ: BMass_SG={len(bary):,}, "
              f"BMass_B={len(bulge):,} – dumps are not aligned, cannot validate.")
        return True

    n = len(bary)
    rtol = 1e-6                      # relative tolerance for float round-off
    ok = True

    def report(mask, name, a, b, extra=""):
        nonlocal ok
        nv = int(np.count_nonzero(mask))
        if nv:
            ok = False
            w = np.argmax(a - b)     # largest absolute overshoot
            print(f"  [FAIL] {name} for {nv:,}/{n:,} ({100*nv/n:.2f}%) galaxies{extra}")
            print(f"         worst row: {a[w]:.3e} vs {b[w]:.3e} (code units, "
                  f"= {a[w]*1e10/hubble_h:.3e} vs {b[w]*1e10/hubble_h:.3e} Msun)")
        else:
            print(f"  [ok]   {name} holds for all {n:,} galaxies")

    # 1) bulge <= baryonic  (always available)
    report(bulge > bary * (1 + rtol), "M_bulge <= M_baryonic", bulge, bary)

    # 2) bulge <= stars  and  stars <= baryonic  (need SMass)
    if stars is not None and len(stars) == n:
        report(bulge > stars * (1 + rtol), "M_bulge <= M_stars", bulge, stars)
        report(stars > bary * (1 + rtol),  "M_stars <= M_baryonic", stars, bary,
               extra="  (a violation here means M_cold < 0)")

        # Component mass densities — these MUST sum to the baryonic density
        if volume:
            f = 1e10 / hubble_h / volume
            rho_b   = np.sum(bary)  * f
            rho_bul = np.sum(bulge) * f
            rho_dsk = np.sum(stars - bulge) * f          # stellar disk
            rho_cld = np.sum(bary - stars)  * f          # cold gas
            print(f"  mass densities [Msun (Mpc/h)^-3]:")
            print(f"     baryonic            = {rho_b:.3e}")
            print(f"     bulge + disk + cold = {rho_bul + rho_dsk + rho_cld:.3e}"
                  f"  (bulge {rho_bul:.2e} + disk {rho_dsk:.2e} + cold {rho_cld:.2e})")
            mismatch = abs(rho_bul + rho_dsk + rho_cld - rho_b) / rho_b
            print(f"     closure residual    = {mismatch:.2e} (should be ~0)")
    elif stars is not None:
        print(f"  (SMass.txt row count {len(stars):,} != {n:,}; "
              f"skipping stars-based checks)")
    else:
        print("  (no SMass.txt — checked M_bulge <= M_baryonic only; dump "
              "SMass for the full M_bulge <= M_stars <= M_baryonic check)")

    if not ok:
        print("  => violations found: the dumped masses are physically "
              "inconsistent, not just differently sampled.")
    print()
    return ok


def plot_baryonic_mass_function(baryonic_all, baryonic_cen, baryonic_sat,
                                 hubble_h, volume, save, out_dir,
                                 whichimf=1):
    """
    Faithful reproduction of SAGE's '2.BaryonicMassFunction' figure:
      - Bell et al. 2003 Schechter curve
      - Model (all galaxies, black solid)
      - Model - Centrals (blue dotted)        [if Type available]
      - Model - Satellites (green dashed)      [if Type available]
    All masses in Msun; volume in (Mpc/h)^3; binwidth fixed at 0.1 dex.
    """
    binwidth = 0.1
    fig, ax = plt.subplots(figsize=(8.34, 6.25))

    # ── Bell et al. 2003 BMF (h=1.0 converted to h=0.73), exactly as SAGE ──
    M = np.arange(7.0, 13.0, 0.01)
    Mstar = np.log10(5.3 * 1.0e10 / hubble_h / hubble_h)
    alpha = -1.21
    phistar = 0.0108 * hubble_h * hubble_h * hubble_h
    xval = 10.0 ** (M - Mstar)
    yval = np.log(10.) * phistar * xval ** (alpha + 1) * np.exp(-xval)
    if whichimf == 0:
        ax.plot(np.log10(10.0**M / 0.7), yval, 'g--', lw=1.5,
                label='Bell et al. 2003')
    else:
        ax.plot(np.log10(10.0**M / 0.7 / 1.8), yval, 'g--', lw=1.5,
                label='Bell et al. 2003')

    # ── Model curves ──
    cen, phi = _mass_function(baryonic_all, binwidth, volume)
    if cen.size:
        ax.plot(cen, phi, 'k-', lw=2.0, label='Model')

    if baryonic_cen is not None:
        cen_c, phi_c = _mass_function(baryonic_cen, binwidth, volume)
        if cen_c.size:
            ax.plot(cen_c, phi_c, 'b:', lw=2.5, label='Model - Centrals')
    if baryonic_sat is not None:
        cen_s, phi_s = _mass_function(baryonic_sat, binwidth, volume)
        if cen_s.size:
            ax.plot(cen_s, phi_s, 'g--', lw=1.2, label='Model - Satellites')

    ax.set_yscale('log')
    ax.axis([8.0, 12.2, 1.0e-6, 1.0e-1])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.set_ylabel(r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$')
    ax.set_xlabel(r'$\log_{10}\ M_{\mathrm{bar}}\ (M_{\odot})$')

    leg = ax.legend(loc='lower left', numpoints=1, labelspacing=0.1)
    leg.draw_frame(False)
    for t in leg.get_texts():
        t.set_fontsize('medium')

    plt.tight_layout()
    if save:
        path = os.path.join(out_dir, "BaryonicMassFunction.png")
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_mass_functions(baryonic_data, bulge_data, disk_s_data, disk_sg_data,
                       volume, n_bins, save, out_dir,
                       hubble_h=0.73, whichimf=1,
                       baryonic_cen=None, baryonic_sat=None):
    r"""
    Reproduce the SAGE baryonic mass function figure (Bell 2003 + Model +
    Centrals + Satellites) and overplot the bulge and disk mass functions
    so the decomposition can be compared directly against the baryonic MF.

    All mass inputs are in physical Msun; volume in (Mpc/h)^3.
    Uses the identical recipe as allresults: binwidth=0.1, phi=N/V/binwidth.
    """
    binwidth = 0.1
    fig, ax = plt.subplots(figsize=(8.34, 6.25))

    # ── Bell et al. 2003 reference (exact SAGE parametrisation) ──
    M = np.arange(7.0, 13.0, 0.01)
    Mstar = np.log10(5.3 * 1.0e10 / hubble_h / hubble_h)
    alpha = -1.21
    phistar = 0.0108 * hubble_h * hubble_h * hubble_h
    xval = 10.0 ** (M - Mstar)
    yval = np.log(10.) * phistar * xval ** (alpha + 1) * np.exp(-xval)
    if whichimf == 0:
        ax.plot(np.log10(10.0**M / 0.7), yval, 'g--', lw=1.5,
                label='Bell et al. 2003')
    else:
        ax.plot(np.log10(10.0**M / 0.7 / 1.8), yval, 'g--', lw=1.5,
                label='Bell et al. 2003')

    # ── Canonical baryonic MF (Model / Centrals / Satellites) ──
    cen, phi = _mass_function(baryonic_data, binwidth, volume)
    if cen.size:
        ax.plot(cen, phi, 'k-', lw=2.0, label='Model (baryonic)')
        print(f"    baryonic  : {len(baryonic_data):>6,} galaxies, "
              f"peak phi={phi.max():.2e} Mpc^-3 dex^-1")
    if baryonic_cen is not None:
        cen_c, phi_c = _mass_function(baryonic_cen, binwidth, volume)
        if cen_c.size:
            ax.plot(cen_c, phi_c, 'b:', lw=2.5, label='Model - Centrals')
    if baryonic_sat is not None:
        cen_s, phi_s = _mass_function(baryonic_sat, binwidth, volume)
        if cen_s.size:
            ax.plot(cen_s, phi_s, color='#228833', ls='--', lw=1.0,
                    label='Model - Satellites')

    # ── New decomposition: bulge and disk mass functions overplotted ──
    for data, colour, label in [
        (bulge_data,  '#EE7733', r'$M_{\rm bulge}$'),
        (disk_s_data, '#EE6677', r'$M_{\rm disk}=M_\star-M_{\rm bulge}$'),
        (disk_sg_data,'#AA3377', r'$M_{\rm disk}=M_\star+M_{\rm cold}-M_{\rm bulge}$'),
    ]:
        if data is None or len(data) == 0:
            continue
        c, p = _mass_function(data, binwidth, volume)
        if c.size:
            ax.plot(c, p, color=colour, lw=2.0, label=label)
            print(f"    {label:40s}: {len(data):>6,} galaxies, "
                  f"peak phi={p.max():.2e} Mpc^-3 dex^-1")

    ax.set_yscale('log')
    ax.axis([8.0, 12.2, 1.0e-6, 1.0e-1])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.set_ylabel(r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$')
    ax.set_xlabel(r'$\log_{10}\ M\ (M_{\odot})$')

    leg = ax.legend(loc='lower left', numpoints=1, labelspacing=0.1)
    leg.draw_frame(False)
    for t in leg.get_texts():
        t.set_fontsize('small')

    plt.tight_layout()
    if save:
        path = os.path.join(out_dir, "mass_functions_comparison.png")
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Figure 1: five-panel timescales (3 top, 2 bottom centred) ────────────────
def plot_five_panels(datasets, bins, log, save, out_dir):
    fig = plt.figure(figsize=(14, 9))
    gs_top = GridSpec(1, 3, figure=fig, left=0.06, right=0.97,
                      top=0.91, bottom=0.54, wspace=0.32)
    gs_bot = GridSpec(1, 2, figure=fig, left=0.20, right=0.83,
                      top=0.44, bottom=0.08, wspace=0.32)

    axes = (
        [fig.add_subplot(gs_top[0, i]) for i in range(3)] +
        [fig.add_subplot(gs_bot[0, i]) for i in range(2)]
    )

    for ax, key in zip(axes, KEYS):
        data = datasets[key]
        if data is None:
            ax.text(0.5, 0.5, f"{key}.txt\nnot found",
                    ha="center", va="center", transform=ax.transAxes, color="grey")
            ax.set_title(LABELS[key])
            continue

        ax.hist(data, bins=bins, color=COLOURS[key], alpha=0.85,
                edgecolor="white", linewidth=0.4)
        med = np.median(data)
        ax.axvline(med, color="k", ls="--", lw=1.2,
                   label=f"median = {med:.2g} Myr")
        ax.set_title(LABELS[key])
        ax.set_xlabel("Time (Myr)")
        ax.set_ylabel("Count")
        ax.legend(loc="upper right", fontsize=9)
        if log:
            apply_log_formatter(ax)

    if save:
        path = os.path.join(out_dir, "timescales_five_panels.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Figure 2: overlaid timescales ─────────────────────────────────────────────
def plot_overlay(datasets, bins, log, save, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for key in KEYS:
        data = datasets[key]
        if data is None:
            continue
        ax.hist(data, bins=bins, color=COLOURS[key], alpha=0.45,
                edgecolor=COLOURS[key], linewidth=0.6,
                label=LABELS[key], histtype="stepfilled")
        ax.hist(data, bins=bins, color=COLOURS[key],
                histtype="step", linewidth=1.5)
        ax.axvline(np.median(data), color=COLOURS[key],
                   ls=":", lw=1.4, alpha=0.9)

    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    if log:
        apply_log_formatter(ax)

    if save:
        path = os.path.join(out_dir, "timescales_overlay.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Figure 4: bulge vs disk tdyn overlaid ────────────────────────────────────
def plot_tdyn_comparison(datasets, bins, log, save, out_dir):
    """Dedicated comparison of bulge and disk dynamical times only."""
    compare_keys = ["tdyn_B_end", "tdyn_disk"]
    present = [k for k in compare_keys if datasets.get(k) is not None]
    if len(present) < 2:
        print("  Skipping tdyn comparison — need both tdyn_B_end and tdyn_disk.")
        return

    # Build bins from just these two distributions
    local_bins = make_bins([datasets[k] for k in present],
                           int((bins[-1] - bins[0]) / (bins[1] - bins[0])) if not log else 80,
                           log)

    fig, ax = plt.subplots(figsize=(7, 5))

    for key in present:
        data = datasets[key]
        ax.hist(data, bins=local_bins, color=COLOURS[key], alpha=0.55,
                edgecolor=COLOURS[key], linewidth=0.6,
                label=LABELS[key], histtype="stepfilled")
        ax.hist(data, bins=local_bins, color=COLOURS[key],
                histtype="step", linewidth=1.8)
        med = np.median(data)
        ax.axvline(med, color=COLOURS[key], ls="--", lw=1.4,
                   label=f"median = {med:.2g} Myr")

    ax.set_xlabel("Time (Myr)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    if log:
        apply_log_formatter(ax)

    if save:
        path = os.path.join(out_dir, "tdyn_comparison.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Figures 3 & 5: generic property panels (bulge / disk) ────────────────────
def plot_property_panels(datasets, keys, colours, labels, log, n_bins,
                         save, out_dir, out_name):
    """Generic 1xN histogram panel figure for galaxy structural properties.

    Keys ending in 'Radii' are treated as radii (kpc, plain tick labels);
    everything else is treated as a mass (Msun, scientific-notation ticks).
    """
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4.7 * n, 4.5))
    fig.subplots_adjust(wspace=0.35)
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        data = datasets[key]
        is_radius = key.endswith("Radii")

        if data is None:
            ax.text(0.5, 0.5, f"{key}.txt\nnot found",
                    ha="center", va="center", transform=ax.transAxes, color="grey")
            ax.set_title(labels[key])
            continue

        if log:
            bins = np.logspace(np.log10(data.min()), np.log10(data.max()), n_bins + 1)
        else:
            bins = np.linspace(data.min(), data.max(), n_bins + 1)

        ax.hist(data, bins=bins, color=colours[key], alpha=0.85,
                edgecolor="white", linewidth=0.4)

        med = np.median(data)
        if is_radius:
            med_label = f"median = {med:.2g} kpc"
        else:
            med_label = f"median = {med:.2e} " + r"$M_\odot$"
        ax.axvline(med, color="k", ls="--", lw=1.2, label=med_label)

        ax.set_title(labels[key])
        ax.set_xlabel(labels[key])
        ax.set_ylabel("Count")
        ax.legend(loc="upper right", fontsize=9)

        if log:
            ax.set_xscale("log")
            if is_radius:
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
                ax.xaxis.get_major_formatter().set_scientific(False)
            else:
                ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
                ax.xaxis.set_major_locator(ticker.LogLocator(numticks=6))

    if save:
        path = os.path.join(out_dir, out_name)
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Plot SAGE26 timescale, bulge, and disk property histograms.")
    parser.add_argument("--dir",   default=".",
                        help="Directory containing the .txt files")
    parser.add_argument("--units", choices=["code", "myr"], default="code",
                        help="Units for timescale files: 'code' or 'myr'")
    parser.add_argument("--log",   action="store_true",
                        help="Log10 x-axis on timescale plots")
    parser.add_argument("--save",  action="store_true",
                        help="Save figures to PNG (don't show interactively)")
    parser.add_argument("--bins",  type=int, default=80,
                        help="Number of histogram bins")
    parser.add_argument("--volume", type=float, default=None,
                        help="Simulation volume in (Mpc/h)^3 (if not provided, read from HDF5)")
    parser.add_argument("--snapshot", type=int, default=None,
                        help="Snapshot number for the HDF5 baryonic MF (default: last/z=0)")
    parser.add_argument("--hdf5-dir", type=str, default=None, dest="hdf5_dir",
                        help="Directory holding the model_*.hdf5 outputs "
                             "(default: same as --dir; searched recursively)")
    parser.add_argument("--imf", type=int, choices=[0, 1], default=1,
                        help="IMF for Bell 2003 conversion: 0=Salpeter, 1=Chabrier [default: 1]")
    args = parser.parse_args()

    whichimf = args.imf

    print(f"Loading files from: {os.path.abspath(args.dir)}")
    print(f"Timescale units: {args.units}  →  Myr")
    print(f"Log x-axis (timescales): {args.log},  bins: {args.bins}")
    print()

    # ── Timescale files ───────────────────────────────────────────────────────
    print("--- Timescales ---")
    datasets = {}
    for key in KEYS:
        raw = load_file(args.dir, key)
        if raw is not None:
            datasets[key] = convert_to_myr(raw, args.units)
            print(f"  {key:12s}: {len(datasets[key]):>8,} values  "
                  f"[{datasets[key].min():.3e}, {datasets[key].max():.3e}] Myr  "
                  f"median = {np.median(datasets[key]):.3e} Myr")
        else:
            datasets[key] = None
    print()

    ts_present = [d for d in datasets.values() if d is not None]
    if ts_present:
        bins = make_bins(ts_present, args.bins, args.log)
        plot_five_panels(datasets, bins, args.log, args.save, args.dir)
        plot_overlay(datasets, bins, args.log, args.save, args.dir)
        plot_tdyn_comparison(datasets, bins, args.log, args.save, args.dir)
    else:
        print("  No timescale files found – skipping timescale figures.")

    # ── Bulge property files ──────────────────────────────────────────────────
    print("--- Bulge properties ---")
    bulge_datasets = {}
    for key in BULGE_KEYS:
        raw = load_file(args.dir, key)
        if raw is not None:
            if key == "BRadii":
                raw = raw * RADII_TO_KPC
            else:
                raw = raw * MASS_TO_MSUN
            bulge_datasets[key] = raw
            unit_str = "kpc" if key == "BRadii" else "Msun"
            print(f"  {key:10s}: {len(raw):>8,} values  "
                  f"[{raw.min():.3e}, {raw.max():.3e}] {unit_str}  "
                  f"median = {np.median(raw):.3e} {unit_str}")
        else:
            bulge_datasets[key] = None
    print()

    if any(d is not None for d in bulge_datasets.values()):
        plot_property_panels(bulge_datasets, BULGE_KEYS, BULGE_COLOURS,
                             BULGE_LABELS, True, args.bins, args.save,
                             args.dir, "bulge_properties.png")
    else:
        print("  No bulge property files found – skipping figure 3.")

    # ── Disk property files ───────────────────────────────────────────────────
    print("--- Disk properties ---")
    disk_datasets = {}
    for key in DISK_KEYS:
        if key == "DRadii":
            raw = load_file(args.dir, key)
            if raw is not None:
                raw = raw * RADII_TO_KPC
        else:
            raw = load_disk_mass(args.dir, key)
            if raw is not None:
                raw = raw * MASS_TO_MSUN

        disk_datasets[key] = raw
        if raw is not None:
            unit_str = "kpc" if key == "DRadii" else "Msun"
            print(f"  {key:10s}: {len(raw):>8,} values  "
                  f"[{raw.min():.3e}, {raw.max():.3e}] {unit_str}  "
                  f"median = {np.median(raw):.3e} {unit_str}")
    print()

    if any(d is not None for d in disk_datasets.values()):
        plot_property_panels(disk_datasets, DISK_KEYS, DISK_COLOURS,
                             DISK_LABELS, True, args.bins, args.save,
                             args.dir, "disk_properties.png")
    else:
        print("  No disk property files found – skipping disk figure.")

    # ── Mass function comparison ──────────────────────────────────────────────
    print("--- Mass function comparison ---")

    # Text-dump-derived masses (physical Msun): used for the M*+Mcold validation
    # overlay and as a fallback when no HDF5 is available.
    baryonic_dump = bulge_datasets.get('BMass_SG', None)
    bulge_dump    = bulge_datasets.get('BMass_B', None)
    disk_s_dump   = disk_datasets.get('DMass_S', None)
    disk_sg_dump  = disk_datasets.get('DMass_SG', None)

    # Prefer HDF5 for the canonical baryonic MF (gives Type split + exact h/volume)
    hdf5_search_dir = args.hdf5_dir if args.hdf5_dir is not None else args.dir
    hdf5_bary = read_baryonic_by_type_from_hdf5(hdf5_search_dir,
                                                snapshot=args.snapshot)

    if hdf5_bary is not None:
        volume = args.volume if args.volume is not None else hdf5_bary['volume']
        hubble_h = hdf5_bary['hubble_h']
        baryonic_all = hdf5_bary['all']
        baryonic_cen = hdf5_bary['cen']
        baryonic_sat = hdf5_bary['sat']
        print(f"  Read masses from HDF5 ({hdf5_bary['snapshot']}): "
              f"{len(baryonic_all):,} galaxies")
        print(f"  Volume = {volume:.3e} (Mpc/h)^3,  Hubble_h = {hubble_h}")
        if baryonic_cen is None:
            print("  (no Type field found – skipping centrals/satellites split)")

        # Bulge & disk from the SAME HDF5 catalogue (single consistent sample)
        if hdf5_bary['bulge'] is not None:
            bulge_mass   = hdf5_bary['bulge']
            disk_s_mass  = hdf5_bary['disk_s']
            disk_sg_mass = hdf5_bary['disk_sg']
            print(f"  Bulge/disk from HDF5 BulgeMass: bulge={len(bulge_mass):,}, "
                  f"disk_s={len(disk_s_mass):,}, disk_sg={len(disk_sg_mass):,}")
        else:
            bulge_mass, disk_s_mass, disk_sg_mass = bulge_dump, disk_s_dump, disk_sg_dump
            print("  (no BulgeMass in HDF5 – bulge/disk fall back to text dumps)")
    else:
        # Fall back entirely to text dumps; need a volume to normalise
        volume = args.volume
        hubble_h = HUBBLE_h
        baryonic_all = baryonic_dump
        baryonic_cen = None
        baryonic_sat = None
        bulge_mass, disk_s_mass, disk_sg_mass = bulge_dump, disk_s_dump, disk_sg_dump
        if h5py is None:
            print("  h5py not installed – cannot read HDF5 (install with: "
                  "pip install h5py).")
        else:
            print(f"  No model_*.hdf5 found under: "
                  f"{os.path.abspath(hdf5_search_dir)}")
            print("  (point --hdf5-dir at your output/<sim>/ folder for the "
                  "Centrals/Satellites split and auto volume)")
        if volume is None and baryonic_all is None:
            print("  Also no BMass_SG.txt dump present; nothing to plot.")
            print("  Provide --volume V (in (Mpc/h)^3) and/or --hdf5-dir.")
        elif volume is None:
            print("  Have text-dump baryonic mass but no volume to normalise.")
            print("  Use --volume V (in (Mpc/h)^3) to enable the mass function plot.")
        else:
            print(f"  Using text-dump masses, "
                  f"volume = {volume:.3e} (Mpc/h)^3")

    have_any = any(d is not None and len(d) > 0
                   for d in (baryonic_all, bulge_mass, disk_s_mass, disk_sg_mass))

    # Physical sanity check on the dumps before plotting anything
    validate_decomposition(args.dir, volume=volume, hubble_h=hubble_h)

    if volume is not None and have_any:
        print("  Plotting baryonic mass function (SAGE style) + decomposition...")
        # 1) Faithful reproduction of the SAGE figure on its own
        if baryonic_all is not None and len(baryonic_all) > 0:
            plot_baryonic_mass_function(
                baryonic_all, baryonic_cen, baryonic_sat,
                hubble_h, volume, args.save, args.dir, whichimf=whichimf)
        # 2) Same figure with bulge/disk MFs overplotted for comparison
        plot_mass_functions(
            baryonic_all, bulge_mass, disk_s_mass, disk_sg_mass,
            volume, args.bins, args.save, args.dir,
            hubble_h=hubble_h, whichimf=whichimf,
            baryonic_cen=baryonic_cen, baryonic_sat=baryonic_sat)
    elif volume is not None:
        print("  Insufficient data for mass function plot.")
    print()

    print("Done.")


if __name__ == "__main__":
    main()