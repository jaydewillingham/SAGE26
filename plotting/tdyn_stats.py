"""
Bulge properties plotter
------------------------
Reads a whitespace-delimited text file with three columns:
    r_bulge   bulge_mass   t_dyn

Produces a two-panel figure:
  Left  – scatter: bulge mass vs bulge radius, coloured by t_dyn (log scale)
  Right – histogram of r_bulge values (resolved only)

Also prints: how many galaxies have r_bulge = 0 with M_bulge > 0.
"""

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import LogLocator, NullFormatter

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    # figure
    "figure.figsize"         : (8.34, 6.25),
    "figure.dpi"             : 140,
    "figure.autolayout"      : True,
    # x ticks
    "xtick.major.size"       : 7.5,
    "xtick.major.width"      : 1.5,
    "xtick.minor.size"       : 5.5,
    "xtick.minor.width"      : 0.5,
    "xtick.direction"        : "in",
    "xtick.top"              : True,
    "xtick.labelsize"        : 16,
    "xtick.major.pad"        : 9,
    # y ticks
    "ytick.major.size"       : 7.5,
    "ytick.major.width"      : 1.5,
    "ytick.minor.size"       : 5.5,
    "ytick.minor.width"      : 0.5,
    "ytick.direction"        : "in",
    "ytick.right"            : True,
    "ytick.labelsize"        : 16,
    # axes
    "axes.linewidth"         : 1.5,
    "axes.labelsize"         : 20,
    "axes.titlesize"         : 12,
    "axes.prop_cycle"        : mpl.cycler("color", [
                                "0C5DA5","00B945","FF9500",
                                "FF2C00","845B97","474747","9e9e9e"]),
    # legend
    "legend.fontsize"        : 14,
    "legend.title_fontsize"  : 16,
    "legend.frameon"         : False,
    # lines
    "grid.linewidth"         : 1,
    "lines.linewidth"        : 2,
    "lines.solid_capstyle"   : "round",
    # font / LaTeX  (mathpazo = Palatino for math + text)
    "font.family"            : "serif",
    "font.size"              : 20.0,
    "text.usetex"            : True,
    "text.latex.preamble"    : r"\usepackage{mathpazo}",
})


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(filepath):
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data[:, 0], data[:, 1], data[:, 2]   # r_bulge, M_bulge, t_dyn


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(r_bulge, bulge_mass, t_dyn):

    mask_pos  = r_bulge > 0
    mask_zero = ~mask_pos
    mask_mpos   = bulge_mass > 0
    mask_zero_mpos = ~mask_mpos

    # ── stats printout ───────────────────────────────────────────────────────
    n_zero_mpos = np.sum(mask_zero & mask_mpos)
    n_total     = len(r_bulge)
    print(f"\n  r_bulge = 0  AND  M_bulge > 0 : {n_zero_mpos} / {n_total}\n")
    print(f"\n  M_bulge = 0 : {mask_zero_mpos.sum()} / {n_total}\n")

    # ── colour mapping ───────────────────────────────────────────────────────
    t_pos = t_dyn[t_dyn > 0]
    vmin, vmax = t_pos.min(), t_dyn.max()
    norm  = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap  = plt.cm.viridis

    # ── layout ───────────────────────────────────────────────────────────────
    fig, (ax_sc, ax_hi) = plt.subplots(
        1, 2,
        figsize=(16, 6.25),
        gridspec_kw={"width_ratios": [1.6, 1]},
    )

    # ════════════════════════════════════════════════════════════════════════
    # Left panel – scatter
    # ════════════════════════════════════════════════════════════════════════
    sc = ax_sc.scatter(
        r_bulge[mask_pos],
        bulge_mass[mask_pos],
        c=t_dyn[mask_pos],
        cmap=cmap, norm=norm,
        s=70, edgecolors="k", linewidths=0.5, zorder=3,
        label=r"resolved ($r_\mathrm{bulge} > 0$)",
    )

    if mask_zero.any():
        ax_sc.scatter(
            np.zeros(mask_zero.sum()),        # jitter=0, all sit at x=0
            bulge_mass[mask_zero],
            c=t_dyn[mask_zero],
            cmap=cmap, norm=norm,
            s=70, marker="v", edgecolors="k", linewidths=0.5, zorder=3,
            label=r"unresolved ($r_\mathrm{bulge} = 0$)",
        )

    ax_sc.set_xlabel(r"$r_\mathrm{bulge}$")
    ax_sc.set_ylabel(r"$M_\mathrm{bulge}$")
    ax_sc.set_title(
        r"Bulge mass vs.\ radius, colour-coded by $t_\mathrm{dyn}$",
        pad=8,
    )
    ax_sc.legend()
    ax_sc.minorticks_on()

    cbar = fig.colorbar(sc, ax=ax_sc, pad=0.02)
    cbar.set_label(r"$t_\mathrm{dyn}$")
    cbar.ax.yaxis.set_minor_locator(LogLocator(subs="auto"))
    cbar.minorticks_on()

    # ════════════════════════════════════════════════════════════════════════
    # Right panel – histogram of r_bulge (resolved only)
    # ════════════════════════════════════════════════════════════════════════
    r_pos = r_bulge[mask_pos]

    if len(r_pos) > 1:
        lo, hi = r_pos.min(), r_pos.max()
        if hi / lo > 10:
            bins = np.logspace(np.log10(lo), np.log10(hi), 15)
            ax_hi.set_xscale("log")
            ax_hi.xaxis.set_minor_locator(LogLocator(subs="auto"))
            ax_hi.xaxis.set_minor_formatter(NullFormatter())
        else:
            bins = 12
    else:
        bins = 5

    ax_hi.hist(
        r_pos,
        bins=bins,
        color="#0C5DA5",
        edgecolor="k",
        linewidth=0.7,
        alpha=0.85,
    )

    # annotate r=0 count
    if mask_zero.any():
        ax_hi.text(
            0.03, 0.95,
            rf"$r_\mathrm{{bulge}}=0$: {mask_zero.sum()}/{n_total}",
            transform=ax_hi.transAxes,
            ha="left", va="top",
            fontsize=13,
            color="#FF2C00",
        )

    ax_hi.set_xlabel(r"$r_\mathrm{bulge}$")
    ax_hi.set_ylabel(r"$N$")
    ax_hi.set_title(r"Distribution of $r_\mathrm{bulge}$", pad=8)
    ax_hi.minorticks_on()

    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "bulge_data.txt"

    print(f"Loading data from: {filepath}")
    r_bulge, bulge_mass, t_dyn = load_data(filepath)
    print(f"  {len(r_bulge)} rows loaded")

    fig = make_figure(r_bulge, bulge_mass, t_dyn)

    outfile = filepath.rsplit(".", 1)[0] + "_plot.png"
    fig.savefig(outfile, dpi=140, bbox_inches="tight")
    print(f"  Figure saved → {outfile}")
    plt.show()