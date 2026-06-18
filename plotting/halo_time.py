import numpy as np
import matplotlib
matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Palatino'],
    'axes.formatter.use_mathtext': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'legend.frameon': False,
})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# Unit conversion
# =============================================================================
UNIT_LENGTH_CM     = 3.08568e24
UNIT_VELOCITY_CM_S = 1.0e5
UNIT_TIME_S        = UNIT_LENGTH_CM / UNIT_VELOCITY_CM_S
SEC_PER_MYR        = 3.155e13
UNIT_TIME_IN_MYR   = UNIT_TIME_S / SEC_PER_MYR   # ≈ 978 029 Myr/code unit

# =============================================================================
# Load data
# =============================================================================
DATA_FILE = './timescales/frac_halotime.txt'

data_code = np.loadtxt(DATA_FILE)
data_myr  = data_code * UNIT_TIME_IN_MYR

# =============================================================================
# Summary statistics
# =============================================================================
med = np.median(data_myr)
mn  = data_myr.mean()
q1  = np.percentile(data_myr, 25)
q3  = np.percentile(data_myr, 75)

print(f"N          = {len(data_myr)}")
print(f"Unit time  = {UNIT_TIME_IN_MYR:.1f} Myr / code unit")
print(f"Min        = {data_myr.min():.1f} Myr")
print(f"Max        = {data_myr.max():.1f} Myr")
print(f"Mean       = {mn:.1f} Myr")
print(f"Median     = {med:.1f} Myr")
print(f"Std        = {data_myr.std():.1f} Myr")
print(f"Q1         = {q1:.1f} Myr")
print(f"Q3         = {q3:.1f} Myr")
print(f"IQR        = {q3 - q1:.1f} Myr")

# =============================================================================
# Colours  (indices 0 and 2 from the standard 7-colour hex cycle)
# =============================================================================
C_BLUE  = '#4C72B0'
C_GREEN = '#55A868'
C_ORG   = '#DD8452'

# =============================================================================
# Figure: stacked histogram (top) + box plot (bottom), sharing the x-axis
# =============================================================================
fig = plt.figure(figsize=(10, 7))
gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)

ax_hist = fig.add_subplot(gs[0])
ax_box  = fig.add_subplot(gs[1], sharex=ax_hist)

# --- Histogram ---
bins = np.linspace(data_myr.min(), data_myr.max(), 30)
ax_hist.hist(data_myr, bins=bins,
             color=C_BLUE, alpha=0.80,
             edgecolor='white', linewidth=0.4)

ax_hist.axvline(med, color=C_ORG,   lw=1.8, ls='--',
                label=fr'Median $= {med:.0f}$ Myr')
ax_hist.axvline(mn,  color=C_GREEN, lw=1.8, ls=':',
                label=fr'Mean $= {mn:.0f}$ Myr')
ax_hist.axvspan(q1, q3, alpha=0.15, color=C_BLUE,
                label=fr'IQR [{q1:.0f}--{q3:.0f}] Myr')

ax_hist.set_ylabel(r'Count', fontsize=12)
ax_hist.legend(fontsize=10)
ax_hist.tick_params(labelbottom=False)

# --- Box plot ---
ax_box.boxplot(data_myr, vert=False, widths=0.55, patch_artist=True,
               medianprops=dict(color=C_ORG, lw=2.0),
               boxprops=dict(facecolor=C_BLUE, alpha=0.6, linewidth=1.2),
               whiskerprops=dict(linewidth=1.2),
               capprops=dict(linewidth=1.2),
               flierprops=dict(marker='.', markersize=2, alpha=0.3,
                               markerfacecolor=C_BLUE, markeredgecolor='none'))

# 5th / 95th percentile guide lines
for pct, label in [(5, r'5\%'), (95, r'95\%')]:
    v = np.percentile(data_myr, pct)
    ax_box.axvline(v, color='grey', lw=0.8, ls=':')
    ax_box.text(v, 1.42, label, ha='center', va='bottom',
                fontsize=7.5, color='grey')

ax_box.set_xlabel(r'0.1 * HaloTime [Myr]', fontsize=12)
ax_box.set_yticks([])

# =============================================================================
# Save
# =============================================================================
OUT = 'frac_halotime_dist'
#plt.savefig(f'{OUT}.pdf', bbox_inches='tight', dpi=200)
plt.savefig(f'{OUT}.png', bbox_inches='tight', dpi=200)
print(f"\nSaved {OUT}.pdf / .png")