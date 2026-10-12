"""Fig. 6 - KW51: certified queries over signal-to-noise ratio and discrepancy fraction."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import figstyle as fs

fs.setup()
R = fs.ROUND / 'results' / 'kw51'
ph = pd.read_csv(R / 'kw51_phase_drift_snr.csv')
cr = pd.read_csv(R / 'kw51_critical_drift.csv')

fig = plt.figure(figsize=(fs.W2, 70 * fs.MM))
axs = [fig.add_axes([0.07, 0.17, 0.26, 0.68]), fig.add_axes([0.37, 0.17, 0.26, 0.68])]
cax = fig.add_axes([0.645, 0.17, 0.010, 0.68])
axC = fig.add_axes([0.74, 0.17, 0.25, 0.68])

rhos = sorted(ph.rho.unique())
snrs = sorted(ph.snr.unique())
bounds = np.arange(-0.5, 8.5, 1)
cmap = ListedColormap(['#f7f6f2'] + fs.SEQ[2::1][:7])
norm = BoundaryNorm(bounds, cmap.N)
for ax, (m, title) in zip(axs, ((0, 'deck only (6 channels)'), (63, 'full array (12 channels)'))):
    Mx = np.zeros((len(snrs), len(rhos)))
    for i, s in enumerate(snrs):
        for j, r in enumerate(rhos):
            Mx[i, j] = ph[(ph['mask'] == m) & (ph.snr == s) & (ph.rho == r)].n_certified.iloc[0]
    im = ax.imshow(Mx, origin='lower', cmap=cmap, norm=norm, aspect='auto')
    for i in range(len(snrs)):
        for j in range(len(rhos)):
            v = int(Mx[i, j])
            ax.text(j, i, str(v), ha='center', va='center', fontsize=5.8,
                    color='white' if v >= 4 else fs.INK)
    ax.set_xticks(range(len(rhos)))
    ax.set_xticklabels([f'{r:g}' for r in rhos], fontsize=5.8, rotation=0)
    ax.set_yticks(range(len(snrs)))
    ax.set_yticklabels([f'{s:g}' for s in snrs], fontsize=6.0)
    ax.set_xlabel(r'discrepancy fraction $\rho$ of the monthly envelope', fontsize=6.4)
    ref = snrs.index(50)
    ax.add_patch(plt.Rectangle((len(rhos) - 1.5, ref - 0.5), 1, 1, fill=False, ec=fs.ORANGE, lw=1.4))
    ax.add_patch(plt.Rectangle((-0.5, ref - 0.5), 1, 1, fill=False, ec=fs.ORANGE, lw=1.4, ls='-'))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=6.8, fontweight='normal', loc='center', pad=3)
axs[0].set_ylabel(r'signal-to-noise ratio $b/\sigma$')
cb = fig.colorbar(im, cax=cax, ticks=range(8))
cb.set_label(r'certified queries ($w^{*}\leq 0.5\,w_0$)', fontsize=6.2)
cb.ax.tick_params(labelsize=5.8, length=0)
cb.outline.set_visible(False)
fs.panel(axs[0], 'a', None, y=1.08)
fs.panel(axs[1], 'b', None, y=1.08)
fig.text(0.07, 0.955, 'Certified modal coordinates over noise and discrepancy (orange: reference SNR, exact and measured)',
         fontsize=7.2, fontweight='bold', va='bottom')

# --------------------------------------------- (c) critical discrepancy ------
d = cr[cr['mask'] == 63].set_index('mode')
modes = np.arange(1, 15)
info = d.loc[modes, 'rho_info'].to_numpy()
cert = d.loc[modes, 'rho_cert'].to_numpy()
axC.barh(modes, info, height=0.7, color=fs.SEQ[3], edgecolor='white', lw=0.5, label='informative ($w^*<w_0$)')
axC.barh(modes, cert, height=0.7, color=fs.SEQ[10], edgecolor='white', lw=0.5,
         label=r'certified ($w^*\leq 0.5\,w_0$)')
for mm, a, c in zip(modes, info, cert):
    if a > 0:
        axC.text(a + 0.02, mm, f'{a:.2f}', va='center', fontsize=5.6, color=fs.INK2)
axC.set_yticks(modes)
axC.set_yticklabels([str(m) for m in modes], fontsize=5.8)
axC.invert_yaxis()
axC.set_xlim(0, 1.15)
axC.set_xlabel(r'largest admissible $\rho$ (full array, $b/\sigma=50$)', fontsize=6.4)
axC.set_ylabel('tracked mode')
axC.legend(loc='lower right', fontsize=5.8)
axC.xaxis.grid(True, color=fs.GRID, lw=0.5)
axC.set_axisbelow(True)
fs.panel(axC, 'c', 'Discrepancy each query tolerates', y=1.02)
fs.save(fig, 'fig06_kw51_phase')
