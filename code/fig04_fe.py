"""Fig. 4 - FE controlled validation: model-exact coverage, linearisation residuals, nonlinear validation."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import figstyle as fs

fs.setup()
R = fs.ROUND / 'results'
A = pd.read_csv(R / 'fe' / 'fe_linear_grid.csv')
B = pd.read_csv(R / 'fe' / 'fe_nonlinear_truth.csv')
res = pd.read_csv(R / 'tables' / 'fe_residual_by_mode.csv')
rng = np.random.default_rng(3)

fig = plt.figure(figsize=(fs.W2, 124 * fs.MM))
axA = fig.add_axes([0.07, 0.635, 0.56, 0.30])
axB = fig.add_axes([0.72, 0.635, 0.27, 0.30])
axC = [fig.add_axes([0.07 + i * 0.31, 0.085, 0.27, 0.31]) for i in range(3)]

# -------------------------------------------- (a) model-exact regime ----------
cells = [(v, m) for v in ('freq', 'shape', 'both') for m in (1.0, 3.0, 10.0)]
for i, (v, m) in enumerate(cells):
    d = A[(A.variant == v) & (A.noise_mult == m)]
    for off, tag, col, mk in ((-0.17, 'naive', fs.ORANGE, 'o'), (0.17, 'cert', fs.BLUE, '^')):
        y = d[f'cov_worst_{tag}'].to_numpy()
        x = i + off + rng.uniform(-0.07, 0.07, len(y))
        axA.plot(x, y, ls='', marker=mk, ms=2.6, mfc=col, mec='white', mew=0.3, alpha=0.9)
    n_rep = int((d.state == 'REPORT').sum())
    axA.text(i, -0.17, f'{n_rep}/{int((d.state == "BOUND").sum())}/{int((d.state == "REFUSE").sum())}',
             ha='center', va='top', fontsize=5.6, color=fs.INK2)
axA.axhline(0.95, color=fs.INK2, lw=0.6)
axA.text(8.55, 0.95, '0.95', fontsize=5.8, color=fs.INK2, va='center', ha='left')
axA.set_xticks(range(9))
axA.set_xticklabels([f'x{int(m)}' for _, m in cells], fontsize=6.2)
for j, v in enumerate(('frequencies only', 'mode shapes only', 'both')):
    axA.text(1 + 3 * j, -0.33, v, ha='center', va='top', fontsize=6.4, color=fs.INK)
    if j:
        axA.axvline(3 * j - 0.5, color=fs.GRID, lw=0.8)
axA.text(-0.6, -0.17, 'R/B/X', ha='right', va='top', fontsize=5.6, color=fs.INK2)
axA.set_xlim(-0.6, 8.9)
axA.set_ylim(-0.02, 1.04)
axA.set_ylabel('coverage at worst-case vertex')
from matplotlib.lines import Line2D
axA.legend(handles=[Line2D([], [], ls='', marker='o', mfc=fs.ORANGE, mec='white', label='naive (noise-only)'),
                    Line2D([], [], ls='', marker='^', mfc=fs.BLUE, mec='white', label='certified')],
           loc='lower left', ncol=2, fontsize=6.0)
fs.hgrid(axA)
fs.panel(axA, 'a', 'Model exact: 270 layout-query cells, noise x1/x3/x10')

# ----------------------------------------- (b) linearisation residuals --------
m = res['mode'].to_numpy()
axB.semilogy(m - 0.12, res.max_freq.clip(lower=1e-2), ls='', marker='o', ms=3.4, color=fs.BLUE,
             label='frequency')
axB.semilogy(m + 0.12, res.max_shape.clip(lower=1e-2), ls='', marker='s', ms=3.4, color=fs.ORANGE,
             label='mode shape')
for _, r in res.iterrows():
    if r.n_swaps > 1:
        axB.text(r['mode'], max(r.max_freq, r.max_shape) * 2.0, f'{int(r.n_swaps)}', ha='center',
                 fontsize=5.6, color=fs.INK2)
axB.axhline(1.0, color=fs.INK2, lw=0.6)
axB.set_xticks(range(1, 15))
axB.set_xticklabels([str(i) for i in range(1, 15)], fontsize=5.8)
axB.set_xlabel('tracked mode')
axB.set_ylabel(r'max |residual| / noise sd')
axB.set_ylim(1e-2, 5e3)
axB.legend(loc='upper left', fontsize=6.0)
axB.text(14.4, 2e-2, 'numbers: distinct FE modes\nmatched across solutions', ha='right', va='bottom',
         fontsize=5.4, color=fs.MUTED)
fs.hgrid(axB)
fs.panel(axB, 'b', 'Linearisation error, 29 FE solutions')

# ------------------------------------------- (c) nonlinear validation at corners ---
rules = [('naive', 'naive', fs.ORANGE, 'o'), ('blind', 'model-error blind', fs.AQUA, 's'),
         ('axis', 'single-axis envelope', fs.YELLOW, 'D'), ('loo', 'combined-design envelope', fs.BLUE, '^')]
for ax, (v, vt) in zip(axC, (('freq', 'frequencies only'), ('shape', 'mode shapes only'),
                             ('both', 'frequencies + shapes'))):
    d = B[(B.variant == v) & (B.set == 'corner')]
    ticks = []
    for i, (tag, lab, col, mk) in enumerate(rules):
        y = d[f'cov_{tag}'].to_numpy()
        x = i + rng.uniform(-0.22, 0.22, len(y))
        ax.plot(x, y, ls='', marker=mk, ms=1.8, mfc=col, mec='none', alpha=0.45)
        ax.plot([i - 0.3, i + 0.3], [np.mean(y)] * 2, color=fs.INK, lw=1.2)
        w = float((d[f'w_{tag}'] / d.w0).median())
        ticks.append(f"{['naive', 'blind', 'single-axis', 'combined'][i]}\n{w:.2f}")
    ax.axhline(0.95, color=fs.INK2, lw=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(ticks, fontsize=5.8)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlim(-0.6, 3.6)
    fs.hgrid(ax)
    ax.set_title(vt, fontsize=6.8, fontweight='normal', loc='center', pad=2)
axC[0].set_ylabel('coverage at the 13 box corners')
axC[0].text(-0.75, -0.115, 'median\nw/w0', ha='right', va='top', fontsize=5.4, color=fs.MUTED,
            transform=axC[0].transData)
fs.panel(axC[0], 'c', 'Nonlinear FE validation, out of sample: interval rule (black bar = mean) x observation set',
         y=1.12)
fs.save(fig, 'fig04_fe')
