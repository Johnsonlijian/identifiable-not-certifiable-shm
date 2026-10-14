"""Fig. 1 - framework: protocol, break-even criterion, error budget of three field queries."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import figstyle as fs

fs.setup()
Z = 1.959963984540054
enum = pd.read_csv(fs.ROUND / 'results' / 'kw51' / 'kw51_enumeration.csv')
e = enum[(enum.snr == 50.0) & (enum['mask'] == 63)].set_index('mode')

fig = plt.figure(figsize=(fs.W2, 78 * fs.MM))
axA = fig.add_axes([0.0, 0.0, 0.525, 0.93])
axB = fig.add_axes([0.63, 0.60, 0.35, 0.32])
axC = fig.add_axes([0.63, 0.08, 0.35, 0.33])

# ------------------------------------------------------------ (a) protocol ---
axA.set_xlim(0, 100)
axA.set_ylim(0, 100)
axA.axis('off')
fs.panel(axA, 'a', 'Query-level certification protocol', x=0.0, y=0.985)


def box(ax, x, y, w, h, title, lines, fc='#f4f8fd', ec=fs.AXIS, tc=fs.INK, lw=0.7, step=6.6, top=10.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=1.6',
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + 2, y + h - 3.2, title, ha='left', va='top', fontsize=7.0, fontweight='bold', color=tc)
    for i, ln in enumerate(lines):
        ax.text(x + 2, y + h - top - step * i, ln, ha='left', va='top', fontsize=6.4, color=fs.INK2)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7,
                                 lw=0.8, color=fs.INK2, shrinkA=0, shrinkB=0))


box(axA, 0, 50, 28, 42, 'Declared inputs',
    [r'layout sensitivity $\mathbf{S}_L$', r'noise covariance $\Sigma$',
     r'prior box $|\theta_i|\leq b$', r'discrepancy bound $\rho$', r'query $\mathbf{q}$ (scalar $\mathbf{q}^{\top}\theta$)'])
box(axA, 34, 74, 66, 18, 'Level 1  rank: is q estimable at all?',
    [r'$\mathbf{q}\in\mathrm{row}(\mathbf{S}_L)$;  rank floor $b\,d_1(\mathbf{q},\mathrm{row}\,\mathbf{S}_L)$'])
box(axA, 34, 51.5, 66, 18.5, 'Level 2  noise: which directions pass screening?',
    [r'Fisher split at $\lambda^{*}=(z/b)^2$;  $p_{\mathrm{null}}(\mathbf{q})\rightarrow$ state'])
box(axA, 34, 25, 66, 22.5, 'Level 3  certified: how precise is the interval?',
    [r'$w^{*}=\min_{\mathbf{a}}\ z\|\mathbf{a}\|_2+b\|\tilde{\mathbf{S}}^{\top}\mathbf{a}-\mathbf{q}\|_1+\tilde{\rho}^{\top}|\mathbf{a}|$',
     'budget = noise + aliasing + discrepancy'])
arrow(axA, 28, 72, 34, 83)
arrow(axA, 67, 74, 67, 70)
arrow(axA, 67, 51.5, 67, 47.5)
# outputs
ow, oy, oh = 31.5, 0.5, 20.5
outs = [('REPORT', ['estimate $\\pm\\,w^{*}$', 'state $p_{\\mathrm{null}}\\leq\\tau_1$']),
        ('BOUND', ['estimate $\\pm\\,w^{*}$', '$\\tau_1<p_{\\mathrm{null}}\\leq\\tau_2$']),
        ('REFUSE', ['prior interval', '+ limiting term'])]
for i, (st, ln) in enumerate(outs):
    x = 1.5 + i * (ow + 2)
    box(axA, x, oy, ow, oh, st, ln, fc='white', ec=fs.STATE[st], tc=fs.INK, lw=1.4, step=5.6, top=9.0)
for i in range(3):
    arrow(axA, 67, 25, 1.5 + i * (ow + 2) + ow / 2, oy + oh + 0.3)
axA.text(14, 47.5, 'publish each scalar\nwith its certificate', ha='center', va='top',
         fontsize=6.2, color=fs.MUTED, style='italic')

# ------------------------------------------------------- (b) break-even criterion --
b = 0.1
lam = np.logspace(0, 6, 400)
noise_only = Z / (b * np.sqrt(lam))
cert = np.minimum(1.0, noise_only)
axB.loglog(lam, noise_only, color=fs.MUTED, lw=0.9, label='noise-only half-width / b')
axB.loglog(lam, cert, color=fs.BLUE, lw=1.6, label=r'certified $w^{*}/b$')
ls = (Z / b) ** 2
axB.axvline(ls, color=fs.INK2, lw=0.6)
axB.text(ls * 1.25, 0.012, r'$\lambda^{*}=(z/b)^2$', fontsize=6.4, color=fs.INK2, va='bottom')
axB.plot([1], [Z / b], marker='o', ms=3.5, color=fs.ORANGE)
axB.annotate('unit eigenvalue floor ($\\lambda=1$):\nhalf-width 19.6 b', xy=(1, Z / b), xytext=(1.4, 0.18),
             fontsize=6.2, color=fs.ORANGE, arrowprops=dict(arrowstyle='-', lw=0.6, color=fs.ORANGE))
axB.fill_betweenx([0.003, 40], 1, ls, color='#f4f2ee', zorder=0)
axB.text(20, 0.03, 'prior wins', fontsize=6.2, color=fs.MUTED, ha='center')
axB.text(4e4, 0.32, 'measurement wins', fontsize=6.2, color=fs.MUTED, ha='center')
axB.set_xlim(1, 1e6)
axB.set_ylim(0.01, 40)
axB.set_xlabel(r'Fisher eigenvalue $\lambda$ along a basis query (b = 0.1)')
axB.set_ylabel('half-width / prior b')
axB.legend(loc='upper right', fontsize=6.2)
fs.panel(axB, 'b', 'Break-even criterion')

# ---------------------------------------------- (c) error budget, KW51 queries --
cases = [(3, 'nodrift'), (3, 'frozen'), (12, 'nodrift'), (12, 'frozen'), (9, 'nodrift'), (9, 'frozen')]
y = np.arange(len(cases))[::-1] * 1.0
for yi, (m, cond) in zip(y, cases):
    r = e.loc[m]
    w0 = r.w0
    if cond == 'nodrift':
        rel, noise, alias, drift = r.rel_nodrift, r.noise_nodrift / w0, r.alias_nodrift / w0, 0.0
    else:
        rel, noise, alias = r.rel_frozen, r.noise_frozen / w0, r.alias_frozen / w0
        drift = max(rel - noise - alias, 0.0)
    if rel >= 1 - 1e-6:
        axC.barh(yi, 1.0, height=0.62, color=fs.PRIOR, edgecolor='white', lw=0.6)
        axC.text(0.78, yi, 'prior (zero estimator)', ha='center', va='center', fontsize=5.9, color=fs.INK)
    else:
        left = 0
        for val, col in ((noise, fs.BUDGET['noise']), (alias, fs.BUDGET['alias']),
                         (drift, fs.BUDGET['drift'])):
            if val > 0:
                axC.barh(yi, val, left=left, height=0.62, color=col, edgecolor='white', lw=0.6)
                left += val
        axC.text(left + 0.02, yi, f'{rel:.2f}', va='center', fontsize=6.2, color=fs.INK)
    axC.plot([r.rank_floor_rel] * 2, [yi - 0.38, yi + 0.38], color=fs.INK, lw=1.0)
labels = [f'mode {m}, ' + ('exact' if c == 'nodrift' else 'monthly') for m, c in cases]
axC.set_yticks(y)
axC.set_yticklabels(labels, fontsize=6.2)
axC.set_xlim(0, 1.0)
axC.set_xlabel(r'certified half-width $w^{*}/w_0$  (KW51 full array, b/$\sigma$ = 50)')
axC.tick_params(axis='y', length=0)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles = [Patch(color=fs.BLUE, label='noise'), Patch(color=fs.ORANGE, label='aliasing'),
           Patch(color=fs.AQUA, label='discrepancy'), Line2D([], [], color=fs.INK, lw=1.0, label='rank floor')]
axC.legend(handles=handles, ncol=2, loc='upper right', bbox_to_anchor=(1.0, 1.03), fontsize=6.0,
           handlelength=1.0, columnspacing=0.9)
fs.panel(axC, 'c', 'Error budget of three field queries')
fs.save(fig, 'fig01_framework')
