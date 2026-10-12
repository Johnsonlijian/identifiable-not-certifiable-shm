"""Fig. 10 - aggregate optimal-design criteria versus query outcomes (KW51, b/sigma = 50)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as fs

fs.setup()
R = fs.ROUND / 'results'
ch = pd.read_csv(R / 'osp' / 'osp_choices.csv')
co = pd.read_csv(R / 'osp' / 'osp_correlations.csv')
enum = pd.read_csv(R / 'kw51' / 'kw51_enumeration.csv')
e = enum[enum.snr == 50.0]
lay = e.groupby('mask').agg(k=('k', 'first'), report=('state', lambda s: int((s == 'REPORT').sum()))).reset_index()
ref = ch[np.isclose(ch.eta_rel, 0.01)]
rng = np.random.default_rng(11)

fig = plt.figure(figsize=(fs.W2, 70 * fs.MM))
axA = fig.add_axes([0.06, 0.16, 0.40, 0.70])
axB = fig.add_axes([0.54, 0.16, 0.20, 0.70])
axC = fig.add_axes([0.82, 0.16, 0.17, 0.70])

# ----------------------------------------------------- (a) choices per budget --
axA.plot(lay.k + rng.uniform(-0.12, 0.12, len(lay)), lay.report, ls='', marker='o', ms=2.6,
         color='#c9c7bf', zorder=1)
styles = [('D-optimal', fs.BLUE, 'o', -0.12), ('A-optimal', fs.ORANGE, 's', 0.0),
          ('greedy-D (EfI-type)', fs.AQUA, 'D', 0.12), ('query-aware (REPORT)', fs.INK, '*', 0.0)]
for rule, col, mk, off in styles:
    d = ref[ref.rule == rule].sort_values('k')
    axA.plot(d.k + off, d.report, color=col, marker=mk, ms=4.6 if mk != '*' else 6.5, lw=0.9,
             label=rule, zorder=3)
axA.text(6.3, 1.35, 'E-optimality undefined: every layout\nhas $\\lambda_{\\min}(F+\\eta I)=\\eta$', ha='right',
         va='bottom', fontsize=5.6, color=fs.MUTED)
axA.set_xlabel('arch channels added to the deck array (budget k)')
axA.set_ylabel('REPORT queries of the chosen layout')
axA.set_xticks(range(7))
axA.set_ylim(1, 11)
axA.legend(loc='upper left', fontsize=5.8)
fs.hgrid(axA)
fs.panel(axA, 'a', r'Layouts chosen by each criterion ($\eta$ = 0.01 trace/14); grey: all 64')

# ------------------------------------------------------- (b) two-way trade-off --
D = ref[ref.rule == 'D-optimal'].set_index('k')
Qa = ref[ref.rule == 'query-aware (REPORT)'].set_index('k')
for k in range(1, 6):
    gain = Qa.loc[k, 'report'] - D.loc[k, 'report']
    loss = Qa.loc[k, 'D_loss']
    axB.plot(loss, gain, marker='o', ms=5, color=fs.INK, mec='white', mew=0.5)
    dy = {2: -0.22, 3: 0.22}.get(k, 0.0)
    axB.text(loss + (0.0 if dy else 0.12), gain + dy, f'k={k}', va='center', ha='center' if dy else 'left',
             fontsize=5.8, color=fs.INK2)
axB.set_xlabel('log-det given up (nats)')
axB.set_ylabel('REPORT queries gained over D-optimal')
axB.set_xlim(-0.2, 3.6)
axB.set_ylim(-0.4, 2.6)
axB.set_yticks([0, 1, 2])
fs.hgrid(axB)
fs.panel(axB, 'b', 'Two-way cost', y=1.02)

# ------------------------------------------------------ (c) rank correlations ---
c = co[np.isclose(co.eta_rel, 0.01)].set_index('criterion')
x = np.arange(2)
axC.bar(x - 0.18, c.loc[['D', 'A'], 'rho_report'], width=0.34, color=fs.STATE['REPORT'],
        edgecolor='white', label='with # REPORT')
axC.bar(x + 0.18, c.loc[['D', 'A'], 'rho_certified'], width=0.34, color=fs.BLUE,
        edgecolor='white', label='with # certified')
for xx, v in zip(list(x - 0.18) + list(x + 0.18),
                 list(c.loc[['D', 'A'], 'rho_report']) + list(c.loc[['D', 'A'], 'rho_certified'])):
    axC.text(xx, v + 0.02, f'{v:.2f}', ha='center', fontsize=5.4, color=fs.INK2)
axC.text(2, 0.05, 'E:\nundefined', ha='center', va='bottom', fontsize=5.6, color=fs.MUTED)
axC.axhline(0, color=fs.INK2, lw=0.5)
axC.set_xticks([0, 1, 2])
axC.set_xlim(-0.6, 2.6)
axC.set_xticklabels(['D', 'A', 'E'])
axC.set_ylim(-0.2, 1.0)
axC.set_ylabel('Spearman correlation (64 layouts)')
axC.legend(loc='upper right', fontsize=5.6)
fs.hgrid(axC)
fs.panel(axC, 'c', 'Agreement', y=1.02)
fs.save(fig, 'fig10_osp')
