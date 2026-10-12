"""Fig. 5 - KW51: evidence hierarchy (funnel), per-mode limits, layout distribution (b/sigma = 50)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as fs

fs.setup()
R = fs.ROUND / 'results'
enum = pd.read_csv(R / 'kw51' / 'kw51_enumeration.csv')
e = enum[enum.snr == 50.0]
rng = np.random.default_rng(5)

fig = plt.figure(figsize=(fs.W2, 118 * fs.MM))
axA = fig.add_axes([0.10, 0.60, 0.33, 0.32])
axB = [fig.add_axes([0.52, 0.775, 0.47, 0.17]), fig.add_axes([0.52, 0.565, 0.47, 0.17])]
axC = fig.add_axes([0.08, 0.08, 0.90, 0.36])

# ----------------------------------------------------------------- (a) funnel --
levels = ['declared queries', 'identifiable\n(rank level, REPORT)', 'certifiable in principle\n(rank floor <= 0.5)',
          'certified, exact dictionary', 'certified, monthly\ndiscrepancy']
counts = {}
for lay, m in (('deck', 0), ('full', 63)):
    d = e[e['mask'] == m]
    counts[lay] = [14, int((d.state_rank == 'REPORT').sum()), int((d.rank_floor_rel <= 0.5).sum()),
                   int((d.rel_nodrift <= 0.5).sum()), int((d.rel_frozen <= 0.5).sum())]
y = np.arange(len(levels))[::-1]
for off, lay in ((0.19, 'full'), (-0.19, 'deck')):
    axA.barh(y + off, counts[lay], height=0.36, color=fs.LAYOUT[lay], edgecolor='white', lw=0.6,
             label='full array (12 ch.)' if lay == 'full' else 'deck only (6 ch.)')
    for yi, c in zip(y, counts[lay]):
        axA.text(c + 0.25, yi + off, str(c), va='center', fontsize=6.2, color=fs.INK)
axA.set_yticks(y)
axA.set_yticklabels(levels, fontsize=6.2)
axA.set_xlim(0, 15.5)
axA.set_xlabel('modal-coordinate queries (of 14)')
axA.tick_params(axis='y', length=0)
axA.legend(loc='lower right', fontsize=6.0)
axA.xaxis.grid(True, color=fs.GRID, lw=0.5)
axA.set_axisbelow(True)
fs.panel(axA, 'a', 'Evidence hierarchy', x=-0.55, y=1.04)

# ------------------------------------------------------- (b) per-mode limits ---
for ax, (lay, m) in zip(axB, (('full', 63), ('deck', 0))):
    d = e[e['mask'] == m].set_index('mode')
    for mode in range(1, 15):
        r = d.loc[mode]
        ax.add_patch(plt.Rectangle((mode - 0.42, 1.08), 0.84, 0.14, color=fs.STATE[r.state], lw=0))
        ax.plot([mode - 0.32, mode + 0.32], [r.rank_floor_rel] * 2, color=fs.INK, lw=1.2)
        ax.plot(mode - 0.12, r.rel_nodrift, marker='o', ms=3.6, color=fs.BLUE, mec='white', mew=0.4)
        ax.plot(mode + 0.12, r.rel_frozen, marker='^', ms=3.6, color=fs.AQUA, mec='white', mew=0.4)
    ax.axhline(0.5, color=fs.INK2, lw=0.6)
    ax.set_xlim(0.4, 14.6)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks(range(1, 15))
    ax.set_xticklabels([str(i) for i in range(1, 15)] if lay == 'deck' else [], fontsize=6.0)
    ax.set_ylabel(r'$w/w_0$', fontsize=6.6)
    ax.text(14.55, 0.05, 'full array' if lay == 'full' else 'deck only', ha='right', fontsize=6.4,
            color=fs.INK, fontweight='bold')
    fs.hgrid(ax)
axB[1].set_xlabel('tracked mode (query = its modal coordinate)')
hs = [plt.Rectangle((0, 0), 1, 1, color=fs.STATE[s]) for s in ('REPORT', 'BOUND', 'REFUSE')]
hl = [Line2D([], [], color=fs.INK, lw=1.2), Line2D([], [], ls='', marker='o', color=fs.BLUE),
      Line2D([], [], ls='', marker='^', color=fs.AQUA)]
axB[0].legend(hs + hl, ['REPORT', 'BOUND', 'REFUSE', 'rank floor', 'exact dictionary', 'monthly discrepancy'],
              ncol=6, loc='lower left', bbox_to_anchor=(-0.01, 1.08), fontsize=5.6, columnspacing=0.7,
              handlelength=1.0, handletextpad=0.3)
fs.panel(axB[0], 'b', 'Per-mode state, rank floor and certified width', x=-0.02, y=1.32)

# ---------------------------------------------------- (c) layout distribution ---
lay = e.groupby('mask').agg(k=('k', 'first'), report=('state', lambda s: int((s == 'REPORT').sum())),
                            cert=('rel_nodrift', lambda s: int((s <= 0.5).sum())),
                            info=('rel_frozen', lambda s: int((s < 1 - 1e-6).sum()))).reset_index()
for _, r in lay.iterrows():
    x = r.k + rng.uniform(-0.16, 0.16)
    axC.plot(x - 0.2, r.report, ls='', marker='s', ms=3.2, color=fs.STATE['REPORT'], alpha=0.75, mec='white', mew=0.3)
    axC.plot(x + 0.2, r.cert, ls='', marker='o', ms=3.4, color=fs.BLUE, alpha=0.75, mec='white', mew=0.3)
for k in range(7):
    s = lay[lay.k == k]
    axC.plot([k - 0.38, k - 0.02], [s.report.mean()] * 2, color=fs.INK, lw=1.0)
    axC.plot([k + 0.02, k + 0.38], [s.cert.mean()] * 2, color=fs.INK, lw=1.0)
    axC.text(k, -0.9, f'{len(s)} layout' + ('s' if len(s) > 1 else ''), ha='center', fontsize=5.6, color=fs.MUTED)
best = lay.loc[lay.report.idxmax()]
axC.annotate('full array: 10 REPORT,\n3 certified (exact), 0 under\nmonthly discrepancy',
             xy=(6 - 0.2, 10), xytext=(4.6, 11.2), fontsize=5.8, color=fs.INK2,
             arrowprops=dict(arrowstyle='-', lw=0.6, color=fs.INK2))
axC.annotate('deck only: 2 REPORT, 1 certified', xy=(0 - 0.2, 2), xytext=(0.35, 7.6), fontsize=5.8,
             color=fs.INK2, arrowprops=dict(arrowstyle='-', lw=0.6, color=fs.INK2))
axC.set_xticks(range(7))
axC.set_xticklabels([str(k) for k in range(7)])
axC.set_xlim(-0.6, 6.6)
axC.set_ylim(-1.3, 13)
axC.set_xlabel('number of arch channels added to the six deck channels')
axC.set_ylabel('queries per layout')
axC.legend(handles=[Line2D([], [], ls='', marker='s', color=fs.STATE['REPORT'], label='REPORT (identifiable)'),
                    Line2D([], [], ls='', marker='o', color=fs.BLUE, label='certified, exact dictionary'),
                    Line2D([], [], color=fs.INK, lw=1.0, label='mean per budget')],
           loc='upper left', fontsize=6.0, ncol=3)
fs.hgrid(axC)
fs.panel(axC, 'c', r'All 64 layouts ($b/\sigma=50$): sensors change what is identifiable much more than what is certified')
fs.save(fig, 'fig05_kw51_hierarchy')
