"""Fig. 9 - HBTA leave-one-excitation-out transfer (full array, session noise)."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as fs

fs.setup()
R = fs.ROUND / 'results' / 'hbta'
B = pd.read_csv(R / 'hbta_loeo_session.csv')
C = pd.read_csv(R / 'hbta_loeo_direction_session.csv')
hb = json.loads((R / 'hbta_gate_summary_session.json').read_text())
DS = [f'DS{k}' for k in range(1, 9)]
combos = sorted(B.held_out.unique())

fig = plt.figure(figsize=(fs.W2, 108 * fs.MM))
axA = fig.add_axes([0.07, 0.58, 0.91, 0.33])
axB = fig.add_axes([0.07, 0.08, 0.42, 0.34])
axC = fig.add_axes([0.62, 0.08, 0.30, 0.36])

# ---------------------------------------------- (a) held-out true-state intervals --
d = B[(B.layout == 'full') & (B['query'] == B.true_state)].copy()
d['ds_i'] = d.true_state.str[2:].astype(int)
d['c_i'] = d.held_out.map({c: i for i, c in enumerate(combos)})
d = d.sort_values(['ds_i', 'c_i'])
xpos = (d.ds_i - 1) * 6 + d.c_i
axA.axhspan(-1, 1, color='#eef4fc', zorder=0)
axA.axhline(1.0, color=fs.INK, lw=0.8)
axA.axhline(0.0, color=fs.INK2, lw=0.5)
cov = np.abs(d.est_noise_only - 1.0) <= d.w_noise_only
for x, est, w, ok in zip(xpos, d.est_noise_only, d.w_noise_only, cov):
    col = fs.AQUA if ok else fs.ORANGE
    axA.plot([x, x], [est - w, est + w], color=col, lw=1.2)
    axA.plot(x, est, marker='s' if ok else 'o', ms=3.2, color=col, mec='white', mew=0.3)
for k in range(1, 9):
    axA.text((k - 1) * 6 + 2, 2.55, f'DS{k}', ha='center', fontsize=6.2, color=fs.INK)
    if k > 1:
        axA.axvline((k - 1) * 6 - 1, color=fs.GRID, lw=0.8)
axA.set_xlim(-1, 47)
axA.set_ylim(-1.6, 2.8)
axA.set_xticks([])
axA.set_ylabel('held-out intensity estimate\n(truth = 1)')
axA.text(46.6, -1.0, 'prior [-1, 1] = discrepancy-aware certificate', ha='right', va='bottom', fontsize=5.8,
         color=fs.BLUE)
axA.legend(handles=[Line2D([], [], color=fs.AQUA, marker='s', ms=3, label='noise-only interval covers truth'),
                    Line2D([], [], color=fs.ORANGE, marker='o', ms=3, label='noise-only interval misses truth')],
           loc='lower left', ncol=2, fontsize=6.0)
fs.panel(axA, 'a', 'Leave-one-excitation-out: 40 held-out contrasts (5 per state, set-ups in alphabetical order)', y=1.02)

# ---------------------------------------------------------- (b) summary bars --
lf = hb['loeo_full']
metrics = [('coverage_true_state', 'covers true\nintensity'), ('detection_rate', 'excludes 0 for\nimposed state'),
           ('false_attribution_rate', 'excludes 0 for a\nnon-imposed state')]
rules = [('naive', 'naive', fs.ORANGE), ('noise_only', 'noise-only certificate', fs.AQUA),
         ('certified', 'with discrepancy', fs.BLUE)]
for j, (tag, lab, col) in enumerate(rules):
    v = [lf[tag][m] for m, _ in metrics]
    xs = np.arange(3) + (j - 1) * 0.26
    axB.bar(xs, v, width=0.24, color=col, edgecolor='white', lw=0.5, label=lab)
    for xx, vv in zip(xs, v):
        axB.text(xx, vv + 0.02, f'{vv:.2f}', ha='center', fontsize=5.4, color=fs.INK2)
axB.set_xticks(range(3))
axB.set_xticklabels([t for _, t in metrics], fontsize=6.0)
axB.set_ylim(0, 1.12)
axB.set_ylabel('fraction of held-out queries')
axB.legend(loc='upper right', fontsize=5.8, ncol=1)
fs.hgrid(axB)
fs.panel(axB, 'b', 'Interval behaviour in transfer (320 queries)')

# ------------------------------------------------ (c) direction attribution ----
M = np.zeros((8, 8))
cf = C[C.layout == 'full']
for _, r in cf.iterrows():
    M[int(r.true_state[2:]) - 1, int(r.nearest[2:]) - 1] += 1
im = axC.imshow(M, cmap=fs.SEQ_CMAP, vmin=0, vmax=5)
for i in range(8):
    for j in range(8):
        if M[i, j] > 0:
            axC.text(j, i, int(M[i, j]), ha='center', va='center', fontsize=5.8,
                     color='white' if M[i, j] >= 3 else fs.INK)
axC.set_xticks(range(8))
axC.set_xticklabels([str(i) for i in range(1, 9)])
axC.set_yticks(range(8))
axC.set_yticklabels([str(i) for i in range(1, 9)])
axC.set_xlabel('nearest training direction')
axC.set_ylabel('imposed state')
for s in axC.spines.values():
    s.set_visible(False)
axC.tick_params(length=0)
acc = hb['direction_attribution']['full_accuracy']
fs.panel(axC, 'c', f'Direction-only attribution: {acc:.0%} (chance 12.5%)', x=-0.25, y=1.03)
fs.save(fig, 'fig09_hbta_loeo')
