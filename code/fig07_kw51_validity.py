"""Fig. 7 - KW51: out-of-sample validity in the winter test window (full array, b/sigma = 50)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as fs

fs.setup()
cov = pd.read_csv(fs.ROUND / 'results' / 'kw51' / 'kw51_coverage.csv')
d = cov[cov['mask'] == 63]
rules = [('naive', 'naive (noise-only)', fs.ORANGE, 'o', -0.24),
         ('blind', 'certificate without discrepancy', fs.AQUA, 's', 0.0),
         ('certified', 'certificate with monthly discrepancy', fs.BLUE, '^', 0.24)]

fig = plt.figure(figsize=(fs.W2, 72 * fs.MM))
axA = fig.add_axes([0.07, 0.17, 0.42, 0.66])
axB = fig.add_axes([0.57, 0.17, 0.42, 0.66])
modes = np.arange(1, 15)
for tag, lab, col, mk, off in rules:
    s = d[d.rule == tag].set_index('mode').loc[modes]
    axA.plot(modes + off, s.coverage_worst, ls='', marker=mk, ms=3.8, color=col, mec='white', mew=0.4)
    axA.plot(modes + off, s.coverage_uniform, ls='', marker=mk, ms=3.8, mfc='white', mec=col, mew=0.8)
    axB.plot(modes + off, s.halfwidth_rel, ls='', marker=mk, ms=3.8, color=col, mec='white', mew=0.4)
axA.axhline(0.95, color=fs.INK2, lw=0.6)
axA.set_ylim(-0.03, 1.05)
axA.set_ylabel('coverage on 35 winter test days')
axB.axhline(1.0, color=fs.PRIOR, lw=1.2)
axB.text(14.6, 1.0, 'prior', fontsize=5.8, color=fs.MUTED, va='bottom', ha='right')
axB.set_ylim(0, 1.08)
axB.set_ylabel(r'half-width / prior $w_0$')
for ax in (axA, axB):
    ax.set_xticks(modes)
    ax.set_xticklabels([str(m) for m in modes], fontsize=6.0)
    ax.set_xlabel('tracked mode')
    ax.set_xlim(0.4, 14.6)
    fs.hgrid(ax)
h = [Line2D([], [], ls='', marker=mk, color=col, label=lab) for _, lab, col, mk, _ in rules]
h += [Line2D([], [], ls='', marker='o', mfc=fs.INK2, mec=fs.INK2, label='filled: worst-case vertex'),
      Line2D([], [], ls='', marker='o', mfc='white', mec=fs.INK2, label='open: uniform draws')]
fig.legend(handles=h, loc='upper left', bbox_to_anchor=(0.06, 0.995), ncol=5, fontsize=5.8,
           columnspacing=0.9, handletextpad=0.3)
fs.panel(axA, 'a', 'Coverage of the true modal coordinate', y=1.03)
fs.panel(axB, 'b', 'Interval half-width', y=1.03)
fs.save(fig, 'fig07_kw51_validity')
