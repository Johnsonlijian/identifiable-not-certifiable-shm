"""Fig. 8 - HBTA: damage signatures vs cross-condition discrepancy, layout map, single-sensor fragility."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as fs

fs.setup()
R = fs.ROUND / 'results'
A = pd.read_csv(R / 'hbta' / 'hbta_insample_map_session.csv')
hb = json.loads((R / 'hbta' / 'hbta_gate_summary_session.json').read_text())
geo = hb['geometry_full']
DS = [f'DS{k}' for k in range(1, 9)]

fig = plt.figure(figsize=(fs.W2, 112 * fs.MM))
axA = fig.add_axes([0.07, 0.60, 0.40, 0.32])
axB = fig.add_axes([0.58, 0.60, 0.41, 0.32])
axC = fig.add_axes([0.16, 0.08, 0.40, 0.34])
axD = fig.add_axes([0.66, 0.08, 0.33, 0.34])

# --------------------------------------- (a) signature vs discrepancy norms ---
x = np.arange(9)
labs = DS + ['benign']
sig = [geo['signature_norm'][k] for k in labs]
env = [geo['envelope_norm'][k] for k in labs]
axA.bar(x - 0.19, sig, width=0.36, color=fs.BLUE, edgecolor='white', lw=0.5, label='damage direction  ||s||')
axA.bar(x + 0.19, env, width=0.36, color=fs.AQUA, edgecolor='white', lw=0.5,
        label='cross-condition envelope  ||e||')
axA.set_xticks(x)
axA.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', 'benign'], fontsize=6.2)
axA.set_xlabel('damage state')
axA.set_ylabel('norm in session-noise units')
axA.legend(loc='upper left', fontsize=6.0)
fs.hgrid(axA)
fs.panel(axA, 'a', 'Signal versus discrepancy (full array)')

# ------------------------------------------- (b) noise-only vs certified width --
f = A[(A.layout == 'full') & (A.kind == 'presence')].set_index('query').loc[DS]
k = np.arange(1, 9)
axB.plot(k - 0.12, f.rel_noise_only, ls='', marker='s', ms=4.2, color=fs.AQUA, mec='white', mew=0.4,
         label='session noise only')
axB.plot(k + 0.12, f.rel_certified, ls='', marker='^', ms=4.4, color=fs.BLUE, mec='white', mew=0.4,
         label='with cross-condition discrepancy')
for kk, v in zip(k, f.rel_certified):
    if v < 0.999:
        axB.text(kk + 0.3, v, f'{v:.2f}', va='center', fontsize=5.8, color=fs.INK2)
axB.axhline(0.5, color=fs.INK2, lw=0.6)
axB.axhline(1.0, color=fs.PRIOR, lw=1.2)
axB.set_ylim(0, 1.08)
axB.set_xticks(k)
axB.set_xticklabels([str(i) for i in k])
axB.set_xlabel('damage-presence query (DS)')
axB.set_ylabel(r'certified half-width $w^{*}/w_0$')
axB.legend(loc='center left', fontsize=6.0, bbox_to_anchor=(0.0, 0.36))
fs.hgrid(axB)
fs.panel(axB, 'b', 'All eight states REPORT; one is certified')

# ------------------------------------------------------ (c) layout-family map --
fams = ['full', 'north wall', 'south wall', 'deck joints', 'top-chord joints', 'west half', 'east half']
cmap = fs.SEQ_CMAP
for i, l in enumerate(fams):
    d = A[(A.layout == l) & (A.kind == 'presence')].set_index('query').loc[DS]
    for j, s in enumerate(DS):
        r = d.loc[s]
        gain = 1.0 - r.rel_certified                    # 0 = prior, 1 = exact
        axC.add_patch(plt.Rectangle((j - 0.45, i - 0.42), 0.9, 0.84, color=cmap(0.08 + 0.92 * gain / 0.6),
                                    lw=0))
        txt = f'{r.rel_certified:.2f}' if r.rel_certified < 0.999 else '-'
        axC.text(j, i, txt, ha='center', va='center', fontsize=5.6,
                 color='white' if gain > 0.4 else fs.INK2,
                 fontweight='bold' if r.rel_certified <= 0.5 else 'normal')
    axC.text(-0.65, i, f'{l} ({int(d.n_sensors.iloc[0])})', ha='right', va='center', fontsize=6.0)
axC.set_xlim(-0.6, 7.6)
axC.set_ylim(len(fams) - 0.5, -0.6)
axC.set_xticks(range(8))
axC.set_xticklabels([str(i) for i in range(1, 9)])
axC.set_yticks([])
axC.set_xlabel('damage state  (every cell is REPORT at the noise level)')
for s in axC.spines.values():
    s.set_visible(False)
axC.tick_params(length=0)
fs.panel(axC, 'c', 'Layout families: certified width with discrepancy', x=-0.42, y=1.03)

# -------------------------------------------------- (d) single-sensor fragility --
single = A[A.layout.str.startswith('single') & (A.kind == 'presence')]
bottom = np.zeros(8)
for st in ('REPORT', 'BOUND', 'REFUSE'):
    v = np.array([np.mean(single[single['query'] == s].state == st) for s in DS])
    axD.bar(k, v, bottom=bottom, width=0.72, color=fs.STATE[st], edgecolor='white', lw=0.6, label=st)
    bottom += v
axD.set_xticks(k)
axD.set_xticklabels([str(i) for i in k])
axD.set_xlabel('damage state')
axD.set_ylabel('share of the 18 single-sensor layouts')
axD.set_ylim(0, 1.0)
axD.legend(loc='upper center', ncol=3, fontsize=5.8, bbox_to_anchor=(0.5, 1.13), columnspacing=0.8)
fs.panel(axD, 'd', 'Single-sensor fragility', y=1.14)
fs.save(fig, 'fig08_hbta_cert')
