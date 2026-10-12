"""Fig. 3 - HBTA: truss and sensors, campaign matrix, FDD spectra with reference modes."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import figstyle as fs

fs.setup()
R = fs.ROUND
pos = json.loads((R / 'cache' / 'hbta_sensor_positions.json').read_text())
runs = pd.read_csv(R / 'results' / 'hbta' / 'hbta_runs.csv', dtype={'rep': str})
runs['rep'] = runs['rep'].str.zfill(2)
z = np.load(R / 'cache' / 'hbta_features.npz')
spec_f, spec = z['spec_f'], z['spec']
ref_f = z['ref_f']
names = list(z['runs'])

fig = plt.figure(figsize=(fs.W2, 118 * fs.MM))
axA = fig.add_axes([0.02, 0.60, 0.50, 0.34])
axB = fig.add_axes([0.60, 0.60, 0.38, 0.32])
axC = fig.add_axes([0.07, 0.08, 0.91, 0.38])

# -------------------------------------------------------------- (a) truss -----
xs = np.arange(-17.5, 17.6, 3.5)
ztop = lambda x: 5.0 - 0.00816 * x ** 2
axA.plot([-17.5, 17.5], [0.6, 0.6], color='#a9a79f', lw=1.0)
xx = np.linspace(-17.5, 17.5, 200)
axA.plot(xx, ztop(xx), color='#a9a79f', lw=1.0)
for x in xs:
    axA.plot([x, x], [0.6, ztop(x)], color='#a9a79f', lw=0.6)
for x0, x1 in zip(xs[:-1], xs[1:]):
    if x1 <= 0:      # diagonals descend towards midspan (N-type panels, schematic)
        axA.plot([x0, x1], [ztop(x0), 0.6], color='#a9a79f', lw=0.6)
    else:
        axA.plot([x0, x1], [0.6, ztop(x1)], color='#a9a79f', lw=0.6)
for k, v in pos.items():
    if not k.startswith('AG'):
        continue
    x, y, zz = v['position']
    north = y > 0
    axA.plot(x, zz, marker='o' if north else 's', ms=5.0, mfc=fs.BLUE if north else 'white',
             mec=fs.BLUE, mew=1.0, zorder=5)
    top = zz > 1.0
    axA.text(x, zz + (0.55 if top else -0.55), k[2:], ha='center', va='bottom' if top else 'top',
             fontsize=5.4, color=fs.INK2)
for lab, xp in (('P1', 7.5), ('P2', 0.0)):
    axA.plot(xp, -1.35, marker='^', ms=5.5, color=fs.ORANGE, zorder=6)
    axA.text(xp + 0.9, -1.4, lab, fontsize=6.0, color=fs.ORANGE, va='center', fontweight='bold')
axA.annotate('', xy=(-17.5, -2.6), xytext=(17.5, -2.6), arrowprops=dict(arrowstyle='<->', lw=0.6,
                                                                         color=fs.MUTED))
axA.text(0, -2.85, '35 m', ha='center', va='top', fontsize=5.8, color=fs.MUTED)
axA.set_xlim(-19, 19)
axA.set_ylim(-3.8, 7.4)
axA.set_aspect('equal')
axA.axis('off')
from matplotlib.lines import Line2D
h = [Line2D([], [], marker='o', ls='', mfc=fs.BLUE, mec=fs.BLUE, label='north wall (AG01-09)'),
     Line2D([], [], marker='s', ls='', mfc='white', mec=fs.BLUE, label='south wall (AG10-18)'),
     Line2D([], [], marker='^', ls='', color=fs.ORANGE, label='shaker positions')]
axA.legend(handles=h, loc='upper left', bbox_to_anchor=(-0.02, 1.10), ncol=3, fontsize=5.8,
           columnspacing=0.8, handletextpad=0.3)
fs.panel(axA, 'a', 'Hell Bridge Test Arena: global accelerometers (y, z)', x=0.0, y=1.13)

# ------------------------------------------------------- (b) campaign matrix ---
combos = ['SM-Y-P1', 'SM-Z-P1', 'SM-Y-P2', 'SM-Z-P2', 'NM-Z-P2']
cols = [('UDS', '01'), ('UDS', '02')] + [(f'DS{k}', '01') for k in range(1, 9)]
T = np.full((5, 10), np.nan)
for i, c in enumerate(combos):
    for j, (st, rep) in enumerate(cols):
        r = runs[(runs.combo == c) & (runs.state == st) & (runs.rep == rep)]
        if len(r):
            T[i, j] = r.temperature.iloc[0]
            axB.text(j, i, r.date.iloc[0][-2:], ha='center', va='center', fontsize=5.4,
                     color='white' if r.temperature.iloc[0] > 16 else fs.INK)
im = axB.imshow(T, cmap=fs.SEQ_CMAP, vmin=6, vmax=22, aspect='auto')
axB.set_xticks(range(10))
axB.set_xticklabels(['U1', 'U2'] + [str(k) for k in range(1, 9)], fontsize=6.0)
axB.set_yticks(range(5))
axB.set_yticklabels(combos, fontsize=6.0)
axB.set_xlabel('state (U = undamaged repeat, 1-8 = DS1-DS8)', fontsize=6.4)
for s in axB.spines.values():
    s.set_visible(False)
axB.tick_params(length=0)
axB.axvline(1.5, color='white', lw=1.5)
cb = fig.colorbar(im, ax=axB, orientation='vertical', fraction=0.05, pad=0.02, aspect=14)
cb.set_label('air temperature (\N{DEGREE SIGN}C)', fontsize=6.0)
cb.ax.tick_params(labelsize=5.8)
cb.outline.set_visible(False)
fs.panel(axB, 'b', '50 runs: set-up x state (cell text = day, Sep 2020)', x=-0.22, y=1.05)

# ------------------------------------------------------------ (c) FDD ---------
for name, col, lab in (('MVS_P1_UDS_SM_Y_01', fs.ORANGE, 'lateral sweep, P1 (UDS)'),
                       ('MVS_P2_UDS_SM_Z_01', fs.BLUE, 'vertical sweep, P2 (UDS)'),
                       ('MVS_P2_UDS_NM_Z_01', fs.AQUA, 'vertical random, P2 (UDS)')):
    s1 = spec[names.index(name)]
    axC.semilogy(spec_f, s1 / np.median(s1), color=col, lw=0.9, label=lab)
for i, fr in enumerate(ref_f):
    axC.axvline(fr, color=fs.GRID, lw=0.9, zorder=0)
    axC.text(fr, 3.2e4 if i % 2 == 0 else 1.2e4, str(i + 1), ha='center', fontsize=5.4, color=fs.INK2)
axC.set_xlim(2.5, 30)
axC.set_ylim(0.2, 1e5)
axC.set_xlabel('frequency (Hz)')
axC.set_ylabel('first singular value / median')
axC.legend(loc='upper right', bbox_to_anchor=(1.0, 0.82), fontsize=6.0)
fs.panel(axC, 'c', 'Frequency-domain decomposition: 17 reference modes (numbered) from the 10 undamaged runs')
fs.save(fig, 'fig03_hbta_data')
