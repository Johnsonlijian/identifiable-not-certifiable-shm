"""Fig. 2 - KW51: geometry and sensors, tracked frequencies, dictionary discrepancy."""
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.io import loadmat
import figstyle as fs
import kw51_data as K

fs.setup()
from paths import KW51_FE_CASES
g = loadmat(KW51_FE_CASES / 'geometry.mat', squeeze_me=True)
nodes = {int(r[0]): r[1:4] for r in np.asarray(g['Nodes'])}
els = np.asarray(g['Elements'])
raw = K.load_raw()
env = np.load(K.ROUND / 'results' / 'kw51' / 'kw51_envelopes.npz')
labels = list(env['labels'])

fig = plt.figure(figsize=(fs.W2, 112 * fs.MM))
axA = fig.add_axes([0.03, 0.55, 0.60, 0.40])
axB = fig.add_axes([0.07, 0.08, 0.56, 0.36])
axC = fig.add_axes([0.71, 0.08, 0.27, 0.80])

# --------------------------------------------------------------- (a) geometry --
# side-A elevation (y ~ 0): arch, hangers/diagonals and deck girder of the official model
for r in els:
    if int(r[1]) in (1, 2):
        a, b = nodes.get(int(r[4])), nodes.get(int(r[5]))
        if a is None or b is None or max(abs(a[1]), abs(b[1])) > 0.6:
            continue
        axA.plot([a[0], b[0]], [a[2], b[2]], color='#a9a79f', lw=0.5, zorder=1)
Sa = np.asarray(g['Sa'])
DOF = np.asarray(g['DOF'])
NN = np.asarray(g['Nodes'])
locs = {}
for r, lab in enumerate(g['labels_a']):
    j = np.flatnonzero(Sa[r])[0]
    node = int(np.floor(DOF[j]))
    p = NN[NN[:, 0] == node, 1:4][0]
    lab = str(lab)[1:]
    key = lab[:-2]                                         # e.g. BD17, AR1516
    locs.setdefault(key, {'x': p[0], 'z': p[2], 'ch': []})['ch'].append(lab[-2:])
for key, v in locs.items():
    arch = key.startswith('AR')
    col = fs.BLUE if arch else fs.ORANGE
    axA.plot(v['x'], v['z'], marker='s' if arch else 'o', ms=5.2, color=col, mec='white', mew=0.7, zorder=5)
    txt = key + '\n' + ', '.join(sorted(v['ch']))
    axA.text(v['x'], v['z'] + (2.2 if arch else -2.2), txt, ha='center', va='bottom' if arch else 'top',
             fontsize=5.6, color=fs.INK2, zorder=6, linespacing=1.1)
axA.annotate('', xy=(0, -6.6), xytext=(116.3, -6.6),
             arrowprops=dict(arrowstyle='<->', lw=0.6, color=fs.MUTED))
axA.text(58, -7.4, '116 m', ha='center', va='top', fontsize=5.8, color=fs.MUTED)
axA.set_xlim(-2, 118)
axA.set_ylim(-17, 25)
axA.set_aspect('equal')
axA.axis('off')
from matplotlib.lines import Line2D
h = [Line2D([], [], marker='o', ls='', color=fs.ORANGE, label='deck channels (6), always present'),
     Line2D([], [], marker='s', ls='', color=fs.BLUE, label='arch channels (6), post-retrofit only')]
axA.legend(handles=h, loc='lower right', ncol=2, fontsize=6.0, bbox_to_anchor=(1.0, -0.02))
axA.text(0.0, -0.2, '',
         transform=axA.transAxes, fontsize=5.8, color=fs.MUTED, va='top')
fs.panel(axA, 'a', 'KW51 bowstring railway bridge: official FE model and accelerometers', x=0.0, y=1.02)

# ------------------------------------------------------ (b) tracked frequencies --
d = raw['dates']
f = raw['freq']
for m in range(14):
    ok = np.isfinite(f[:, m])
    axB.plot(d[ok], f[ok, m], ls='', marker='.', ms=0.6, color=fs.INK2, alpha=0.35, rasterized=True)
axB.axvspan(datetime(2019, 5, 15), datetime(2019, 9, 28), color='#f1efe9', zorder=0)
axB.text(datetime(2019, 7, 20), 7.55, 'retrofit works', ha='center', va='center', fontsize=6.2,
         color=fs.MUTED)
wl = {'train': 'train', 'calibration': 'calib.', 'test': 'test'}
for (lab, (a, b)), col in zip(K.WINDOWS.items(), ('#dbe8fa', '#d6f2e7', '#fbe3d8')):
    a, b = datetime.fromisoformat(a), datetime.fromisoformat(b)
    axB.axvspan(a, b, color=col, zorder=0, lw=0)
    axB.text(a + (b - a) / 2, 7.55, wl[lab], ha='center', va='center', fontsize=5.4,
             color=fs.INK2, zorder=4)
axB.set_ylim(0.2, 7.85)
axB.set_xlim(datetime(2018, 10, 1), datetime(2020, 1, 16))
axB.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
axB.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axB.set_ylabel('tracked natural frequency (Hz)')
fs.hgrid(axB)
fs.panel(axB, 'b', 'Hourly SSI identification of 14 tracked modes (11,328 records)')

# --------------------------------------------------- (c) discrepancy envelope ---
E = env['E_frozen']                                      # 12 x 14, absolute (unit-norm shapes)
im = axC.imshow(E, cmap=fs.SEQ_CMAP, aspect='auto', vmin=0, vmax=np.quantile(E, 0.98))
axC.set_xticks(range(14))
axC.set_xticklabels([str(m) for m in range(1, 15)], fontsize=6.0)
axC.set_yticks(range(12))
axC.set_yticklabels([l[1:] for l in labels], fontsize=5.8)
axC.set_xlabel('tracked mode')
axC.axhline(5.5, color='white', lw=1.2)
for s in axC.spines.values():
    s.set_visible(False)
axC.tick_params(length=0)
cb = fig.colorbar(im, ax=axC, orientation='horizontal', fraction=0.05, pad=0.10, aspect=30)
cb.set_label(r'envelope $\max_d|\phi^{(d)}-\phi^{\mathrm{train}}|$ (unit-norm shapes)', fontsize=6.0)
cb.ax.tick_params(labelsize=5.8)
cb.outline.set_visible(False)
imax = np.unravel_index(np.argmax(E), E.shape)
axC.text(imax[1], imax[0], f'{E.max():.2f}', ha='center', va='center', fontsize=5.2, color='white')
fs.panel(axC, 'c', 'Calibration-month discrepancy', x=-0.30)
fs.save(fig, 'fig02_kw51_data')
