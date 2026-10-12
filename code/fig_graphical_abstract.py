"""Graphical abstract: identifiable versus certified queries on the two bridges."""
import json
import pandas as pd
import matplotlib.pyplot as plt
import figstyle as fs

fs.setup()
R = fs.ROUND / 'results'
e = pd.read_csv(R / 'kw51' / 'kw51_enumeration.csv')
d = e[(e.snr == 50.0) & (e['mask'] == 63)]
kw = [('declared', 14), ('identifiable', int((d.state_rank == 'REPORT').sum())),
      ('certifiable in principle', int((d.rank_floor_rel <= 0.5).sum())),
      ('certified, exact model', int((d.rel_nodrift <= 0.5).sum())),
      ('certified, measured drift', int((d.rel_frozen <= 0.5).sum()))]
hb = json.loads((R / 'hbta' / 'hbta_gate_summary_session.json').read_text())
hb_list = [('declared', 8), ('identifiable', hb['insample_state_counts_presence']['full'].get('REPORT', 0)),
           ('certified, noise only', hb['insample_certified_presence']['full']['noise_only']),
           ('certified, across excitations', hb['insample_certified_presence']['full']['certified']),
           ('certified in transfer', 0)]

fig = plt.figure(figsize=(13.28 / 2.54 * 1.4, 5.31 / 2.54 * 1.4))
fig.text(0.02, 0.93, 'Identifiable is not certifiable', fontsize=13, fontweight='bold', color=fs.INK)
fig.text(0.02, 0.84, 'Query-level certification: REPORT / BOUND / REFUSE with an interval guaranteed over the prior '
         'range and a declared model-discrepancy bound', fontsize=7.5, color=fs.INK2)
for k, (title, items, col, tot) in enumerate((
        ('KW51 bridge, full array: 14 modal-coordinate queries', kw, fs.BLUE, 14),
        ('Hell Bridge Test Arena, full array: 8 damage states', hb_list, fs.AQUA, 8))):
    ax = fig.add_axes([0.20 + 0.50 * k, 0.10, 0.27, 0.62])
    y = list(range(len(items)))[::-1]
    vals = [v for _, v in items]
    ax.barh(y, vals, color=[col] * 2 + [fs.SEQ[6]] * (len(items) - 2), height=0.62, edgecolor='white')
    for yi, (lab, v) in zip(y, items):
        ax.text(v + tot * 0.03, yi, str(v), va='center', fontsize=8, color=fs.INK, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _ in items], fontsize=7.2)
    ax.set_xlim(0, tot * 1.15)
    ax.set_xticks([])
    for s in ('bottom',):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_title(title, fontsize=7.8, fontweight='bold', loc='left', x=-0.62)
fs.save(fig, 'graphical_abstract')
