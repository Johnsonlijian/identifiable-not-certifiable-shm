"""KW51: does the retrofit change what the deck array can answer?

Arch-channel shapes exist only after the retrofit, so the comparison uses the
six deck channels with a deck-only normalisation applied identically to the
pre-retrofit (Oct 2018 - May 2019), retrofit (May - Sep 2019) and
post-retrofit (Sep 2019 - Jan 2020) stages. For every stage the rank-level and
noise-level states and the drift-free certificates of the 14 modal-coordinate
queries are computed at b / sigma = 50 and 5.

Also reported: state persistence of the post-retrofit 12-channel analysis over
the daily dictionaries of the held-out test window.

Outputs: results/kw51/kw51_prepost.csv, kw51_persistence.csv, kw51_prepost_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import qcert as Q
import kw51_data as K

OUT = K.ROUND / 'results' / 'kw51'
BOX = 0.1
P = K.N_MODE

stages = K.stage_dictionaries_deck()
rows = []
for snr in (50.0, 5.0):
    sigma = BOX / snr
    for st, (Dst, nrec) in stages.items():
        S_w = Dst / sigma
        g = Q.geometry(S_w, BOX)
        sol = Q.CertificateSolver(S_w.shape[0], P)
        for m in range(P):
            q = np.eye(P)[m]
            c = sol.solve(S_w, q, BOX, None)
            rows.append({'snr': snr, 'stage': st, 'n_records': nrec, 'mode': m + 1,
                         'pnull_exact': g.pnull_exact(q), 'pnull': g.pnull(q),
                         'state_rank': Q.state_from_pnull(g.pnull_exact(q)),
                         'state': Q.state_from_pnull(g.pnull(q)), 'rel_nodrift': c.rel})
pp = pd.DataFrame(rows)
pp.to_csv(OUT / 'kw51_prepost.csv', index=False)

# MAC between stage dictionaries (deck channels) per mode
mac = {}
for a, b in (('pre', 'retrofit'), ('pre', 'post'), ('retrofit', 'post')):
    A, B = stages[a][0], stages[b][0]
    mac[f'{a}-{b}'] = [float((A[:, m] @ B[:, m]) ** 2 / ((A[:, m] @ A[:, m]) * (B[:, m] @ B[:, m])))
                       for m in range(P)]

# ------------------------------------- state persistence over test days -----
D = K.build_post_retrofit()
test = D['daily'][D['in_window']['test']]
pers = []
for snr in (50.0, 5.0):
    sigma = BOX / snr
    for mask in (0, 63):
        ids = K.layout_ids(mask)
        g_tr = Q.geometry(D['phi_train'][ids] / sigma, BOX)
        st_tr = [Q.state_from_pnull(g_tr.pnull(np.eye(P)[m])) for m in range(P)]
        same = np.zeros(P)
        for Sd in test:
            g = Q.geometry(Sd[ids] / sigma, BOX)
            same += [Q.state_from_pnull(g.pnull(np.eye(P)[m])) == st_tr[m] for m in range(P)]
        for m in range(P):
            pers.append({'snr': snr, 'mask': mask, 'mode': m + 1, 'state_train': st_tr[m],
                         'persistence': same[m] / len(test)})
pers = pd.DataFrame(pers)
pers.to_csv(OUT / 'kw51_persistence.csv', index=False)

summary = {}
for snr in (50.0, 5.0):
    d = pp[pp.snr == snr]
    piv = d.pivot_table(index='mode', columns='stage', values='state', aggfunc='first')
    pivp = d.pivot_table(index='mode', columns='stage', values='pnull')
    changed = [int(m) for m in piv.index if len(set(piv.loc[m])) > 1]
    summary[str(snr)] = {
        'counts': {st: [int((d[d.stage == st].state == s).sum()) for s in (Q.REPORT, Q.BOUND, Q.REFUSE)]
                   for st in ('pre', 'retrofit', 'post')},
        'counts_rank': {st: [int((d[d.stage == st].state_rank == s).sum())
                             for s in (Q.REPORT, Q.BOUND, Q.REFUSE)] for st in ('pre', 'retrofit', 'post')},
        'modes_changing_state': changed,
        'max_abs_pnull_shift': float((pivp.max(1) - pivp.min(1)).max()),
        'rel_nodrift_by_stage_mode3': d[d['mode'] == 3].set_index('stage').rel_nodrift.round(4).to_dict(),
        'persistence_full_min': float(pers[(pers.snr == snr) & (pers['mask'] == 63)].persistence.min()),
        'persistence_deck_min': float(pers[(pers.snr == snr) & (pers['mask'] == 0)].persistence.min()),
        'persistence_full_mean': float(pers[(pers.snr == snr) & (pers['mask'] == 63)].persistence.mean()),
    }
summary['n_records'] = {st: int(v[1]) for st, v in stages.items()}
summary['mac_deck_dictionaries'] = {k: [round(x, 4) for x in v] for k, v in mac.items()}
summary['min_mac'] = {k: float(min(v)) for k, v in mac.items()}
(OUT / 'kw51_prepost_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=1))
print(pp[pp.snr == 50.0].pivot_table(index='mode', columns='stage', values=['pnull', 'state'],
                                    aggfunc='first'))
