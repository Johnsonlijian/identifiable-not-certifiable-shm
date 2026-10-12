"""Aggregate optimal-sensor-placement criteria versus query-level outcomes (KW51).

For every arch-channel budget k = 0..6 (deck channels always present) the
layout chosen by each criterion is compared with the query-aware choice, in
both directions:
  * classical criteria scored by query outcomes (REPORT states, certified
    queries, mean relative certified width without discrepancy);
  * query-aware layouts scored by the classical criteria.

Criteria on F = Phi_L^T Phi_L / sigma^2 (reference b / sigma = 50):
  D   max log det(F + eta I)        A   min trace((F + eta I)^-1)
  E   max lambda_min(F + eta I)     G   greedy forward selection on D (EfI-type)
Because at most 12 channels observe 14 modal coordinates, F is singular for
every layout: D and A require the regulariser eta and E is degenerate
(lambda_min = eta for all layouts). eta is swept over five decades.

Outputs: results/osp/osp_choices.csv, osp_correlations.csv, osp_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import kw51_data as K

ROUND = K.ROUND
OUT = ROUND / 'results' / 'osp'
OUT.mkdir(parents=True, exist_ok=True)
BOX, SNR = 0.1, 50.0
SIGMA = BOX / SNR
KAPPA = 0.5

enum = pd.read_csv(ROUND / 'results' / 'kw51' / 'kw51_enumeration.csv')
enum = enum[enum.snr == SNR]
phi = np.load(ROUND / 'results' / 'kw51' / 'kw51_envelopes.npz')['phi_train']

outcome = enum.groupby('mask').agg(
    report=('state', lambda s: int((s == 'REPORT').sum())),
    refuse=('state', lambda s: int((s == 'REFUSE').sum())),
    certified=('rel_nodrift', lambda s: int((s <= KAPPA).sum())),
    informative=('rel_nodrift', lambda s: int((s < 1 - 1e-6).sum())),
    mean_rel=('rel_nodrift', 'mean'),
    mean_rank_floor=('rank_floor_rel', 'mean')).reset_index()
outcome['k'] = outcome['mask'].map(lambda m: bin(m).count('1'))
modes_rep = {m: set(enum[(enum['mask'] == m) & (enum.state == 'REPORT')]['mode']) for m in range(64)}
modes_cert = {m: set(enum[(enum['mask'] == m) & (enum.rel_nodrift <= KAPPA)]['mode']) for m in range(64)}


def fisher(mask):
    J = phi[K.layout_ids(mask)]
    return J.T @ J / SIGMA ** 2


lam_scale = float(np.trace(fisher(63)) / 14)
ETAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
rows, cors = [], []
for eta_rel in ETAS:
    eta = eta_rel * lam_scale
    crit = {}
    for m in range(64):
        F = fisher(m) + eta * np.eye(14)
        w = np.linalg.eigvalsh(F)
        crit[m] = {'D': float(np.sum(np.log(w))), 'A': float(np.sum(1 / w)), 'E': float(w[0])}
    # greedy forward selection on D (EfI-type)
    greedy, cur = [0], 0
    for _ in range(6):
        best = max((cur | (1 << c) for c in range(6) if not cur >> c & 1),
                   key=lambda mm: crit[mm]['D'])
        greedy.append(best)
        cur = best
    cm = pd.DataFrame(crit).T.reset_index().rename(columns={'index': 'mask'})
    cm = cm.merge(outcome, on='mask')
    for col, sign in (('D', 1), ('A', -1), ('E', 1)):
        cors.append({'eta_rel': eta_rel, 'criterion': col,
                     'rho_report': float(spearmanr(sign * cm[col], cm.report).statistic),
                     'rho_certified': float(spearmanr(sign * cm[col], cm.certified).statistic),
                     'rho_mean_rel': float(spearmanr(sign * cm[col], -cm.mean_rel).statistic),
                     'n_distinct_values': int(cm[col].round(9).nunique())})
    for k in range(7):
        cand = cm[cm.k == k]
        picks = {
            'D-optimal': int(cand.loc[cand.D.idxmax(), 'mask']),
            'A-optimal': int(cand.loc[cand.A.idxmin(), 'mask']),
            'E-optimal': int(cand.loc[cand.E.idxmax(), 'mask']),
            'greedy-D (EfI-type)': greedy[k],
            'query-aware (REPORT)': int(cand.sort_values(['report', 'certified', 'mean_rel'],
                                                         ascending=[False, False, True]).iloc[0]['mask']),
            'query-aware (certified)': int(cand.sort_values(['certified', 'mean_rel'],
                                                            ascending=[False, True]).iloc[0]['mask']),
        }
        e_ties = int((cand.E.round(6) == cand.E.max().round(6)).sum())
        best_rep = int(cand.report.max())
        best_cert = int(cand.certified.max())
        Dmax = float(cand.D.max())
        for name, m in picks.items():
            o = outcome.set_index('mask').loc[m]
            rows.append({'eta_rel': eta_rel, 'k': k, 'rule': name, 'mask': m,
                         'channels': 'D+' + ''.join(f'A{c + 1}' for c in range(6) if m >> c & 1),
                         'report': int(o.report), 'certified': int(o.certified),
                         'mean_rel': float(o.mean_rel), 'report_regret': best_rep - int(o.report),
                         'certified_regret': best_cert - int(o.certified),
                         'D_loss': Dmax - crit[m]['D'],
                         'lost_report_modes': sorted(set().union(*[modes_rep[mm] for mm in cand['mask']
                                                                  if outcome.set_index('mask').loc[mm].report == best_rep][:1])
                                                     - modes_rep[m]),
                         'E_ties_at_budget': e_ties})
ch = pd.DataFrame(rows)
ch.to_csv(OUT / 'osp_choices.csv', index=False)
co = pd.DataFrame(cors)
co.to_csv(OUT / 'osp_correlations.csv', index=False)

ref = ch[ch.eta_rel == 1e-2]
mid = ref[(ref.k >= 1) & (ref.k <= 5)]
summary = {
    'snr': SNR, 'kappa': KAPPA, 'eta_reference_rel': 1e-2,
    'mean_report_regret_k1to5': mid.groupby('rule').report_regret.mean().round(3).to_dict(),
    'max_report_regret_k1to5': mid.groupby('rule').report_regret.max().to_dict(),
    'mean_certified_regret_k1to5': mid.groupby('rule').certified_regret.mean().round(3).to_dict(),
    'D_loss_of_query_aware_k1to5': mid[mid.rule.str.startswith('query')].groupby('rule')
    .D_loss.mean().round(3).to_dict(),
    'choices_reference_eta': ref[['k', 'rule', 'channels', 'report', 'certified', 'report_regret',
                                  'certified_regret', 'D_loss', 'lost_report_modes']].to_dict('records'),
    'regret_by_eta': ch[(ch.k >= 1) & (ch.k <= 5)].groupby(['eta_rel', 'rule']).report_regret
    .mean().round(3).unstack().to_dict('index'),
    'spearman': co.to_dict('records'),
    'E_degenerate_all_eta': bool((co[co.criterion == 'E'].n_distinct_values <= 2).all()),
    'full_array_singular_values': np.linalg.svd(phi, compute_uv=False).round(4).tolist(),
}
(OUT / 'osp_summary.json').write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps({k: summary[k] for k in ('mean_report_regret_k1to5', 'max_report_regret_k1to5',
                                           'mean_certified_regret_k1to5', 'D_loss_of_query_aware_k1to5',
                                           'regret_by_eta', 'E_degenerate_all_eta')}, indent=1))
print(co.round(3).to_string(index=False))
print(ref[['k', 'rule', 'channels', 'report', 'certified', 'report_regret', 'D_loss']].to_string(index=False))
