"""Derived tables and the key-number ledger used by the manuscript.

Reads results/{kw51,fe,hbta,osp}; writes results/tables/*.csv and
results/key_numbers.json (every number quoted in the manuscript is taken from
this ledger).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import qcert as Q

ROUND = Path(__file__).resolve().parent.parent
RES = ROUND / 'results'
TAB = RES / 'tables'
TAB.mkdir(exist_ok=True)
J = lambda p: json.loads((RES / p).read_text())

kw = J('kw51/kw51_summary.json')
kwc = J('kw51/kw51_coverage_summary.json')
kwp = J('kw51/kw51_prepost_summary.json')
fe = J('fe/fe_summary.json')
hb = J('hbta/hbta_gate_summary_session.json')
hb_id = J('hbta/hbta_gate_summary.json')
hbm = J('hbta/hbta_modal_summary.json')
osp = J('osp/osp_summary.json')
enum = pd.read_csv(RES / 'kw51' / 'kw51_enumeration.csv')

# ------------------------------------------------ KW51 threshold sensitivity --
rows = []
for snr in (50.0, 5.0):
    e = enum[enum.snr == snr]
    for t1 in (0.1, 0.2, 0.3):
        for t2 in (0.5, 0.6, 0.7):
            for lay, m in (('deck', 0), ('full', 63)):
                p = e[e['mask'] == m].pnull
                rows.append({'snr': snr, 'tau1': t1, 'tau2': t2, 'layout': lay,
                             'REPORT': int((p <= t1).sum()), 'BOUND': int(((p > t1) & (p <= t2)).sum()),
                             'REFUSE': int((p > t2).sum()),
                             'all_layout_REPORT': int((e.pnull <= t1).sum())})
thr = pd.DataFrame(rows)
thr.to_csv(TAB / 'kw51_threshold_sensitivity.csv', index=False)
kap = []
for snr in (50.0,):
    e = enum[enum.snr == snr]
    for kappa in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        for lay, m in (('deck', 0), ('full', 63)):
            d = e[e['mask'] == m]
            kap.append({'kappa': kappa, 'layout': lay, 'cert_nodrift': int((d.rel_nodrift <= kappa).sum()),
                        'cert_frozen': int((d.rel_frozen <= kappa).sum())})
pd.DataFrame(kap).to_csv(TAB / 'kw51_kappa_sensitivity.csv', index=False)

# ------------------------------------------------ KW51 per-mode table -------
e50 = enum[enum.snr == 50.0]
mt = []
for m in range(1, 15):
    d0 = e50[(e50['mask'] == 0) & (e50['mode'] == m)].iloc[0]
    d1 = e50[(e50['mask'] == 63) & (e50['mode'] == m)].iloc[0]
    sub = e50[e50['mode'] == m]
    mt.append({'mode': m, 'freq_hz': None,
               'pnull_deck': d0.pnull_exact, 'state_deck': d0.state,
               'pnull_full': d1.pnull_exact, 'state_full': d1.state,
               'n_layouts_report': int((sub.state == 'REPORT').sum()),
               'rank_floor_deck': d0.rank_floor_rel, 'rank_floor_full': d1.rank_floor_rel,
               'rel_nodrift_full': d1.rel_nodrift, 'rel_frozen_full': d1.rel_frozen,
               'rel_nodrift_deck': d0.rel_nodrift})
mt = pd.DataFrame(mt)
fr = np.load(RES / 'kw51' / 'kw51_envelopes.npz')['freq_train']
mt['freq_hz'] = np.round(fr, 3)
mt.to_csv(TAB / 'kw51_per_mode.csv', index=False)

# ------------------------------------------------ HBTA per-state table ------
A = pd.read_csv(RES / 'hbta' / 'hbta_insample_map_session.csv')
B = pd.read_csv(RES / 'hbta' / 'hbta_loeo_session.csv')
C = pd.read_csv(RES / 'hbta' / 'hbta_loeo_direction_session.csv')
geo = hb['geometry_full']
ht = []
for k in range(1, 9):
    s = f'DS{k}'
    full = A[(A.layout == 'full') & (A['query'] == s)].iloc[0]
    single = A[A.layout.str.startswith('single') & (A['query'] == s)]
    lo = B[(B.layout == 'full') & (B.true_state == s) & (B['query'] == s)]
    ht.append({'state': s, 'signature_norm': geo['signature_norm'][s],
               'envelope_norm': geo['envelope_norm'][s],
               'single_report_rate': float((single.state == 'REPORT').mean()),
               'single_refuse_rate': float((single.state == 'REFUSE').mean()),
               'rel_noise_only_full': full.rel_noise_only, 'rel_certified_full': full.rel_certified,
               'loeo_cover_noise_only': float(lo.cover_noise_only.mean()),
               'loeo_detect_noise_only': float(lo.excl0_noise_only.mean()),
               'loeo_cover_certified': float(lo.cover_certified.mean()),
               'direction_accuracy': float(C[(C.layout == 'full') & (C.true_state == s)].correct.mean())})
ht = pd.DataFrame(ht)
ht.to_csv(TAB / 'hbta_per_state.csv', index=False)
lay = []
for l in ['full', 'north wall', 'south wall', 'deck joints', 'top-chord joints', 'west half', 'east half']:
    d = A[(A.layout == l) & (A.kind == 'presence')]
    lay.append({'layout': l, 'n_sensors': int(d.n_sensors.iloc[0]), 'n_rows': int(d.n_rows.iloc[0]),
                'REPORT': int((d.state == 'REPORT').sum()), 'BOUND': int((d.state == 'BOUND').sum()),
                'REFUSE': int((d.state == 'REFUSE').sum()),
                'cert_noise_only': int((d.rel_noise_only <= 0.5).sum()),
                'cert_cross_condition': int((d.rel_certified <= 0.5).sum()),
                'informative_cross_condition': int((d.rel_certified < 1 - 1e-6).sum()),
                'certified_states': ','.join(d[d.rel_certified <= 0.5]['query'])})
pd.DataFrame(lay).to_csv(TAB / 'hbta_layout_families.csv', index=False)

# ------------------------------------------------ FE tables -----------------
FA = pd.read_csv(RES / 'fe' / 'fe_linear_grid.csv')
fa = FA.groupby(['variant', 'noise_mult']).agg(
    report=('state', lambda s: int((s == 'REPORT').sum())),
    bound=('state', lambda s: int((s == 'BOUND').sum())),
    refuse=('state', lambda s: int((s == 'REFUSE').sum())),
    cert_cov_worst_min=('cov_worst_cert', 'min'), naive_cov_worst_min=('cov_worst_naive', 'min'),
    naive_cov_uniform_min=('cov_uniform_naive', 'min'),
    median_rel_width=('rel_width', 'median')).reset_index()
fa.to_csv(TAB / 'fe_linear_cells.csv', index=False)
FB = pd.read_csv(RES / 'fe' / 'fe_nonlinear_truth.csv')
fb = []
for v in ('freq', 'shape', 'both'):
    for sset in ('axis', 'corner'):
        d = FB[(FB.variant == v) & (FB.set == sset)]
        for t in ('naive', 'blind', 'axis', 'loo'):
            if f'cov_{t}' not in d or d[f'cov_{t}'].isna().all():
                continue
            dd = d.dropna(subset=[f'cov_{t}'])
            fb.append({'variant': v, 'set': sset, 'rule': t, 'n': int(len(dd)),
                       'mean_cov': float(dd[f'cov_{t}'].mean()), 'min_cov': float(dd[f'cov_{t}'].min()),
                       'frac_below_095': float((dd[f'cov_{t}'] < 0.95).mean()),
                       'median_rel_width': float((dd[f'w_{t}'] / dd.w0).median())})
pd.DataFrame(fb).to_csv(TAB / 'fe_nonlinear_rules.csv', index=False)
res = pd.read_csv(RES / 'fe' / 'fe_linearisation_residuals.csv')
rm = res.groupby('mode').agg(max_freq=('resid_freq_sigma', lambda s: float(np.abs(s).max())),
                             max_shape=('resid_shape_sigma_max', 'max'),
                             min_mac=('mac', 'min'),
                             n_swaps=('fe_mode_matched', 'nunique')).reset_index()
rm.to_csv(TAB / 'fe_residual_by_mode.csv', index=False)

# ------------------------------------------------ key-number ledger ---------
s50, s5 = kw['by_snr']['50.0'], kw['by_snr']['5.0']
key = {
    'kw51': {
        'records': 11328, 'calibration_days': kw['n_calibration_days'], 'test_days': kw['n_test_days'],
        'rejected_days': kw['rejected_days'],
        'envelope_abs_frozen': kw['envelope_abs']['frozen'],
        'envelope_scale': kw['envelope_scale'],
        'snr50': {k: s50[k] for k in ('states_rank', 'states_noise', 'certified', 'informative',
                                     'n_problems_at_prior_frozen', 'min_rel_frozen', 'argmin_rel_frozen',
                                     'layout_report_distribution', 'always_report_modes',
                                     'never_report_modes', 'rel_nodrift_full', 'rank_floor_full',
                                     'rank_floor_deck', 'noise_equals_rank_all_layouts')},
        'snr5': {k: s5[k] for k in ('states_rank', 'states_noise', 'certified', 'informative',
                                    'n_problems_at_prior_frozen')},
        'legacy_floor_snr5_full_REPORT': int(thr[(thr.snr == 5.0) & (thr.layout == 'full') & (thr.tau1 == 0.2)
                                                 & (thr.tau2 == 0.6)].REPORT.iloc[0]),
        'critical_drift': kw['critical_drift'],
        'coverage': kwc, 'prepost': kwp,
        'singular_values': kw['singular_values'],
    },
    'fe': {'partA': fe['partA'], 'partB': fe['partB'], 'n_cases': fe['n_cases'],
           'residual_by_mode': rm.to_dict('records')},
    'hbta': {'modal': {k: hbm[k] for k in ('n_reference_modes', 'reference_freqs_hz', 'median_mac_full',
                                          'median_mpc_full', 'nperseg', 'band_hz')},
             'session': {k: hb[k] for k in ('noise', 'usable_observables_all', 'insample_state_counts_presence',
                                            'insample_certified_presence', 'single_sensor_report_rate',
                                            'loeo_full', 'loeo_single', 'loeo_attribution_full',
                                            'direction_attribution')},
             'session_geometry': {'signature_norm': geo['signature_norm'], 'envelope_norm': geo['envelope_norm'],
                                  'max_abs_cos_benign': geo['max_abs_cos_benign']},
             'id_only': {k: hb_id[k] for k in ('noise', 'loeo_full', 'insample_certified_presence',
                                               'direction_attribution')},
             'id_only_geometry': {'signature_norm': hb_id['geometry_full']['signature_norm'],
                                  'envelope_norm': hb_id['geometry_full']['envelope_norm']},
             'per_state': ht.to_dict('records')},
    'osp': {k: osp[k] for k in ('mean_report_regret_k1to5', 'max_report_regret_k1to5',
                                'mean_certified_regret_k1to5', 'D_loss_of_query_aware_k1to5',
                                'regret_by_eta', 'E_degenerate_all_eta')},
    'osp_spearman_ref': [r for r in osp['spearman'] if abs(r['eta_rel'] - 0.01) < 1e-12],
}
(RES / 'key_numbers.json').write_text(json.dumps(key, indent=1, default=float))
print(thr[(thr.tau1 == 0.2) & (thr.tau2 == 0.6)].to_string(index=False))
print(mt.round(3).to_string(index=False))
print(ht.round(3).to_string(index=False))
print(pd.DataFrame(lay).to_string(index=False))
print(fa.round(3).to_string(index=False))
print(pd.DataFrame(fb).round(3).to_string(index=False))
print(rm.round(2).to_string(index=False))
