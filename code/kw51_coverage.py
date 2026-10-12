"""KW51 out-of-sample validity under the real dictionary variability.

Estimators and certificates are built on the training dictionary (Sep-Oct
2019) with the discrepancy envelope of the calibration month (Nov 2019). They
are applied to synthetic modal-coordinate vectors observed through the daily
dictionaries of the held-out winter window (Dec 2019 - Jan 2020), i.e. the
true observation operator is the measured dictionary of the test day. Three
interval rules are compared for every layout and query:

  naive      retained-space projection estimator, noise-only 95 % interval;
  blind      certified estimator without discrepancy term (E = 0);
  certified  certified estimator with the calibration-month envelope.

Two truth generators are used: theta uniform in the prior box (typical case)
and theta at the box vertex that maximises the bias of the given estimator on
the given test day (worst case, the regime the certificate is designed for).

Outputs: results/kw51/kw51_coverage.csv, kw51_coverage_summary.json
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
SNR = 50.0
SIGMA = BOX / SNR
N_DRAW = 2000
SEED = 20261011
P = K.N_MODE

D = K.build_post_retrofit()
phi = D['phi_train']
E_frozen = K.envelope_frozen(D)
test = D['daily'][D['in_window']['test']]
print('test days', len(test))
rng = np.random.default_rng(SEED)

rows = []
for mask in (0, 7, 56, 63):
    ids = K.layout_ids(mask)
    S = phi[ids]
    S_w = S / SIGMA
    g = Q.geometry(S_w, BOX)
    sol = Q.CertificateSolver(len(ids), P)
    for m in range(P):
        q = np.eye(P)[m]
        rules = {}
        aP = g.proj_weights(S_w, q)
        rules['naive'] = (aP, Q.Z95 * np.linalg.norm(aP))
        c0 = sol.solve(S_w, q, BOX, None)
        rules['blind'] = (c0.a, c0.width)
        cE = sol.solve(S_w, q, BOX, E_frozen[ids] / SIGMA)
        rules['certified'] = (cE.a, cE.width)
        for rule, (a, w) in rules.items():
            cov_u, cov_w, n_u, n_w = 0, 0, 0, 0
            for Sd in test:
                Sd_w = Sd[ids] / SIGMA
                # typical truth: theta uniform in the box
                th = rng.uniform(-BOX, BOX, (N_DRAW, P))
                y = th @ Sd_w.T + rng.standard_normal((N_DRAW, len(ids)))
                err = y @ a - th @ q
                cov_u += int(np.sum(np.abs(err) <= w))
                n_u += N_DRAW
                # worst-case truth: vertex maximising the bias of this estimator
                bias_dir = Sd_w.T @ a - q
                th_w = BOX * np.where(bias_dir >= 0, 1.0, -1.0)
                yw = th_w @ Sd_w.T + rng.standard_normal((N_DRAW, len(ids)))
                errw = yw @ a - th_w @ q
                cov_w += int(np.sum(np.abs(errw) <= w))
                n_w += N_DRAW
            rows.append({'mask': mask, 'mode': m + 1, 'rule': rule,
                         'halfwidth_rel': w / BOX, 'informative': bool(w < BOX * (1 - 1e-6)),
                         'coverage_uniform': cov_u / n_u, 'coverage_worst': cov_w / n_w,
                         'state': Q.state_from_pnull(g.pnull(q))})
    print('layout', mask, 'done')
cov = pd.DataFrame(rows)
cov.to_csv(OUT / 'kw51_coverage.csv', index=False)


def agg(df):
    return {'n': int(len(df)),
            'mean_cov_uniform': float(df.coverage_uniform.mean()),
            'min_cov_uniform': float(df.coverage_uniform.min()),
            'mean_cov_worst': float(df.coverage_worst.mean()),
            'min_cov_worst': float(df.coverage_worst.min()),
            'n_below_0.95_worst': int((df.coverage_worst < 0.95).sum()),
            'n_below_0.95_uniform': int((df.coverage_uniform < 0.95).sum()),
            'median_halfwidth_rel': float(df.halfwidth_rel.median())}


summary = {'snr': SNR, 'n_draw_per_day': N_DRAW, 'n_test_days': int(len(test)),
           'by_rule': {r: agg(cov[cov.rule == r]) for r in ('naive', 'blind', 'certified')},
           'by_rule_layout': {f'{r}_{m}': agg(cov[(cov.rule == r) & (cov['mask'] == m)])
                              for r in ('naive', 'blind', 'certified') for m in (0, 63)},
           'informative_rows': {r: agg(cov[(cov.rule == r) & cov.informative])
                                for r in ('naive', 'blind', 'certified')},
           'report_state_rows': {r: agg(cov[(cov.rule == r) & (cov.state == Q.REPORT)])
                                 for r in ('naive', 'blind', 'certified')}}
(OUT / 'kw51_coverage_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=1))
print(cov[cov['mask'] == 63].pivot_table(index='mode', columns='rule',
                                         values=['coverage_worst', 'halfwidth_rel']).round(3))
