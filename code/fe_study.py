"""Controlled-truth study on the official KW51 finite-element model.

Estimand: relative perturbations z = (E_s, h_b, I_d, A_d) of the steel modulus,
ballast thickness, diagonal moment of inertia and diagonal area of the official
model (Zenodo 17366772), prior box b = 0.1. Observables: relative changes of
the 14 tracked frequencies (sd 0.005) and changes of the unit-norm mode shapes
at the layout channels (sd 0.02), linearised by central differences (h = 1e-3).

Part A - linear-Gaussian regime (the model is exact):
    3 observation variants x 3 noise levels x 3 layouts x 10 queries; Monte
    Carlo coverage of the naive (projection, noise-only) interval and of the
    certificate, for theta uniform in the box and at the worst-case vertex.

Part B - nonlinear validation (the model is wrong):
    45 full finite-element solutions at known parameter points (single-axis
    injections of +-0.02 and +-0.1 and 13 box corners) replace the linear
    prediction. The additive linearisation residual is bounded by an envelope
    calibrated on the 16 single-axis solutions only (2p solves an analyst would
    run) and validated on the 13 corner solutions.

Outputs: results/fe/*.csv, fe_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).parent))
import qcert as Q
from paths import ROUND, KW51_FE_DIR as FE, KW51_FE_CASES as CASES

OUT = ROUND / 'results' / 'fe'
OUT.mkdir(parents=True, exist_ok=True)

BOX = 0.1
N_DRAW = 4000
SEED = 20261012
rng = np.random.default_rng(SEED)

s = loadmat(FE / 'kw51_fe_sensitivity.mat', squeeze_me=True)
Jf = np.asarray(s['Jf'], float)              # (14, 4)
Jphi = np.asarray(s['Jphi'], float)          # (12, 14, 4)
f0 = np.asarray(s['base_f'], float)
phi0 = np.asarray(s['base_phi'], float)
SF0, SP0 = float(s['sigma_f']), float(s['sigma_phi'])
NM, NP = Jf.shape
PNAMES = ['E_s', 'h_b', 'I_d', 'A_d']

LAYOUTS = {'deck': 0, 'half': 0b010101, 'full': 63}


def layout_ids(mask):
    return list(range(6)) + [6 + c for c in range(6) if mask >> c & 1]


def build(mask, variant, mult=1.0):
    ids = layout_ids(mask)
    S, sd = [], []
    if variant in ('both', 'freq'):
        S.append(Jf)
        sd.append(np.full(NM, SF0 * mult))
    if variant in ('both', 'shape'):
        S.append(Jphi[ids].reshape(-1, NP))
        sd.append(np.full(len(ids) * NM, SP0 * mult))
    return np.vstack(S), np.concatenate(sd), ids


def observe(f, phi, ids, variant):
    parts = []
    if variant in ('both', 'freq'):
        parts.append((f - f0) / f0)
    if variant in ('both', 'shape'):
        parts.append((phi[ids] - phi0[ids]).ravel())
    return np.concatenate(parts)


# ------------------------------------------------------------- queries ------
Qd = {f'{n}': np.eye(NP)[i] for i, n in enumerate(PNAMES)}
Qd['I_d+A_d'] = np.array([0, 0, 1, 1.]) / np.sqrt(2)
Qd['I_d-A_d'] = np.array([0, 0, 1, -1.]) / np.sqrt(2)
Qd['E_s-h_b'] = np.array([1, -1, 0, 0.]) / np.sqrt(2)
Qd['global'] = np.ones(NP) / 2
S_deck, sd_deck, _ = build(0, 'both')
lam, V = np.linalg.eigh((S_deck / sd_deck[:, None]).T @ (S_deck / sd_deck[:, None]))
for j in range(2):
    v = V[:, j] * np.sign(V[np.argmax(np.abs(V[:, j])), j])
    Qd[f'weak{j + 1}'] = v
QN = list(Qd)
QM = np.array([Qd[k] for k in QN])

# --------------------------------------------------- derivative consistency --
cases = []
for i in range(1, 47):
    c = loadmat(CASES / f'case_{i:03d}.mat', squeeze_me=True)
    ph = np.asarray(c['phi'], float).copy()
    # the stored solutions use their own sign reference; align every mode with
    # the base shapes used for the sensitivities (shape sign is arbitrary)
    flip = np.sign(np.sum(ph * phi0, axis=0))
    flip[flip == 0] = 1.0
    ph *= flip[None, :]
    cases.append({'name': str(c['case_name']), 'z': np.atleast_1d(np.asarray(c['z'], float)),
                  'f': np.asarray(c['f'], float), 'phi': ph,
                  'mac': np.asarray(c['mac'], float), 'match': np.asarray(c['match']).astype(int),
                  'mac_min': float(np.min(c['mac']))})
byname = {c['name']: c for c in cases}
chk = {}
for j in range(NP):
    cp_, cm_ = byname[f'derivative_h0.001_j{j + 1}_s1'], byname[f'derivative_h0.001_j{j + 1}_s-1']
    dJf = ((cp_['f'] - cm_['f']) / f0) / 0.002
    dJp = (cp_['phi'] - cm_['phi']) / 0.002
    chk[PNAMES[j]] = {'freq_maxabs': float(np.abs(dJf - Jf[:, j]).max()),
                      'shape_maxabs': float(np.abs(dJp - Jphi[:, :, j]).max()),
                      'shape_scale': float(np.abs(Jphi[:, :, j]).max())}
print('derivative consistency', json.dumps(chk, indent=1))

# ================================================================ Part A ====
recA = []
for variant in ('freq', 'shape', 'both'):
    for mult in (1.0, 3.0, 10.0):
        for lname, mask in LAYOUTS.items():
            S, sd, ids = build(mask, variant, mult)
            S_w = S / sd[:, None]
            g = Q.geometry(S_w, BOX)
            sol = Q.CertificateSolver(S_w.shape[0], NP)
            th = rng.uniform(-BOX, BOX, (N_DRAW, NP))
            Y = th @ S_w.T + rng.standard_normal((N_DRAW, S_w.shape[0]))
            for qn, q in zip(QN, QM):
                c = sol.solve(S_w, q, BOX)
                aP = g.proj_weights(S_w, q)
                wN = Q.Z95 * np.linalg.norm(aP)
                rec = {'variant': variant, 'noise_mult': mult, 'layout': lname, 'query': qn,
                       'pnull': g.pnull(q), 'state': Q.state_from_pnull(g.pnull(q)),
                       'w0': c.w0, 'halfwidth': c.width, 'rel_width': c.rel,
                       'naive_halfwidth': wN, 'limitation': c.limitation()}
                for tag, a, w in (('cert', c.a, c.width), ('naive', aP, wN)):
                    rec[f'cov_uniform_{tag}'] = float(np.mean(np.abs(Y @ a - th @ q) <= w))
                    vert = BOX * np.where(S_w.T @ a - q >= 0, 1.0, -1.0)
                    yv = vert @ S_w.T + rng.standard_normal((N_DRAW, S_w.shape[0]))
                    rec[f'cov_worst_{tag}'] = float(np.mean(np.abs(yv @ a - vert @ q) <= w))
                recA.append(rec)
A = pd.DataFrame(recA)
A.to_csv(OUT / 'fe_linear_grid.csv', index=False)
print(A.groupby(['variant', 'noise_mult'])[['cov_worst_cert', 'cov_worst_naive',
                                             'cov_uniform_naive']].min().round(3))

# ================================================================ Part B ====
# 16 single-axis solutions (+-0.02, +-0.1 per parameter) and 13 box corners
inj = [c for c in cases if c['name'].startswith('injection')]
corners = [c for c in cases if c['name'].startswith('corner')]
evalset = inj + corners
# per-mode linearisation residual (whitened, max over channels) for the record
resid_rows = []
for c in evalset:
    rf = ((c['f'] - f0) / f0 - Jf @ c['z']) / SF0
    rp = np.abs(c['phi'] - phi0 - Jphi @ c['z']).max(0) / SP0
    for m in range(NM):
        resid_rows.append({'case': c['name'], 'set': 'axis' if c in inj else 'corner',
                           'mode': m + 1, 'resid_freq_sigma': float(rf[m]),
                           'resid_shape_sigma_max': float(rp[m]), 'mac': float(c['mac'][m]),
                           'fe_mode_matched': int(c['match'][m])})
pd.DataFrame(resid_rows).to_csv(OUT / 'fe_linearisation_residuals.csv', index=False)

recB, envB = [], []
for variant in ('freq', 'shape', 'both'):
    for lname, mask in LAYOUTS.items():
        S, sd, ids = build(mask, variant)
        S_w = S / sd[:, None]
        g = Q.geometry(S_w, BOX)
        sol = Q.CertificateSolver(S_w.shape[0], NP)
        # whitened linearisation residuals r = (y_FE - S z) / sd
        R = {c['name']: (observe(c['f'], c['phi'], ids, variant) - S @ c['z']) / sd for c in evalset}
        rho_axis = np.max(np.abs([R[c['name']] for c in inj]), axis=0)
        envB.append({'variant': variant, 'layout': lname, 'n_obs': int(len(rho_axis)),
                     'axis_env_median': float(np.median(rho_axis)),
                     'axis_env_max': float(rho_axis.max()),
                     'frac_obs_axis_env_gt3': float(np.mean(rho_axis > 3)),
                     'frac_obs_corner_exceeds_axis': float(np.mean(
                         np.max(np.abs([R[c['name']] for c in corners]), 0) > rho_axis + 1e-9))})
        for qn, q in zip(QN, QM):
            c_blind = sol.solve(S_w, q, BOX)
            c_axis = sol.solve(S_w, q, BOX, Q.additive_envelope(rho_axis, BOX))
            aP = g.proj_weights(S_w, q)
            wN = Q.Z95 * np.linalg.norm(aP)
            for case in evalset:
                ytrue = observe(case['f'], case['phi'], ids, variant) / sd
                Y = ytrue[None] + rng.standard_normal((N_DRAW, S_w.shape[0]))
                truth = case['z'] @ q
                rules = [('naive', aP, wN), ('blind', c_blind.a, c_blind.width),
                         ('axis', c_axis.a, c_axis.width)]
                if case in corners:
                    # leave-one-corner-out envelope: axis solutions + the other 12 corners
                    others = [o for o in corners if o is not case]
                    rho_loo = np.max(np.abs([R[o['name']] for o in inj + others]), axis=0)
                    c_loo = sol.solve(S_w, q, BOX, Q.additive_envelope(rho_loo, BOX))
                    rules.append(('loo', c_loo.a, c_loo.width))
                row = {'variant': variant, 'layout': lname, 'query': qn,
                       'case': case['name'], 'set': 'axis' if case in inj else 'corner',
                       'mac_min': case['mac_min'], 'truth': truth,
                       'state': Q.state_from_pnull(g.pnull(q)), 'w0': c_axis.w0}
                for tag, a, w in rules:
                    e = Y @ a - truth
                    row[f'w_{tag}'] = w
                    row[f'cov_{tag}'] = float(np.mean(np.abs(e) <= w))
                    row[f'bias_{tag}'] = float(np.mean(e))
                recB.append(row)
        print('Part B', variant, lname, 'done')
B = pd.DataFrame(recB)
B.to_csv(OUT / 'fe_nonlinear_truth.csv', index=False)
pd.DataFrame(envB).to_csv(OUT / 'fe_nonlinear_envelopes.csv', index=False)


def summ(df, tag):
    if f'cov_{tag}' not in df or df[f'cov_{tag}'].isna().all():
        return None
    d = df.dropna(subset=[f'cov_{tag}'])
    return {'n': int(len(d)), 'mean': float(d[f'cov_{tag}'].mean()),
            'min': float(d[f'cov_{tag}'].min()),
            'frac_below_0.95': float((d[f'cov_{tag}'] < 0.95).mean()),
            'median_rel_width': float((d[f'w_{tag}'] / d.w0).median())}


summary = {
    'derivative_consistency': chk,
    'n_cases': {'axis': len(inj), 'corner': len(corners)},
    'case_min_mac': {c['name']: c['mac_min'] for c in evalset},
    'partA': {
        'n_rows': int(len(A)),
        'state_counts': A.state.value_counts().to_dict(),
        'min_cov_worst_cert_report': float(A[A.state == Q.REPORT].cov_worst_cert.min()),
        'min_cov_worst_cert_all': float(A.cov_worst_cert.min()),
        'min_cov_uniform_cert_all': float(A.cov_uniform_cert.min()),
        'median_cov_uniform_cert': float(A.cov_uniform_cert.median()),
        'naive_nonreport': {'n': int((A.state != Q.REPORT).sum()),
                            'n_naive_uniform_below_0.5': int(((A.state != Q.REPORT)
                                                              & (A.cov_uniform_naive < 0.5)).sum()),
                            'mean_gain_uniform': float((A.cov_uniform_cert - A.cov_uniform_naive)
                                                       [A.state != Q.REPORT].mean()),
                            'mean_gain_worst': float((A.cov_worst_cert - A.cov_worst_naive)
                                                     [A.state != Q.REPORT].mean())},
        'full_both_x1_halfwidths': A[(A.variant == 'both') & (A.noise_mult == 1.0)
                                     & (A.layout == 'full')].set_index('query').halfwidth
        .round(5).to_dict(),
        'deck_both_x1_halfwidths': A[(A.variant == 'both') & (A.noise_mult == 1.0)
                                     & (A.layout == 'deck')].set_index('query').halfwidth
        .round(5).to_dict(),
        'worst_cov_by_cell': A.groupby(['variant', 'noise_mult']).agg(
            cert=('cov_worst_cert', 'min'), naive=('cov_worst_naive', 'min'),
            naive_uniform=('cov_uniform_naive', 'min')).round(4).reset_index().to_dict('records'),
    },
    'partB': {
        'axis_set': {t: summ(B[B.set == 'axis'], t) for t in ('naive', 'blind', 'axis')},
        'corner_set': {t: summ(B[B.set == 'corner'], t) for t in ('naive', 'blind', 'axis', 'loo')},
        'corner_by_variant': {v: {t: summ(B[(B.variant == v) & (B.set == 'corner')], t)
                                  for t in ('naive', 'blind', 'axis', 'loo')}
                              for v in ('freq', 'shape', 'both')},
        'corner_by_layout_both': {l: {t: summ(B[(B.layout == l) & (B.variant == 'both')
                                                & (B.set == 'corner')], t)
                                      for t in ('naive', 'blind', 'axis', 'loo')}
                                  for l in LAYOUTS},
        'envelopes': envB,
    },
}
(OUT / 'fe_summary.json').write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps(summary['partA'], indent=1, default=float)[:3000])
print(json.dumps(summary['partB'], indent=1, default=float))
