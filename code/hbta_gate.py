"""HBTA real-damage audit: damage-attribution queries under cross-condition discrepancy.

Observation model. For excitation set-up c and damage state k the contrast
x_kc = features(DS_k, c) - mean(features(UDS_01, c), features(UDS_02, c)) is
modelled as x_kc = S theta + r_c + eps, with
  S        the empirical damage directions (mean contrast over set-ups) and the
           benign direction (mean of UDS_02 - UDS_01),
  theta    (theta_1..theta_8, theta_b): damage intensities in units of the
           imposed damage and the benign amplitude, prior box b = 1,
  eps      identification noise, sd from the half-record scatter,
  r_c      the set-up-specific deviation of the directions (cross-condition
           discrepancy), bounded by the envelope of the training set-ups.

Features per reference mode: relative frequency change and change of the unit-
norm shape at the 35 channels. Layouts select sensors; a frequency row enters
a layout only if the reference shape carries >= 5 % of its energy on the
layout's channels. Observables seen in fewer than MIN_COMBO set-ups are dropped.

Queries: presence of DS_k (basis e_k, k = 1..8) and the 28 pairwise
discriminants (e_i - e_j)/sqrt(2).

Analyses
  A  in-sample map: directions and envelope from all 5 set-ups, every layout
  B  leave-one-excitation-out (LOEO): directions, envelope from 4 set-ups,
     held-out contrasts of the 5th; interval coverage of the known truth
     (theta_k = 1 for the imposed state, 0 for the others) for the naive,
     noise-only-certified and discrepancy-aware certified rules.

Outputs: results/hbta/*.csv, hbta_gate_summary.json
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import qcert as Q

ROUND = Path(__file__).resolve().parent.parent
OUT = ROUND / 'results' / 'hbta'
z = np.load(ROUND / 'cache' / 'hbta_features.npz', allow_pickle=False)
runs = list(z['runs'])
Fq, Sh = z['freq'], z['shape']
REF_F, REF_SH = z['ref_f'], z['ref_shape']
chans = list(z['chan_names'])
inv = pd.read_csv(OUT / 'hbta_runs.csv', dtype={'rep': str}).set_index('run').loc[runs].reset_index()
inv['rep'] = inv['rep'].str.zfill(2)
R, NC = REF_SH.shape
BOX = 1.0
MIN_COMBO = 2
DS = [f'DS{k}' for k in range(1, 9)]
COMBOS = sorted(inv.combo.unique())
SENSORS = [f'AG{i:02d}' for i in range(1, 19)]
sensor_of = np.array([c.split('.')[0] for c in chans])
rng = np.random.default_rng(20261013)


def feature_vector(ri, tag=0):
    """[relative frequency (R), shapes (R x NC)] of run ri; NaN where unmatched."""
    f = Fq[ri, tag] / REF_F - 1.0
    return np.concatenate([f, Sh[ri, tag].ravel()])


NOBS = R + R * NC
obs_mode = np.concatenate([np.arange(R), np.repeat(np.arange(R), NC)])
obs_kind = np.array(['f'] * R + ['phi'] * (R * NC))
obs_chan = np.array([-1] * R + list(np.tile(np.arange(NC), R)))
obs_sensor = np.array(['FREQ'] * R + list(np.tile(sensor_of, R)))

# ------------------------------------------- identification noise (half split)
d = []
for ri in range(len(runs)):
    a, b = feature_vector(ri, 1), feature_vector(ri, 2)
    d.append(a - b)
d = np.array(d)
# robust sd of the half-to-half difference, pooled over runs; full-record sd = sd(d) / 2
mad = np.nanmedian(np.abs(d - np.nanmedian(d, 0)), 0) * 1.4826
n_pairs = np.sum(np.isfinite(d), 0)
# pool per mode and kind to stabilise observables with few pairs
sd_full = np.full(NOBS, np.nan)
for m in range(R):
    for kind in ('f', 'phi'):
        sel = (obs_mode == m) & (obs_kind == kind)
        vals = d[:, sel]
        pooled = np.nanmedian(np.abs(vals - np.nanmedian(vals))) * 1.4826 if np.isfinite(vals).sum() > 5 else np.nan
        own = np.where(n_pairs[sel] >= 5, mad[sel], pooled)
        sd_full[sel] = 0.5 * np.where(np.isfinite(own), own, pooled)
fallback_f = np.nanmedian(sd_full[obs_kind == 'f'])
fallback_p = np.nanmedian(sd_full[obs_kind == 'phi'])
sd_full = np.where(np.isfinite(sd_full) & (sd_full > 0), sd_full,
                   np.where(obs_kind == 'f', fallback_f, fallback_p))
print('identification noise: freq median sd', np.median(sd_full[obs_kind == 'f']),
      ' shape median sd', np.median(sd_full[obs_kind == 'phi']))

# ------------------------------------------------------------- contrasts ----
idx = {(r.combo, r.state, r.rep): i for i, r in inv.iterrows()}
X = np.full((len(COMBOS), 8, NOBS), np.nan)     # damage contrasts
Bn = np.full((len(COMBOS), NOBS), np.nan)       # benign contrasts
for ci, c in enumerate(COMBOS):
    u1, u2 = feature_vector(idx[(c, 'UDS', '01')]), feature_vector(idx[(c, 'UDS', '02')])
    um = 0.5 * (u1 + u2)
    Bn[ci] = u2 - u1
    for k, ds in enumerate(DS):
        X[ci, k] = feature_vector(idx[(c, ds, '01')]) - um

# ----------------------------------------------- session-level variability --
# scatter of the undamaged-repeat contrasts around their mean, pooled per mode
# and feature kind (5 set-ups per observable are too few for per-entry sds)
bres = Bn - np.nanmean(Bn, 0, keepdims=True)
sd_sess = np.full(NOBS, np.nan)
for m in range(R):
    for kind in ('f', 'phi'):
        sel = (obs_mode == m) & (obs_kind == kind)
        v = bres[:, sel]
        if np.isfinite(v).sum() >= 4:
            sd_sess[sel] = np.sqrt(np.nanmean(v ** 2)) / np.sqrt(2.0)
sd_sess = np.where(np.isfinite(sd_sess), sd_sess,
                   np.where(obs_kind == 'f', np.nanmedian(sd_sess[obs_kind == 'f']),
                            np.nanmedian(sd_sess[obs_kind == 'phi'])))
NOISE_MODEL = sys.argv[1] if len(sys.argv) > 1 else 'id'
if NOISE_MODEL == 'session':
    sd_use = np.sqrt(sd_full ** 2 + sd_sess ** 2)
else:
    sd_use = sd_full
SD_CONTRAST = sd_use * np.sqrt(1.5)          # DS run minus mean of two UDS runs
SD_BENIGN = sd_use * np.sqrt(2.0)
SUFFIX = '' if NOISE_MODEL == 'id' else '_session'
print('noise model', NOISE_MODEL, 'session sd: freq median', np.median(sd_sess[obs_kind == 'f']),
      'shape median', np.median(sd_sess[obs_kind == 'phi']))


def directions(train):
    """Directions S (NOBS x 9, unwhitened), envelope E (NOBS x 9), usable mask."""
    Xt, Bt = X[train], Bn[train]
    cnt = np.sum(np.isfinite(Xt), axis=0)                    # (8, NOBS)
    S = np.column_stack([np.nanmean(Xt[:, k], 0) for k in range(8)] + [np.nanmean(Bt, 0)])
    dev = np.abs(Xt - S[:, :8].T[None])                       # (n_tr, 8, NOBS)
    E = np.column_stack([np.nanmax(dev[:, k], 0) for k in range(8)]
                        + [np.nanmax(np.abs(Bt - S[:, 8][None]), 0)])
    usable = (cnt.min(0) >= MIN_COMBO) & np.isfinite(S).all(1) & np.isfinite(E).all(1)
    return np.nan_to_num(S), np.nan_to_num(E), usable


def layout_rows(sensors, usable):
    """Observable rows of a layout (frequency rows need >= 5 % shape energy on the layout)."""
    ch = np.isin(sensor_of, sensors)
    energy = (REF_SH[:, ch] ** 2).sum(1)
    sees = energy >= 0.05
    rows = np.flatnonzero(usable & (
        ((obs_kind == 'f') & sees[obs_mode]) |
        ((obs_kind == 'phi') & np.isin(obs_sensor, sensors))))
    return rows


Qmat, Qname = [], []
for k in range(8):
    Qmat.append(np.eye(9)[k])
    Qname.append(f'DS{k + 1}')
for i, j in combinations(range(8), 2):
    v = np.zeros(9)
    v[i], v[j] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    Qmat.append(v)
    Qname.append(f'DS{i + 1}-DS{j + 1}')
Qmat = np.array(Qmat)

LAYOUTS = {'full': SENSORS,
           'north wall': SENSORS[:9], 'south wall': SENSORS[9:],
           'deck joints': ['AG01', 'AG03', 'AG05', 'AG07', 'AG09', 'AG11', 'AG13', 'AG15', 'AG17'],
           'top-chord joints': ['AG02', 'AG04', 'AG06', 'AG08', 'AG10', 'AG12', 'AG14', 'AG16', 'AG18'],
           'west half': ['AG01', 'AG02', 'AG03', 'AG04', 'AG10', 'AG11', 'AG12', 'AG13'],
           'east half': ['AG06', 'AG07', 'AG08', 'AG09', 'AG15', 'AG16', 'AG17', 'AG18']}
for s in SENSORS:
    LAYOUTS[f'single {s}'] = [s]
for k in (2, 3, 4, 6, 9, 12):
    for r in range(10):
        LAYOUTS[f'random{k}-{r}'] = sorted(rng.choice(SENSORS, k, replace=False).tolist())

# ================================================== A: in-sample map ========
S_all, E_all, use_all = directions(list(range(len(COMBOS))))
recA = []
for lname, sens in LAYOUTS.items():
    rows = layout_rows(sens, use_all)
    if len(rows) == 0:
        continue
    sd = SD_CONTRAST[rows]
    S_w = S_all[rows] / sd[:, None]
    E_w = E_all[rows] / sd[:, None]
    g = Q.geometry(S_w, BOX)
    sol = Q.CertificateSolver(len(rows), 9)
    for qn, q in zip(Qname, Qmat):
        c0 = sol.solve(S_w, q, BOX)
        cE = sol.solve(S_w, q, BOX, E_w)
        recA.append({'layout': lname, 'n_sensors': len(sens), 'n_rows': int(len(rows)),
                     'query': qn, 'kind': 'presence' if '-' not in qn else 'discriminant',
                     'pnull': g.pnull(q), 'state': Q.state_from_pnull(g.pnull(q)),
                     'rel_noise_only': c0.rel, 'rel_certified': cE.rel,
                     'noise': cE.noise, 'alias': cE.alias, 'drift': cE.drift,
                     'limitation': cE.limitation(), 'signature_norm':
                         float(np.linalg.norm(S_w[:, int(np.argmax(np.abs(q)))]))})
A = pd.DataFrame(recA)
A.to_csv(OUT / f'hbta_insample_map{SUFFIX}.csv', index=False)

# signature geometry (full layout)
rows_full = layout_rows(SENSORS, use_all)
Sw_full = S_all[rows_full] / SD_CONTRAST[rows_full][:, None]
norms = np.linalg.norm(Sw_full, axis=0)
cosm = (Sw_full.T @ Sw_full) / np.outer(norms, norms)
Ew_full = E_all[rows_full] / SD_CONTRAST[rows_full][:, None]
geo = {'n_rows_full': int(len(rows_full)),
       'signature_norm': dict(zip(DS + ['benign'], norms.round(2).tolist())),
       'envelope_norm': dict(zip(DS + ['benign'], np.linalg.norm(Ew_full, axis=0).round(2).tolist())),
       'max_abs_cos_benign': float(np.max(np.abs(cosm[8, :8]))),
       'cos_matrix': cosm.round(3).tolist()}

# ================================================== B: LOEO =================
recB, recC = [], []
for hc in range(len(COMBOS)):
    train = [i for i in range(len(COMBOS)) if i != hc]
    S_tr, E_tr, use_tr = directions(train)
    for lname in ['full', 'north wall', 'south wall', 'deck joints', 'top-chord joints',
                  'west half', 'east half'] + [f'single {s}' for s in SENSORS]:
        sens = LAYOUTS[lname]
        base_rows = layout_rows(sens, use_tr)
        for k in range(8):
            y_raw = X[hc, k]
            rows = base_rows[np.isfinite(y_raw[base_rows])]
            if len(rows) < 3:
                continue
            sd = SD_CONTRAST[rows]
            S_w = S_tr[rows] / sd[:, None]
            E_w = E_tr[rows] / sd[:, None]
            y_w = y_raw[rows] / sd
            g = Q.geometry(S_w, BOX)
            sol = Q.CertificateSolver(len(rows), 9)
            truth = np.zeros(9)
            truth[k] = 1.0
            # direction-only attribution: nearest training damage direction by cosine
            cosv = (S_w[:, :8].T @ y_w) / (np.linalg.norm(S_w[:, :8], axis=0)
                                           * np.linalg.norm(y_w) + 1e-300)
            recC.append({'held_out': COMBOS[hc], 'layout': lname, 'true_state': DS[k],
                         'nearest': DS[int(np.argmax(cosv))], 'cos_true': float(cosv[k]),
                         'cos_best_other': float(np.max(np.delete(cosv, k))),
                         'amplitude_ratio': float((S_w[:, k] @ y_w) / (S_w[:, k] @ S_w[:, k]))})
            for qi in range(8):                       # presence queries only
                q = Qmat[qi]
                aP = g.proj_weights(S_w, q)
                c0 = sol.solve(S_w, q, BOX)
                cE = sol.solve(S_w, q, BOX, E_w)
                rec = {'held_out': COMBOS[hc], 'layout': lname, 'n_sensors': len(sens),
                       'true_state': DS[k], 'query': DS[qi], 'truth': truth[qi],
                       'n_rows': int(len(rows)), 'pnull': g.pnull(q),
                       'state': Q.state_from_pnull(g.pnull(q))}
                for tag, a, w in (('naive', aP, Q.Z95 * np.linalg.norm(aP)),
                                  ('noise_only', c0.a, c0.width), ('certified', cE.a, cE.width)):
                    est = float(a @ y_w)
                    rec[f'est_{tag}'] = est
                    rec[f'w_{tag}'] = float(w)
                    rec[f'cover_{tag}'] = bool(abs(est - truth[qi]) <= w)
                    rec[f'excl0_{tag}'] = bool(abs(est) > w)
                recB.append(rec)
    print('LOEO held out', COMBOS[hc], 'done')
B = pd.DataFrame(recB)
B.to_csv(OUT / f'hbta_loeo{SUFFIX}.csv', index=False)
C = pd.DataFrame(recC)
C['correct'] = C.nearest == C.true_state
C.to_csv(OUT / f'hbta_loeo_direction{SUFFIX}.csv', index=False)


def loeo_summary(df):
    out = {'n': int(len(df))}
    for tag in ('naive', 'noise_only', 'certified'):
        on = df[df.truth == 1.0]
        off = df[df.truth == 0.0]
        out[tag] = {
            'coverage': float(df[f'cover_{tag}'].mean()),
            'coverage_true_state': float(on[f'cover_{tag}'].mean()),
            'coverage_other_states': float(off[f'cover_{tag}'].mean()),
            'detection_rate': float(on[f'excl0_{tag}'].mean()),
            'false_attribution_rate': float(off[f'excl0_{tag}'].mean()),
            'median_halfwidth': float(df[f'w_{tag}'].median())}
    return out


def attribution(df, tag):
    """Fraction of held-out contrasts whose interval pattern singles out the true state."""
    ok = []
    for (ho, lay, ts), g in df.groupby(['held_out', 'layout', 'true_state']):
        if len(g) < 8:
            continue
        on = g[g['query'] == ts]
        off = g[g['query'] != ts]
        ok.append(bool(on[f'excl0_{tag}'].all() and not off[f'excl0_{tag}'].any()))
    return float(np.mean(ok)) if ok else float('nan')


summary = {
    'n_reference_modes': int(R), 'n_channels': int(NC), 'combos': COMBOS,
    'noise': {'freq_median_sd_rel': float(np.median(sd_full[obs_kind == 'f'])),
              'shape_median_sd': float(np.median(sd_full[obs_kind == 'phi']))},
    'usable_observables_all': int(use_all.sum()),
    'geometry_full': geo,
    'insample_full': A[A.layout == 'full'][['query', 'state', 'rel_noise_only', 'rel_certified',
                                             'limitation']].to_dict('records'),
    'insample_state_counts_presence': {
        l: A[(A.layout == l) & (A.kind == 'presence')].state.value_counts().to_dict()
        for l in ['full', 'north wall', 'south wall', 'deck joints', 'top-chord joints',
                  'west half', 'east half']},
    'insample_certified_presence': {
        l: {'noise_only': int((A[(A.layout == l) & (A.kind == 'presence')].rel_noise_only <= 0.5).sum()),
            'certified': int((A[(A.layout == l) & (A.kind == 'presence')].rel_certified <= 0.5).sum())}
        for l in ['full', 'north wall', 'south wall', 'deck joints', 'top-chord joints',
                  'west half', 'east half']},
    'single_sensor_report_rate': A[A.layout.str.startswith('single') & (A.kind == 'presence')]
    .groupby('query').state.apply(lambda s: float(np.mean(s == Q.REPORT))).to_dict(),
    'single_sensor_cert_rate': A[A.layout.str.startswith('single') & (A.kind == 'presence')]
    .groupby('query').rel_certified.apply(lambda s: float(np.mean(s <= 0.5))).to_dict(),
    'loeo_full': loeo_summary(B[B.layout == 'full']),
    'loeo_multi_sensor': loeo_summary(B[~B.layout.str.startswith('single')]),
    'loeo_single': loeo_summary(B[B.layout.str.startswith('single')]),
    'loeo_attribution_full': {t: attribution(B[B.layout == 'full'], t)
                              for t in ('naive', 'noise_only', 'certified')},
    'loeo_full_by_heldout': {
        h: loeo_summary(B[(B.layout == 'full') & (B.held_out == h)]) for h in COMBOS},
    'direction_attribution': {
        'full_accuracy': float(C[C.layout == 'full'].correct.mean()),
        'full_by_heldout': C[C.layout == 'full'].groupby('held_out').correct.mean().to_dict(),
        'full_by_state': C[C.layout == 'full'].groupby('true_state').correct.mean().to_dict(),
        'full_amplitude_ratio_median': float(C[C.layout == 'full'].amplitude_ratio.median()),
        'full_amplitude_ratio_iqr': [float(C[C.layout == 'full'].amplitude_ratio.quantile(.25)),
                                     float(C[C.layout == 'full'].amplitude_ratio.quantile(.75))],
        'multi_sensor_accuracy': C[~C.layout.str.startswith('single')].groupby('layout')
        .correct.mean().to_dict(),
        'single_sensor_accuracy': float(C[C.layout.str.startswith('single')].correct.mean()),
        'chance': 1 / 8},
    'loeo_full_by_state': {
        s: {t: {'coverage_true': float(B[(B.layout == 'full') & (B.true_state == s)
                                         & (B['query'] == s)][f'cover_{t}'].mean()),
                'detect': float(B[(B.layout == 'full') & (B.true_state == s)
                                  & (B['query'] == s)][f'excl0_{t}'].mean())}
            for t in ('naive', 'noise_only', 'certified')} for s in DS},
}
(OUT / f'hbta_gate_summary{SUFFIX}.json').write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps({k: summary[k] for k in ('noise', 'usable_observables_all', 'insample_state_counts_presence',
                                           'insample_certified_presence', 'loeo_full',
                                           'loeo_attribution_full', 'loeo_single',
                                           'direction_attribution')}, indent=1, default=float))
print(json.dumps(geo['signature_norm']), json.dumps(geo['envelope_norm']), geo['max_abs_cos_benign'])
print(A[A.layout == 'full'][['query', 'state', 'rel_noise_only', 'rel_certified', 'limitation']]
      .head(8).to_string(index=False))
