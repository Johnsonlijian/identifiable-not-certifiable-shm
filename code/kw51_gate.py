"""KW51 field audit: modal-coordinate queries over all 64 arch-channel layouts.

Estimand: theta in R^14, the modal coordinates of the 14 tracked modes in a
response vector measured at the array, y = Phi_L theta + eps (the operation
behind modal filtering, modal expansion and virtual sensing). Query m asks for
the coordinate of mode m (basis query e_m). The prior box is b = 0.1. The
observation noise is declared through the signal-to-noise ratio b / sigma:
the reference operating point is b / sigma = 50 and the noise-limited case
b / sigma = 5 (the setting of earlier analyses) is reported alongside; the
phase diagram sweeps 5 - 250.

Outputs (results/kw51/):
  kw51_enumeration.csv        64 layouts x 14 queries x 2 SNR: rank-, noise- and
                              certified-level quantities and error budgets
  kw51_envelopes.npz          dictionary-discrepancy envelopes
  kw51_envelope_scale.csv     envelope size versus dictionary averaging window
  kw51_phase_drift_snr.csv    certified-query counts on a discrepancy x SNR grid
  kw51_critical_drift.csv     per query: largest discrepancy fraction still certifiable
  kw51_policy.csv             per layout: state and certification counts
  kw51_summary.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import qcert as Q
import kw51_data as K

OUT = K.ROUND / 'results' / 'kw51'
OUT.mkdir(parents=True, exist_ok=True)

BOX = 0.1
SNR_REF, SNR_LOW = 50.0, 5.0
KAPPA_CERT = 0.5            # a query is "certified" when w* <= 0.5 w0
P = K.N_MODE
QUERIES = np.eye(P)

D = K.build_post_retrofit()
phi = D['phi_train']                                        # (12, 14)
E_frozen = K.envelope_frozen(D)
E_roll = {h: K.envelope_rolling(D, h) for h in (1, 7)}

# -------------------------------- envelope size versus averaging window -----
# k-day block dictionaries in the calibration month compared with the training
# dictionary: identification scatter shrinks with k, environmental drift does not
cal_idx = np.flatnonzero(D['in_window']['calibration'])
al = D['aligned']
dates = D['raw']['dates']
scale_rows = []
for kdays in (1, 2, 3, 5, 7, 10, 15):
    blocks = []
    start = datetime(2019, 11, 1)
    while True:
        a = start
        b = datetime.fromordinal(a.toordinal() + kdays)
        if b > datetime(2019, 12, 1):
            break
        sel = (dates >= a) & (dates < b)
        Dk = K.aggregate(al, sel)
        if Dk is not None:
            blocks.append(Dk)
        start = b
    dev = np.abs(np.array(blocks) - phi[None])
    scale_rows.append({'window_days': kdays, 'n_blocks': len(blocks),
                       'median_abs_dev': float(np.median(dev)),
                       'p90_abs_dev': float(np.quantile(dev, .9)),
                       'max_abs_dev': float(dev.max()),
                       'mean_block_max': float(dev.max(0).mean())})
scale = pd.DataFrame(scale_rows)
scale.to_csv(OUT / 'kw51_envelope_scale.csv', index=False)
print(scale.round(5).to_string(index=False))

np.savez_compressed(OUT / 'kw51_envelopes.npz', phi_train=phi, E_frozen=E_frozen,
                    E_roll1=E_roll[1][0], E_roll7=E_roll[7][0], days=np.array(D['days']),
                    daily=D['daily'], labels=np.array(D['labels']),
                    freq_train=D['freq_train'])
print('calibration days', int(D['in_window']['calibration'].sum()),
      'test days', int(D['in_window']['test'].sum()), 'rejected', D['rejected_days'])
for nm, E in (('frozen', E_frozen), ('roll1', E_roll[1][0]), ('roll7', E_roll[7][0])):
    print(f'envelope {nm}: median {np.median(E):.4f} p90 {np.quantile(E, .9):.4f} max {E.max():.4f}'
          f' (prior b = {BOX})')

# ------------------------------------------------------------ enumeration ---
rows = []
solvers = {}
for snr in (SNR_REF, SNR_LOW):
    sigma = BOX / snr
    for mask in range(64):
        ids = K.layout_ids(mask)
        n = len(ids)
        if n not in solvers:
            solvers[n] = Q.CertificateSolver(n, P)
        sol = solvers[n]
        S_w = phi[ids] / sigma
        g = Q.geometry(S_w, BOX)
        for m in range(P):
            q = QUERIES[m]
            c_fr = sol.solve(S_w, q, BOX, E_frozen[ids] / sigma)
            c_r1 = sol.solve(S_w, q, BOX, E_roll[1][0][ids] / sigma)
            c_r7 = sol.solve(S_w, q, BOX, E_roll[7][0][ids] / sigma)
            c_00 = sol.solve(S_w, q, BOX, None)
            rf = Q.rank_floor(S_w, q, BOX)
            rows.append({
                'snr': snr, 'mask': mask, 'k': mask.bit_count(), 'n_channels': n, 'mode': m + 1,
                'rank': g.rank, 'retained_rank': g.retained_rank,
                'pnull_exact': g.pnull_exact(q), 'state_rank': Q.state_from_pnull(g.pnull_exact(q)),
                'pnull': g.pnull(q), 'state': Q.state_from_pnull(g.pnull(q)),
                'naive_halfwidth': Q.Z95 * g.proj_sd(q),
                'w0': c_fr.w0, 'rank_floor_rel': rf / c_fr.w0,
                'rel_nodrift': c_00.rel, 'noise_nodrift': c_00.noise, 'alias_nodrift': c_00.alias,
                'rel_frozen': c_fr.rel, 'noise_frozen': c_fr.noise, 'alias_frozen': c_fr.alias,
                'drift_frozen': c_fr.drift, 'limitation_frozen': c_fr.limitation(),
                'rel_roll7': c_r7.rel, 'rel_roll1': c_r1.rel,
                'limitation_nodrift': c_00.limitation(),
            })
        if mask % 32 == 31:
            print(f'snr {snr}: enumerated through mask {mask}')
enum = pd.DataFrame(rows)
enum.to_csv(OUT / 'kw51_enumeration.csv', index=False)


def counts(df, col):
    v = df[col].value_counts()
    return [int(v.get(s, 0)) for s in (Q.REPORT, Q.BOUND, Q.REFUSE)]


# ------------------------------------- phase diagram discrepancy x SNR ------
RHO = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
SNRS = [5, 10, 25, 50, 100, 250]
phase = []
for mask in (0, 7, 56, 63):
    ids = K.layout_ids(mask)
    sol = solvers[len(ids)]
    for snr in SNRS:
        sig = BOX / snr
        S_w = phi[ids] / sig
        for rho in RHO:
            Ew = rho * E_frozen[ids] / sig
            rel = np.array([sol.solve(S_w, QUERIES[m], BOX, Ew).rel for m in range(P)])
            phase.append({'mask': mask, 'snr': snr, 'rho': rho,
                          'n_certified': int(np.sum(rel <= KAPPA_CERT)),
                          'n_informative': int(np.sum(rel < 1 - 1e-6)),
                          'rel_widths': ';'.join(f'{r:.4f}' for r in rel)})
pd.DataFrame(phase).to_csv(OUT / 'kw51_phase_drift_snr.csv', index=False)

# ------------------ critical discrepancy fraction per query (reference SNR) --
crit = []
sigma = BOX / SNR_REF
for mask in (0, 63):
    ids = K.layout_ids(mask)
    sol = solvers[len(ids)]
    S_w = phi[ids] / sigma
    for m in range(P):
        q = QUERIES[m]
        f0 = sol.solve(S_w, q, BOX, None).rel
        rec = {'mask': mask, 'mode': m + 1, 'rel_nodrift': f0}
        for thr, tag in ((KAPPA_CERT, 'cert'), (1 - 1e-6, 'info')):
            if f0 > thr:
                rec[f'rho_{tag}'] = 0.0
                continue
            if sol.solve(S_w, q, BOX, E_frozen[ids] / sigma).rel <= thr:
                rec[f'rho_{tag}'] = 1.0
                continue
            lo, hi = 0.0, 1.0
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if sol.solve(S_w, q, BOX, mid * E_frozen[ids] / sigma).rel <= thr:
                    lo = mid
                else:
                    hi = mid
            rec[f'rho_{tag}'] = lo
        crit.append(rec)
crit = pd.DataFrame(crit)
crit.to_csv(OUT / 'kw51_critical_drift.csv', index=False)

# ------------------------------------------------------- per-layout table ---
def layout_table(df):
    return df.groupby('mask').agg(
        k=('k', 'first'),
        report_rank=('state_rank', lambda s: int((s == Q.REPORT).sum())),
        report=('state', lambda s: int((s == Q.REPORT).sum())),
        bound=('state', lambda s: int((s == Q.BOUND).sum())),
        refuse=('state', lambda s: int((s == Q.REFUSE).sum())),
        cert_nodrift=('rel_nodrift', lambda s: int((s <= KAPPA_CERT).sum())),
        cert_frozen=('rel_frozen', lambda s: int((s <= KAPPA_CERT).sum())),
        info_nodrift=('rel_nodrift', lambda s: int((s < 1 - 1e-6).sum())),
        info_frozen=('rel_frozen', lambda s: int((s < 1 - 1e-6).sum())),
        mean_rel_nodrift=('rel_nodrift', 'mean'),
        mean_rank_floor=('rank_floor_rel', 'mean'),
    ).reset_index()


pol = pd.concat([layout_table(enum[enum.snr == s]).assign(snr=s) for s in (SNR_REF, SNR_LOW)])
pol.to_csv(OUT / 'kw51_policy.csv', index=False)

# ---------------------------------------------------------------- summary ---
summary = {'box': BOX, 'snr_ref': SNR_REF, 'snr_low': SNR_LOW, 'kappa_cert': KAPPA_CERT,
           'breakeven_floor': Q.breakeven_floor(BOX),
           'n_calibration_days': int(D['in_window']['calibration'].sum()),
           'n_test_days': int(D['in_window']['test'].sum()),
           'rejected_days': D['rejected_days'],
           'envelope_abs': {nm: {'median': float(np.median(E)), 'p90': float(np.quantile(E, .9)),
                                 'max': float(E.max())}
                            for nm, E in (('frozen', E_frozen), ('roll1', E_roll[1][0]),
                                          ('roll7', E_roll[7][0]))},
           'envelope_scale': scale.to_dict('records'),
           'singular_values': {'deck': np.linalg.svd(phi[K.layout_ids(0)], compute_uv=False).tolist(),
                               'full': np.linalg.svd(phi, compute_uv=False).tolist()},
           'by_snr': {}}
for snr in (SNR_REF, SNR_LOW):
    e = enum[enum.snr == snr]
    deck, full = e[e['mask'] == 0], e[e['mask'] == 63]
    pt = layout_table(e)
    summary['by_snr'][str(snr)] = {
        'states_rank': {'deck': counts(deck, 'state_rank'), 'full': counts(full, 'state_rank')},
        'states_noise': {'deck': counts(deck, 'state'), 'full': counts(full, 'state')},
        'noise_equals_rank_all_layouts': bool((e.state == e.state_rank).all()),
        'certified': {lay: {pl: int((d[f'rel_{pl}'] <= KAPPA_CERT).sum())
                            for pl in ('nodrift', 'roll1', 'roll7', 'frozen')}
                      for lay, d in (('deck', deck), ('full', full))},
        'informative': {lay: {pl: int((d[f'rel_{pl}'] < 1 - 1e-6).sum())
                              for pl in ('nodrift', 'roll1', 'roll7', 'frozen')}
                        for lay, d in (('deck', deck), ('full', full))},
        'n_problems_at_prior_frozen': int((e.rel_frozen >= 1 - 1e-6).sum()),
        'n_problems': int(len(e)),
        'min_rel_frozen': float(e.rel_frozen.min()),
        'argmin_rel_frozen': e.loc[e.rel_frozen.idxmin(), ['mask', 'mode']].astype(int).tolist(),
        'rel_nodrift_full': full.set_index('mode').rel_nodrift.round(4).to_dict(),
        'rel_frozen_full': full.set_index('mode').rel_frozen.round(4).to_dict(),
        'rel_nodrift_deck': deck.set_index('mode').rel_nodrift.round(4).to_dict(),
        'rank_floor_full': full.set_index('mode').rank_floor_rel.round(4).to_dict(),
        'rank_floor_deck': deck.set_index('mode').rank_floor_rel.round(4).to_dict(),
        'pnull_full': full.set_index('mode').pnull.round(4).to_dict(),
        'pnull_deck': deck.set_index('mode').pnull.round(4).to_dict(),
        'layout_report_distribution': pt.report.value_counts().sort_index().to_dict(),
        'layouts_reaching_full_report': pt.loc[pt.report == pt.report.max(), 'mask'].tolist(),
        'always_report_modes': [int(m) for m in range(1, P + 1)
                                if (e[e['mode'] == m].state == Q.REPORT).all()],
        'never_report_modes': [int(m) for m in range(1, P + 1)
                               if (e[e['mode'] == m].state != Q.REPORT).all()],
        'limitation_frozen_full': full.limitation_frozen.value_counts().to_dict(),
    }
summary['critical_drift'] = crit.to_dict('records')
(OUT / 'kw51_summary.json').write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary['by_snr'], indent=1, default=str)[:6000])
print(crit.round(4).to_string(index=False))
