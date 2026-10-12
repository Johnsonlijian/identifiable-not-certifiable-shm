"""KW51 modal dictionaries from the public tracked-mode archive (Zenodo 3745914).

The archive contains hourly covariance-driven SSI identifications of 14 tracked
modes at 12 accelerometer channels (6 deck, 6 arch). Arch channels exist only
after the 2019 retrofit. Mode shapes are complex; every record is phase-aligned
to a reference record of the same mode, projected on its real part and
normalised to unit Euclidean norm over the channel set of the convention.

Conventions
-----------
* 12-channel convention (post-retrofit analyses): normalisation over all 12
  channels before any layout masking, so that masking never renormalises.
* 6-channel deck convention (pre/during/post retrofit comparison): normalisation
  over the 6 deck channels only, applied identically to every stage.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

from paths import ROUND, PROJECT, KW51_TRACKEDMODES as RAW

CACHE = ROUND / 'cache' / 'kw51_raw.npz'

N_DECK, N_ARCH, N_MODE = 6, 6, 14
WINDOWS = {'train': ('2019-09-28', '2019-11-01'),
           'calibration': ('2019-11-01', '2019-12-01'),
           'test': ('2019-12-01', '2020-01-16')}
STAGES = {'pre': (None, '2019-05-15'), 'retrofit': ('2019-05-15', '2019-09-28'),
          'post': ('2019-09-28', None)}


def load_raw():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        m, sdn, f, env = z['m'], z['sdn'], z['f'], z['env']
        labels_m, labels_env = list(z['labels_m']), list(z['labels_env'])
    else:                                                     # pragma: no cover
        from scipy.io import loadmat
        x = loadmat(RAW, squeeze_me=True, struct_as_record=False)['modes']
        m, sdn, f, env = np.asarray(x.m, complex), np.asarray(x.sdn), np.asarray(x.f), np.asarray(x.env)
        labels_m, labels_env = list(x.labels_m), list(x.labels_env)
    dates = np.array([datetime.fromordinal(int(t) - 366) + timedelta(days=float(t) - int(t))
                      for t in sdn])
    dates = np.array([datetime.fromtimestamp(round(t.timestamp())) for t in dates])
    return {'shapes': m, 'dates': dates, 'freq': f, 'env': env,
            'labels_m': labels_m, 'labels_env': labels_env}


def window_mask(dates, a, b):
    sel = np.ones(len(dates), bool)
    if a is not None:
        sel &= dates >= datetime.fromisoformat(a)
    if b is not None:
        sel &= dates < datetime.fromisoformat(b)
    return sel


def align(shapes, dates, chans, ref_window=WINDOWS['train']):
    """Phase-aligned, real, unit-norm shapes over the channel set `chans`.

    Returns an array (n_records, n_modes, len(chans)) with NaN where a record
    has no complete vector on `chans`.
    """
    ref_sel = window_mask(dates, *ref_window)
    out = np.full((shapes.shape[0], shapes.shape[1], len(chans)), np.nan)
    for q in range(shapes.shape[1]):
        v_all = shapes[:, q, :][:, chans]
        valid = np.isfinite(v_all).all(1)
        ix = np.flatnonzero(valid & ref_sel)
        if not len(ix):
            ix = np.flatnonzero(valid)
            if not len(ix):
                continue
        ref = v_all[ix[0]].copy()
        ref *= np.exp(-1j * np.angle(ref[np.argmax(abs(ref))]))
        for i in np.flatnonzero(valid):
            v = v_all[i] * np.exp(-1j * np.angle(np.vdot(ref, v_all[i])))
            re = v.real
            nn = np.linalg.norm(re)
            if nn > 1e-12:
                out[i, q] = re / nn
    return out


def normalise(P):
    return P / np.linalg.norm(P, axis=0, keepdims=True)


def aggregate(aligned, sel):
    """Window dictionary (channels x modes): mean of aligned shapes, renormalised."""
    means = np.nanmean(aligned[sel], axis=0).T
    if not np.isfinite(means).all():
        return None
    return normalise(means)


def daily_dictionaries(aligned, dates, sel):
    days = np.array([t.date().isoformat() for t in dates])
    out, used, rejected = [], [], []
    for d in np.unique(days[sel]):
        ix = sel & (days == d)
        if np.isfinite(aligned[ix]).all(2).any(0).all():
            out.append(aggregate(aligned, ix))
            used.append(d)
        else:
            rejected.append(d)
    return np.array(out), used, rejected


def build_post_retrofit():
    """12-channel dictionaries for the train/calibration/test protocol."""
    raw = load_raw()
    chans = list(range(N_DECK + N_ARCH))
    al = align(raw['shapes'], raw['dates'], chans)
    sel = {k: window_mask(raw['dates'], *v) for k, v in WINDOWS.items()}
    phi_train = aggregate(al, sel['train'])
    post = window_mask(raw['dates'], '2019-09-28', '2020-01-16')
    daily, days, rejected = daily_dictionaries(al, raw['dates'], post)
    day_dates = np.array([datetime.fromisoformat(d) for d in days])
    in_win = {k: (day_dates >= datetime.fromisoformat(a)) & (day_dates < datetime.fromisoformat(b))
              for k, (a, b) in WINDOWS.items()}
    return {'raw': raw, 'aligned': al, 'phi_train': phi_train, 'daily': daily,
            'days': days, 'day_dates': day_dates, 'rejected_days': rejected,
            'in_window': in_win, 'labels': raw['labels_m'],
            'freq_train': np.nanmedian(raw['freq'][sel['train']], 0)}


def envelope_frozen(D):
    """Frozen-dictionary policy: max over calibration days of |Phi_d - Phi_train|."""
    cal = D['daily'][D['in_window']['calibration']]
    return np.max(np.abs(cal - D['phi_train'][None]), axis=0)


def lag_pairs(D, h, window):
    """Index pairs (d, d - h days) with both daily dictionaries available."""
    idx = {d: i for i, d in enumerate(D['days'])}
    pairs = []
    for i, d in enumerate(D['days']):
        if not D['in_window'][window][i]:
            continue
        prev = (datetime.fromisoformat(d) - timedelta(days=h)).date().isoformat()
        if prev in idx:
            pairs.append((i, idx[prev]))
    return pairs


def envelope_rolling(D, h):
    """Rolling-dictionary policy: max over calibration days of |Phi_d - Phi_{d-h}|."""
    pairs = lag_pairs(D, h, 'calibration')
    diffs = np.array([np.abs(D['daily'][i] - D['daily'][j]) for i, j in pairs])
    return diffs.max(0), len(pairs)


def layout_ids(mask: int) -> list[int]:
    """Deck channels always present; arch channel c present when bit c of mask is set."""
    return list(range(N_DECK)) + [N_DECK + c for c in range(N_ARCH) if mask >> c & 1]


def stage_dictionaries_deck():
    """6-channel deck dictionaries for the pre / retrofit / post stages."""
    raw = load_raw()
    chans = list(range(N_DECK))
    al = align(raw['shapes'], raw['dates'], chans, ref_window=(None, None))
    out = {}
    for st, (a, b) in STAGES.items():
        sel = window_mask(raw['dates'], a, b)
        out[st] = (aggregate(al, sel), int(sel.sum()))
    return out
