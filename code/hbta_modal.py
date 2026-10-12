"""HBTA modal identification: frequency-domain decomposition with signed shapes.

Data: Hell Bridge Test Arena benchmark (Zenodo 10.5281/zenodo.10507957, v1),
50 runs at 100 Hz: 5 excitation set-ups (sine sweep lateral/vertical at shaker
positions P1/P2, random excitation vertical at P2) x (8 damage states + 2
undamaged repeats). 18 tri-axial global accelerometers on the two truss walls,
lateral (y) and vertical (z) components, 35 usable channels (AG09 has z only).

Processing per run
  1. remove the mean of every channel;
  2. cross-spectral density matrix, Welch averaging, Hann window, 4096 samples
     (0.024 Hz), 50 % overlap, 2.5-30 Hz;
  3. singular value decomposition per frequency line; peaks of the first
     singular value (prominence >= 0.2 decades, top 25);
  4. frequency refined by parabolic interpolation of log S1 around the peak;
     shape = first singular vector rotated to its maximal real projection,
     real part, unit norm; modal phase collinearity (MPC) recorded;
  5. the same on two halves of the record (up-sweep / down-sweep for the
     swept set-ups, first / second half of the excited segment for random
     excitation) - the half-to-half scatter gives the identification noise.

Reference modes: peaks of the 10 undamaged runs clustered by frequency
(+-0.25 Hz) and MAC >= 0.8; a cluster present in >= 4 undamaged runs is a
reference mode. Every run (and half) is matched to the reference modes by the
highest MAC within +-0.3 Hz (MAC >= 0.8, one peak per reference).

Outputs: results/hbta/hbta_runs.csv, hbta_reference_modes.csv,
         cache/hbta_features.npz, results/hbta/hbta_modal_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import sys
sys.path.insert(0, str(Path(__file__).parent))
from paths import ROUND, HBTA_H5 as H5

OUT = ROUND / 'results' / 'hbta'
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROUND / 'cache'

FS = 100.0
NSEG = 4096
BAND = (2.5, 30.0)
PROM = 0.2               # decades
NPEAK = 25
F_TOL_REF, F_TOL_MATCH = 0.25, 0.30
MAC_MIN = 0.8
MIN_UDS = 4


def mac(a, b):
    return float((a @ b) ** 2 / ((a @ a) * (b @ b)))


def load_run(f, name):
    acc = f[name]['acceleration']
    chans, names = [], []
    for i in range(1, 19):
        g = acc[f'AG{i:02d}']
        for comp in ('y', 'z'):
            if comp in g:
                chans.append(np.asarray(g[comp], float))
                names.append(f'AG{i:02d}.{comp}')
    L = min(len(c) for c in chans)
    X = np.stack([c[:L] for c in chans], 1)
    return X - X.mean(0, keepdims=True), names


def excitation_segments(f, name, n):
    """Two halves of the excited part of a record."""
    a = f[name].attrs
    if 'Noise' in str(a['mvs_mode']):
        acc = f[name]['acceleration']['AS']
        s = np.asarray(acc[list(acc.keys())[0]], float)
        s = s - s.mean()
        w = 200
        rms = np.sqrt(np.convolve(s ** 2, np.ones(w) / w, mode='same'))
        on = np.flatnonzero(rms > 0.25 * np.median(rms[rms > np.quantile(rms, 0.5)]))
        i0, i1 = (int(on[0]), int(on[-1])) if len(on) else (0, n)
        mid = (i0 + i1) // 2
        return (i0, mid), (mid, i1)
    # swept sine: up-sweep then down-sweep, turning point near the middle
    return (0, n // 2), (n // 2, n)


def fdd(X):
    n, nc = X.shape
    step = NSEG // 2
    nseg = (n - NSEG) // step + 1
    win = np.hanning(NSEG)
    freqs = np.fft.rfftfreq(NSEG, 1 / FS)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    G = np.zeros((int(band.sum()), nc, nc), complex)
    for i in range(nseg):
        seg = X[i * step:i * step + NSEG] * win[:, None]
        Fd = np.fft.rfft(seg, axis=0)[band]
        G += Fd[:, :, None] * Fd[:, None, :].conj()
    G /= max(nseg, 1)
    U, s, _ = np.linalg.svd(G)
    return freqs[band], s[:, 0], s[:, 1], U[:, :, 0]


def real_shape(u):
    """Rotate a complex vector to its maximal real projection; return shape and MPC."""
    ang = 0.5 * np.angle(np.sum(u ** 2))
    v = u * np.exp(-1j * ang)
    re = v.real
    # modal phase collinearity
    sxx, syy, sxy = np.sum(v.real ** 2), np.sum(v.imag ** 2), np.sum(v.real * v.imag)
    eps = (syy - sxx) / (2 * sxy) if abs(sxy) > 1e-30 else 0.0
    th = np.arctan(abs(eps) + np.sign(sxy) * np.sqrt(1 + eps ** 2)) if abs(sxy) > 1e-30 else 0.0
    mpc = ((sxx + sxy / (2 * eps + 1e-300) * (eps ** 2 + 1) * (2 * np.sin(th) ** 2 - 1)) /
           (sxx + syy)) if abs(sxy) > 1e-30 else 1.0
    re = re / np.linalg.norm(re)
    if re[np.argmax(np.abs(re))] < 0:
        re = -re
    return re, float(np.clip(mpc, 0, 1))


def peaks(fb, s1, s2, u1):
    ls = np.log10(s1)
    pk, prop = find_peaks(ls, prominence=PROM)
    order = np.argsort(-s1[pk])[:NPEAK]
    out = []
    for p in sorted(pk[order]):
        if 0 < p < len(fb) - 1:
            y0, y1, y2 = ls[p - 1], ls[p], ls[p + 1]
            den = y0 - 2 * y1 + y2
            dp = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
            fr = fb[p] + np.clip(dp, -0.5, 0.5) * (fb[1] - fb[0])
        else:
            fr = fb[p]
        shp, mpc = real_shape(u1[p])
        out.append({'f': float(fr), 'shape': shp, 'mpc': mpc, 's1': float(s1[p]),
                    's1s2': float(s1[p] / s2[p])})
    return out


f = h5py.File(H5, 'r')
names = sorted(f.keys())
inv = []
ident = {}
for name in names:
    a = f[name].attrs
    combo = (('NM' if 'Noise' in str(a['mvs_mode']) else 'SM') + '-'
             + ('Y' if 'Horizontal' in str(a['mvs_direction']) else 'Z') + '-'
             + str(a['mvs_position'])[-2:])
    X, chan_names = load_run(f, name)
    h1, h2 = excitation_segments(f, name, X.shape[0])
    res = {}
    for tag, (i0, i1) in (('full', (0, X.shape[0])), ('h1', h1), ('h2', h2)):
        fb, s1, s2, u1 = fdd(X[i0:i1])
        res[tag] = peaks(fb, s1, s2, u1)
        if tag == 'full':
            res['spectrum'] = (fb, s1)
    ident[name] = res
    inv.append({'run': name, 'combo': combo, 'state': str(a['damage_state']),
                'rep': name.split('_')[-1], 'date': str(a['date']),
                'temperature': float(a['temperature']), 'train': str(a['train_passing']),
                'wind': str(a.get('wind_range', '')), 'duration_s': float(a['duration']),
                'n_peaks_full': len(res['full'])})
    print(name, combo, len(res['full']), 'peaks')
f.close()
inv = pd.DataFrame(inv)
inv.to_csv(OUT / 'hbta_runs.csv', index=False)

# ------------------------------------------------------ reference modes -----
uds = inv[inv.state == 'UDS'].run.tolist()
pool = [(p['s1'], p['f'], p['shape'], r) for r in uds for p in ident[r]['full']]
pool.sort(key=lambda t: -t[0])
clusters = []
for s1, fr, sh, r in pool:
    for c in clusters:
        if abs(fr - c['f']) < F_TOL_REF and mac(c['anchor'], sh) >= MAC_MIN:
            if r not in c['runs']:
                c['runs'].add(r)
                c['fs'].append(fr)
                c['shapes'].append(sh * np.sign(sh @ c['anchor']))
            break
    else:
        clusters.append({'f': fr, 'anchor': sh, 'fs': [fr], 'shapes': [sh], 'runs': {r}})
refs = [c for c in clusters if len(c['runs']) >= MIN_UDS]
for c in refs:
    m = np.mean(c['shapes'], 0)
    c['shape'] = m / np.linalg.norm(m)
    c['f'] = float(np.median(c['fs']))
refs.sort(key=lambda c: c['f'])
# merge references that are the same physical mode (close and high MAC)
merged = []
for c in refs:
    if merged and abs(c['f'] - merged[-1]['f']) < F_TOL_REF and mac(c['shape'], merged[-1]['shape']) >= 0.9:
        if len(c['runs']) > len(merged[-1]['runs']):
            merged[-1] = c
        continue
    merged.append(c)
refs = merged
ref_tbl = pd.DataFrame([{'ref_mode': i + 1, 'freq_hz': c['f'], 'n_uds_runs': len(c['runs']),
                         'freq_sd_hz': float(np.std(c['fs'])),
                         'uds_runs_Y': sum('_Y_' in r for r in c['runs']),
                         'uds_runs_Z': sum('_Z_' in r for r in c['runs'])}
                        for i, c in enumerate(refs)])
ref_tbl.to_csv(OUT / 'hbta_reference_modes.csv', index=False)
print(ref_tbl.round(3).to_string(index=False))
R = len(refs)
REF_SH = np.array([c['shape'] for c in refs])
REF_F = np.array([c['f'] for c in refs])
NC = REF_SH.shape[1]


def match(plist):
    """One-to-one assignment of identified peaks to reference modes."""
    cand = []
    for i, c in enumerate(refs):
        for j, p in enumerate(plist):
            if abs(p['f'] - c['f']) < F_TOL_MATCH:
                mm = mac(c['shape'], p['shape'])
                if mm >= MAC_MIN:
                    cand.append((mm, i, j))
    cand.sort(reverse=True)
    used_r, used_p, out = set(), set(), {}
    for mm, i, j in cand:
        if i in used_r or j in used_p:
            continue
        used_r.add(i)
        used_p.add(j)
        p = plist[j]
        sh = p['shape'] * np.sign(p['shape'] @ refs[i]['shape'])
        out[i] = (p['f'], sh, mm, p['mpc'])
    return out


Fq = np.full((len(names), 3, R), np.nan)          # frequency [run, full/h1/h2, mode]
Sh = np.full((len(names), 3, R, NC), np.nan)      # shape
Mc = np.full((len(names), 3, R), np.nan)          # MAC to reference
Mp = np.full((len(names), 3, R), np.nan)          # MPC
for ri, name in enumerate(names):
    for ti, tag in enumerate(('full', 'h1', 'h2')):
        for i, (fr, sh, mm, mpc) in match(ident[name][tag]).items():
            Fq[ri, ti, i], Sh[ri, ti, i], Mc[ri, ti, i], Mp[ri, ti, i] = fr, sh, mm, mpc
spec_f = ident[names[0]]['spectrum'][0]
spec = np.array([ident[n]['spectrum'][1] for n in names])
np.savez_compressed(CACHE / 'hbta_features.npz', runs=np.array(names), freq=Fq, shape=Sh,
                    mac=Mc, mpc=Mp, ref_f=REF_F, ref_shape=REF_SH,
                    chan_names=np.array(chan_names), spec_f=spec_f, spec=spec)
det = pd.DataFrame({'run': names, **{f'm{i + 1}': np.isfinite(Fq[:, 0, i]) for i in range(R)}})
det = det.merge(inv[['run', 'combo', 'state']], on='run')
summary = {
    'n_runs': len(names), 'n_channels': NC, 'channels': chan_names,
    'nperseg': NSEG, 'band_hz': BAND, 'prominence_decades': PROM,
    'n_reference_modes': R, 'reference_freqs_hz': REF_F.round(4).tolist(),
    'reference_n_uds_runs': ref_tbl.n_uds_runs.tolist(),
    'detection_rate_by_combo': det.groupby('combo')[[f'm{i + 1}' for i in range(R)]].mean()
    .round(2).to_dict('index'),
    'median_mac_full': float(np.nanmedian(Mc[:, 0])),
    'median_mpc_full': float(np.nanmedian(Mp[:, 0])),
    'half_split_matched_fraction': float(np.mean(np.isfinite(Fq[:, 1]) & np.isfinite(Fq[:, 2]))),
}
(OUT / 'hbta_modal_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != 'channels'}, indent=1))
