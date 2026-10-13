"""Generate the LaTeX tables of the Supplementary Information from the result files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROUND = Path(__file__).resolve().parent.parent
RES = ROUND / 'results'
OUT = ROUND / 'manuscript' / 'si_tables'
OUT.mkdir(parents=True, exist_ok=True)
J = lambda p: json.loads((RES / p).read_text())
ARCH = ['AR0910A', 'AR0910C', 'AR1516A', 'AR1516C', 'AR2122A', 'AR2122C']


def write(name, body):
    if body.lstrip().startswith(r'\begin{longtable}'):
        body = '{\\footnotesize\n' + body + '\n}'
    (OUT / f'{name}.tex').write_text(body, encoding='utf-8')


def f(x, d=2):
    return f'{x:.{d}f}'


# ---------------------------------------------------------- S1 threshold grid --
thr = pd.read_csv(RES / 'tables' / 'kw51_threshold_sensitivity.csv')
rows = []
for snr in (50.0, 5.0):
    for t1 in (0.1, 0.2, 0.3):
        for t2 in (0.5, 0.6, 0.7):
            d = thr[(thr.snr == snr) & (thr.tau1 == t1) & (thr.tau2 == t2)]
            dk = d[d.layout == 'deck'].iloc[0]
            fl = d[d.layout == 'full'].iloc[0]
            rows.append(f'{int(snr)} & {t1:.1f} & {t2:.1f} & {dk.REPORT}/{dk.BOUND}/{dk.REFUSE} & '
                        f'{fl.REPORT}/{fl.BOUND}/{fl.REFUSE} & {dk.all_layout_REPORT} \\\\')
write('tab_s1_thresholds', '\n'.join([
    r'\begin{longtable}{cccccc}',
    r'\caption{KW51: reporting states of the deck-only and full arrays (\REPORT/\BOUND/\REFUSE) and the total number of \REPORT{} states over the 896 layout--query problems, for a grid of thresholds $(\tau_1,\tau_2)$ and two signal-to-noise ratios. The operating point of the main text is $(0.2,0.6)$.}\label{tab:s1}\\',
    r'\toprule $b/\sigma$ & $\tau_1$ & $\tau_2$ & deck & full array & \REPORT, all layouts\\ \midrule\endfirsthead',
    r'\toprule $b/\sigma$ & $\tau_1$ & $\tau_2$ & deck & full array & \REPORT, all layouts\\ \midrule\endhead',
    *rows, r'\bottomrule', r'\end{longtable}']))

kap = pd.read_csv(RES / 'tables' / 'kw51_kappa_sensitivity.csv')
rows = []
for k in sorted(kap.kappa.unique()):
    d = kap[kap.kappa == k]
    dk, fl = d[d.layout == 'deck'].iloc[0], d[d.layout == 'full'].iloc[0]
    rows.append(f'{k:.1f} & {dk.cert_nodrift} & {dk.cert_frozen} & {fl.cert_nodrift} & {fl.cert_frozen} \\\\')
write('tab_s2_kappa', '\n'.join([
    r'\begin{table}[h]\centering\footnotesize',
    r'\caption{KW51 ($b/\sigma=50$): number of modal coordinates with $w^{*}\le\kappa\,w_0$ for several certification levels $\kappa$, with an exact dictionary and with the monthly discrepancy envelope.}\label{tab:s2}',
    r'\begin{tabular}{ccccc}\toprule',
    r'$\kappa$ & \multicolumn{2}{c}{deck only} & \multicolumn{2}{c}{full array}\\',
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5} & exact & monthly & exact & monthly\\ \midrule',
    *rows, r'\bottomrule\end{tabular}\end{table}']))

# ------------------------------------------------------- S3 envelope scale ----
kw = J('kw51/kw51_summary.json')
sc = pd.DataFrame(kw['envelope_scale'])
rows = [f'{int(r.window_days)} & {int(r.n_blocks)} & {r.median_abs_dev:.4f} & {r.p90_abs_dev:.4f} & '
        f'{r.max_abs_dev:.4f} \\\\' for _, r in sc.iterrows()]
env = kw['envelope_abs']
write('tab_s3_envelope', '\n'.join([
    r'\begin{table}[h]\centering\footnotesize',
    r'\caption{KW51: absolute deviation of block-averaged dictionaries of the calibration month from the training dictionary (unit-norm shapes), as a function of the averaging window. The frozen-dictionary envelope used in the main text is the 1-day row. Lagged envelopes, built from the differences between daily dictionaries $h$ days apart, have medians of ' + f"{env['roll1']['median']:.4f}" + r' ($h=1$) and ' + f"{env['roll7']['median']:.4f}" + r' ($h=7$) and maxima of ' + f"{env['roll1']['max']:.3f}" + r' and ' + f"{env['roll7']['max']:.3f}" + r'.}\label{tab:s3}',
    r'\begin{tabular}{ccccc}\toprule',
    r'window (days) & blocks & median & 90th percentile & maximum\\ \midrule',
    *rows, r'\bottomrule\end{tabular}\end{table}']))

# ---------------------------------------------------- S4 per-layout table -----
pol = pd.read_csv(RES / 'kw51' / 'kw51_policy.csv')
p50, p5 = pol[pol.snr == 50.0].set_index('mask'), pol[pol.snr == 5.0].set_index('mask')
rows = []
for m in range(64):
    ch = ', '.join(ARCH[c] for c in range(6) if m >> c & 1) or '--'
    r, r5 = p50.loc[m], p5.loc[m]
    rows.append(f'{m} & {int(r.k)} & {ch} & {int(r.report)}/{int(r.bound)}/{int(r.refuse)} & '
                f'{int(r.cert_nodrift)} & {int(r.info_frozen)} & {int(r5.report)}/{int(r5.bound)}/{int(r5.refuse)} \\\\')
write('tab_s4_layouts', '\n'.join([
    r'\begin{longtable}{rcp{5.4cm}cccc}',
    r"\caption{KW51: all 64 layouts (deck channels always present plus the listed arch channels). States are \REPORT/\BOUND/\REFUSE{} counts of the 14 modal-coordinate queries; `cert.' counts queries with $w^{*}\le0.5\,w_0$ with an exact dictionary and `inf.' queries with $w^{*}<w_0$ under the monthly discrepancy, both at $b/\sigma=50$.}\label{tab:s4}\\",
    r'\toprule mask & $k$ & arch channels & states ($b/\sigma=50$) & cert. & inf. & states ($b/\sigma=5$)\\ \midrule\endfirsthead',
    r'\toprule mask & $k$ & arch channels & states ($b/\sigma=50$) & cert. & inf. & states ($b/\sigma=5$)\\ \midrule\endhead',
    *rows, r'\bottomrule', r'\end{longtable}']))

# ------------------------------------------------------- S5 retrofit stages ---
pp = pd.read_csv(RES / 'kw51' / 'kw51_prepost.csv')
d = pp[pp.snr == 50.0]
rows = []
for m in range(1, 15):
    s = d[d['mode'] == m].set_index('stage')
    rows.append(f'{m} & ' + ' & '.join(f'{s.loc[st].pnull:.3f}' for st in ('pre', 'retrofit', 'post'))
                + f' & {s.loc["post"].state} \\\\')
write('tab_s5_prepost', '\n'.join([
    r'\begin{table}[h]\centering\footnotesize',
    r'\caption{KW51 deck array, six-channel normalisation, $b/\sigma=50$: noise-level null participation of each modal-coordinate query in the three structural stages. The state is the same in all three stages.}\label{tab:s5}',
    r'\begin{tabular}{ccccc}\toprule mode & pre-retrofit & retrofit & post-retrofit & state\\ \midrule',
    *rows, r'\bottomrule\end{tabular}\end{table}']))

# ------------------------------------------------- S6 critical discrepancy ----
cr = pd.read_csv(RES / 'kw51' / 'kw51_critical_drift.csv')
rows = []
for m in range(1, 15):
    a, b = cr[(cr['mask'] == 0) & (cr['mode'] == m)].iloc[0], cr[(cr['mask'] == 63) & (cr['mode'] == m)].iloc[0]
    rows.append(f'{m} & {a.rel_nodrift:.3f} & {a.rho_info:.3f} & {a.rho_cert:.3f} & '
                f'{b.rel_nodrift:.3f} & {b.rho_info:.3f} & {b.rho_cert:.3f} \\\\')
write('tab_s6_critical', '\n'.join([
    r'\begin{table}[h]\centering\footnotesize',
    r'\caption{KW51 ($b/\sigma=50$): certified width with an exact dictionary and the largest fraction $\rho$ of the monthly envelope under which each query remains informative ($w^{*}<w_0$) or certified ($w^{*}\le0.5\,w_0$).}\label{tab:s6}',
    r'\begin{tabular}{ccccccc}\toprule',
    r'mode & \multicolumn{3}{c}{deck only} & \multicolumn{3}{c}{full array}\\',
    r'\cmidrule(lr){2-4}\cmidrule(lr){5-7} & $w^{*}/w_0$ exact & $\rho$ inf. & $\rho$ cert. & $w^{*}/w_0$ exact & $\rho$ inf. & $\rho$ cert.\\ \midrule',
    *rows, r'\bottomrule\end{tabular}\end{table}']))

# ---------------------------------------------------------- S7 FE queries -----
grid = pd.read_csv(RES / 'fe' / 'fe_linear_grid.csv')
qn = list(dict.fromkeys(grid['query']))
qdesc = {'E_s': '$(1,0,0,0)$', 'h_b': '$(0,1,0,0)$', 'I_d': '$(0,0,1,0)$', 'A_d': '$(0,0,0,1)$',
         'I_d+A_d': r'$(0,0,1,1)/\sqrt{2}$', 'I_d-A_d': r'$(0,0,1,-1)/\sqrt{2}$',
         'E_s-h_b': r'$(1,-1,0,0)/\sqrt{2}$', 'global': '$(1,1,1,1)/2$',
         'weak1': 'weakest eigenvector, deck-only Fisher', 'weak2': 'second weakest eigenvector'}
full = grid[(grid.variant == 'both') & (grid.noise_mult == 1.0)]
rows = []
for q in qn:
    w_full = full[(full.layout == 'full') & (full['query'] == q)].halfwidth.iloc[0]
    w_deck = full[(full.layout == 'deck') & (full['query'] == q)].halfwidth.iloc[0]
    qtex = q.replace('_', '\\_')
    rows.append(f"{qtex} & {qdesc[q]} & {w_deck:.4f} & {w_full:.4f} \\\\")
write('tab_s7_fequeries', '\n'.join([
    r'\begin{table}[h]\centering\footnotesize',
    r'\caption{FE model: query directions in $(E_s,h_b,I_d,A_d)$ and certified half-widths with frequencies and shapes at nominal noise (prior half-width $b\|\mathbf{q}\|_1$; $b=0.1$).}\label{tab:s7}',
    r'\begin{tabular}{llcc}\toprule query & direction & $w^{*}$ deck & $w^{*}$ full array\\ \midrule',
    *rows, r'\bottomrule\end{tabular}\end{table}']))

# ------------------------------------------------- S8 residuals by mode -------
rm = pd.read_csv(RES / 'tables' / 'fe_residual_by_mode.csv')
fe_modes = [1, 2, 3, 4, 5, 6, 7, 15, 20, 25, 27, 30, 39, 41]
rows = [f'{int(r["mode"])} & {fe_modes[int(r["mode"]) - 1]} & {r.max_freq:.2f} & {r.max_shape:.2f} & '
        f'{r.min_mac:.3f} & {int(r.n_swaps)} \\\\' for _, r in rm.iterrows()]
write('tab_s8_residuals', '\n'.join([
    r'\begin{table}[h]\centering\footnotesize',
    r'\caption{FE model: largest linearisation residual of each tracked mode over the 29 nonlinear solutions, in units of the observation noise ($\sigma_f=0.005$ relative frequency, $\sigma_\phi=0.02$ shape entry), smallest MAC to the base shape, and number of distinct FE modes matched to the tracked mode.}\label{tab:s8}',
    r'\begin{tabular}{cccccc}\toprule tracked mode & FE mode (base) & frequency & shape & min MAC & matched FE modes\\ \midrule',
    *rows, r'\bottomrule\end{tabular}\end{table}']))

# ------------------------------------------------- S9 nonlinear rules ---------
nr = pd.read_csv(RES / 'tables' / 'fe_nonlinear_rules.csv')
lab = {'naive': 'naive', 'blind': 'model-error blind', 'axis': 'single-axis envelope', 'loo': 'combined (leave-one-corner-out)'}
rows = []
for v, vn in (('freq', 'frequencies'), ('shape', 'mode shapes'), ('both', 'both')):
    for s in ('axis', 'corner'):
        for t in ('naive', 'blind', 'axis', 'loo'):
            d = nr[(nr.variant == v) & (nr.set == s) & (nr.rule == t)]
            if not len(d):
                continue
            r = d.iloc[0]
            rows.append(f'{vn} & {"single-axis" if s == "axis" else "corners"} & {lab[t]} & {int(r.n)} & '
                        f'{r.mean_cov:.3f} & {r.min_cov:.3f} & {r.frac_below_095:.2f} & {r.median_rel_width:.2f} \\\\')
write('tab_s9_nonlinear', '\n'.join([
    r'\begin{longtable}{lllccccc}',
    r'\caption{FE model, nonlinear validation: coverage of the four interval rules on the 16 single-axis solutions (in-sample for the single-axis envelope) and on the 13 box corners (out of sample), with 4000 noise draws per solution; columns give the number of layout--query--solution rows, mean and minimum coverage, fraction of rows below 0.95, and median half-width relative to the prior.}\label{tab:s9}\\',
    r'\toprule observables & solutions & rule & rows & mean & min & $<0.95$ & $w/w_0$\\ \midrule\endfirsthead',
    r'\toprule observables & solutions & rule & rows & mean & min & $<0.95$ & $w/w_0$\\ \midrule\endhead',
    *rows, r'\bottomrule', r'\end{longtable}']))

# --------------------------------------------------- S10 HBTA reference modes --
ref = pd.read_csv(RES / 'hbta' / 'hbta_reference_modes.csv')
hm = J('hbta/hbta_modal_summary.json')
det = hm['detection_rate_by_combo']
combos = ['SM-Y-P1', 'SM-Z-P1', 'SM-Y-P2', 'SM-Z-P2', 'NM-Z-P2']
rows = []
for _, r in ref.iterrows():
    m = int(r.ref_mode)
    rows.append(f'{m} & {r.freq_hz:.2f} & {r.freq_sd_hz:.3f} & {int(r.n_uds_runs)} & '
                + ' & '.join(f'{det[c][f"m{m}"]:.1f}' for c in combos) + r' \\')
write('tab_s10_refmodes', '\n'.join([
    r'\begin{table}[h]\centering\footnotesize',
    r'\caption{HBTA: reference modes from the ten undamaged runs (median frequency, standard deviation across undamaged runs, number of undamaged runs in the cluster) and the fraction of the ten runs of each excitation set-up in which the mode is matched.}\label{tab:s10}',
    r'\begin{tabular}{cccccccccc}\toprule',
    r'mode & $f$ (Hz) & sd (Hz) & U runs & SM-Y-P1 & SM-Z-P1 & SM-Y-P2 & SM-Z-P2 & NM-Z-P2\\ \midrule',
    *rows, r'\bottomrule\end{tabular}\end{table}']))

# ------------------------------------------------------ S11 layout families ---
lf = pd.read_csv(RES / 'tables' / 'hbta_layout_families.csv')
rows = [f"{r.layout} & {int(r.n_sensors)} & {int(r.n_rows)} & {int(r.REPORT)}/{int(r.BOUND)}/{int(r.REFUSE)} & "
        f"{int(r.cert_noise_only)} & {int(r.cert_cross_condition)} & {int(r.informative_cross_condition)} \\\\"
        for _, r in lf.iterrows()]
write('tab_s11_families', '\n'.join([
    r'\begin{table}[h]\centering\footnotesize',
    r'\caption{HBTA layout families (session noise): number of sensors and observables, noise-level states of the eight presence queries, and the number of presence queries certified ($w^{*}\le0.5\,w_0$) with session noise only and with the cross-condition discrepancy, and informative with discrepancy.}\label{tab:s11}',
    r'\begin{tabular}{lcccccc}\toprule layout & sensors & observables & states & cert. (noise only) & cert. (discrepancy) & inf. (discrepancy)\\ \midrule',
    *rows, r'\bottomrule\end{tabular}\end{table}']))

# --------------------------------------------------------- S12 LOEO by set-up --
hb = J('hbta/hbta_gate_summary_session.json')
hbi = J('hbta/hbta_gate_summary.json')
rows = []
for c in sorted(hb['loeo_full_by_heldout']):
    s = hb['loeo_full_by_heldout'][c]['noise_only']
    rows.append(f'{c} & {s["coverage_true_state"]:.3f} & {s["detection_rate"]:.3f} & '
                f'{s["false_attribution_rate"]:.3f} & {hb["direction_attribution"]["full_by_heldout"][c]:.3f} \\\\')
write('tab_s12_loeo', '\n'.join([
    r'\begin{table}[h]\centering\footnotesize',
    r'\caption{HBTA leave-one-excitation-out transfer per held-out set-up (full array, session noise): true-state coverage, exclusion of zero for the imposed state and for non-imposed states by the noise-only certificate, and direction-only attribution accuracy. The discrepancy-aware certificate returns the prior for every held-out query.}\label{tab:s12}',
    r'\begin{tabular}{lcccc}\toprule held-out set-up & coverage (true) & excludes 0 (imposed) & excludes 0 (other) & direction accuracy\\ \midrule',
    *rows, r'\bottomrule\end{tabular}\end{table}']))

# ------------------------------------------------ S13 noise-model comparison --
gi, gs = hbi['geometry_full'], hb['geometry_full']
li, ls = hbi['loeo_full'], hb['loeo_full']
rows = [
    f"signature norms (DS1--DS8) & {min(gi['signature_norm'][f'DS{k}'] for k in range(1, 9)):.0f}--{max(gi['signature_norm'][f'DS{k}'] for k in range(1, 9)):.0f} & "
    f"{min(gs['signature_norm'][f'DS{k}'] for k in range(1, 9)):.0f}--{max(gs['signature_norm'][f'DS{k}'] for k in range(1, 9)):.0f} \\\\",
    f"envelope norms (DS1--DS8) & {min(gi['envelope_norm'][f'DS{k}'] for k in range(1, 9)):.0f}--{max(gi['envelope_norm'][f'DS{k}'] for k in range(1, 9)):.0f} & "
    f"{min(gs['envelope_norm'][f'DS{k}'] for k in range(1, 9)):.0f}--{max(gs['envelope_norm'][f'DS{k}'] for k in range(1, 9)):.0f} \\\\",
    f"certified in sample, noise only / with discrepancy & {hbi['insample_certified_presence']['full']['noise_only']} / {hbi['insample_certified_presence']['full']['certified']} & "
    f"{hb['insample_certified_presence']['full']['noise_only']} / {hb['insample_certified_presence']['full']['certified']} \\\\",
    f"LOEO true-state coverage, noise-only certificate & {li['noise_only']['coverage_true_state']:.3f} & {ls['noise_only']['coverage_true_state']:.3f} \\\\",
    f"LOEO false exclusion of zero, noise-only certificate & {li['noise_only']['false_attribution_rate']:.3f} & {ls['noise_only']['false_attribution_rate']:.3f} \\\\",
    f"LOEO direction attribution & {hbi['direction_attribution']['full_accuracy']:.3f} & {hb['direction_attribution']['full_accuracy']:.3f} \\\\",
]
write('tab_s13_noisemodels', '\n'.join([
    r'\begin{table}[h]\centering\footnotesize',
    r'\caption{HBTA full array: identification noise only (half-record scatter) versus session noise (identification plus undamaged-repeat scatter).}\label{tab:s13}',
    r'\begin{tabular}{lcc}\toprule quantity & identification noise & session noise\\ \midrule',
    *rows, r'\bottomrule\end{tabular}\end{table}']))

# --------------------------------------------------------- S14 random layouts --
A = pd.read_csv(RES / 'hbta' / 'hbta_insample_map_session.csv')
rnd = A[A.layout.str.startswith('random') & (A.kind == 'presence')]
rows = []
for k, g in rnd.groupby('n_sensors'):
    rows.append(f'{int(k)} & {g.layout.nunique()} & {np.mean(g.state == "REPORT"):.2f} & '
                f'{np.mean(g.rel_noise_only <= 0.5):.2f} & {np.mean(g.rel_certified <= 0.5):.2f} \\\\')
write('tab_s14_random', '\n'.join([
    r'\begin{table}[h]\centering\footnotesize',
    r'\caption{HBTA random sensor subsets (session noise): share of presence queries in \REPORT, certified with session noise only, and certified with the cross-condition discrepancy.}\label{tab:s14}',
    r'\begin{tabular}{ccccc}\toprule sensors & layouts & \REPORT & cert. (noise only) & cert. (discrepancy)\\ \midrule',
    *rows, r'\bottomrule\end{tabular}\end{table}']))

# --------------------------------------------------------------- S15 OSP ------
ch = pd.read_csv(RES / 'osp' / 'osp_choices.csv')
ref = ch[np.isclose(ch.eta_rel, 0.01) & (ch.rule != 'E-optimal')]
rows = []
for k in range(7):
    for rule in ('D-optimal', 'A-optimal', 'greedy-D (EfI-type)', 'query-aware (REPORT)'):
        r = ref[(ref.k == k) & (ref.rule == rule)].iloc[0]
        chs = r.channels.replace('D+', '').replace('A1', 'AR0910A ').replace('A2', 'AR0910C ') \
            .replace('A3', 'AR1516A ').replace('A4', 'AR1516C ').replace('A5', 'AR2122A ').replace('A6', 'AR2122C ')
        rows.append(f'{k} & {rule} & {chs.strip() or "--"} & {int(r.report)} & {int(r.certified)} & '
                    f'{r.D_loss:.2f} \\\\')
write('tab_s15_osp', '\n'.join([
    r'\begin{longtable}{clp{5.6cm}ccc}',
    r'\caption{KW51 ($b/\sigma=50$, $\eta=0.01$ of the mean full-array eigenvalue): arch channels chosen by each rule for every budget $k$, the resulting numbers of \REPORT{} and certified queries, and the log-determinant given up relative to the D-optimal layout of the same budget.}\label{tab:s15}\\',
    r'\toprule $k$ & rule & arch channels & \REPORT & cert. & $\Delta\log\det$\\ \midrule\endfirsthead',
    r'\toprule $k$ & rule & arch channels & \REPORT & cert. & $\Delta\log\det$\\ \midrule\endhead',
    *rows, r'\bottomrule', r'\end{longtable}']))
osp = J('osp/osp_summary.json')
rows = []
for eta, d in sorted(osp['regret_by_eta'].items(), key=lambda t: float(t[0])):
    rows.append(f'{float(eta):g} & {d["D-optimal"]:.1f} & {d["A-optimal"]:.1f} & {d["greedy-D (EfI-type)"]:.1f} \\\\')
write('tab_s16_eta', '\n'.join([
    r'\begin{table}[h]\centering\footnotesize',
    r'\caption{Mean \REPORT{} regret over $k=1$--5 against the query-aware layout, for regularisers $\eta$ given as fractions of the mean full-array Fisher eigenvalue.}\label{tab:s16}',
    r'\begin{tabular}{cccc}\toprule $\eta$ (relative) & D-optimal & A-optimal & greedy-D\\ \midrule',
    *rows, r'\bottomrule\end{tabular}\end{table}']))
print('SI tables written to', OUT)
