"""Run the full analysis pipeline in order and compare the key-number ledger with a reference.

Usage: python run_all.py [reference_key_numbers.json]

The HBTA identification (hbta_modal.py) needs the raw HDF5 file; when it is absent and
cache/hbta_features.npz exists, the step is skipped and the cached features are used.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

from paths import HBTA_H5

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
STEPS = [['test_qcert.py'], ['kw51_gate.py'], ['kw51_coverage.py'], ['kw51_prepost.py'], ['fe_study.py'],
         ['hbta_modal.py'], ['hbta_gate.py', 'id'], ['hbta_gate.py', 'session'], ['osp_compare.py'],
         ['derive_tables.py'], ['make_si_tables.py'],
         ['fig01_framework.py'], ['fig02_kw51_data.py'], ['fig03_hbta_data.py'], ['fig04_fe.py'],
         ['fig05_kw51_hierarchy.py'], ['fig06_kw51_phase.py'], ['fig07_kw51_validity.py'],
         ['fig08_hbta_cert.py'], ['fig09_hbta_loeo.py'], ['fig10_osp.py'], ['fig_graphical_abstract.py']]
log = []
failed = False
for step in STEPS:
    if step[0] == 'hbta_modal.py' and not HBTA_H5.exists() and (ROUND / 'cache' / 'hbta_features.npz').exists():
        log.append({'step': step[0], 'returncode': None, 'seconds': 0.0,
                    'stderr_tail': 'skipped: raw HDF5 not found, cached features used'})
        print(f"{step[0]:32s} skipped (raw HDF5 not found; using cache/hbta_features.npz)", flush=True)
        continue
    t0 = time.time()
    r = subprocess.run([sys.executable, '-W', 'ignore', str(HERE / step[0])] + step[1:], cwd=HERE,
                       capture_output=True, text=True)
    log.append({'step': ' '.join(step), 'returncode': r.returncode, 'seconds': round(time.time() - t0, 1),
                'stderr_tail': r.stderr[-600:] if r.returncode else ''})
    print(f"{' '.join(step):32s} rc={r.returncode} {time.time() - t0:7.1f}s", flush=True)
    if r.returncode:
        print(r.stderr[-2000:])
        failed = True
        break
(ROUND / 'results' / 'run_all_log.json').write_text(json.dumps(log, indent=1))

if len(sys.argv) > 1:
    ref = json.loads(Path(sys.argv[1]).read_text())
    new = json.loads((ROUND / 'results' / 'key_numbers.json').read_text())
    diffs = []

    def walk(a, b, path=''):
        if isinstance(a, dict):
            for k in set(a) | set(b if isinstance(b, dict) else {}):
                walk(a.get(k), (b or {}).get(k) if isinstance(b, dict) else None, f'{path}/{k}')
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                diffs.append((path, 'length/type', None))
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f'{path}[{i}]')
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(a - b) > 1e-9 * max(1.0, abs(a)):
                diffs.append((path, a, b))
        elif a != b:
            diffs.append((path, a, b))
    walk(ref, new)
    print(f'key-number differences: {len(diffs)}')
    for d in diffs[:40]:
        print('  ', d)
    if diffs:
        failed = True

if failed:
    sys.exit(1)
