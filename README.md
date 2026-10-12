# Reproducibility package

Manuscript: *Identifiable is not certifiable: query-level certification of structural
inferences from vibration sensor arrays* (L. Ren, S. Tang, Y. Wang), prepared for Mechanical Systems
and Signal Processing.

The public release is versioned at https://github.com/Johnsonlijian/identifiable-not-certifiable-shm.
It contains code, derived tables, cached features and figures. Raw third-party archives are
not redistributed.

## Contents

| Folder | Content |
|---|---|
| `code/` | `qcert.py` (certification library), `test_qcert.py` (11 unit tests), analysis scripts, figure scripts, `run_all.py` (whole pipeline), `paths.py` (input locations) |
| `cache/` | The identified HBTA modal features (`hbta_features.npz`), which let the pipeline run without the raw HBTA records, the HBTA sensor coordinates read from the data file, and the reference key-number ledger (`key_numbers_reference.json`) |
| `fe_inputs/` | Sensitivities of the official KW51 finite-element model (`kw51_fe_sensitivity.mat`), the 46 stored FE solutions (`case_001.mat` ... `case_046.mat`), and the model geometry and sensor selection (`geometry.mat`, `sensor_definition.mat`) |
| `results/` | Every derived table behind the figures and tables (`kw51/`, `fe/`, `hbta/`, `osp/`, `tables/`), the key-number ledger `key_numbers.json`, the unit-test log, the pipeline log and the software versions |
| `figures/` | The ten figures and the graphical abstract (PDF and PNG) |
| `manuscript/si_tables/` | LaTeX sources of the 16 Supplementary-Information tables, written by `make_si_tables.py` |
| `requirements.txt` | Python packages and the versions used |
| `SHA256SUMS.txt` | SHA-256 hash of every file in the package |

## External data (not redistributed)

| Data | Source | Environment variable |
|---|---|---|
| KW51 tracked modes, `trackedmodes.mat` (in `trackedmodes.zip`) - **required** | Zenodo, doi:10.5281/zenodo.3745914 | `KW51_TRACKEDMODES` |
| Hell Bridge Test Arena, `data_100Hz.h5`, version 1 - optional | Zenodo, doi:10.5281/zenodo.10507957 | `HBTA_H5` |
| KW51 official FE model - only to regenerate `fe_inputs/` with MATLAB and StaBIL 3.1 | Zenodo, doi:10.5281/zenodo.17366772 | - |

Download `trackedmodes.zip`, extract `trackedmodes.mat` and set `KW51_TRACKEDMODES` to its path.
`run_all.py` skips the HBTA identification `hbta_modal.py` when `HBTA_H5` does not point to an
existing file and uses `cache/hbta_features.npz` instead; set it to re-identify the modes from the
raw records. The FE study uses the stored derived inputs in `fe_inputs/`. Regenerating those inputs
from the original KU Leuven model and StaBIL 3.1 is outside this repository because the original
model/toolbox files are third-party materials; the source records and their licences are listed in
`DATASETS_AND_LINKS.csv`.

## Environment

Python 3.11 with NumPy 2.4, SciPy 1.17, pandas 3.0, h5py 3.16, Matplotlib 3.10, CVXPY 1.9 and
Clarabel 0.11 (exact versions in `requirements.txt` and `results/environment.txt`).

## Running

From the package root, with `KW51_TRACKEDMODES` set:

```
python code/test_qcert.py
python code/run_all.py cache/key_numbers_reference.json
```

`run_all.py` runs, in order: the unit tests; the KW51 audit (`kw51_gate.py`), its out-of-sample
coverage test (`kw51_coverage.py`) and the retrofit comparison (`kw51_prepost.py`); the FE study
(`fe_study.py`); the HBTA identification (`hbta_modal.py`, see above) and certification under the
two noise models (`hbta_gate.py id`, `hbta_gate.py session`); the sensor-placement comparison
(`osp_compare.py`); the derived tables and the key-number ledger (`derive_tables.py`); the
Supplementary-Information tables (`make_si_tables.py`); and the figures. It then compares the new
ledger with the reference and prints every number that differs (relative tolerance 1e-9). All
random draws use fixed seeds. The whole run takes about five minutes on a desktop computer
(about eight with the HBTA identification from the raw records).

## Licences

KW51 monitoring data and FE model: CC BY-NC-SA 4.0 (Zenodo records 3745914 and 17366772).
Hell Bridge Test Arena data: CC BY 4.0 (Zenodo record 10507957). Derived data in `cache/`,
`fe_inputs/` and `results/` are retained under the applicable source terms. The original code
and documentation are covered by the repository `LICENSE`; that notice does not relicense the
third-party data. See `DATASETS_AND_LINKS.csv` and `REPRODUCIBLE_RUNBOOK.md` for the complete
source and run instructions.
