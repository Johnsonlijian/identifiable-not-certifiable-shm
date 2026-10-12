# Reproducible runbook

This repository contains the analysis code, derived tables, and publication figures for *Identifiable is not certifiable: query-level certification of structural inferences from vibration sensor arrays*. It does not redistribute the raw KW51 monitoring archive, the raw Hell Bridge HDF5 file, or the original KU Leuven model/toolbox files.

## Environment

Use Python 3.11 with the pinned packages in `requirements.txt`. The reference environment was NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.3, h5py 3.16.0, Matplotlib 3.10.9, CVXPY 1.9.3, and Clarabel 0.11.1. All analysis seeds are fixed in the scripts.

## Inputs and licences

1. Download `trackedmodes.mat` from Zenodo record 10.5281/zenodo.3745914 under its CC BY-NC-SA 4.0 terms, and set `KW51_TRACKEDMODES` to that file.
2. The default HBTA run uses `cache/hbta_features.npz`. To re-identify modes from the raw records, download `data_100Hz.h5` from Zenodo record 10.5281/zenodo.10507957 under its CC BY 4.0 terms and set `HBTA_H5` to that file.
3. The stored KW51 FE sensitivity and case files under `fe_inputs/` are derived inputs associated with Zenodo record 10.5281/zenodo.17366772 (CC BY-NC-SA 4.0). Regenerating them from the original model requires the source model/toolbox and MATLAB; those original files and regeneration scripts are not redistributed here.

The authoritative source and licence records are in `DATASETS_AND_LINKS.csv`. The original code/documentation licence is stated in `LICENSE` and is separate from the third-party data licences.

## Run

From the repository root:

```text
python code/test_qcert.py
python code/run_all.py cache/key_numbers_reference.json
```

The runner executes the unit tests, KW51 gate and held-out coverage, FE study, HBTA gate, sensor-placement comparison, derived tables, supplementary tables, and figures. If the raw HBTA file is absent it uses the cached modal features. A successful run exits with code 0 and reports zero key-number differences. If a step fails or any reference value differs, it exits with code 1.

## Expected outputs

Derived tables are under `results/`; figures are under `figures/`; supplementary table sources are under `manuscript/si_tables/`. The reference ledger is `cache/key_numbers_reference.json`. `SHA256SUMS.txt` records the release contents.

## Scope

The repository is a reproducibility aid for the submitted analysis. It does not itself establish publication, peer review, acceptance, or a licence to redistribute source datasets. Please cite the manuscript and the three Zenodo records when reusing the materials.
