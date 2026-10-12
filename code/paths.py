"""Locations of the external inputs. Override with environment variables when the
package is used outside the original project tree.

  KW51_TRACKEDMODES   trackedmodes.mat from Zenodo 10.5281/zenodo.3745914 (trackedmodes.zip)
  KW51_FE_DIR         folder with kw51_fe_sensitivity.mat (official FE model, run_kw51_fe_sensitivity.m)
  KW51_FE_CASES       folder with case_001.mat ... case_046.mat and geometry.mat (run_fe_contract.m)
  HBTA_H5             data_100Hz.h5 from Zenodo 10.5281/zenodo.10507957 (version 1)

In the reproducibility package the FE inputs are stored in fe_inputs/. Raw external
inputs are deliberately resolved only from external_data/ or explicit environment
variables so that the package has no dependency on the authors' project tree.
"""
import os
from pathlib import Path

ROUND = Path(__file__).resolve().parent.parent
PROJECT = ROUND.parent  # retained for the private package builder; public inputs use package-local paths
FE_INPUTS = ROUND / 'fe_inputs'


def _p(var, default):
    return Path(os.environ.get(var, default))


KW51_TRACKEDMODES = _p('KW51_TRACKEDMODES', ROUND / 'external_data' / 'trackedmodes.mat')
KW51_FE_DIR = _p('KW51_FE_DIR', FE_INPUTS)
KW51_FE_CASES = _p('KW51_FE_CASES', FE_INPUTS)
HBTA_H5 = _p('HBTA_H5', ROUND / 'external_data' / 'data_100Hz.h5')
