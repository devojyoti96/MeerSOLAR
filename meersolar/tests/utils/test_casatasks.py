import pytest
import psutil
import numpy as np
import os
import traceback
from casatasks import casalog
from casatools import ms as casamstool, table
from unittest.mock import patch, MagicMock
from meersolar.utils.casatasks import *

try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except BaseException:
    traceback.print_exc()
    pass


def test_check_scan_in_caltable(dummy_caltables):
    assert check_scan_in_caltable(dummy_caltables[0], 1) == False
    assert check_scan_in_caltable(dummy_caltables[0], 3) == True


def test_reset_weights_and_flags(dummy_msname):
    if os.path.exists(f"{dummy_msname}/.reset"):
        os.system(f"rm -rf {dummy_msname}/.reset")
    reset_weights_and_flags(dummy_msname)
    assert os.path.exists(f"{dummy_msname}/.reset") == True


def test_correct_missing_col_subms(dummy_submsname):
    correct_missing_col_subms(dummy_submsname)


@patch("meersolar.utils.casatasks.psutil.Process")
@patch("meersolar.utils.casatasks.os.path.exists", return_value=False)
@patch("meersolar.utils.casatasks.os.system")
@patch("meersolar.utils.casatasks.limit_threads")
@patch("meersolar.utils.casatasks.suppress_casa_output")
@patch("casatasks.mstransform")
@patch("casatasks.initweights")
@patch("casatasks.flagdata")
def test_single_mstransform(
    mock_flagdata,
    mock_initweights,
    mock_mstransform,
    mock_suppress,
    mock_limit_threads,
    mock_system,
    mock_exists,
    mock_psutil_process,
):
    # Mock memory return for dry_run
    mock_process = MagicMock()
    mock_process.memory_info.return_value.rss = 3 * 1024**3  # 3 GB
    mock_psutil_process.return_value = mock_process

    # Run only dry_run path
    mem = single_mstransform(msname="mock.ms", dry_run=True)

    assert isinstance(mem, float)
    assert mem == 3.0
