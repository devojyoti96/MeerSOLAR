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


@patch("casatasks.flagdata")
@patch("casatasks.initweights")
@patch("casatasks.mstransform")
@patch("meersolar.utils.casatasks.suppress_output")
@patch("meersolar.utils.casatasks.limit_threads")
@patch("meersolar.utils.casatasks.os.system")
@patch("meersolar.utils.casatasks.os.path.exists", return_value=False)
@patch("meersolar.utils.casatasks.psutil.Process")
@patch("meersolar.utils.casatasks.msmetadata")
def test_single_mstransform(
    mock_msmetadata,
    mock_psutil_process,
    mock_exists,
    mock_system,
    mock_limit_threads,
    mock_suppress,
    mock_mstransform,
    mock_initweights,
    mock_flagdata,
):
    # Setup mock for msmetadata
    mock_msmd = MagicMock()
    mock_msmd.fieldsforscan.return_value = [0]
    mock_msmetadata.return_value = mock_msmd

    # Call the function and check return
    outputms = single_mstransform(msname="mock.ms", outputms="mock_output.ms", scan="1")
    assert outputms == "mock_output.ms"
