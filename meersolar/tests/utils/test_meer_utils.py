import pytest
import psutil
import numpy as np
import os
import traceback
from casatasks import casalog
from casatools import ms as casamstool, table
from unittest.mock import patch, MagicMock, mock_open, call
from meersolar.utils.meer_utils import *

try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except BaseException:
    traceback.print_exc()
    pass


def test_get_band_name(dummy_msname):
    assert get_band_name(dummy_msname) == "U"


def test_get_bad_chans(dummy_msname):
    assert get_bad_chans(dummy_msname) == ""


def test_get_good_chans(dummy_msname):
    assert get_good_chans(dummy_msname) == "0:0~10"


def test_get_fluxcals(dummy_msname):
    fields, scans = get_fluxcals(dummy_msname)
    assert fields == ["J0408-6545"]
    assert scans == {"J0408-6545": [1, 2, 3]}


def test_get_polcals(dummy_msname):
    fields, scans = get_polcals(dummy_msname)
    assert fields == []
    assert scans == {}


def test_get_pol_names(dummy_msname):
    assert get_pol_names(dummy_msname) == ["XX", "XY", "YX", "YY"]


@patch("meersolar.utils.meer_utils.get_datadir", return_value="/mock/datadir")
@patch("meersolar.utils.meer_utils.get_band_name", return_value="L")
@patch("meersolar.utils.meer_utils.np.load")
@patch("meersolar.utils.meer_utils.glob.glob")
@patch("meersolar.utils.meer_utils.os.path.exists")
@patch("meersolar.utils.meer_utils.msmetadata")
def test_get_phasecals(
    mock_msmetadata,
    mock_exists,
    mock_glob,
    mock_np_load,
    mock_get_band_name,
    mock_get_datadir,
):
    # Mock inputs
    msname = "test.ms"
    mock_exists.return_value = True
    mock_glob.return_value = ["test.ms/SUBMSS/file1.ms", "test.ms/SUBMSS/file2.ms"]
    # Mock np.load return value
    mock_np_load.return_value.tolist.return_value = (
        ["J1234-5678", "J2345-6789"],
        [1.23, 4.56],
    )
    # Mock msmetadata object and its methods
    mock_msmd_instance = MagicMock()
    mock_msmd_instance.fieldnames.return_value = np.array(["J1234-5678", "J0408-6545"])
    mock_msmd_instance.scansforfield.return_value = np.array([1, 2, 3])
    mock_msmetadata.return_value = mock_msmd_instance
    # Run test
    fields, scans, fluxes = get_phasecals(msname)
    # Assertions
    assert fields == ["J1234-5678"]
    assert scans == {"J1234-5678": [1, 2, 3]}
    assert fluxes == {"J1234-5678": 1.23}
    expected_call = call("/mock/datadir/L_band_cal.npy", allow_pickle=True)
    assert mock_np_load.call_args_list.count(expected_call) == 2


@patch("meersolar.utils.meer_utils.np.load")
def test_get_valid_scans(mock_npload, dummy_msname, mock_npy_data):
    mock_npload.return_value = mock_npy_data
    assert get_valid_scans(dummy_msname, field="J0408-6545") == [1, 3]
    assert get_valid_scans(dummy_msname, field="J0431+2037") == [5, 12, 19, 26]


@patch("meersolar.utils.meer_utils.np.load")
def test_get_target_fields(mock_npload, dummy_msname, mock_npy_data):
    mock_npload.return_value = mock_npy_data
    fields, scans = get_target_fields(dummy_msname)
    assert fields == ["Sun"]
    assert scans == {"Sun": [7, 8, 9, 10, 14, 15, 16, 17, 21, 22, 23, 24]}


def test_get_caltable_fields(dummy_caltables):
    assert get_caltable_fields(dummy_caltables[0]) == ["J0408-6545", "J0431+2037"]


@patch("meersolar.utils.meer_utils.np.load")
def test_get_cal_target_scans(mock_npload, dummy_msname, mock_npy_data):
    mock_npload.return_value = mock_npy_data
    targets, cals, fluxcals, phasecals, polcals = get_cal_target_scans(dummy_msname)
    assert targets == [7, 8, 9, 10, 14, 15, 16, 17, 21, 22, 23, 24]
    assert cals == [1, 2, 3, 4, 5, 6, 11, 12, 13, 18, 19, 20, 25, 26]
    assert fluxcals == [1, 2, 3]
    assert phasecals == [4, 5, 6, 11, 12, 13, 18, 19, 20, 25, 26]
    assert polcals == []


@patch("meersolar.utils.meer_utils.os.system")
@patch("meersolar.utils.meer_utils.os.path.exists", return_value=False)
@patch("casatasks.split")
@patch("meersolar.utils.meer_utils.table")
@patch("meersolar.utils.meer_utils.casamstool")
@patch("meersolar.utils.meer_utils.mjdsec_to_timestamp")
def test_split_noise_diode_scans(
    mock_timestamp,
    mock_mstool,
    mock_table,
    mock_split,
    mock_exists,
    mock_system,
):
    dummy_times = np.array([1.0, 2.0, 3.0, 4.0])
    mock_table_instance = MagicMock()
    mock_table.return_value = mock_table_instance
    mock_table_instance.getcol.return_value = dummy_times
    mock_timestamp.side_effect = (
        lambda t, str_format=1: f"2024-06-30T00:00:{int(t):02d}"
    )
    mock_ms = MagicMock()
    mock_ms.getdata.return_value = {"data": np.array([10.0])}
    mock_mstool.return_value = mock_ms
    noise_on_ms, noise_off_ms = split_noise_diode_scans(
        msname="dummy.ms",
        field="0",
        scan="3",
    )
    assert "noise_on.ms" in noise_on_ms or "noise_off.ms" in noise_off_ms
    assert mock_split.call_count == 2  # Called for even and odd splits
    assert mock_system.call_count >= 2  # Called for mv operations
    mock_table_instance.open.assert_called_once()
    mock_table_instance.getcol.assert_called_with("TIME")
    mock_timestamp.assert_called()


def test_determine_noise_diode_cal_scan(dummy_msname):
    assert determine_noise_diode_cal_scan(dummy_msname, 1) == True
    assert determine_noise_diode_cal_scan(dummy_msname, 5) == False
