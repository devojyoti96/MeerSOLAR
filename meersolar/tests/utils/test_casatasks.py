import pytest
from unittest.mock import patch, MagicMock
from meersolar.utils.casatasks import *


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


@patch("meersolar.utils.casatasks.os.system")
@patch("meersolar.utils.casatasks.os.path.exists", return_value=False)
@patch("casatasks.split")
@patch("meersolar.utils.casatasks.table")
@patch("meersolar.utils.casatasks.casamstool")
@patch("meersolar.utils.casatasks.mjdsec_to_timestamp")
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
        msname="dummy.ms", field="0", scan="3", dry_run=False
    )
    assert "noise_on.ms" in noise_on_ms or "noise_off.ms" in noise_off_ms
    assert mock_split.call_count == 2  # Called for even and odd splits
    assert mock_system.call_count >= 2  # Called for mv operations
    mock_table_instance.open.assert_called_once()
    mock_table_instance.getcol.assert_called_with("TIME")
    mock_timestamp.assert_called()
