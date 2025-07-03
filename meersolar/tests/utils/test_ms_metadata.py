import pytest
from unittest.mock import patch
from meersolar.utils.ms_metadata import *


def test_get_phasecenter(dummy_msname):
    ra, dec = get_phasecenter(dummy_msname, "0")
    assert ra == 62.08492
    assert dec == 294.24747


def test_get_timeranges_for_scan(dummy_msname):
    t = get_timeranges_for_scan(dummy_msname, 1, 5, 60)
    assert t == ["2024/06/10/09:58:29.20"]
    t = get_timeranges_for_scan(dummy_msname, 1, 20, 40)
    assert t == [
        "2024/06/10/09:58:29.20~2024/06/10/09:59:09.20",
        "2024/06/10/09:58:45.18~2024/06/10/09:59:25.18",
    ]
    t = get_timeranges_for_scan(dummy_msname, 1, 40, 40)
    assert t == ["2024/06/10/09:58:29.20~2024/06/10/09:59:09.20"]


def test_calc_fractional_bandwidth(dummy_msname):
    assert calc_fractional_bandwidth(dummy_msname) == 0.7


def test_baseline_names(dummy_msname):
    bs_names = baseline_names(dummy_msname)
    for bs in bs_names:
        assert "&&" in bs


def test_get_ms_size(dummy_msname):
    assert get_ms_size(dummy_msname, only_autocorr=True) == 0.21
    assert get_ms_size(dummy_msname, only_autocorr=False) == 6.06


def test_get_column_size(dummy_msname):
    assert get_column_size(dummy_msname, only_autocorr=True) == 0.21
    assert get_column_size(dummy_msname, only_autocorr=False) == 6.06


def test_get_ms_scan_size(dummy_msname):
    assert get_ms_scan_size(dummy_msname, 1) == 0.04


def test_get_chunk_size(dummy_msname):
    assert get_chunk_size(dummy_msname, memory_limit=1) == 6


def test_check_datacolumn_valid(dummy_msname):
    assert check_datacolumn_valid(dummy_msname, datacolumn="DATA") == True
    assert check_datacolumn_valid(dummy_msname, datacolumn="CORRECTED_DATA") == False


def test_get_band_name(dummy_msname):
    assert get_band_name(dummy_msname) == "U"


def test_get_bad_chans(dummy_msname):
    assert get_bad_chans(dummy_msname) == ""


def test_get_good_chans(dummy_msname):
    assert get_good_chans(dummy_msname) == "0:0~10"


def test_get_bad_ants(dummy_msname):
    ant_list, ant_str = get_bad_ants(dummy_msname)
    assert ant_list == []
    assert ant_str == ""


def test_get_common_spw():
    assert get_common_spw("0:0~100", "0:50~70") == "0:50~70"


def test_scans_in_timerange(dummy_msname):
    assert scans_in_timerange(
        dummy_msname, timerange="2024/06/10/10:20:00~2024/06/10/10:30:00"
    ) == {8: "2024/06/10/10:20:00.00~2024/06/10/10:30:00.00"}


def test_get_refant(dummy_submsname):
    assert get_refant(f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms") == "2"


def test_get_ms_scans(dummy_submsname):
    assert get_ms_scans(f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms") == [3]
    assert get_ms_scans(f"{dummy_submsname}/SUBMSS/test_subms.ms.0001.ms") == [5]


def test_get_submsname_scans(dummy_submsname):
    mslist, scanlist = get_submsname_scans(dummy_submsname)
    assert len(mslist) == len(scanlist)
    assert scanlist == [3, 5, 12, 19, 26]


def test_get_fluxcals(dummy_msname):
    fields, scans = get_fluxcals(dummy_msname)
    assert fields == ["J0408-6545"]
    assert scans == {"J0408-6545": [1, 2, 3]}


def test_get_polcals(dummy_msname):
    fields, scans = get_polcals(dummy_msname)
    assert fields == []
    assert scans == {}


@patch("meersolar.utils.ms_metadata.np.load")
def test_get_phasecals(mock_npload, dummy_msname, mock_npy_data):
    mock_npload.return_value = mock_npy_data
    fields, scans, fluxes = get_phasecals(dummy_msname)
    assert fields == ["J0431+2037"]
    assert scans == {"J0431+2037": [4, 5, 6, 11, 12, 13, 18, 19, 20, 25, 26]}
    assert fluxes == {"J0431+2037": "3.412"}


@patch("meersolar.utils.ms_metadata.np.load")
def test_get_valid_scans(mock_npload, dummy_msname, mock_npy_data):
    mock_npload.return_value = mock_npy_data
    assert get_valid_scans(dummy_msname, field="J0408-6545") == [1, 3]
    assert get_valid_scans(dummy_msname, field="J0431+2037") == [5, 12, 19, 26]


@patch("meersolar.utils.ms_metadata.np.load")
def test_get_target_fields(mock_npload, dummy_msname, mock_npy_data):
    mock_npload.return_value = mock_npy_data
    fields, scans = get_target_fields(dummy_msname)
    assert fields == ["Sun"]
    assert scans == {"Sun": [7, 8, 9, 10, 14, 15, 16, 17, 21, 22, 23, 24]}


def test_get_caltable_fields(dummy_caltables):
    assert get_caltable_fields(dummy_caltables[0]) == ["J0408-6545", "J0431+2037"]


@patch("meersolar.utils.ms_metadata.np.load")
def test_get_cal_target_scans(mock_npload, dummy_msname, mock_npy_data):
    mock_npload.return_value = mock_npy_data
    targets, cals, fluxcals, phasecals, polcals = get_cal_target_scans(dummy_msname)
    assert targets == [7, 8, 9, 10, 14, 15, 16, 17, 21, 22, 23, 24]
    assert cals == [1, 2, 3, 4, 5, 6, 11, 12, 13, 18, 19, 20, 25, 26]
    assert fluxcals == [1, 2, 3]
    assert phasecals == [4, 5, 6, 11, 12, 13, 18, 19, 20, 25, 26]
    assert polcals == []
