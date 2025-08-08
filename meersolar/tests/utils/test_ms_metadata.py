import pytest
import os
from unittest.mock import patch
from meersolar.utils.ms_metadata import *


@pytest.mark.parametrize(
    "uvrange_input, expected_output, expect_error",
    [
        (">200lambda", ["<200lambda"], False),
        ("<100lambda", [">100lambda"], False),
        ("10~1000lambda", ["<10lambda", ">1000lambda"], False),
        ("   >300lambda  ", ["<300lambda"], False),
        ("50~500lambda", ["<50lambda", ">500lambda"], False),
        ("500", None, True),
        ("20-100lambda", None, True),
        ("100~lambda", None, True),
        ("lambda~200", None, True),
        ("", None, True),
        ("300~10lambda", None, True),
    ],
)
def test_get_uvrange_exclude(uvrange_input, expected_output, expect_error):
    if expect_error:
        with pytest.raises(ValueError):
            get_uvrange_exclude(uvrange_input)
    else:
        assert get_uvrange_exclude(uvrange_input) == expected_output


def test_get_phasecenter(dummy_msname):
    ra, dec = get_phasecenter(dummy_msname, "0")
    assert ra == 62.08492
    assert dec == 294.24747


def test_get_observatory_name(dummy_msname):
    assert get_observatory_name(dummy_msname) == "MEERKAT"


def test_get_observatory_name(dummy_msname):
    lat, lon, height = get_observatory_coord(dummy_msname)
    assert lat == -30.713
    assert lon == 21.444
    assert height == 1050.0


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
    assert get_column_size(dummy_msname, only_autocorr=True) == 0.34
    assert get_column_size(dummy_msname, only_autocorr=False) == 10.02


def test_get_ms_scan_size(dummy_msname):
    assert get_ms_scan_size(dummy_msname, 1) == 0.07


def test_get_chunk_size(dummy_msname):
    assert get_chunk_size(dummy_msname, memory_limit=1) == 10


def test_check_datacolumn_valid(dummy_msname):
    assert check_datacolumn_valid(dummy_msname, datacolumn="DATA") == True
    assert check_datacolumn_valid(dummy_msname, datacolumn="CORRECTED_DATA") == False


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
    assert get_refant(f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms") == "1"


def test_get_ms_scans(dummy_submsname):
    assert get_ms_scans(f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms") == [3]
    assert get_ms_scans(f"{dummy_submsname}/SUBMSS/test_subms.ms.0001.ms") == [5]


def test_get_submsname_scans(dummy_submsname):
    mslist, scanlist = get_submsname_scans(dummy_submsname)
    assert len(mslist) == len(scanlist)
    assert scanlist == [3, 5, 12, 19, 26]
