import pytest
from meersolar.utils.calibration import *
from unittest.mock import MagicMock, patch


def test_merge_caltables_from_fixture(dummy_caltables, tmp_path):
    merged = tmp_path / "merged.K"
    result = merge_caltables(
        dummy_caltables.copy(), str(merged), append=False, keepcopy=True
    )
    assert os.path.exists(result)
    tb = table()
    tb.open(result)
    merged_rows = tb.nrows()
    tb.close()
    tb.open(dummy_caltables[0])
    single_rows = tb.nrows()
    tb.close()
    assert merged_rows == 2 * single_rows


def test_determine_noise_diode_cal_scan(dummy_msname):
    assert determine_noise_diode_cal_scan(dummy_msname, 1) == True
    assert determine_noise_diode_cal_scan(dummy_msname, 5) == False


def test_get_psf_size(dummy_msname):
    assert get_psf_size(dummy_msname) == 12.68


def test_calc_bw_smearing_freqwidth(dummy_msname):
    assert calc_bw_smearing_freqwidth(dummy_msname, full_FoV=False, FWHM=False) == 2.66
    assert calc_bw_smearing_freqwidth(dummy_msname, full_FoV=False, FWHM=True) == 2.66
    assert calc_bw_smearing_freqwidth(dummy_msname, full_FoV=True, FWHM=True) == 1.06
    assert calc_bw_smearing_freqwidth(dummy_msname, full_FoV=True, FWHM=False) == 0.53


def test_calc_time_smearing_timewidth(dummy_msname):
    assert (
        calc_time_smearing_timewidth(dummy_msname, full_FoV=False, FWHM=False) == 11.98
    )
    assert (
        calc_time_smearing_timewidth(dummy_msname, full_FoV=False, FWHM=True) == 11.98
    )
    assert calc_time_smearing_timewidth(dummy_msname, full_FoV=True, FWHM=True) == 3.99
    assert calc_time_smearing_timewidth(dummy_msname, full_FoV=True, FWHM=False) == 2.0


def test_max_time_solar_smearing(dummy_msname):
    assert max_time_solar_smearing(dummy_msname) == 152.16


def test_delaycal(dummy_submsname):
    caltable = delaycal(
        msname=f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms",
        caltable=f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.kcal",
        refant="0",
    )
    if os.path.exists(f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.kcal"):
        os.system(f"rm -rf {dummy_submsname}/SUBMSS/test_subms.ms.0000.kcal")
        assert caltable == f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.kcal"
    else:
        assert caltable == None
