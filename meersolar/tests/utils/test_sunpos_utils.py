import pytest
import glob
import os
from unittest.mock import patch, MagicMock
from meersolar.utils.sunpos_utils import *


def test_get_solar_elevation():
    result = get_solar_elevation(-30, 21, 1050, "2024-06-10T09:30:00")
    assert isinstance(result, float)
    assert result == 34.651


def test_radec_sun(dummy_msname):
    sun_radec_string, sun_ra, sun_dec, radeg, decdeg = radec_sun(dummy_msname)
    assert isinstance(sun_radec_string, str)
    assert "J2000" in sun_radec_string
    assert isinstance(sun_ra, str)
    assert isinstance(sun_dec, str)
    assert isinstance(radeg, float)
    assert isinstance(decdeg, float)


@patch("meersolar.utils.sunpos_utils.run_chgcenter", return_value=0)
@patch(
    "meersolar.utils.sunpos_utils.radec_sun",
    return_value=("RADEC_STRING", "12h00m00s", "-20d00m00s", 180.0, -20.0),
)
def test_move_to_sun(mock_radec_sun, mock_run_chgcenter):
    msname = "mock.ms"
    result = move_to_sun(msname, only_uvw=True)

    # Check that mocked functions were called
    mock_radec_sun.assert_called_once_with(msname)
    mock_run_chgcenter.assert_called_once_with(
        msname, "12h00m00s", "-20d00m00s", only_uvw=True, container_name="solarwsclean"
    )

    assert result == 0


@patch("meersolar.utils.sunpos_utils.run_solar_sidereal_cor", return_value=0)
@patch("meersolar.utils.sunpos_utils.os.system")
@patch("meersolar.utils.sunpos_utils.os.path.exists", return_value=False)
def test_correct_solar_sidereal_motion(mock_exists, mock_system, mock_run):
    msname = "mock.ms"
    result = correct_solar_sidereal_motion(msname, verbose=True)
    mock_exists.assert_called_once_with("mock.ms/.sidereal_cor")
    mock_run.assert_called_once_with(
        msname="mock.ms", container_name="solarwsclean", verbose=True
    )
    mock_system.assert_called_once_with("touch mock.ms/.sidereal_cor")
    assert result == 0
