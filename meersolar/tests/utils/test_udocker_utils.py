import pytest
import psutil
import traceback
import tempfile
import os
from unittest.mock import patch, MagicMock
from meersolar.utils.udocker_utils import *


@patch("meersolar.utils.udocker_utils.os.makedirs")
@patch("meersolar.utils.udocker_utils.get_datadir", return_value="/mock/data")
def test_set_udocker_env(mock_get_datadir, mock_makedirs):
    # Backup original env
    original_env = dict(os.environ)
    try:
        set_udocker_env()
        mock_get_datadir.assert_called_once()
        mock_makedirs.assert_called_once_with("/mock/data/udocker", exist_ok=True)
        assert os.environ["UDOCKER_DIR"] == "/mock/data/udocker"
        assert (
            os.environ["UDOCKER_TARBALL"] == "/mock/data/udocker-englib-1.2.11.tar.gz"
        )
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


@patch("meersolar.utils.udocker_utils.set_udocker_env")
def test_init_udocker(mock_env):
    init_udocker()


@pytest.mark.parametrize(
    "system_return, expected",
    [
        (0, True),  # Container present
        (1, False),  # Container absent
    ],
)
@patch("meersolar.utils.udocker_utils.os.system")
@patch("meersolar.utils.udocker_utils.set_udocker_env")
def test_check_udocker_container(mock_env, system_mock, system_return, expected):
    # First call: udocker inspect, Second call: cleanup
    system_mock.side_effect = [system_return, None]
    result = check_udocker_container("test_container")
    assert result is expected
    assert system_mock.call_count == 2


@pytest.mark.parametrize(
    "check_container, container_present, expected_return",
    [
        (True, False, 1),  # container check fails, fallback fails
        (False, True, 0),  # skip check, run successfully
    ],
)
@patch("meersolar.utils.udocker_utils.traceback.print_exc")
@patch("meersolar.utils.udocker_utils.psutil.Process")
@patch("meersolar.utils.udocker_utils.os.system")
@patch("meersolar.utils.udocker_utils.initialize_wsclean_container")
@patch("meersolar.utils.udocker_utils.check_udocker_container")
@patch("meersolar.utils.udocker_utils.tempfile.mkdtemp", return_value="/mock/temp")
@patch("meersolar.utils.udocker_utils.os.getcwd", return_value="/mock")
@patch(
    "meersolar.utils.udocker_utils.os.path.abspath", side_effect=lambda x: f"/abs/{x}"
)
@patch("meersolar.utils.udocker_utils.os.path.dirname", side_effect=lambda x: "/abs")
@patch("meersolar.utils.udocker_utils.set_udocker_env")
def test_run_wsclean_param_cases(
    mock_env,
    mock_dirname,
    mock_abspath,
    mock_getcwd,
    mock_mkdtemp,
    mock_check,
    mock_init,
    mock_system,
    mock_process,
    mock_traceback,
    check_container,
    container_present,
    expected_return,
):
    mock_check.return_value = container_present
    mock_init.return_value = None if not container_present else "solarwsclean"
    mock_system.return_value = 0
    mock_process.return_value.memory_info.return_value.rss = 2.5 * 1024**3  # 2.5 GB
    result = run_wsclean(
        "wsclean -name mock test.ms",
        container_name="solarwsclean",
        check_container=check_container,
        verbose=False,
    )
    assert result == expected_return


@pytest.mark.parametrize(
    "container_present, expected",
    [
        (False, 0),  # Container not found, init fails
        (True, 0),  # Normal run success
    ],
)
@patch("meersolar.utils.udocker_utils.traceback.print_exc")
@patch("meersolar.utils.udocker_utils.psutil.Process")
@patch("meersolar.utils.udocker_utils.os.system")
@patch("meersolar.utils.udocker_utils.initialize_wsclean_container")
@patch("meersolar.utils.udocker_utils.check_udocker_container")
@patch("meersolar.utils.udocker_utils.tempfile.mkdtemp", return_value="/mock/temp")
@patch("meersolar.utils.udocker_utils.os.getcwd", return_value="/mock")
@patch(
    "meersolar.utils.udocker_utils.os.path.abspath", side_effect=lambda x: f"/abs/{x}"
)
@patch("meersolar.utils.udocker_utils.os.path.dirname", side_effect=lambda x: "/abs")
@patch("meersolar.utils.udocker_utils.set_udocker_env")
def test_run_solar_sidereal_cor(
    mock_env,
    mock_dirname,
    mock_abspath,
    mock_getcwd,
    mock_mkdtemp,
    mock_check,
    mock_init,
    mock_system,
    mock_process,
    mock_traceback,
    container_present,
    expected,
):
    mock_check.return_value = container_present
    mock_init.return_value = None if not container_present else "solarwsclean"
    mock_system.return_value = 0
    mock_process.return_value.memory_info.return_value.rss = 2.5 * 1024**3
    result = run_solar_sidereal_cor(
        msname="test.ms",
        only_uvw=False,
        container_name="solarwsclean",
        verbose=False,
    )
    assert result == expected


@pytest.mark.parametrize(
    "container_present, expected",
    [
        (False, 0),  # container missing, init fails
        (True, 0),  # normal run, successful
    ],
)
@patch("meersolar.utils.udocker_utils.traceback.print_exc")
@patch("meersolar.utils.udocker_utils.psutil.Process")
@patch("meersolar.utils.udocker_utils.os.system")
@patch("meersolar.utils.udocker_utils.initialize_wsclean_container")
@patch("meersolar.utils.udocker_utils.check_udocker_container")
@patch("meersolar.utils.udocker_utils.tempfile.mkdtemp", return_value="/mock/temp")
@patch("meersolar.utils.udocker_utils.os.getcwd", return_value="/mock")
@patch(
    "meersolar.utils.udocker_utils.os.path.abspath", side_effect=lambda x: f"/abs/{x}"
)
@patch("meersolar.utils.udocker_utils.os.path.dirname", side_effect=lambda x: "/abs")
@patch("meersolar.utils.udocker_utils.set_udocker_env")
def test_run_chgcenter_param_cases(
    mock_env,
    mock_dirname,
    mock_abspath,
    mock_getcwd,
    mock_mkdtemp,
    mock_check,
    mock_init,
    mock_system,
    mock_process,
    mock_traceback,
    container_present,
    expected,
):
    mock_check.return_value = container_present
    mock_init.return_value = None if not container_present else "solarwsclean"
    mock_system.return_value = 0
    mock_process.return_value.memory_info.return_value.rss = 2.5 * 1024**3
    result = run_chgcenter(
        msname="test.ms",
        ra="00:00:00.0",
        dec="-30:00:00.0",
        only_uvw=False,
        container_name="solarwsclean",
        verbose=False,
    )
    assert result == expected
