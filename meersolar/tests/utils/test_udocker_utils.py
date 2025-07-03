import pytest
from unittest.mock import patch, MagicMock
from meersolar.utils.udocker_utils import *


def test_init_udocker():
    init_udocker()


@pytest.mark.parametrize(
    "system_return, expected",
    [
        (0, True),  # Container present
        (1, False),  # Container absent
    ],
)
@patch("meersolar.utils.udocker_utils.os.system")
def test_check_udocker_container(system_mock, system_return, expected):
    # First call: udocker inspect, Second call: cleanup
    system_mock.side_effect = [system_return, None]
    result = check_udocker_container("test_container")
    assert result is expected
    assert system_mock.call_count == 2


@pytest.mark.parametrize(
    "check_container, container_present, dry_run, expected_return",
    [
        (True, False, False, 1),  # container check fails, fallback fails
        (True, True, True, 2.5),  # dry run, mock memory
        (False, True, False, 0),  # skip check, run successfully
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
def test_run_wsclean_param_cases(
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
    dry_run,
    expected_return,
):
    mock_check.return_value = container_present
    mock_init.return_value = None if not container_present else "meerwsclean"
    mock_system.return_value = 0
    mock_process.return_value.memory_info.return_value.rss = 2.5 * 1024**3  # 2.5 GB
    result = run_wsclean(
        "wsclean -name mock test.ms",
        container_name="meerwsclean",
        check_container=check_container,
        verbose=False,
        dry_run=dry_run,
    )
    if dry_run:
        assert isinstance(result, float)
        assert round(result, 1) == expected_return
    else:
        assert result == expected_return


@pytest.mark.parametrize(
    "container_present, dry_run, expected",
    [
        (False, False, 1),  # Container not found, init fails
        (True, True, 2.5),  # Dry run returns mock memory
        (True, False, 0),  # Normal run success
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
def test_run_solar_sidereal_cor(
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
    dry_run,
    expected,
):
    mock_check.return_value = container_present
    mock_init.return_value = None if not container_present else "meerwsclean"
    mock_system.return_value = 0
    mock_process.return_value.memory_info.return_value.rss = 2.5 * 1024**3
    result = run_solar_sidereal_cor(
        msname="test.ms",
        only_uvw=False,
        container_name="meerwsclean",
        verbose=False,
        dry_run=dry_run,
    )
    if dry_run:
        assert isinstance(result, float)
        assert round(result, 1) == expected
    else:
        assert result == expected


@pytest.mark.parametrize(
    "container_present, dry_run, expected",
    [
        (False, False, 1),  # container missing, init fails
        (True, True, 2.5),  # dry run, returns memory usage
        (True, False, 0),  # normal run, successful
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
def test_run_chgcenter_param_cases(
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
    dry_run,
    expected,
):
    mock_check.return_value = container_present
    mock_init.return_value = None if not container_present else "meerwsclean"
    mock_system.return_value = 0
    mock_process.return_value.memory_info.return_value.rss = 2.5 * 1024**3
    result = run_chgcenter(
        msname="test.ms",
        ra="00:00:00.0",
        dec="-30:00:00.0",
        only_uvw=False,
        container_name="meerwsclean",
        verbose=False,
        dry_run=dry_run,
    )
    if dry_run:
        assert isinstance(result, float)
        assert round(result, 1) == expected
    else:
        assert result == expected
