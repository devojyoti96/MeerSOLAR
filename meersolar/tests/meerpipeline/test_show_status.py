import pytest
from unittest.mock import patch, MagicMock, mock_open
from meersolar.meerpipeline.show_status import *


@pytest.mark.parametrize(
    "pid_alive, clean_old_jobs, expect_rm",
    [
        (True, False, False),
        (False, False, False),
        (False, True, True),
    ],
)
@patch("meersolar.meerpipeline.show_status.drop_cache")
@patch("meersolar.meerpipeline.show_status.os.path.exists", return_value=True)
@patch("meersolar.meerpipeline.show_status.os.system")
@patch("meersolar.meerpipeline.show_status.psutil.pid_exists")
@patch(
    "meersolar.meerpipeline.show_status.open",
    new_callable=mock_open,
    read_data="1234 5678 dummy workdir outdir",
)
@patch("meersolar.meerpipeline.show_status.glob.glob")
@patch("meersolar.meerpipeline.show_status.get_cachedir", return_value="/mock/cache")
def test_show_job_status(
    mock_cachedir,
    mock_glob,
    mock_open_func,
    mock_pid_exists,
    mock_system,
    mock_exists,
    mock_drop_cache,
    pid_alive,
    clean_old_jobs,
    expect_rm,
):
    # Mock job file present
    mock_glob.return_value = ["/mock/cache/main_pids_1234.txt"]
    mock_pid_exists.return_value = pid_alive
    show_job_status(clean_old_jobs=clean_old_jobs)
    # Check psutil.pid_exists was called
    mock_pid_exists.assert_called_once_with(5678)
    if expect_rm:
        mock_system.assert_any_call("rm -rf /mock/cache/main_pids_1234.txt")
        mock_system.assert_any_call("rm -rf /mock/cache/pids/pids_1234.txt")
    else:
        mock_system.assert_not_called()


@pytest.mark.parametrize(
    "argv_args, expect_show_called, expect_exit_called",
    [
        (["show_meersolar_status"], False, True),  # No args, should exit
        (["show_meersolar_status", "--show"], True, False),  # Show only
        (
            ["show_meersolar_status", "--show", "--clean_old_jobs"],
            True,
            False,
        ),  # Show + clean
    ],
)
@patch("meersolar.meerpipeline.show_status.show_job_status")
@patch("meersolar.meerpipeline.show_status.sys.exit")
@patch("meersolar.meerpipeline.show_status.sys.argv", new_callable=list)
def test_cli_show_job_status(
    mock_argv,
    mock_exit,
    mock_show_status,
    argv_args,
    expect_show_called,
    expect_exit_called,
):
    # Patch argv directly
    sys.argv[:] = argv_args

    cli()

    if expect_show_called:
        mock_show_status.assert_called_once()
    else:
        mock_show_status.assert_not_called()

    if expect_exit_called:
        mock_exit.assert_called_once_with(1)
    else:
        mock_exit.assert_not_called()
