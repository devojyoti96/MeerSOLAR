import pytest
from unittest.mock import patch, MagicMock, mock_open
from meersolar.meerpipeline.kill_job import *


@pytest.mark.parametrize(
    "process_exists, child_count",
    [
        (True, 2),  # Normal case: process exists with children
        (False, 0),  # Error case: NoSuchProcess
    ],
)
@patch("meersolar.meerpipeline.kill_job.psutil.Process")
def test_kill_process_and_children(mock_process_cls, process_exists, child_count):
    mock_parent = MagicMock()
    mock_child = MagicMock()

    if process_exists:
        mock_process_cls.return_value = mock_parent
        mock_parent.children.return_value = [mock_child] * child_count
    else:
        mock_process_cls.side_effect = psutil.NoSuchProcess(pid=12345)

    kill_process_and_children(pid=12345)

    if process_exists:
        assert mock_parent.children.call_count == 1
        assert mock_child.kill.call_count == child_count
        mock_parent.kill.assert_called_once()
    else:
        mock_process_cls.assert_called_once_with(12345)


@pytest.mark.parametrize(
    "pid_file_exists, expected_force_kill_called",
    [
        (True, True),
        (False, False),
    ],
)
@patch("meersolar.meerpipeline.kill_job.drop_cache")
@patch("meersolar.meerpipeline.kill_job.os.system")
@patch("meersolar.meerpipeline.kill_job.force_kill_pids_with_children")
@patch("meersolar.meerpipeline.kill_job.os.path.exists")
@patch("meersolar.meerpipeline.kill_job.os.kill")
@patch("meersolar.meerpipeline.kill_job.np.loadtxt")
@patch("meersolar.meerpipeline.kill_job.get_cachedir", return_value="/mock/cache")
@patch("sys.argv", ["kill_meersolar_job", "--jobid", "123"])
def test_kill_meerjob(
    mock_cachedir,
    mock_loadtxt,
    mock_kill,
    mock_exists,
    mock_force_kill,
    mock_system,
    mock_drop_cache,
    pid_file_exists,
    expected_force_kill_called,
):
    # Mock the loadtxt return for main_pids and pids file
    mock_loadtxt.side_effect = [
        ["123", "9999", "test.ms", "/mock/work", "/mock/out"],
        [111, 222],
    ]

    # Mock file existence
    def exists_side_effect(path):
        if "pids/pids_123.txt" in path:
            return pid_file_exists
        return True

    mock_exists.side_effect = exists_side_effect

    # Run the function
    kill_meerjob()

    mock_kill.assert_called_once_with(9999, signal.SIGKILL)
    if expected_force_kill_called:
        mock_force_kill.assert_called_once_with([111, 222])
    else:
        mock_force_kill.assert_not_called()
    assert mock_system.call_args[0][0].startswith("rm -rf /mock/work/tmp_meersolar_")
    assert mock_drop_cache.call_count == 4
