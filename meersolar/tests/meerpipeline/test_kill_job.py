import pytest
from unittest.mock import patch, MagicMock, call
import psutil
import numpy as np

from meersolar.meerpipeline.kill_job import (
    terminate_process_and_children,
    force_kill_pids_with_children,
    kill_meerjob,
)


@pytest.mark.parametrize("has_process", [True, False])
@patch("meersolar.meerpipeline.kill_job.psutil.wait_procs")
@patch("meersolar.meerpipeline.kill_job.psutil.Process")
def test_terminate_process_and_children(mock_process_cls, mock_wait_procs, has_process):
    mock_parent = MagicMock()
    mock_children = [MagicMock(), MagicMock()]
    mock_process_cls.return_value = mock_parent
    mock_parent.children.return_value = mock_children
    mock_wait_procs.return_value = ([], mock_children)

    if not has_process:
        mock_process_cls.side_effect = psutil.NoSuchProcess(9999)

    terminate_process_and_children(9999)

    if has_process:
        assert mock_parent.terminate.call_count == 1
        assert all(child.terminate.call_count == 1 for child in mock_children)
        assert all(child.kill.call_count == 1 for child in mock_children)
    else:
        mock_process_cls.assert_called_once()


@patch("meersolar.meerpipeline.kill_job.terminate_process_and_children")
@patch("meersolar.meerpipeline.kill_job.psutil.pid_exists")
def test_force_kill_pids_with_children(mock_pid_exists, mock_terminate):
    mock_pid_exists.side_effect = [True, False]

    pids = [111]
    force_kill_pids_with_children(pids, max_tries=2, wait_time=0.1)

    mock_terminate.assert_called_with(111)
    assert mock_pid_exists.call_count >= 1


@pytest.mark.parametrize("pid_file_exists", [True, False])
@patch("meersolar.meerpipeline.kill_job.drop_cache")
@patch("meersolar.meerpipeline.kill_job.os.system")
@patch("meersolar.meerpipeline.kill_job.force_kill_pids_with_children")
@patch("meersolar.meerpipeline.kill_job.os.path.exists")
@patch("meersolar.meerpipeline.kill_job.terminate_process_and_children")
@patch("meersolar.meerpipeline.kill_job.np.loadtxt")
@patch("meersolar.meerpipeline.kill_job.get_cachedir", return_value="/mock/cache")
@patch("sys.argv", ["kill_meersolar_job", "--jobid", "123"])
def test_kill_meerjob(
    mock_cachedir,
    mock_loadtxt,
    mock_terminate,
    mock_exists,
    mock_force_kill,
    mock_system,
    mock_drop_cache,
    pid_file_exists,
):
    # Simulate file contents
    mock_loadtxt.side_effect = [
        ["123", "9999", "test.ms", "/mock/work", "/mock/out"],  # main_pids file
        [111, 222],  # pids file
    ]

    # Simulate file existence
    def exists_side_effect(path):
        if "pids/pids_123.txt" in path:
            return pid_file_exists
        return True

    mock_exists.side_effect = exists_side_effect

    kill_meerjob()

    mock_terminate.assert_called_once_with(9999)

    if pid_file_exists:
        mock_force_kill.assert_called_once_with([111, 222])
    else:
        mock_force_kill.assert_not_called()

    mock_system.assert_called_once_with("rm -rf /mock/work/tmp_meersolar_*")
    mock_drop_cache.assert_has_calls(
        [
            call("test.ms"),
            call("/mock/work"),
            call("/mock/out"),
            call("/mock/cache"),
        ],
        any_order=False,
    )
