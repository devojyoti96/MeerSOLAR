import pytest
import os
import sys
import tempfile
import numpy as np
import time
from pathlib import Path
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
from datetime import datetime as dt
from unittest.mock import patch, MagicMock, mock_open, call
from itertools import chain, repeat
from meersolar.utils.proc_manage_utils import *


def test_get_total_worker():
    mock_cluster = MagicMock()
    mock_cluster.workers = {"worker-1": {}, "worker-2": {}}
    assert get_total_worker(mock_cluster) == 2


@pytest.mark.parametrize(
    "current_workers,expected_result,description",
    [
        ([0, 1, 2], 0, "successfully scaled within timeout"),
        ([0, 0, 0], 1, "failed to scale within timeout"),
    ],
)
def test_scale_worker_and_wait(current_workers, expected_result, description):
    mock_cluster = MagicMock()
    mock_cluster.workers = {}

    # Side effect to simulate worker count increasing over time
    def get_workers_side_effect():
        count = current_workers.pop(0) if current_workers else 0
        return {f"worker-{i}": {} for i in range(count)}

    with (
        patch(
            "meersolar.utils.proc_manage_utils.get_total_worker"
        ) as mock_get_total_worker,
        patch("meersolar.utils.proc_manage_utils.time.sleep", return_value=None),
    ):

        mock_get_total_worker.side_effect = lambda cluster: len(
            get_workers_side_effect()
        )
        result = scale_worker_and_wait(mock_cluster, 2, timeout=3, poll_interval=1)

        assert (
            result == expected_result
        ), f"Expected {expected_result} but got {result} for: {description}"
        mock_cluster.scale.assert_called_with(2)


def test_save_pid():
    os.system("rm -rf /tmp/test_pid.txt")
    save_pid(10, "/tmp/test_pid.txt")
    assert os.path.exists("/tmp/test_pid.txt") == True
    a = np.loadtxt("/tmp/test_pid.txt", dtype="int")
    assert a == 10
    os.system(f"rm -rf /tmp/test_pid.txt")


@patch("meersolar.utils.proc_manage_utils.psutil.pid_exists")
@patch("meersolar.utils.proc_manage_utils.np.loadtxt")
@patch("meersolar.utils.proc_manage_utils.get_cachedir")
def get_nprocess_solarpipe(mock_get_cachedir, mock_loadtxt, mock_pid_exists):
    mock_get_cachedir.return_value = "/mock/.meersolar"
    mock_loadtxt.return_value = [101, 102, 103]
    mock_pid_exists.side_effect = lambda pid: pid in [102, 103]
    result = get_nprocess_solarpipe(jobid=42)
    assert result == 2
    mock_loadtxt.assert_called_once_with(
        "/mock/.meersolar/pids/pids_42.txt", unpack=True
    )


@patch("meersolar.utils.proc_manage_utils.np.savetxt")
@patch("meersolar.utils.proc_manage_utils.np.loadtxt")
@patch("meersolar.utils.proc_manage_utils.os.path.exists")
@patch("meersolar.utils.proc_manage_utils.get_cachedir")
@patch("meersolar.utils.proc_manage_utils.dt")
def test_get_jobid(mock_dt, mock_getdir, mock_exists, mock_loadtxt, mock_savetxt):
    fake_time = dt(2025, 7, 1, 15, 30, 45, 123456)
    mock_dt.utcnow.return_value = fake_time
    mock_getdir.return_value = "/mock/.meersolar"
    mock_exists.return_value = False
    mock_loadtxt.return_value = []
    jobid = get_jobid()
    expected = int("20250701153045123")
    assert jobid == expected
    mock_savetxt.assert_called_once()


@patch("meersolar.utils.proc_manage_utils.dt")
@patch("builtins.open", new_callable=mock_open)
@patch("meersolar.utils.proc_manage_utils.os.system")
@patch("meersolar.utils.proc_manage_utils.os.path.exists")
@patch("meersolar.utils.proc_manage_utils.glob.glob")
@patch(
    "meersolar.utils.proc_manage_utils.get_cachedir",
    return_value="/mock/.meersolar",
)
def test_save_main_process_info(
    mock_get_cachedir,
    mock_glob,
    mock_exists,
    mock_system,
    mock_openfile,
    mock_dt,
):
    mock_glob.return_value = ["/mock/.meersolar/main_pids_20250625000000000000.txt"]
    mock_exists.return_value = True
    fake_now = dt(2025, 7, 1, 0, 0, 0)
    mock_dt.utcnow.return_value = fake_now
    mock_dt.strptime.side_effect = lambda s, fmt: dt.strptime(s, fmt)
    result = save_main_process_info(
        pid=1234,
        jobid="20250701010101010101",
        msname="test.ms",
        workdir="/mock/workdir",
        outdir="/mock/outdir",
        cpu_frac=0.5,
        mem_frac=0.6,
    )
    expected_file = "/mock/.meersolar/main_pids_20250701010101010101.txt"
    assert result == expected_file
    mock_openfile().write.assert_called_once_with(
        "20250701010101010101 1234 test.ms /mock/workdir /mock/outdir 0.5 0.6"
    )
    mock_glob.return_value = ["/mock/.meersolar/main_pids_20250625000000000000.txt"]
    mock_system.assert_any_call(
        "rm -rf /mock/.meersolar/main_pids_20250625000000000000.txt"
    )
    mock_system.assert_any_call(
        "rm -rf /mock/.meersolar/pids/pids_20250625000000000000.txt"
    )


def calc_sum(i):
    time.sleep(0.5)
    return np.nansum(i)


def test_get_local_dask_cluster():
    client, cluster, dask_dir = get_local_dask_cluster(
        1,
        dask_dir="/tmp/test_dask",
    )
    assert client is not None
    cluster.adapt(minimum=2, maximum=5)
    expected_results = [np.nansum(i) for i in range(10)]
    tasks = [delayed(calc_sum)(i) for i in range(10)]
    results = compute(*tasks)
    results = list(results)
    client.close()
    cluster.close()
    assert results == expected_results
    assert os.path.exists(dask_dir)
    os.system(f"rm -rf /tmp/test_dask")
    assert os.path.exists(dask_dir) == False


def dummy_task():
    time.sleep(2)
    return sum(range(1000000))


@pytest.mark.parametrize(
    "env_type, mock_env, expected_line",
    [
        (
            "conda",
            {"CONDA_DEFAULT_ENV": "myenv"},
            "conda activate myenv",
        ),
        (
            "virtualenv",
            {"VIRTUAL_ENV": "/fake/venv"},
            "source /fake/venv/bin/activate",
        ),
        (
            "plain",
            {},
            f"export PATH=/usr/bin:$PATH",
        ),
    ],
)
def test_generate_activate_env(env_type, mock_env, expected_line):
    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = os.path.join(tmpdir, "test_env.sh")

        # Patch environment variables
        with (
            patch.dict(os.environ, mock_env, clear=True),
            patch(
                "sys.executable",
                new=sys.executable if env_type != "plain" else "/usr/bin/python3",
            ),
            patch("subprocess.run") as mock_run,
        ):

            # For conda case, simulate successful `module avail`
            mock_run.return_value = MagicMock(returncode=0)

            result = generate_activate_env(outfile)
            content = Path(result).read_text()

            assert expected_line in content
            assert os.access(result, os.X_OK)
