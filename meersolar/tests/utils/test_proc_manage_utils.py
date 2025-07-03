import pytest
from unittest.mock import patch, MagicMock, mock_open, call
from meersolar.utils.proc_manage_utils import *


@patch("meersolar.utils.proc_manage_utils.psutil.pid_exists")
@patch("meersolar.utils.proc_manage_utils.np.loadtxt")
@patch("meersolar.utils.proc_manage_utils.get_meersolar_cachedir")
def test_get_nprocess_meersolar(mock_get_cachedir, mock_loadtxt, mock_pid_exists):
    mock_get_cachedir.return_value = "/mock/.meersolar"
    mock_loadtxt.return_value = [101, 102, 103]
    mock_pid_exists.side_effect = lambda pid: pid in [102, 103]
    result = get_nprocess_meersolar(jobid=42)
    assert result == 2
    mock_loadtxt.assert_called_once_with(
        "/mock/.meersolar/pids/pids_42.txt", unpack=True
    )


def test_save_pid():
    save_pid(10, "/tmp/test_pid.txt")
    assert os.path.exists("/tmp/test_pid.txt") == True
    a = np.loadtxt("/tmp/test_pid.txt", dtype="int")
    assert a == 10
    os.system(f"rm -rf /tmp/test_pid.txt")


@patch("meersolar.utils.proc_manage_utils.np.savetxt")
@patch("meersolar.utils.proc_manage_utils.np.loadtxt")
@patch("meersolar.utils.proc_manage_utils.os.path.exists")
@patch("meersolar.utils.proc_manage_utils.get_meersolar_cachedir")
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
    "meersolar.utils.proc_manage_utils.get_meersolar_cachedir",
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


@patch("meersolar.utils.proc_manage_utils.os.system")
@patch("meersolar.utils.proc_manage_utils.os.path.exists")
@patch("meersolar.utils.proc_manage_utils.os.makedirs")
@patch("meersolar.utils.proc_manage_utils.os.path.isdir")
@patch("builtins.open", new_callable=mock_open)
def test_create_batch_script_nonhpc(
    mock_openfile,
    mock_isdir,
    mock_makedirs,
    mock_exists,
    mock_system,
):
    cmd = "python script.py"
    workdir = "/mock/work"
    basename = "testjob"
    mock_isdir.return_value = False
    mock_exists.return_value = False
    batch_path = f"{workdir}/{basename}.batch"
    cmd_batch_path = f"{workdir}/{basename}_cmd.batch"
    finished_prefix = f"{workdir}/.Finished_{basename}"
    outputfile = f"{workdir}/logs/{basename}.log"
    returned_batch_file = create_batch_script_nonhpc(cmd, workdir, basename)
    assert returned_batch_file == batch_path
    mock_isdir.assert_called_once_with(f"{workdir}/logs")
    mock_makedirs.assert_called_once_with(f"{workdir}/logs")
    handle = mock_openfile()
    expected_cmd = (
        f"{cmd}; exit_code=$?; if [ $exit_code -ne 0 ]; then touch {finished_prefix}_1; "
        f"else touch {finished_prefix}_0; fi"
    )
    expected_batch = (
        f"export PYTHONUNBUFFERED=1\n"
        f"nohup sh {cmd_batch_path}> {outputfile} 2>&1 &\n"
        f"sleep 2\n rm -rf {batch_path}\n rm -rf {cmd_batch_path}"
    )
    handle.write.assert_has_calls(
        [
            call(expected_cmd),  # first write: cmd batch file
            call(expected_batch),  # second write: launch batch file
        ]
    )
    mock_system.assert_any_call(f"rm -rf {finished_prefix}*")
    mock_system.assert_any_call(f"chmod a+rwx {batch_path}")
    mock_system.assert_any_call(f"chmod a+rwx {cmd_batch_path}")


@patch("meersolar.utils.proc_manage_utils.resource.setrlimit")
@patch(
    "meersolar.utils.proc_manage_utils.resource.getrlimit", return_value=(4096, 8192)
)
@patch("meersolar.utils.proc_manage_utils.psutil.swap_memory")
@patch("meersolar.utils.proc_manage_utils.psutil.virtual_memory")
@patch("meersolar.utils.proc_manage_utils.psutil.cpu_percent", return_value=10)
@patch("meersolar.utils.proc_manage_utils.psutil.cpu_count", return_value=16)
@patch("meersolar.utils.proc_manage_utils.os.makedirs")
def test_get_dask_client(
    mock_makedirs,
    mock_cpu_count,
    mock_cpu_percent,
    mock_virtual_memory,
    mock_swap_memory,
    mock_getrlimit,
    mock_setrlimit,
):
    mock_virtual_memory.return_value = MagicMock(
        total=64 * 1024**3, available=60 * 1024**3
    )
    mock_swap_memory.return_value = MagicMock(total=8 * 1024**3)
    client, cluster, n_workers, threads_per_worker, mem_per_worker = get_dask_client(
        n_jobs=4,
        dask_dir="/mock/tmp",
        only_cal=True,
        min_mem_per_job=4,
    )
    assert client is None
    assert cluster is None
    assert n_workers >= 1
    assert threads_per_worker >= 1
    assert mem_per_worker > 0


def test_run_limited_memory_task():
    def slow_function():
        time.sleep(2)
        return sum(range(1000000))

    task = delayed(slow_function)()
    mem_gb = run_limited_memory_task(task, dask_dir="/tmp", timeout=5)
    assert mem_gb is not None
    assert isinstance(mem_gb, float)
    assert mem_gb > 0
