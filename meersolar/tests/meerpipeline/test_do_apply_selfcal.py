import pytest
from unittest.mock import patch, MagicMock
from meersolar.meerpipeline.do_apply_selfcal import *


@patch("meersolar.meerpipeline.do_apply_selfcal.drop_cache")
@patch("meersolar.meerpipeline.do_apply_selfcal.os.system")
@patch("meersolar.meerpipeline.do_apply_selfcal.os.chdir")
@patch(
    "meersolar.meerpipeline.do_apply_selfcal.check_datacolumn_valid", return_value=True
)
@patch("meersolar.meerpipeline.do_apply_selfcal.msmetadata")
@patch("meersolar.meerpipeline.do_apply_selfcal.get_local_dask_cluster")
@patch("meersolar.meerpipeline.do_apply_selfcal.get_column_size", return_value=1.0)
@patch("meersolar.meerpipeline.do_apply_selfcal.delayed")
@patch(
    "meersolar.meerpipeline.do_apply_selfcal.glob.glob",
    return_value=["/tmp/caldir/selfcal_scan_1.gcal"],
)
@patch(
    "meersolar.meerpipeline.do_apply_selfcal.os.path.basename",
    side_effect=lambda x: x.split("/")[-1],
)
def test_run_all_applysol(
    mock_basename,
    mock_glob,
    mock_delayed,
    mock_get_column_size,
    mock_get_dask_client,
    mock_msmetadata,
    mock_check_datacol,
    mock_chdir,
    mock_os_system,
    mock_drop_cache,
):
    mock_msmd = MagicMock()
    mock_msmd.open.return_value = None
    mock_msmd.close.return_value = None
    mock_msmd.scannumbers.return_value = [1]
    mock_msmetadata.return_value = mock_msmd

    mock_delayed.side_effect = lambda fn=None, *args, **kwargs: MagicMock()

    mock_dask_client = MagicMock()
    mock_dask_cluster = MagicMock()
    mock_get_dask_client.return_value = (
        mock_dask_client,
        mock_dask_cluster,
        1,
        1,
        4.0,
        "/mock/dask_dir",
    )

    result = run_all_applysol(
        ["mock.ms"],
        mock_dask_client,
        workdir="/tmp/work",
        caldir="/tmp/caldir",
        overwrite_datacolumn=True,
        applymode="calonly",
        force_apply=True,
        cpu_frac=0.8,
        mem_frac=0.8,
    )

    assert result == 0


@pytest.mark.parametrize(
    "mslist_str, caldir_exists, run_ok, expected_msg",
    [
        ("ms1.ms,ms2.ms", True, True, 0),
        ("ms1.ms", False, True, 1),
        ("ms1.ms", True, False, 1),
    ],
)
@patch("meersolar.meerpipeline.do_apply_selfcal.save_pid")
@patch(
    "meersolar.meerpipeline.do_apply_selfcal.get_cachedir", return_value="/mock/cache"
)
@patch("os.makedirs")
@patch("os.path.exists")
@patch("os.getpid", return_value=5678)
@patch("meersolar.meerpipeline.do_apply_selfcal.run_all_applysol")
@patch("meersolar.meerpipeline.do_apply_selfcal.drop_cache")
@patch("meersolar.meerpipeline.do_apply_selfcal.clean_shutdown")
@patch("time.sleep", return_value=None)
@patch("traceback.print_exc", return_value=None)
def test_main_applysol(
    mock_trace,
    mock_sleep,
    mock_shutdown,
    mock_drop,
    mock_run_all,
    mock_getpid,
    mock_exists,
    mock_makedirs,
    mock_cachedir,
    mock_save_pid,
    mslist_str,
    caldir_exists,
    run_ok,
    expected_msg,
):
    mslist = mslist_str.split(",")
    workdir = "/mock/work"
    caldir = "/mock/caldir" if caldir_exists else f"{workdir}/caltables"

    def exists_side_effect(path):
        if "jobname_password.npy" in path:
            return False
        if path == caldir:
            return caldir_exists
        if path in mslist or path == workdir:
            return True
        return False

    mock_exists.side_effect = exists_side_effect
    mock_run_all.return_value = 0 if run_ok else 1

    dask_client = MagicMock()
    msg = main(
        mslist=mslist_str,
        workdir=workdir,
        caldir=caldir,
        applymode="calonly",
        overwrite_datacolumn=False,
        force_apply=False,
        start_remote_log=False,
        cpu_frac=0.5,
        mem_frac=0.5,
        logfile=None,
        jobid=42,
        dask_client=dask_client,
    )

    assert msg == expected_msg
    mock_makedirs.assert_any_call(workdir, exist_ok=True)


@pytest.mark.parametrize(
    "argv, should_exit",
    [
        (["prog.py"], True),
        (
            [
                "prog.py",
                "ms1.ms,ms2.ms",
                "--workdir",
                "/mock/work",
                "--caldir",
                "/mock/caldir",
                "--overwrite_datacolumn",
                "--force_apply",
                "--cpu_frac",
                "0.6",
                "--mem_frac",
                "0.7",
                "--jobid",
                "321",
            ],
            False,
        ),
    ],
)
@patch("meersolar.meerpipeline.do_apply_selfcal.main", return_value=0)
@patch("meersolar.meerpipeline.do_apply_selfcal.sys.exit")
@patch("meersolar.meerpipeline.do_apply_selfcal.argparse.ArgumentParser.print_help")
def test_cli_apply_selfcal(mock_print_help, mock_exit, mock_main, argv, should_exit):
    with patch("sys.argv", argv):
        from meersolar.meerpipeline import do_apply_selfcal

        result = do_apply_selfcal.cli()
        assert result == should_exit
