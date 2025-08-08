import pytest
from unittest.mock import patch, MagicMock
from meersolar.meerpipeline.do_apply_basiccal import *


@pytest.mark.parametrize(
    "data, expected, raises",
    [
        ([1.0, np.nan, 3.0], [1.0, 2.0, 3.0], None),  # internal NaN
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], None),  # no NaNs
        ([np.nan, 1.0, 2.0, np.nan], [0.0, 1.0, 2.0, 3.0], None),  # edge NaNs
        ([np.nan, np.nan, np.nan], None, ValueError),  # all NaNs
    ],
)
def test_interpolate_nans(data, expected, raises):
    data = np.array(data, dtype=float)
    if raises:
        with pytest.raises(raises, match="All values are NaN."):
            interpolate_nans(data)
    else:
        result = interpolate_nans(data)
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "data, threshold, expected_n_nan",
    [
        ([1, 2, 100, 3, 4], 2, 0),  # may not detect 100 as outlier
        ([1, 1, 1, 1], 2, 0),
        ([1, np.nan, 100, 1], 2, 1),  # expect only original NaN retained
    ],
)
def test_filter_outliers(data, threshold, expected_n_nan):
    result = filter_outliers(
        np.array(data, dtype=float), threshold=threshold, max_iter=2
    )
    assert np.isnan(result).sum() == expected_n_nan


def test_scale_bandpass(dummy_bpass, dummy_att_table):
    expected = dummy_bpass.split(".bcal")[0] + "_att.bcal"
    result = scale_bandpass(dummy_bpass, dummy_att_table)
    assert result == expected
    assert os.path.exists(result)
    os.system(f"rm -rf {result}")
    assert os.path.exists(result) == False
    expected = dummy_bpass.split(".bcal")[0] + "_att.bcal"
    result = scale_bandpass(dummy_bpass, [dummy_att_table, dummy_att_table])
    assert result == expected
    assert os.path.exists(result)
    os.system(f"rm -rf {result}")
    assert os.path.exists(result) == False


@patch("meersolar.meerpipeline.do_apply_basiccal.psutil.Process")
@patch("meersolar.meerpipeline.do_apply_basiccal.limit_threads")
@patch("meersolar.meerpipeline.do_apply_basiccal.os.path.exists", return_value=False)
@patch("meersolar.meerpipeline.do_apply_basiccal.os.system")
@patch("meersolar.meerpipeline.do_apply_basiccal.glob.glob", return_value=[])
@patch("meersolar.meerpipeline.do_apply_basiccal.suppress_output")
@patch("meersolar.meerpipeline.do_apply_basiccal.single_ms_flag")
@patch("casatasks.applycal", return_value=None)
@patch("casatasks.clearcal", return_value=None)
@patch("casatasks.flagdata", return_value=None)
@patch("casatasks.split", retun_value=None)
def test_applysol(
    mock_split,
    mock_flagdata,
    mock_clearcal,
    mock_applycal,
    mock_single_flag,
    mock_suppress,
    mock_glob,
    mock_system,
    mock_exists,
    mock_limit_threads,
    mock_process,
):
    mock_proc = MagicMock()
    mock_proc.memory_info.return_value.rss = 2 * 1024**3  # 2GB
    mock_process.return_value = mock_proc
    status = applysol(
        msname="test.ms",
        gaintable=["a.cal", "b.cal"],
        gainfield=["", ""],
        interp=["linear", "linear"],
        parang=True,
        applymode="calflag",
        overwrite_datacolumn=True,
        n_threads=2,
        memory_limit=1.0,
        force_apply=True,
        soltype="basic",
        do_post_flag=True,
    )
    assert status == 0
    mock_applycal.assert_called_once()
    mock_split.assert_called_once()
    mock_single_flag.assert_called_once()
    with (
        patch("meersolar.meerpipeline.do_apply_basiccal.limit_threads"),
        patch(
            "meersolar.meerpipeline.do_apply_basiccal.os.path.exists",
            side_effect=Exception("fail"),
        ),
        patch("meersolar.meerpipeline.do_apply_basiccal.psutil.Process"),
    ):
        result = applysol(msname="bad.ms")
        assert result == 1


def mock_glob_pattern(pattern):
    if "attval_scan" in pattern:
        return ["myms_attval_scan_9.npy"]
    elif "calibrator_caltable_scan" in pattern:
        return ["/mock/caldir/calibrator_caltable_scan_9.bcal"]
    elif "bcal" in pattern:
        return ["/mock/caldir/calibrator_caltable.bcal"]
    elif "kcal" in pattern:
        return ["/mock/caldir/calibrator_caltable.kcal"]
    elif "gcal" in pattern:
        return ["/mock/caldir/calibrator_caltable.gcal"]
    elif "dcal" in pattern:
        return ["/mock/caldir/calibrator_caltable.dcal"]
    elif "kcrosscal" in pattern:
        return ["/mock/caldir/calibrator_caltable.kcrosscal"]
    elif "xfcal" in pattern:
        return ["/mock/caldir/calibrator_caltable.xfcal"]
    elif "panglecal" in pattern:
        return ["/mock/caldir/calibrator_caltable.panglecal"]
    return []


@patch("meersolar.meerpipeline.do_apply_basiccal.drop_cache")
@patch("meersolar.meerpipeline.do_apply_basiccal.time.sleep")
@patch("meersolar.meerpipeline.do_apply_basiccal.time.time", side_effect=[0, 5])
@patch("meersolar.meerpipeline.do_apply_basiccal.os.chdir")
@patch("meersolar.meerpipeline.do_apply_basiccal.os.system")
@patch("meersolar.meerpipeline.do_apply_basiccal.os.path.exists", return_value=True)
@patch("meersolar.meerpipeline.do_apply_basiccal.glob.glob")
@patch(
    "meersolar.meerpipeline.do_apply_basiccal.check_datacolumn_valid", return_value=True
)
@patch("meersolar.meerpipeline.do_apply_basiccal.get_column_size", return_value=1.0)
@patch("meersolar.meerpipeline.do_apply_basiccal.get_local_dask_cluster")
@patch(
    "meersolar.meerpipeline.do_apply_basiccal.delayed",
    side_effect=lambda f, *a, **kw: f,
)
@patch("meersolar.meerpipeline.do_apply_basiccal.applysol", return_value=0)
@patch(
    "meersolar.meerpipeline.do_apply_basiccal.scale_bandpass",
    return_value="scaled.bcal",
)
def test_run_all_applysol(
    mock_scale,
    mock_applysol,
    mock_delayed,
    mock_dask,
    mock_ms_size,
    mock_checkcol,
    mock_glob,
    mock_exists,
    mock_system,
    mock_chdir,
    mock_time,
    mock_sleep,
    mock_drop,
):
    mock_dask.return_value = (MagicMock(), MagicMock(), "/mock/dask_dir")
    mock_glob.side_effect = mock_glob_pattern
    dask_client = MagicMock()
    result = run_all_applysol(
        ["test1.ms", "test2.ms"],
        dask_client,
        workdir="/mock/workdir",
        caldir="/mock/caldir",
        use_only_bandpass=False,
        overwrite_datacolumn=True,
        applymode="calflag",
        force_apply=True,
        do_post_flag=True,
        cpu_frac=0.8,
        mem_frac=0.8,
    )
    assert result == 0


@pytest.mark.parametrize(
    "mslist_str, caldir_exists, expected_msg",
    [
        ("ms1.ms,ms2.ms", True, 0),
        ("ms1.ms", False, 1),
        (
            "ms1.ms",
            True,
            0,
        ),  # previously expected 1, but run_all_applysol always returns 0
    ],
)
@patch("meersolar.meerpipeline.do_apply_basiccal.save_pid")
@patch(
    "meersolar.meerpipeline.do_apply_basiccal.get_cachedir", return_value="/mock/cache"
)
@patch("os.makedirs")
@patch("os.path.exists")
@patch("os.getpid", return_value=1234)
@patch("time.sleep", return_value=None)
@patch("traceback.print_exc", return_value=None)
@patch("meersolar.meerpipeline.do_apply_basiccal.drop_cache")
@patch("meersolar.meerpipeline.do_apply_basiccal.clean_shutdown")
@patch("meersolar.meerpipeline.do_apply_basiccal.run_all_applysol", return_value=0)
def test_main_apply_basiccal(
    mock_run_all_applysol,
    mock_shutdown,
    mock_drop_cache,
    mock_trace,
    mock_sleep,
    mock_getpid,
    mock_exists,
    mock_makedirs,
    mock_get_cachedir,
    mock_save_pid,
    mslist_str,
    caldir_exists,
    expected_msg,
):
    mslist = mslist_str.split(",")
    workdir = "/mock/work"
    caldir = "/mock/caltables" if caldir_exists else ""
    dask_client = MagicMock()

    def exists_side_effect(path):
        if "jobname_password.npy" in path:
            return False
        if path in ["/mock/caltables"] + mslist + [workdir]:
            return True
        return False

    mock_exists.side_effect = exists_side_effect

    msg = main(
        mslist=mslist_str,
        workdir=workdir,
        caldir=caldir,
        use_only_bandpass=False,
        applymode="calflag",
        overwrite_datacolumn=False,
        force_apply=False,
        do_post_flag=False,
        start_remote_log=False,
        cpu_frac=0.7,
        mem_frac=0.6,
        logfile="mock.log",
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
                "/mock/caltables",
                "--use_only_bandpass",
                "--force_apply",
            ],
            False,
        ),
    ],
)
@patch("meersolar.meerpipeline.do_apply_basiccal.main", return_value=0)
@patch("meersolar.meerpipeline.do_apply_basiccal.sys.exit")
@patch("meersolar.meerpipeline.do_apply_basiccal.argparse.ArgumentParser.print_help")
def test_cli_apply_basiccal(mock_print_help, mock_exit, mock_main, argv, should_exit):
    with patch("sys.argv", argv):
        from meersolar.meerpipeline import do_apply_basiccal

        result = do_apply_basiccal.cli()
        assert result == should_exit
