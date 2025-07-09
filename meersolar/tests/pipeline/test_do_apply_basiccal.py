import pytest
from unittest.mock import patch, MagicMock
from meersolar.pipeline.do_apply_basiccal import *

@pytest.mark.parametrize(
    "data, expected, raises",
    [
        ([1.0, np.nan, 3.0], [1.0, 2.0, 3.0], None),                        # internal NaN
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], None),                           # no NaNs
        ([np.nan, 1.0, 2.0, np.nan], [0.0, 1.0, 2.0, 3.0], None),           # edge NaNs
        ([np.nan, np.nan, np.nan], None, ValueError),                      # all NaNs
    ]
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
        ([1, 2, 100, 3, 4], 2, 0),        # may not detect 100 as outlier
        ([1, 1, 1, 1], 2, 0),
        ([1, np.nan, 100, 1], 2, 1),      # expect only original NaN retained
    ]
)
def test_filter_outliers(data, threshold, expected_n_nan):
    result = filter_outliers(np.array(data, dtype=float), threshold=threshold, max_iter=2)
    assert np.isnan(result).sum() == expected_n_nan
    
    
def test_scale_bandpass(dummy_bpass,dummy_att_table):
    expected=dummy_bpass.split(".bcal")[0]+"_scan_15.bcal"
    result=scale_bandpass(dummy_bpass, dummy_att_table, freqavg=10)
    assert result==expected
    assert os.path.exists(result)
    os.system(f"rm -rf {result}")
    assert os.path.exists(result)==False
    

@patch("meersolar.pipeline.do_apply_basiccal.psutil.Process")
@patch("meersolar.pipeline.do_apply_basiccal.limit_threads")
@patch("meersolar.pipeline.do_apply_basiccal.os.path.exists", return_value=False)
@patch("meersolar.pipeline.do_apply_basiccal.os.system")
@patch("meersolar.pipeline.do_apply_basiccal.glob.glob", return_value=[])
@patch("meersolar.pipeline.do_apply_basiccal.suppress_casa_output")
@patch("meersolar.pipeline.do_apply_basiccal.single_ms_flag")
@patch("casatasks.applycal",return_value=None)
@patch("casatasks.clearcal",return_value=None)
@patch("casatasks.flagdata",return_value=None)
@patch("casatasks.split",retun_value=None)
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
    mem = applysol(msname="test.ms", dry_run=True)
    assert mem == 2.0
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
        dry_run=False,
    )
    assert status == 0
    mock_applycal.assert_called_once()
    mock_split.assert_called_once()
    mock_single_flag.assert_called_once()
    with patch("meersolar.pipeline.do_apply_basiccal.limit_threads"), \
         patch("meersolar.pipeline.do_apply_basiccal.os.path.exists", side_effect=Exception("fail")), \
         patch("meersolar.pipeline.do_apply_basiccal.psutil.Process"):
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
        
@patch("meersolar.pipeline.do_apply_basiccal.drop_cache")
@patch("meersolar.pipeline.do_apply_basiccal.time.sleep")
@patch("meersolar.pipeline.do_apply_basiccal.time.time", side_effect=[0, 5])
@patch("meersolar.pipeline.do_apply_basiccal.os.chdir")
@patch("meersolar.pipeline.do_apply_basiccal.os.system")
@patch("meersolar.pipeline.do_apply_basiccal.os.path.exists", return_value=True)
@patch("meersolar.pipeline.do_apply_basiccal.glob.glob")
@patch("meersolar.pipeline.do_apply_basiccal.check_datacolumn_valid", return_value=True)
@patch("meersolar.pipeline.do_apply_basiccal.msmetadata")
@patch("meersolar.pipeline.do_apply_basiccal.get_ms_size", return_value=1.0)
@patch("meersolar.pipeline.do_apply_basiccal.run_limited_memory_task", return_value=1.0)
@patch("meersolar.pipeline.do_apply_basiccal.get_dask_client")
@patch("meersolar.pipeline.do_apply_basiccal.delayed", side_effect=lambda f, *a, **kw: f)
@patch("meersolar.pipeline.do_apply_basiccal.compute", side_effect=lambda *args: [0] * len(args))
@patch("meersolar.pipeline.do_apply_basiccal.applysol", return_value=0)
@patch("meersolar.pipeline.do_apply_basiccal.scale_bandpass", return_value="scaled.bcal")
def test_run_all_applysol(
    mock_scale,
    mock_applysol,
    mock_compute,
    mock_delayed,
    mock_dask,
    mock_memtask,
    mock_ms_size,
    mock_msmd,
    mock_checkcol,
    mock_glob,
    mock_exists,
    mock_system,
    mock_chdir,
    mock_time,
    mock_sleep,
    mock_drop,
):
    mock_dask.return_value = (MagicMock(), MagicMock(), 1, 1, 1.0)
    mock_glob.side_effect = mock_glob_pattern
    mock_msmd_inst = MagicMock()
    mock_msmd_inst.scannumbers.return_value = [1]
    mock_msmd.return_value = mock_msmd_inst
    result = run_all_applysol(
        mslist=["test1.ms", "test2.ms"],
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
    "argv_args, expected_return",
    [
        (
            [
                "run_applysol",
                "test1.ms,test2.ms",
                "--workdir", "/mock/workdir",
                "--caldir", "/mock/caldir",
                "--logfile", "/mock/logfile.log",
                "--jobid", "99",
            ],
            0,
        )
    ]
)
@patch("meersolar.pipeline.do_apply_basiccal.run_all_applysol", return_value=0)
@patch("meersolar.pipeline.do_apply_basiccal.clean_shutdown")
@patch("meersolar.pipeline.do_apply_basiccal.drop_cache")
@patch("meersolar.pipeline.do_apply_basiccal.time.sleep")
@patch("meersolar.pipeline.do_apply_basiccal.init_logger")
@patch("meersolar.pipeline.do_apply_basiccal.np.load", return_value=("jobname", "password"))
@patch("meersolar.pipeline.do_apply_basiccal.os.path.exists", return_value=True)
@patch("meersolar.pipeline.do_apply_basiccal.os.makedirs")
@patch("meersolar.pipeline.do_apply_basiccal.os.getpid", return_value=1234)
@patch("meersolar.pipeline.do_apply_basiccal.save_pid")
@patch("meersolar.pipeline.do_apply_basiccal.sys")
def test_main(
    mock_sys,
    mock_save_pid,
    mock_getpid,
    mock_makedirs,
    mock_exists,
    mock_npload,
    mock_logger,
    mock_sleep,
    mock_drop_cache,
    mock_shutdown,
    mock_run_apply,
    argv_args,
    expected_return,
):
    with patch("sys.argv", argv_args):
        from meersolar.pipeline import do_apply_basiccal
        result = do_apply_basiccal.main()
        assert result == expected_return
    
    
    
