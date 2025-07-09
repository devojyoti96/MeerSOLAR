import pytest
from unittest.mock import patch, MagicMock
from meersolar.pipeline.do_target_split import *

def test_chanlist_to_str():
    result=chanlist_to_str([0,1,2,10,45])
    assert result=="0~2;10;45"
    
@patch("meersolar.pipeline.do_target_split.psutil.Process")
@patch("meersolar.pipeline.do_target_split.os.path.exists")
@patch("meersolar.pipeline.do_target_split.os.system")
@patch("meersolar.pipeline.do_target_split.limit_threads")
@patch("meersolar.pipeline.do_target_split.suppress_casa_output")
@patch("meersolar.pipeline.do_target_split.msmetadata")
@patch("casatasks.split")
@patch("casatasks.initweights")
@patch("casatasks.flagdata")
def test_split_scan(
    mock_flagdata,
    mock_initweights,
    mock_split,
    mock_msmetadata,
    mock_suppress,
    mock_limit,
    mock_system,
    mock_exists,
    mock_psutil_process,
):
    mock_msmd_instance = MagicMock()
    mock_msmd_instance.fieldsforscan.return_value = [0, 1]
    mock_msmd_instance.open = MagicMock()
    mock_msmd_instance.close = MagicMock()
    mock_msmetadata.return_value = mock_msmd_instance

    mock_exists.side_effect = lambda path: False if ".splited" in path else True
    mock_process = MagicMock()
    mock_process.memory_info.return_value.rss = 4 * 1024**3
    mock_psutil_process.return_value = mock_process

    result = split_scan(
        msname="test.ms",
        outputvis="split.ms",
        scan="3",
        width="2",
        timebin="4s",
        datacolumn="corrected",
        spw="0",
        corr="RR,LL",
        timerange="",
        n_threads=2,
        dry_run=False,
    )
    assert result == "split.ms"
    mock_split.assert_called_once()
    mock_initweights.assert_called_once()
    mock_flagdata.assert_called_once()
    mock_system.assert_any_call("touch split.ms/.splited")
    
    
@patch("meersolar.pipeline.do_target_split.get_dask_client")
@patch("meersolar.pipeline.do_target_split.compute", return_value=["mock.ms"])
@patch("meersolar.pipeline.do_target_split.run_limited_memory_task", return_value=1.0)
@patch("meersolar.pipeline.do_target_split.get_cal_target_scans", return_value=([1, 2], [], [], [], []))
@patch("meersolar.pipeline.do_target_split.get_valid_scans", return_value=[1, 2])
@patch("meersolar.pipeline.do_target_split.get_bad_chans", return_value="0:0~100;200~300")
@patch("meersolar.pipeline.do_target_split.get_pol_names", return_value="XX,YY")
@patch("meersolar.pipeline.do_target_split.get_common_spw", return_value="0:0~100;200~300")
@patch("meersolar.pipeline.do_target_split.get_timeranges_for_scan", return_value=["10s", "20s"])
@patch("meersolar.pipeline.do_target_split.split_into_chunks", return_value=[[0, 1, 2]])
@patch("meersolar.pipeline.do_target_split.chanlist_to_str", return_value="0~2")
@patch("meersolar.pipeline.do_target_split.single_mstransform", return_value="mock.ms")
@patch("meersolar.pipeline.do_target_split.drop_cache")
@patch("meersolar.pipeline.do_target_split.msmetadata")
@patch("meersolar.pipeline.do_target_split.os.chdir")
def test_split_target_scans(
    mock_chdir,
    mock_msmetadata,
    mock_drop_cache,
    mock_single_mstransform,
    mock_chanlist_to_str,
    mock_split_into_chunks,
    mock_get_timeranges_for_scan,
    mock_get_common_spw,
    mock_get_pol_names,
    mock_get_bad_chans,
    mock_get_valid_scans,
    mock_get_cal_target_scans,
    mock_run_mem,
    mock_compute,
    mock_get_dask_client,
):
    mock_dask_client = MagicMock()
    mock_dask_cluster = MagicMock()
    mock_get_dask_client.return_value = (mock_dask_client, mock_dask_cluster, 1, 1, 1.0)
    mock_msmd = MagicMock()
    mock_msmd.chanres.return_value = [0.1]
    mock_msmd.chanfreqs.return_value = [100.0, 200.0, 300.0]
    mock_msmd.nchan.return_value = 3
    mock_msmetadata.return_value = mock_msmd

    msg, result = split_target_scans(
        msname="mock.ms",
        workdir="/mock/workdir",
        timeres=1.0,
        freqres=1.0,
        datacolumn="DATA",
        scans=[],
    )
    assert msg == 0
    assert result == ["mock.ms"]
    
    
@pytest.mark.parametrize(
    "argv_args, expected_return",
    [
        (
            [
                "run_target_split",
                "mock.ms",
                "--workdir",
                "/mock/workdir",
                "--timeres",
                "2",
                "--freqres",
                "1",
                "--datacolumn",
                "data",
                "--scans",
                "1,2",
            ],
            0,
        )
    ]
)
@patch("meersolar.pipeline.do_target_split.split_target_scans", return_value=(0, ["mock.ms"]))
@patch("meersolar.pipeline.do_target_split.os.makedirs")
@patch("meersolar.pipeline.do_target_split.os.path.exists", return_value=True)
@patch("meersolar.pipeline.do_target_split.save_pid")
@patch("meersolar.pipeline.do_target_split.init_logger", return_value=None)
@patch("meersolar.pipeline.do_target_split.clean_shutdown")
@patch("meersolar.pipeline.do_target_split.drop_cache")
@patch("meersolar.pipeline.do_target_split.np.load", return_value=["job", "pass"])
@patch("meersolar.pipeline.do_target_split.time.sleep")
def test_main(
    mock_sleep,
    mock_np_load,
    mock_drop_cache,
    mock_clean_shutdown,
    mock_init_logger,
    mock_save_pid,
    mock_exists,
    mock_makedirs,
    mock_split_target_scans,
    argv_args,
    expected_return,
):
    with patch("sys.argv", argv_args):
        from meersolar.pipeline import do_target_split
        result = do_target_split.main()
        assert result == expected_return
        
    



