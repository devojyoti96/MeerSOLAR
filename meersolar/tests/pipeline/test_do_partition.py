import pytest
from unittest.mock import patch, MagicMock
from meersolar.pipeline.do_partition import *

@patch("meersolar.pipeline.do_partition.msmetadata")
@patch("meersolar.pipeline.do_partition.get_valid_scans", return_value=[1, 2])
@patch("meersolar.pipeline.do_partition.get_pol_names", return_value="XX,YY")
@patch("meersolar.pipeline.do_partition.get_ms_scan_size", return_value=1.0)
@patch("meersolar.pipeline.do_partition.run_limited_memory_task", return_value=1.0)
@patch("meersolar.pipeline.do_partition.get_dask_client")
@patch("meersolar.pipeline.do_partition.tmp_with_cache_rel")
@patch("meersolar.pipeline.do_partition.compute", return_value=["mock.ms", "mock.ms2"])
@patch("meersolar.pipeline.do_partition.single_mstransform")
@patch("meersolar.pipeline.do_partition.suppress_casa_output")
@patch("meersolar.pipeline.do_partition.os")
@patch("meersolar.pipeline.do_partition.time.sleep")
@patch("meersolar.pipeline.do_partition.drop_cache")
@patch("casatasks.virtualconcat")
def test_partion_ms(
    mock_virtualconcat,
    mock_drop_cache,
    mock_sleep,
    mock_os,
    mock_suppress,
    mock_single_transform,
    mock_compute,
    mock_tmpdir,
    mock_get_dask_client,
    mock_memtask,
    mock_get_scan_size,
    mock_get_pol,
    mock_get_valid,
    mock_msmetadata,
):
    mock_msmd = MagicMock()
    mock_msmd.scannumbers.return_value = [1, 2]
    mock_msmd.scansforfield.return_value.tolist.return_value = [1]
    mock_msmd.fieldnames.return_value = ["Field1", "Field2"]
    mock_msmd.fieldsforscan.return_value = [0]
    mock_msmetadata.return_value = mock_msmd
    mock_context = MagicMock()
    mock_context.__enter__.return_value = "/mock/tmp"
    mock_context.__exit__.return_value = None
    mock_tmpdir.return_value = mock_context
    mock_dask_client = MagicMock()
    mock_dask_cluster = MagicMock()
    mock_get_dask_client.return_value = (mock_dask_client, mock_dask_cluster, 2, 2, 1.0)
    result = partion_ms(
        msname="mock.ms",
        outputms="final.ms",
        workdir="/mock/tmp",
        fields="",
        scans="",
        width=1,
        timebin="5s",
        fullpol=True,
        datacolumn="DATA",
        cpu_frac=0.5,
        mem_frac=0.5,
    )
    mock_virtualconcat.assert_called_once()
    assert result == "final.ms"
    
    
@pytest.mark.parametrize(
    "argv_args, expected_return",
    [
        (
            [
                "run_partition",
                "mock.ms",
                "--outputms",
                "mock_output.ms",
                "--workdir",
                "/mock/workdir",
                "--fields",
                "0,1",
                "--scans",
                "2,3",
                "--width",
                "4",
                "--timebin",
                "10s",
                "--datacolumn",
                "data",
                "--split_fullpol",
                "--cpu_frac",
                "0.5",
                "--mem_frac",
                "0.6",
                "--logfile",
                "/mock/log.txt",
                "--jobid",
                "123",
            ],
            0,
        )
    ]
)
@patch("meersolar.pipeline.do_partition.clean_shutdown")
@patch("meersolar.pipeline.do_partition.drop_cache")
@patch("meersolar.pipeline.do_partition.init_logger")
@patch("meersolar.pipeline.do_partition.save_pid")
@patch("meersolar.pipeline.do_partition.os.path.exists", return_value=True)
@patch("meersolar.pipeline.do_partition.os.makedirs")
@patch("meersolar.pipeline.do_partition.os.path.abspath", side_effect=lambda x: f"/abs/{x}")
@patch("meersolar.pipeline.do_partition.partion_ms", return_value="/mock/outputms.ms")
@patch("meersolar.pipeline.do_partition.np.load", return_value=["mock_job", "mock_pass"])
@patch("meersolar.pipeline.do_partition.time.sleep")
def test_main(
    mock_sleep,
    mock_np_load,
    mock_partion_ms,
    mock_abspath,
    mock_makedirs,
    mock_exists,
    mock_save_pid,
    mock_init_logger,
    mock_drop_cache,
    mock_clean_shutdown,
    argv_args, 
    expected_return,
):
    with patch("sys.argv", argv_args):
        from meersolar.pipeline import do_partition
        result = do_partition.main()
        assert result == expected_return
    
    
    

    
    
