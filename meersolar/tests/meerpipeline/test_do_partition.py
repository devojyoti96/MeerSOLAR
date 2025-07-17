import pytest
from unittest.mock import patch, MagicMock
from meersolar.meerpipeline.do_partition import *


@patch("meersolar.meerpipeline.do_partition.msmetadata")
@patch("meersolar.meerpipeline.do_partition.get_valid_scans", return_value=[1, 2])
@patch("meersolar.meerpipeline.do_partition.get_pol_names", return_value=["XX", "YY"])
@patch("meersolar.meerpipeline.do_partition.get_ms_scan_size", return_value=1.0)
@patch("meersolar.meerpipeline.do_partition.run_limited_memory_task", return_value=1.0)
@patch("meersolar.meerpipeline.do_partition.get_dask_client")
@patch("meersolar.meerpipeline.do_partition.single_mstransform")
@patch("meersolar.meerpipeline.do_partition.suppress_casa_output")
@patch("meersolar.meerpipeline.do_partition.os")
@patch("meersolar.meerpipeline.do_partition.time.sleep")
@patch("meersolar.meerpipeline.do_partition.drop_cache")
@patch("casatasks.virtualconcat")
def test_partion_ms(
    mock_virtualconcat,
    mock_drop_cache,
    mock_sleep,
    mock_os,
    mock_suppress,
    mock_single_transform,
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
    mock_dask_client = MagicMock()
    mock_dask_cluster = MagicMock()
    mock_get_dask_client.return_value = (mock_dask_client, mock_dask_cluster, 2, 2, 1.0)
    mock_dask_client.compute.return_value = ["mock.ms", "mock.ms2"]
    result = partion_ms(
        msname="mock.ms",
        outputms="final.ms",
        workdir="/mock/tmp",
        fields="",
        scans="",
        width=1,
        timebin="5s",
        datacolumn="DATA",
        cpu_frac=0.5,
        mem_frac=0.5,
    )
    mock_get_dask_client.assert_called_once()
    mock_dask_client.compute.assert_called()
    mock_virtualconcat.assert_called_once()
    assert result == "final.ms"


@pytest.mark.parametrize(
    "ms_exists, partition_success, expected_msg",
    [
        (True, True, 0),  # Successful partitioning
        (True, False, 1),  # partion_ms returns None or file missing
        (False, False, 1),  # Input MS does not exist
    ],
)
@patch("meersolar.meerpipeline.do_partition.partion_ms")
@patch("meersolar.meerpipeline.do_partition.get_cachedir", return_value="/mock/cache")
@patch("meersolar.meerpipeline.do_partition.save_pid")
@patch("meersolar.meerpipeline.do_partition.os.makedirs")
@patch("meersolar.meerpipeline.do_partition.os.path.exists")
@patch("meersolar.meerpipeline.do_partition.os.getpid", return_value=12345)
@patch(
    "meersolar.meerpipeline.do_partition.os.path.abspath",
    side_effect=lambda x: f"/abs/{x}",
)
@patch("meersolar.meerpipeline.do_partition.os.path.dirname", return_value="/mock")
@patch("meersolar.meerpipeline.do_partition.drop_cache")
@patch("meersolar.meerpipeline.do_partition.clean_shutdown")
@patch("time.sleep", return_value=None)
@patch("traceback.print_exc", return_value=None)
def test_main_partition(
    mock_trace,
    mock_sleep,
    mock_shutdown,
    mock_drop_cache,
    mock_dirname,
    mock_abspath,
    mock_getpid,
    mock_exists,
    mock_makedirs,
    mock_save_pid,
    mock_cachedir,
    mock_partion_ms,
    ms_exists,
    partition_success,
    expected_msg,
):
    msname = "input.ms"
    outputms = "multi.ms"

    def exists_side_effect(path):
        if path == msname:
            return ms_exists
        if path == outputms:
            return partition_success
        if "jobname_password.npy" in path:
            return False
        return True

    mock_exists.side_effect = exists_side_effect
    mock_partion_ms.return_value = outputms if partition_success else None

    msg = main(
        msname=msname,
        outputms=outputms,
        workdir="",
        fields="0",
        scans="1",
        width=2,
        timebin="10s",
        datacolumn="data",
        cpu_frac=0.5,
        mem_frac=0.5,
        logfile=None,
        jobid="42",
        start_remote_log=False,
    )

    assert msg == expected_msg
