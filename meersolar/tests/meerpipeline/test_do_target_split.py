import pytest
from unittest.mock import patch, MagicMock
from meersolar.meerpipeline.do_target_split import *


def test_chanlist_to_str():
    result = chanlist_to_str([0, 1, 2, 10, 45])
    assert result == "0~2;10;45"


@patch("meersolar.meerpipeline.do_target_split.get_local_dask_cluster")
@patch(
    "meersolar.meerpipeline.do_target_split.get_cal_target_scans",
    return_value=([1, 2], [], [], [], []),
)
@patch("meersolar.meerpipeline.do_target_split.get_valid_scans", return_value=[1, 2])
@patch(
    "meersolar.meerpipeline.do_target_split.get_bad_chans",
    return_value="0:0~100;200~300",
)
@patch(
    "meersolar.meerpipeline.do_target_split.get_pol_names", return_value=["XX", "YY"]
)
@patch(
    "meersolar.meerpipeline.do_target_split.get_common_spw",
    return_value="0:0~100;200~300",
)
@patch(
    "meersolar.meerpipeline.do_target_split.get_timeranges_for_scan",
    return_value=["10s", "20s"],
)
@patch(
    "meersolar.meerpipeline.do_target_split.split_into_chunks", return_value=[[0, 1, 2]]
)
@patch("meersolar.meerpipeline.do_target_split.chanlist_to_str", return_value="0~2")
@patch(
    "meersolar.meerpipeline.do_target_split.single_mstransform", return_value="mock.ms"
)
@patch("meersolar.meerpipeline.do_target_split.drop_cache")
@patch("meersolar.meerpipeline.do_target_split.msmetadata")
@patch("meersolar.meerpipeline.do_target_split.os.chdir")
@patch("meersolar.meerpipeline.do_target_split.get_ms_scan_size", return_value=1.0)
def test_split_target_scans(
    mock_get_ms_scan_size,
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
    mock_get_dask_client,
):
    mock_dask_client = MagicMock()
    mock_dask_cluster = MagicMock()
    mock_get_dask_client.return_value = (
        mock_dask_client,
        mock_dask_cluster,
        "/mock/dask_dir",
    )
    mock_dask_client.compute.return_value = [MagicMock()]
    mock_dask_client.gather.return_value = ["mock.ms"]
    mock_msmd = MagicMock()
    mock_msmd.chanres.return_value = [0.1]
    mock_msmd.chanfreqs.return_value = [100.0, 200.0, 300.0]
    mock_msmd.nchan.return_value = 3
    mock_msmetadata.return_value = mock_msmd
    msg, result = split_target_scans(
        "mock.ms",
        mock_dask_client,
        workdir="/mock/workdir",
        timeres=1.0,
        freqres=1.0,
        datacolumn="DATA",
        scans=[],
    )
    mock_dask_client.compute.assert_called()
    mock_dask_client.gather.assert_called()
    assert msg == 0
    assert result == ["mock.ms"]


@pytest.mark.parametrize(
    "ms_exists, split_success, expected_msg",
    [
        (True, True, 0),  # Successful run
        (True, False, 1),  # Split fails internally but returns handled result
        (False, False, 1),  # Invalid MS path
    ],
)
@patch("meersolar.meerpipeline.do_target_split.split_target_scans")
@patch("meersolar.meerpipeline.do_target_split.save_pid")
@patch(
    "meersolar.meerpipeline.do_target_split.get_cachedir", return_value="/mock/cache"
)
@patch("os.makedirs")
@patch("os.path.exists")
@patch("os.getpid", return_value=1234)
@patch("meersolar.meerpipeline.do_target_split.drop_cache")
@patch("meersolar.meerpipeline.do_target_split.clean_shutdown")
@patch("time.sleep", return_value=None)
@patch("traceback.print_exc", return_value=None)
def test_main_split_target_scans(
    mock_traceback,
    mock_sleep,
    mock_shutdown,
    mock_drop_cache,
    mock_getpid,
    mock_exists,
    mock_makedirs,
    mock_cachedir,
    mock_save_pid,
    mock_split_target_scans,
    ms_exists,
    split_success,
    expected_msg,
):
    from meersolar.meerpipeline.do_target_split import main

    msname = "mock.ms"
    workdir = "/mock/work"

    def exists_side_effect(path):
        return path == msname if ms_exists else False

    mock_exists.side_effect = exists_side_effect
    mock_split_target_scans.return_value = (
        0 if split_success else 1,
        ["out1.ms", "out2.ms"],
    )
    dask_client = MagicMock()
    msg = main(
        msname=msname,
        workdir=workdir,
        datacolumn="data",
        spw="",
        scans="1,2",
        time_window=-1,
        time_interval=-1,
        quack_timestamps=-1,
        spectral_chunk=-1,
        n_spectral_chunk=-1,
        freqres=-1,
        timeres=-1,
        prefix="targets",
        merge_spws=False,
        cpu_frac=0.8,
        mem_frac=0.8,
        logfile=None,
        jobid=0,
        start_remote_log=False,
        dask_client=dask_client,
    )
    assert msg == expected_msg


@pytest.mark.parametrize(
    "argv, should_exit",
    [
        (["prog.py"], True),  # Missing args
        (
            [
                "prog.py",
                "mock.ms",
                "--workdir",
                "/mock/work",
                "--scans",
                "1,2",
                "--prefix",
                "targets",
            ],
            False,
        ),  # Normal CLI call
    ],
)
@patch("meersolar.meerpipeline.do_target_split.main", return_value=0)
@patch("meersolar.meerpipeline.do_target_split.sys.exit")
@patch("meersolar.meerpipeline.do_target_split.argparse.ArgumentParser.print_help")
def test_cli_split_target_scans(
    mock_print_help,
    mock_exit,
    mock_main,
    argv,
    should_exit,
):
    with patch("sys.argv", argv):
        from meersolar.meerpipeline import do_target_split

        result = do_target_split.cli()
        assert result == should_exit
