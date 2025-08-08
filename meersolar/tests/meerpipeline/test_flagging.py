import pytest
from unittest.mock import patch, MagicMock
from casatools import table
from meersolar.meerpipeline.flagging import *


def test_single_ms_flag(dummy_submsname):
    result = single_ms_flag(
        msname=f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms",
        badspw="0:0;1",
        bad_ants_str="1,2",
        datacolumn="data",
        use_tfcrop=True,
        use_rflag=True,
        flagdimension="freqtime",
        flag_autocorr=True,
        n_threads=-1,
        memory_limit=-1,
    )
    assert result == 0
    tb = table()
    tb.open(f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms", nomodify=False)
    flag = tb.getcol("FLAG")
    flag *= False
    tb.putcol("FLAG", flag)
    tb.flush()
    tb.close()
    os.system(f"rm -rf {dummy_submsname}/SUBMSS/test_subms.ms.0000.ms.flagversions")
    assert (
        os.path.exists(f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms.flagversions")
        == False
    )


def test_do_flagging(dummy_submsname):
    workdir = os.getcwd()
    dask_client = MagicMock()
    result = do_flagging(
        dummy_submsname,
        dask_client,
        workdir,
        datacolumn="data",
        flag_bad_ants=True,
        flag_bad_spw=True,
        use_tfcrop=True,
        use_rflag=True,
        flagdimension="freqtime",
        flag_autocorr=True,
        flag_backup=True,
        cpu_frac=0.8,
        mem_frac=0.8,
    )
    assert result == 0
    tb = table()
    tb.open(dummy_submsname, nomodify=False)
    flag = tb.getcol("FLAG")
    flag *= False
    tb.putcol("FLAG", flag)
    tb.flush()
    tb.close()
    os.system(f"rm -rf {dummy_submsname}.flagversions")
    os.system(f"rm -rf {workdir}/dask-scratch-space {workdir}/tmp")
    assert os.path.exists(f"{dummy_submsname}.flagversions") == False


@pytest.mark.parametrize(
    "ms_exists, flag_result, expected_msg",
    [
        (True, 0, 0),  # Success case
        (True, 1, 1),  # CASA flagging failed
        (False, None, 1),  # MS not found
    ],
)
@patch("meersolar.meerpipeline.flagging.do_flagging")
@patch("meersolar.meerpipeline.flagging.save_pid")
@patch("meersolar.meerpipeline.flagging.get_cachedir", return_value="/mock/cache")
@patch("os.makedirs")
@patch("os.path.exists")
@patch("os.getpid", return_value=9999)
@patch("meersolar.meerpipeline.flagging.drop_cache")
@patch("meersolar.meerpipeline.flagging.clean_shutdown")
@patch("time.sleep", return_value=None)
@patch("traceback.print_exc", return_value=None)
def test_main_flagging(
    mock_trace,
    mock_sleep,
    mock_shutdown,
    mock_drop_cache,
    mock_getpid,
    mock_exists,
    mock_makedirs,
    mock_cachedir,
    mock_save_pid,
    mock_do_flagging,
    ms_exists,
    flag_result,
    expected_msg,
):
    msname = "mock.ms"
    workdir = "/mock/work"

    def exists_side_effect(path):
        return path == msname if ms_exists else False

    mock_exists.side_effect = exists_side_effect
    mock_do_flagging.return_value = flag_result
    dask_client = MagicMock()
    msg = main(
        msname=msname,
        workdir=workdir,
        datacolumn="DATA",
        flag_bad_ants=True,
        flag_bad_spw=True,
        use_tfcrop=False,
        use_rflag=False,
        flag_autocorr=True,
        flagbackup=True,
        flagdimension="freqtime",
        cpu_frac=0.8,
        mem_frac=0.8,
        logfile=None,
        jobid=1,
        start_remote_log=False,
        dask_client=dask_client,
    )
    assert msg == expected_msg


@pytest.mark.parametrize(
    "argv_args, expect_main_called, expected_exit",
    [
        (["prog"], False, 1),  # No args: expect sys.exit(1)
        (
            ["prog", "mock.ms", "--workdir", "mockdir"],
            True,
            0,  # Valid: expect main() call and return value
        ),
    ],
)
@patch("meersolar.meerpipeline.flagging.main", return_value=0)
@patch("meersolar.meerpipeline.flagging.sys.exit")
@patch("meersolar.meerpipeline.flagging.argparse.ArgumentParser.print_help")
def test_cli_flagging(
    mock_print_help,
    mock_exit,
    mock_main,
    argv_args,
    expect_main_called,
    expected_exit,
):
    with patch("sys.argv", argv_args):
        from meersolar.meerpipeline import flagging

        result = flagging.cli()
        assert result == expected_exit
