import pytest
from unittest.mock import patch, MagicMock
from casatools import table
from meersolar.meerpipeline.flagging import *


def test_single_ms_flag(dummy_submsname):
    result=single_ms_flag(
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
        dry_run=False,
    )
    assert result==0
    tb=table()
    tb.open(f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms",nomodify=False)
    flag=tb.getcol("FLAG")
    flag*=False
    tb.putcol("FLAG",flag)
    tb.flush()
    tb.close()
    os.system(f"rm -rf {dummy_submsname}/SUBMSS/test_subms.ms.0000.ms.flagversions")
    assert os.path.exists(f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms.flagversions")==False
    
def test_do_flagging(dummy_submsname):
    result = do_flagging(
    dummy_submsname,
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
    assert result==0
    tb=table()
    tb.open(dummy_submsname,nomodify=False)
    flag=tb.getcol("FLAG")
    flag*=False
    tb.putcol("FLAG",flag)
    tb.flush()
    tb.close()
    os.system(f"rm -rf {dummy_submsname}.flagversions")
    assert os.path.exists(f"{dummy_submsname}.flagversions")==False
    
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
    )
    assert msg == expected_msg


@pytest.mark.parametrize(
    "argv, should_exit",
    [
        (["prog.py"], True),  # No args → help and exit
        ([
            "prog.py", "mock.ms",
            "--workdir", "/mock/work",
            "--no_flag_bad_ants",
            "--no_flag_bad_spw",
            "--use_tfcrop",
            "--use_rflag",
            "--no_flag_autocorr",
            "--no_flagbackup",
            "--flagdimension", "time",
        ], False),  # Valid call
    ],
)
@patch("meersolar.meerpipeline.flagging.do_flagging", return_value=0)
@patch("meersolar.meerpipeline.flagging.save_pid")
@patch("meersolar.meerpipeline.flagging.get_cachedir", return_value="/mock/cache")
@patch("os.makedirs")
@patch("os.path.exists", return_value=True)
@patch("os.getpid", return_value=1234)
@patch("meersolar.meerpipeline.flagging.drop_cache")
@patch("meersolar.meerpipeline.flagging.clean_shutdown")
@patch("time.sleep", return_value=None)
def test_cli_flagging(
    mock_sleep,
    mock_shutdown,
    mock_drop_cache,
    mock_getpid,
    mock_exists,
    mock_makedirs,
    mock_cachedir,
    mock_save_pid,
    mock_do_flagging,
    argv,
    should_exit,
):
    with patch.object(sys, "argv", argv):
        if should_exit:
            with pytest.raises(SystemExit) as e:
                cli()
            assert e.value.code == 1
        else:
            result = cli()
            assert result == 0
        
    
    
