import pytest
from unittest.mock import patch, MagicMock
from casatools import table
from meersolar.pipeline.flagging import *


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
    "argv_args, expected_return",
    [
        (
            [
                "do_flagging.py",  # Fake script name
                "mock.ms",
                "--workdir", "/mock/workdir",
                "--datacolumn", "DATA",
                "--no_flag_bad_ants",
                "--use_tfcrop",
                "--flagdimension", "freqtime",
                "--cpu_frac", "0.5",
                "--mem_frac", "0.5",
                "--jobid", "1",
            ],
            0,  # Assuming success
        )
    ]
)
@patch("meersolar.pipeline.flagging.save_pid")
@patch("meersolar.pipeline.flagging.do_flagging", return_value=0)
@patch("meersolar.pipeline.flagging.os.path.exists", return_value=True)
@patch("meersolar.pipeline.flagging.os.makedirs")
@patch("meersolar.pipeline.flagging.np.load", return_value=("jobname", "password"))
@patch("meersolar.pipeline.flagging.init_logger")
@patch("meersolar.pipeline.flagging.drop_cache")
@patch("meersolar.pipeline.flagging.clean_shutdown")
@patch("meersolar.pipeline.flagging.time.sleep")
def test_main(
    mock_sleep,
    mock_shutdown,
    mock_drop_cache,
    mock_logger,
    mock_npload,
    mock_makedirs,
    mock_exists,
    mock_doflagging,
    mock_save_pid,
    argv_args,
    expected_return,
):
    with patch("sys.argv", argv_args):
        from meersolar.pipeline import flagging as do_flagging
        result = do_flagging.main()
        assert result == expected_return
        
    
    
