import pytest
from unittest.mock import patch, MagicMock
from meersolar.pipeline.do_sidereal_cor import *

@patch("meersolar.pipeline.do_sidereal_cor.drop_cache")
@patch("meersolar.pipeline.do_sidereal_cor.time.sleep")
@patch("meersolar.pipeline.do_sidereal_cor.os.path.exists", return_value=True)
@patch("meersolar.pipeline.do_sidereal_cor.compute", side_effect=lambda *args: [0] * len(args))
@patch("meersolar.pipeline.do_sidereal_cor.get_dask_client")
@patch("meersolar.pipeline.do_sidereal_cor.run_limited_memory_task", return_value=1.0)
@patch("meersolar.pipeline.do_sidereal_cor.delayed")
def test_cor_sidereal_motion(
    mock_delayed,
    mock_mem_task,
    mock_get_client,
    mock_compute,
    mock_exists,
    mock_sleep,
    mock_drop_cache,
):
    mock_delayed.side_effect = lambda fn: fn
    mock_client = MagicMock()
    mock_cluster = MagicMock()
    mock_get_client.return_value = (mock_client, mock_cluster, 2, 1, 1.0)
    mslist = ["test1.ms", "test2.ms"]
    workdir = "/mock/workdir"
    status, corrected = cor_sidereal_motion(mslist, workdir)

    assert status == 0
    assert corrected == mslist
    mock_client.close.assert_called_once()
    mock_cluster.close.assert_called_once()
    
@pytest.mark.parametrize(
    "argv_args, expected_return",
    [
        (
            [
                "run_solar_siderealcor",
                "test1.ms,test2.ms",
                "--workdir", "/mock/workdir",
                "--logfile", "/mock/logfile.log",
                "--jobid", "42"
            ],
            0,
        )
    ]
)
@patch("meersolar.pipeline.do_sidereal_cor.sys")
@patch("meersolar.pipeline.do_sidereal_cor.clean_shutdown")
@patch("meersolar.pipeline.do_sidereal_cor.drop_cache")
@patch("meersolar.pipeline.do_sidereal_cor.time.sleep")
@patch("meersolar.pipeline.do_sidereal_cor.init_logger")
@patch("meersolar.pipeline.do_sidereal_cor.np.load", return_value=("job", "pwd"))
@patch("meersolar.pipeline.do_sidereal_cor.os.makedirs")
@patch("meersolar.pipeline.do_sidereal_cor.os.path.exists", return_value=True)
@patch("meersolar.pipeline.do_sidereal_cor.save_pid")
@patch("meersolar.pipeline.do_sidereal_cor.cor_sidereal_motion", return_value=(0, ["test1.ms", "test2.ms"]))
def test_main(
    mock_cor,
    mock_save_pid,
    mock_exists,
    mock_makedirs,
    mock_npload,
    mock_init_logger,
    mock_sleep,
    mock_drop,
    mock_shutdown,
    mock_sys,
    argv_args,
    expected_return,
):
    with patch("sys.argv", argv_args):
        from meersolar.pipeline import do_sidereal_cor
        result = do_sidereal_cor.main()
        assert result == expected_return
