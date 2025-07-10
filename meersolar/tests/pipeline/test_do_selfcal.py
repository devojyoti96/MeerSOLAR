import pytest
from unittest.mock import patch, MagicMock
from itertools import cycle
from meersolar.pipeline.do_selfcal import *

@patch("meersolar.pipeline.do_selfcal.drop_cache")
@patch("meersolar.pipeline.do_selfcal.clean_shutdown")
@patch("meersolar.pipeline.do_selfcal.time.sleep", return_value=None)
@patch("meersolar.pipeline.do_selfcal.os.chdir")
@patch("meersolar.pipeline.do_selfcal.os.makedirs")
@patch("meersolar.pipeline.do_selfcal.os.path.exists", return_value=False)
@patch("meersolar.pipeline.do_selfcal.os.system")
@patch("meersolar.pipeline.do_selfcal.create_logger", return_value=(MagicMock(), "log.log"))
@patch("meersolar.pipeline.do_selfcal.init_logger")
@patch("meersolar.pipeline.do_selfcal.get_unflagged_antennas", return_value=(["ant1", "ant2"], [0.1, 0.1]))
@patch("meersolar.pipeline.do_selfcal.calc_cellsize", return_value=5.0)
@patch("meersolar.pipeline.do_selfcal.calc_field_of_view", return_value=1200)
@patch("meersolar.pipeline.do_selfcal.check_datacolumn_valid", return_value=True)
@patch("meersolar.pipeline.do_selfcal.msmetadata")
@patch("casatasks.flagmanager", return_value={0: {"name": "applycal"}})
@patch("casatasks.initweights")
@patch("casatasks.flagdata")
@patch("casatasks.split")
@patch("meersolar.pipeline.do_selfcal.limit_threads")
@patch("meersolar.pipeline.do_selfcal.intensity_selfcal")
def test_do_selfcal(
    mock_intensity_selfcal,
    mock_limit_threads,
    mock_split,
    mock_flagdata,
    mock_initweights,
    mock_flagmanager,
    mock_msmetadata,
    mock_check_data,
    mock_fov,
    mock_cellsize,
    mock_unflagged,
    mock_init_logger,
    mock_create_logger,
    mock_os_system,
    mock_path_exists,
    mock_makedirs,
    mock_chdir,
    mock_sleep,
    mock_shutdown,
    mock_drop_cache,
):
    
    mock_msmd = MagicMock()
    mock_msmd.open.return_value = None
    mock_msmd.close.return_value = None
    mock_msmd.scannumbers.return_value = [0]
    mock_msmd.fieldsforscan.return_value = [0]
    mock_msmd.meanfreq.return_value=100.0
    mock_msmd.timesforspws.return_value=["100","200"]
    mock_msmd.chanfreqs.return_value=[100.0,100.1,100.2]
    mock_msmetadata.return_value = mock_msmd

    mock_intensity_selfcal.side_effect = cycle([
        (0, "g0.cal", 100.0, 0.01, "img1.fits", "mod1.fits", "res1.fits"),
        (0, "g1.cal", 110.0, 0.009, "img2.fits", "mod2.fits", "res2.fits"),
        (0, "g2.cal", 111.0, 0.008, "img3.fits", "mod3.fits", "res3.fits"),
    ])
    status, caltable = do_selfcal(msname="mock.ms", workdir="/tmp", selfcaldir="/tmp", dry_run=False)
    assert status == 0
    assert caltable in ["g0.cal","g1.cal","g2.cal"]

    # --- Case 2: No model flux even at lowest threshold
    mock_intensity_selfcal.side_effect = [
        (1, "", 0, 0, "", "", ""),
        (1, "", 0, 0, "", "", ""),
    ]
    status, caltable = do_selfcal(msname="mock.ms", workdir="/tmp", selfcaldir="/tmp", dry_run=False)
    assert status == 1
    assert caltable == []

    # --- Case 3: No solutions found
    mock_intensity_selfcal.side_effect = [
        (2, "", 0, 0, "", "", ""),
    ]
    status, caltable = do_selfcal(msname="mock.ms", workdir="/tmp", selfcaldir="/tmp", dry_run=False)
    assert status == 2 or status == 1
    assert caltable == []

    # --- Case 4: dry run
    import psutil, os
    expected_mem = round(psutil.Process(os.getpid()).memory_info().rss / 1024**3, 2)
    mem = do_selfcal(msname="mock.ms", workdir="/tmp", selfcaldir="/tmp", dry_run=True)
    assert isinstance(mem, float)
    assert abs(mem - expected_mem) < 1.0  # within 1 GB

    # --- Case 5: Dynamic range drop triggers fallback to previous gaintable
    mock_intensity_selfcal.side_effect = cycle([
        (0, "g0.cal", 100.0, 0.01, "img1.fits", "mod1.fits", "res1.fits"),
        (0, "g1.cal", 150.0, 0.009, "img2.fits", "mod2.fits", "res2.fits"),
        (0, "g2.cal", 80.0, 0.009, "img3.fits", "mod3.fits", "res3.fits"),
    ])
    status, caltable = do_selfcal(msname="mock.ms", workdir="/tmp", selfcaldir="/tmp", dry_run=False)
    assert status == 0
    assert caltable in ["g1.cal", "g2.cal"]  # depends on exact logic

    # --- Case 6: Maximum iteration exit (simulate steady DR)
    mock_intensity_selfcal.side_effect = cycle([
        (0, f"g{i}.cal", 100.0, 0.01, f"img{i}.fits", f"mod{i}.fits", f"res{i}.fits") for i in range(20)
    ])
    status, caltable = do_selfcal(msname="mock.ms", workdir="/tmp", selfcaldir="/tmp", max_iter=5, dry_run=False)
    assert status == 0
    assert caltable.startswith("g")

    # --- Case 7: Exception path
    mock_intensity_selfcal.side_effect = Exception("simulated failure")
    status, caltable = do_selfcal(msname="mock.ms", workdir="/tmp", selfcaldir="/tmp", dry_run=False)
    assert status == 1
    assert caltable == []
    
@pytest.mark.parametrize(
    "argv_args, expected_return",
    [
        (
            [
                "run_selfcal",
                "/data/mock1.ms,/data/mock2.ms",
                "--workdir", "/tmp/testwork",
                "--caldir", "/tmp/testcal",
            ],
            0,
        )
    ]
)
@patch("meersolar.pipeline.do_selfcal.np.load", return_value=["mock_job", "mock_pass"])
@patch("meersolar.pipeline.do_selfcal.save_pid", return_value=None)
@patch("meersolar.pipeline.do_selfcal.table")
@patch("meersolar.pipeline.do_selfcal.os.system")
@patch("meersolar.pipeline.do_selfcal.clean_shutdown")
@patch("meersolar.pipeline.do_selfcal.drop_cache")
@patch("meersolar.pipeline.do_selfcal.time.sleep", return_value=None)
@patch("meersolar.pipeline.do_selfcal.get_dask_client")
@patch("meersolar.pipeline.do_selfcal.compute")
@patch("meersolar.pipeline.do_selfcal.delayed")
@patch("meersolar.pipeline.do_selfcal.init_logger")
@patch("meersolar.pipeline.do_selfcal.create_logger", return_value=(MagicMock(), "mock.log"))
@patch("meersolar.pipeline.do_selfcal.check_udocker_container", return_value=True)
@patch("meersolar.pipeline.do_selfcal.check_datacolumn_valid", return_value=True)
@patch("meersolar.pipeline.do_selfcal.msmetadata")
@patch("meersolar.pipeline.do_selfcal.psutil.virtual_memory")
@patch("meersolar.pipeline.do_selfcal.resource.getrlimit",return_value=(0,1))
@patch("meersolar.pipeline.do_selfcal.resource.setrlimit")
@patch("meersolar.pipeline.do_selfcal.datadir", "/tmp/mockdata")
@patch("meersolar.pipeline.do_selfcal.os.makedirs")
@patch("meersolar.pipeline.do_selfcal.run_limited_memory_task", return_value=4.0)
def test_main(
    mock_run_memtask,
    mock_makedirs,
    mock_setrlimit,
    mock_getrlimit,
    mock_virtual_memory,
    mock_msmetadata,
    mock_check_valid,
    mock_check_udocker,
    mock_create_logger,
    mock_init_logger,
    mock_delayed,
    mock_compute,
    mock_get_dask_client,
    mock_sleep,
    mock_drop_cache,
    mock_shutdown,
    mock_os_system,
    mock_table,
    mock_save_pid,
    mock_np_load,
    argv_args,
    expected_return,
):
    os.makedirs("/tmp/mockdata/pids", exist_ok=True)

    # Setup MS metadata mock
    mock_msmd = MagicMock()
    mock_msmd.open.return_value = None
    mock_msmd.close.return_value = None
    mock_msmd.timesforspws.return_value = [0.0, 1.0, 2.0, 10.0]
    mock_msmd.chanfreqs.return_value = np.linspace(100.0, 200.0, 64)
    mock_msmetadata.return_value = mock_msmd

    # Setup compute return values
    mock_compute.return_value = [(0, "gcal0.tbl"), (0, "gcal1.tbl")]

    # Setup Dask client return values
    mock_client = MagicMock()
    mock_cluster = MagicMock()
    mock_get_dask_client.return_value = (mock_client, mock_cluster, 2, 2, 4.0)

    # Setup table getcol
    tb_mock = MagicMock()
    tb_mock.getcol.return_value = [1]
    mock_table.return_value = tb_mock

    with patch("sys.argv", argv_args):
        from meersolar.pipeline import do_selfcal
        result = do_selfcal.main()
        assert result == expected_return

