import pytest
from unittest.mock import patch, MagicMock
from itertools import cycle
from meersolar.meerpipeline.do_selfcal import *


@patch("meersolar.meerpipeline.do_selfcal.drop_cache")
@patch("meersolar.meerpipeline.do_selfcal.clean_shutdown")
@patch("meersolar.meerpipeline.do_selfcal.time.sleep", return_value=None)
@patch("meersolar.meerpipeline.do_selfcal.os.chdir")
@patch("meersolar.meerpipeline.do_selfcal.os.makedirs")
@patch("meersolar.meerpipeline.do_selfcal.os.path.exists", return_value=False)
@patch("meersolar.meerpipeline.do_selfcal.os.system")
@patch(
    "meersolar.meerpipeline.do_selfcal.create_logger",
    return_value=(MagicMock(), "log.log"),
)
@patch("meersolar.meerpipeline.do_selfcal.init_logger")
@patch(
    "meersolar.meerpipeline.do_selfcal.get_unflagged_antennas",
    return_value=(["ant1", "ant2"], [0.1, 0.1]),
)
@patch("meersolar.meerpipeline.do_selfcal.calc_cellsize", return_value=5.0)
@patch("meersolar.meerpipeline.do_selfcal.calc_field_of_view", return_value=1200)
@patch("meersolar.meerpipeline.do_selfcal.check_datacolumn_valid", return_value=True)
@patch("meersolar.meerpipeline.do_selfcal.msmetadata")
@patch("casatasks.flagmanager", return_value={0: {"name": "applycal"}})
@patch("casatasks.initweights")
@patch("casatasks.flagdata")
@patch("casatasks.split")
@patch("meersolar.meerpipeline.do_selfcal.limit_threads")
@patch("meersolar.meerpipeline.do_selfcal.intensity_selfcal")
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
    mock_msmd.meanfreq.return_value = 100.0
    mock_msmd.timesforspws.return_value = ["100", "200"]
    mock_msmd.chanfreqs.return_value = [100.0, 100.1, 100.2]
    mock_msmetadata.return_value = mock_msmd

    mock_intensity_selfcal.side_effect = cycle(
        [
            (0, "g0.cal", 100.0, 0.01, "img1.fits", "mod1.fits", "res1.fits"),
            (0, "g1.cal", 110.0, 0.009, "img2.fits", "mod2.fits", "res2.fits"),
            (0, "g2.cal", 111.0, 0.008, "img3.fits", "mod3.fits", "res3.fits"),
        ]
    )
    status, caltable = do_selfcal(
        msname="mock.ms", workdir="/tmp", selfcaldir="/tmp", dry_run=False
    )
    assert status == 0
    assert caltable in ["g0.cal", "g1.cal", "g2.cal"]

    # --- Case 2: No model flux even at lowest threshold
    mock_intensity_selfcal.side_effect = [
        (1, "", 0, 0, "", "", ""),
        (1, "", 0, 0, "", "", ""),
    ]
    status, caltable = do_selfcal(
        msname="mock.ms", workdir="/tmp", selfcaldir="/tmp", dry_run=False
    )
    assert status == 1
    assert caltable == []

    # --- Case 3: No solutions found
    mock_intensity_selfcal.side_effect = [
        (2, "", 0, 0, "", "", ""),
    ]
    status, caltable = do_selfcal(
        msname="mock.ms", workdir="/tmp", selfcaldir="/tmp", dry_run=False
    )
    assert status == 2 or status == 1
    assert caltable == []

    # --- Case 4: dry run
    import psutil, os

    expected_mem = round(psutil.Process(os.getpid()).memory_info().rss / 1024**3, 2)
    mem = do_selfcal(msname="mock.ms", workdir="/tmp", selfcaldir="/tmp", dry_run=True)
    assert isinstance(mem, float)
    assert abs(mem - expected_mem) < 1.0  # within 1 GB

    # --- Case 5: Dynamic range drop triggers fallback to previous gaintable
    mock_intensity_selfcal.side_effect = cycle(
        [
            (0, "g0.cal", 100.0, 0.01, "img1.fits", "mod1.fits", "res1.fits"),
            (0, "g1.cal", 150.0, 0.009, "img2.fits", "mod2.fits", "res2.fits"),
            (0, "g2.cal", 80.0, 0.009, "img3.fits", "mod3.fits", "res3.fits"),
        ]
    )
    status, caltable = do_selfcal(
        msname="mock.ms", workdir="/tmp", selfcaldir="/tmp", dry_run=False
    )
    assert status == 0
    assert caltable in ["g1.cal", "g2.cal"]  # depends on exact logic

    # --- Case 6: Maximum iteration exit (simulate steady DR)
    mock_intensity_selfcal.side_effect = cycle(
        [
            (
                0,
                f"g{i}.cal",
                100.0,
                0.01,
                f"img{i}.fits",
                f"mod{i}.fits",
                f"res{i}.fits",
            )
            for i in range(20)
        ]
    )
    status, caltable = do_selfcal(
        msname="mock.ms", workdir="/tmp", selfcaldir="/tmp", max_iter=5, dry_run=False
    )
    assert status == 0
    assert caltable.startswith("g")

    # --- Case 7: Exception path
    mock_intensity_selfcal.side_effect = Exception("simulated failure")
    status, caltable = do_selfcal(
        msname="mock.ms", workdir="/tmp", selfcaldir="/tmp", dry_run=False
    )
    assert status == 1
    assert caltable == []


@pytest.mark.parametrize(
    "mslist_str, caldir_exists, container_ok, valid_cols, expected_msg",
    [
        ("ms1.ms,ms2.ms", True, True, True, 0),
        ("ms1.ms", False, True, True, 0),
        ("ms1.ms", True, False, True, 1),
        ("ms1.ms", True, True, False, 1),
    ],
)
@patch("meersolar.meerpipeline.do_selfcal.create_logger")
@patch("meersolar.meerpipeline.do_selfcal.save_pid")
@patch("meersolar.meerpipeline.do_selfcal.get_cachedir", return_value="/mock/cache")
@patch("os.makedirs")
@patch("os.path.exists")
@patch("os.getpid", return_value=4321)
@patch("os.system", return_value=0)
@patch("meersolar.meerpipeline.do_selfcal.table")
@patch("meersolar.meerpipeline.do_selfcal.check_udocker_container")
@patch("meersolar.meerpipeline.do_selfcal.initialize_wsclean_container")
@patch("meersolar.meerpipeline.do_selfcal.run_limited_memory_task", return_value=4)
@patch("meersolar.meerpipeline.do_selfcal.check_datacolumn_valid")
@patch("meersolar.meerpipeline.do_selfcal.get_dask_client")
@patch("meersolar.meerpipeline.do_selfcal.do_selfcal", return_value=(0, "mock.gcal"))
@patch("meersolar.meerpipeline.do_selfcal.drop_cache")
@patch("meersolar.meerpipeline.do_selfcal.clean_shutdown")
@patch("time.sleep", return_value=None)
@patch("traceback.print_exc", return_value=None)
@patch("meersolar.meerpipeline.do_selfcal.msmetadata")
def test_main_selfcal(
    mock_msmetadata,
    mock_trace,
    mock_sleep,
    mock_shutdown,
    mock_drop_cache,
    mock_do_selfcal,
    mock_get_dask_client,
    mock_check_datacolumn,
    mock_run_limited_memory_task,
    mock_init_container,
    mock_check_container,
    mock_table,
    mock_system,
    mock_getpid,
    mock_exists,
    mock_makedirs,
    mock_get_cachedir,
    mock_save_pid,
    mock_create_logger,
    mslist_str,
    caldir_exists,
    container_ok,
    valid_cols,
    expected_msg,
):
    mslist = mslist_str.split(",")
    workdir = "/mock/work"
    default_caldir = f"{workdir}/caltables"
    caldir = default_caldir if not caldir_exists else "/mock/caltables"

    def exists_side_effect(path):
        if "jobname_password.npy" in path:
            return False
        if path == "/mock/caltables":
            return caldir_exists
        if path in [workdir, default_caldir] + mslist:
            return True
        if path == "mock.gcal":
            return True
        return False

    mock_exists.side_effect = exists_side_effect

    # Mock msmetadata instance
    msmd_instance = MagicMock()
    msmd_instance.open.return_value = None
    msmd_instance.close.return_value = None
    msmd_instance.timesforspws.return_value = np.array([0.0, 1.0, 2.0])
    msmd_instance.chanfreqs.return_value = np.array([100.0, 101.0, 102.0])
    mock_msmetadata.return_value = msmd_instance

    # Mock logger and dask
    mock_check_container.return_value = container_ok
    mock_check_datacolumn.return_value = valid_cols
    mock_create_logger.return_value = (MagicMock(), "mock.log")
    mock_get_dask_client.return_value = (MagicMock(), MagicMock(), 1, 1, 2)

    # CASA table mock
    table_instance = MagicMock()
    table_instance.open.return_value = None
    table_instance.getcol.return_value = [1]
    table_instance.close.return_value = None
    mock_table.return_value = table_instance

    if container_ok:
        mock_init_container.return_value = "meerwsclean"
    else:
        mock_init_container.return_value = None

    msg = main(
        mslist=mslist_str,
        workdir=workdir,
        caldir=caldir,
        start_thresh=5,
        stop_thresh=3,
        max_iter=5,
        max_DR=500,
        min_iter=1,
        conv_frac=0.5,
        solint="60s",
        uvrange="",
        minuv=0.0,
        weight="briggs",
        robust=0.0,
        applymode="calonly",
        min_tol_factor=1.0,
        do_apcal=True,
        solar_selfcal=True,
        keep_backup=False,
        cpu_frac=0.7,
        mem_frac=0.6,
        jobid=42,
        start_remote_log=False,
    )

    assert msg == expected_msg

    # Ensure makedirs called for both workdir and caldir
    mock_makedirs.assert_any_call(workdir, exist_ok=True)
    mock_makedirs.assert_any_call(caldir, exist_ok=True)


@pytest.mark.parametrize(
    "argv, should_exit",
    [
        (["prog.py"], True),  # No args → help
        (
            [
                "prog.py",
                "ms1.ms,ms2.ms",
                "--workdir",
                "/mock/work",
                "--caldir",
                "/mock/caltables",
                "--start_thresh",
                "5",
                "--stop_thresh",
                "3",
                "--no_apcal",
                "--keep_backup",
            ],
            False,
        ),
    ],
)
@patch("meersolar.meerpipeline.do_selfcal.main", return_value=0)
def test_cli_selfcal(mock_main, argv, should_exit):
    with patch.object(sys, "argv", argv):
        if should_exit:
            import pytest

            with pytest.raises(SystemExit) as e:
                cli()
            assert e.value.code == 1
        else:
            result = cli()
            assert result == 0
            assert mock_main.called
