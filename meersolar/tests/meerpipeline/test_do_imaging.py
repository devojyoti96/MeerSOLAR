import pytest
from unittest.mock import patch, MagicMock
from itertools import cycle
from meersolar.meerpipeline.do_imaging import *


@pytest.mark.parametrize(
    "msname, expected_status",
    [
        ("mock.ms", 0),
    ],
)
@patch(
    "meersolar.meerpipeline.do_imaging.rename_meersolar_image",
    side_effect=lambda *args, **kwargs: f"/mock/renamed/{os.path.basename(args[0])}",
)
@patch(
    "meersolar.meerpipeline.do_imaging.make_stokes_wsclean_imagecube",
    side_effect=lambda images, output, **kwargs: output,
)
@patch(
    "meersolar.meerpipeline.do_imaging.glob.glob",
    side_effect=lambda pattern: [pattern.replace("*", "I")],
)
@patch("meersolar.meerpipeline.do_imaging.run_wsclean", return_value=0)
@patch("meersolar.meerpipeline.do_imaging.get_multiscale_bias", return_value=0.7)
@patch(
    "meersolar.meerpipeline.do_imaging.calc_multiscale_scales", return_value=[0, 4, 16]
)
@patch("meersolar.meerpipeline.do_imaging.calc_sun_dia", return_value=30)
@patch("meersolar.meerpipeline.do_imaging.calc_npix_in_psf", return_value=3)
@patch(
    "meersolar.meerpipeline.do_imaging.create_circular_mask",
    return_value="solar_mask.fits",
)
@patch("meersolar.meerpipeline.do_imaging.psutil.virtual_memory")
@patch("meersolar.meerpipeline.do_imaging.psutil.cpu_count", return_value=4)
@patch("meersolar.meerpipeline.do_imaging.init_logger")
@patch(
    "meersolar.meerpipeline.do_imaging.create_logger",
    return_value=(MagicMock(), "mock.log"),
)
@patch("meersolar.meerpipeline.do_imaging.timestamp_to_mjdsec", side_effect=lambda t: 0)
@patch("meersolar.meerpipeline.do_imaging.get_band_name", return_value="L")
@patch("meersolar.meerpipeline.do_imaging.msmetadata")
@patch("meersolar.meerpipeline.do_imaging.clean_shutdown")
@patch("meersolar.meerpipeline.do_imaging.time.sleep", return_value=None)
@patch("meersolar.meerpipeline.do_imaging.drop_cache")
def test_perform_imaging(
    mock_drop_cache,
    mock_sleep,
    mock_shutdown,
    mock_msmd,
    mock_get_band,
    mock_ts_to_mjd,
    mock_create_logger,
    mock_init_logger,
    mock_cpu_count,
    mock_virt_mem,
    mock_create_mask,
    mock_psf,
    mock_sun_dia,
    mock_multiscale,
    mock_bias,
    mock_run_wsclean,
    mock_glob,
    mock_stokes_cube,
    mock_rename,
    msname,
    expected_status,
):
    # Setup mocks
    mem_mock = MagicMock()
    mem_mock.total = 16 * 1024**3  # 16 GB
    mock_virt_mem.return_value = mem_mock

    msmd_inst = MagicMock()
    msmd_inst.meanfreq.return_value = 1400
    msmd_inst.chanfreqs.return_value = np.linspace(1000, 1800, 128)
    msmd_inst.timesforspws.return_value = np.linspace(0, 60, 10)
    msmd_inst.ncorrforpol.return_value = [4]
    mock_msmd.return_value = msmd_inst

    # Call function
    code, output = perform_imaging(
        msname=msname,
        workdir="/tmp/work",
        imagedir="/tmp/images",
        freqrange="1200~1500",
        timerange="2021-01-01T00:00:00~2021-01-01T01:00:00",
        nchan=1,
        ntime=1,
        pol="I",
    )

    assert code == expected_status
    assert isinstance(output, dict)
    assert all(isinstance(v, list) for v in output.values())


@pytest.mark.parametrize(
    "container_exists, corrupted_ms, freqres, timeres, expected",
    [
        (False, False, 1.0, 2.0, 0),  # Normal case
        (True, True, -1, -1, 1),  # Corrupted MS should return failure
    ],
)
@patch("meersolar.meerpipeline.do_imaging.drop_cache")
@patch("meersolar.meerpipeline.do_imaging.os.system")
@patch("meersolar.meerpipeline.do_imaging.os.makedirs")
@patch("meersolar.meerpipeline.do_imaging.calc_sun_dia", return_value=32.0)
@patch("meersolar.meerpipeline.do_imaging.calc_field_of_view", return_value=1500.0)
@patch("meersolar.meerpipeline.do_imaging.calc_cellsize", return_value=5.0)
@patch("meersolar.meerpipeline.do_imaging.calc_npix_in_psf", return_value=3)
@patch("meersolar.meerpipeline.do_imaging.resource.setrlimit")
@patch(
    "meersolar.meerpipeline.do_imaging.resource.getrlimit", return_value=(1024, 4096)
)
@patch(
    "meersolar.meerpipeline.do_imaging.initialize_wsclean_container",
    return_value="meerwsclean",
)
@patch("meersolar.meerpipeline.do_imaging.check_udocker_container")
@patch("meersolar.meerpipeline.do_imaging.check_datacolumn_valid")
@patch("meersolar.meerpipeline.do_imaging.init_logger")
@patch("meersolar.meerpipeline.do_imaging.create_logger")
@patch("meersolar.meerpipeline.do_imaging.msmetadata")
@patch("meersolar.meerpipeline.do_imaging.get_local_dask_cluster")
@patch("meersolar.meerpipeline.do_imaging.np.load", return_value=["job", "pass"])
@patch("meersolar.meerpipeline.do_imaging.perform_imaging")
def test_run_all_imaging(
    mock_perform_imaging,
    mock_npload,
    mock_get_dask,
    mock_msmd,
    mock_create_logger,
    mock_init_logger,
    mock_check_col,
    mock_check_udocker,
    mock_init_container,
    mock_getrlimit,
    mock_setrlimit,
    mock_npix,
    mock_cellsize,
    mock_fov,
    mock_sundia,
    mock_makedirs,
    mock_system,
    mock_drop,
    container_exists,
    corrupted_ms,
    freqres,
    timeres,
    expected,
):
    # Control branch
    mock_check_udocker.return_value = container_exists
    mock_check_col.return_value = not corrupted_ms

    # MSMD behavior
    msmd_inst = MagicMock()
    msmd_inst.open.return_value = None
    msmd_inst.close.return_value = None
    msmd_inst.timesforspws.return_value = np.linspace(0, 10, 5)
    msmd_inst.chanfreqs.return_value = np.linspace(100, 200, 64)
    msmd_inst.meanfreq.return_value = 1400.0
    mock_msmd.return_value = msmd_inst

    # Logger mock
    logger = MagicMock()
    mock_create_logger.return_value = (logger, "/tmp/fake.log")

    # Dask client
    client = MagicMock()
    cluster = MagicMock()
    mock_get_dask.return_value = (client, cluster, "/mock/dask_dir")

    # Imaging result
    client.compute.return_value = [MagicMock()]
    client.gather.return_value = [
        (0, {"image": ["img.fits"], "model": ["mod.fits"], "residual": ["res.fits"]})
    ]

    mslist = ["mock.ms"]
    workdir = "/tmp/mockwork"
    outdir = "/tmp/mockout"

    result = run_all_imaging(
        mslist,
        client,
        workdir=workdir,
        outdir=outdir,
        freqres=freqres,
        timeres=timeres,
        pol="IQUV",
    )
    if not corrupted_ms:
        client.compute.assert_called()

    assert result == expected

    if container_exists:
        mock_init_container.assert_not_called()
    else:
        mock_init_container.assert_called_once()


@pytest.mark.parametrize(
    "mslist_str, workdir_exists, container_ok, compute_success, expected_msg",
    [
        ("ms1.ms,ms2.ms", True, True, True, 0),
        ("ms1.ms", False, True, True, 0),
        ("ms1.ms", True, False, True, 0),
        ("ms1.ms", True, True, False, 1),
    ],
)
@patch("meersolar.meerpipeline.do_imaging.create_logger")
@patch("meersolar.meerpipeline.do_imaging.save_pid")
@patch("meersolar.meerpipeline.do_imaging.get_cachedir", return_value="/mock/cache")
@patch("os.makedirs")
@patch("os.path.exists")
@patch("os.getpid", return_value=1234)
@patch("meersolar.meerpipeline.do_imaging.drop_cache")
@patch("meersolar.meerpipeline.do_imaging.clean_shutdown")
@patch("meersolar.meerpipeline.do_imaging.np.load", return_value=("job", "pass"))
@patch("meersolar.meerpipeline.do_imaging.time.sleep", return_value=None)
@patch("meersolar.meerpipeline.do_imaging.traceback.print_exc", return_value=None)
@patch("meersolar.meerpipeline.do_imaging.run_all_imaging")
def test_main_do_imaging(
    mock_run_all_imaging,
    mock_trace,
    mock_sleep,
    mock_np_load,
    mock_shutdown,
    mock_drop_cache,
    mock_getpid,
    mock_exists,
    mock_makedirs,
    mock_get_cachedir,
    mock_save_pid,
    mock_create_logger,
    mslist_str,
    workdir_exists,
    container_ok,
    compute_success,
    expected_msg,
):
    mslist = mslist_str.split(",")
    workdir = "/mock/work/imaging"
    outdir = "/mock/output"

    def exists_side_effect(path):
        if "jobname_password.npy" in path:
            return False
        if path in ["/mock/work/imaging", "/mock/work/imaging/logs"]:
            return True
        if path == "/mock/output":
            return False  # Force creation
        if path in mslist:
            return True
        return False

    mock_exists.side_effect = exists_side_effect

    mock_create_logger.return_value = (MagicMock(), "/mock/logfile")
    if compute_success:
        mock_run_all_imaging.return_value = 0
    else:
        mock_run_all_imaging.side_effect = Exception("Mock failure")
    dask_client = MagicMock()
    msg = main(
        mslist=mslist_str,
        workdir=workdir,
        outdir=outdir,
        start_remote_log=False,
        jobid=42,
        dask_client=dask_client,
    )

    assert msg == expected_msg


@pytest.mark.parametrize(
    "argv, should_exit",
    [
        (["prog.py"], True),  # No args → help triggers sys.exit(1)
        (
            [
                "prog.py",
                "ms1.ms,ms2.ms",
                "--workdir",
                "/mock/work",
                "--outdir",
                "/mock/output",
                "--cpu_frac",
                "0.5",
                "--mem_frac",
                "0.6",
                "--no_multiscale",
                "--no_solar_mask",
                "--no_make_overlay",
                "--no_make_plots",
                "--no_saveres",
                "--no_savemodel",
            ],
            False,
        ),
    ],
)
@patch("meersolar.meerpipeline.do_imaging.main", return_value=0)
@patch("meersolar.meerpipeline.do_imaging.sys.exit")
@patch("meersolar.meerpipeline.do_imaging.argparse.ArgumentParser.print_help")
def test_cli_do_imaging(mock_print_help, mock_exit, mock_main, argv, should_exit):
    with patch("sys.argv", argv):
        from meersolar.meerpipeline import do_imaging

        result = do_imaging.cli()
        assert result == should_exit
