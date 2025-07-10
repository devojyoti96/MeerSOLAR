import pytest
from unittest.mock import patch, MagicMock
from itertools import cycle
from meersolar.pipeline.do_imaging import *


@pytest.mark.parametrize("msname, expected_status", [
    ("mock.ms", 0),
])
@patch("meersolar.pipeline.do_imaging.rename_image", side_effect=lambda *args, **kwargs: f"/mock/renamed/{os.path.basename(args[0])}")
@patch("meersolar.pipeline.do_imaging.make_stokes_wsclean_imagecube", side_effect=lambda images, output, **kwargs: output)
@patch("meersolar.pipeline.do_imaging.glob.glob", side_effect=lambda pattern: [pattern.replace("*", "I")])
@patch("meersolar.pipeline.do_imaging.run_wsclean", return_value=0)
@patch("meersolar.pipeline.do_imaging.get_multiscale_bias", return_value=0.7)
@patch("meersolar.pipeline.do_imaging.calc_multiscale_scales", return_value=[0, 4, 16])
@patch("meersolar.pipeline.do_imaging.calc_sun_dia", return_value=30)
@patch("meersolar.pipeline.do_imaging.calc_npix_in_psf", return_value=3)
@patch("meersolar.pipeline.do_imaging.create_circular_mask", return_value="solar_mask.fits")
@patch("meersolar.pipeline.do_imaging.psutil.virtual_memory")
@patch("meersolar.pipeline.do_imaging.psutil.cpu_count", return_value=4)
@patch("meersolar.pipeline.do_imaging.init_logger")
@patch("meersolar.pipeline.do_imaging.create_logger", return_value=(MagicMock(), "mock.log"))
@patch("meersolar.pipeline.do_imaging.timestamp_to_mjdsec", side_effect=lambda t: 0)
@patch("meersolar.pipeline.do_imaging.get_band_name", return_value="L")
@patch("meersolar.pipeline.do_imaging.msmetadata")
@patch("meersolar.pipeline.do_imaging.clean_shutdown")
@patch("meersolar.pipeline.do_imaging.time.sleep", return_value=None)
@patch("meersolar.pipeline.do_imaging.drop_cache")
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
    expected_status
):
    # Setup mocks
    mem_mock = MagicMock()
    mem_mock.total = 16 * 1024 ** 3  # 16 GB
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
        dry_run=False
    )

    assert code == expected_status
    assert isinstance(output, list)
    assert all(isinstance(sublist, list) for sublist in output)
    
    
@patch("meersolar.pipeline.do_imaging.drop_cache")
@patch("meersolar.pipeline.do_imaging.os.system")
@patch("meersolar.pipeline.do_imaging.os.makedirs")
@patch("meersolar.pipeline.do_imaging.calc_sun_dia", return_value=32.0)
@patch("meersolar.pipeline.do_imaging.calc_field_of_view", return_value=1500.0)
@patch("meersolar.pipeline.do_imaging.calc_cellsize", return_value=5.0)
@patch("meersolar.pipeline.do_imaging.calc_npix_in_psf", return_value=3)
@patch("meersolar.pipeline.do_imaging.resource.setrlimit")
@patch("meersolar.pipeline.do_imaging.resource.getrlimit", return_value=(1024, 4096))
@patch("meersolar.pipeline.do_imaging.check_udocker_container", return_value=True)
@patch("meersolar.pipeline.do_imaging.check_datacolumn_valid", return_value=True)
@patch("meersolar.pipeline.do_imaging.init_logger")
@patch("meersolar.pipeline.do_imaging.create_logger")
@patch("meersolar.pipeline.do_imaging.msmetadata")
@patch("meersolar.pipeline.do_imaging.compute")
@patch("meersolar.pipeline.do_imaging.get_dask_client")
@patch("meersolar.pipeline.do_imaging.run_limited_memory_task", return_value=4.0)
@patch("meersolar.pipeline.do_imaging.np.load", return_value=["mockjob", "mockpass"])
@patch("meersolar.pipeline.do_imaging.perform_imaging")
def test_run_all_imaging(
    mock_perform_imaging,
    mock_np_load,
    mock_run_limited,
    mock_get_dask_client,
    mock_compute,
    mock_msmetadata,
    mock_create_logger,
    mock_init_logger,
    mock_check_datacolumn_valid,
    mock_check_udocker,
    mock_getrlimit,
    mock_setrlimit,
    mock_calc_npix_in_psf,
    mock_calc_cellsize,
    mock_calc_fov,
    mock_calc_sun_dia,
    mock_makedirs,
    mock_system,
    mock_drop_cache,
):
    os.makedirs("/tmp/mockdata/pids", exist_ok=True)

    # Mock msmetadata behavior
    mock_msmd = MagicMock()
    mock_msmd.open.return_value = None
    mock_msmd.close.return_value = None
    mock_msmd.timesforspws.return_value = np.linspace(0, 10, 5)
    mock_msmd.chanfreqs.return_value = np.linspace(100, 200, 64)
    mock_msmd.meanfreq.return_value = 1400.0
    mock_msmetadata.return_value = mock_msmd

    # Mock logger
    mock_logger = MagicMock()
    mock_create_logger.return_value = (mock_logger, "/tmp/fake.log")

    # Mock Dask client and cluster
    mock_client = MagicMock()
    mock_cluster = MagicMock()
    mock_get_dask_client.return_value = (mock_client, mock_cluster, 2, 2, 4.0)

    # Mock imaging result
    mock_compute.return_value = [(0, [["img1.fits"], ["mod1.fits"], ["res1.fits"]])]

    # Call the function
    result = run_all_imaging(
        mslist=["mock.ms"],
        workdir="/tmp/work",
        outdir="/tmp/out"
    )

    assert result == 0
    mock_logger.info.assert_any_call("Imaging successfully done for: 1 measurement sets.")
    
    
@pytest.mark.parametrize(
    "argv_args, expected_return",
    [
        (
            [
                "run_imaging",
                "/data/mock1.ms,/data/mock2.ms",
                "--workdir", "/tmp/testwork",
                "--outdir", "/tmp/testcal",
            ],
            0,
        )
    ]
)    
@patch("meersolar.pipeline.do_imaging.clean_shutdown")
@patch("meersolar.pipeline.do_imaging.drop_cache")
@patch("meersolar.pipeline.do_imaging.run_all_imaging", return_value=0)
@patch("meersolar.pipeline.do_imaging.init_logger", return_value=MagicMock())
@patch("meersolar.pipeline.do_imaging.create_logger", return_value=(MagicMock(), "/tmp/mainlog.log"))
@patch("meersolar.pipeline.do_imaging.os.makedirs")
@patch("meersolar.pipeline.do_imaging.os.path.exists", return_value=True)
@patch("meersolar.pipeline.do_imaging.save_pid")
@patch("meersolar.pipeline.do_imaging.np.load", return_value=["jobX", "passX"])
@patch("meersolar.pipeline.do_imaging.time.sleep")
def test_main(
    mock_sleep,
    mock_npload,
    mock_save_pid,
    mock_exists,
    mock_makedirs,
    mock_create_logger,
    mock_init_logger,
    mock_run_all_imaging,
    mock_drop_cache,
    mock_clean_shutdown,
    argv_args, 
    expected_return,
):
    with patch("sys.argv", argv_args):
        from meersolar.pipeline import do_imaging
        result = do_imaging.main()
        assert result == expected_return
        
    
    
