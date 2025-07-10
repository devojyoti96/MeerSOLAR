import pytest
from unittest.mock import patch, MagicMock
from itertools import cycle
from meersolar.pipeline.make_ds import *


@patch("meersolar.pipeline.make_ds.os.makedirs")
@patch("meersolar.pipeline.make_ds.get_cal_target_scans", return_value=([1], [], [], [], []))
@patch("meersolar.pipeline.make_ds.get_valid_scans", return_value=[1])
@patch("meersolar.pipeline.make_ds.msmetadata")
@patch("meersolar.pipeline.make_ds.casamstool")
@patch("meersolar.pipeline.make_ds.check_datacolumn_valid", return_value=True)
@patch("meersolar.pipeline.make_ds.get_dask_client")
@patch("meersolar.pipeline.make_ds.make_ds_file_per_scan", return_value="mockfile.npy")
@patch("meersolar.pipeline.make_ds.make_ds_plot", return_value="mockfile.png")
@patch("meersolar.pipeline.make_ds.glob.glob", return_value=["sci_mock.nc"])
@patch("meersolar.pipeline.make_ds.os.system")
def test_make_solar_DS(
    mock_system,
    mock_glob,
    mock_plot,
    mock_make_ds,
    mock_get_dask,
    mock_check_col,
    mock_mstool_class,
    mock_msmd_class,
    mock_valid_scans,
    mock_get_scans,
    mock_makedirs,
):
    # Mock msmd and mstool behaviors
    mock_msmd = MagicMock()
    mock_msmd.nchan.return_value = 10
    mock_msmd.nantennas.return_value = 64
    mock_msmd_class.return_value = mock_msmd

    mock_mstool = MagicMock()
    mock_mstool.nrow.return_value = 10000
    mock_mstool_class.return_value = mock_mstool

    # Mock Dask client return values
    mock_client = MagicMock()
    mock_cluster = MagicMock()
    mock_get_dask.return_value = (mock_client, mock_cluster, 1, 1, 1)

    from meersolar.pipeline.make_ds import make_solar_DS

    make_solar_DS(
        msname="mock.ms",
        workdir="/mock/workdir",
        ds_file_name="mockDS",
        target_scans=[],
        merge_scan=False,
        showgui=False,
    )

    # Assertions
    mock_makedirs.assert_called_once_with("/mock/workdir/dynamic_spectra", exist_ok=True)
    mock_get_scans.assert_called_once()
    mock_valid_scans.assert_called_once()
    mock_check_col.assert_called_once_with("mock.ms", datacolumn="CORRECTED_DATA")
    mock_make_ds.assert_called_once()
    mock_plot.assert_called_once()
    mock_system.assert_any_call("rm -rf /mock/workdir/dask-scratch-space /mock/workdir/tmp")


@patch("meersolar.pipeline.make_ds.drop_cache")
@patch("meersolar.pipeline.make_ds.os.system")
@patch("meersolar.pipeline.make_ds.os.path.samefile", return_value=False)
@patch("meersolar.pipeline.make_ds.glob.glob", return_value=["/outdir/dynamic_spectra/mock_DS_scan_1.png"])
@patch("meersolar.pipeline.make_ds.make_solar_DS")
def test_make_ds(
    mock_make_solar_DS,
    mock_glob,
    mock_samefile,
    mock_os_system,
    mock_drop_cache,
):
    mock_msname = "/data/mock.ms"
    mock_workdir = "/workdir"
    mock_outdir = "/outdir"

    result = make_ds(
        msname=mock_msname,
        workdir=mock_workdir,
        outdir=mock_outdir,
        extension="png",
        target_scans=[1],
        merge_scans=True,
        seperate_scans=True,
        cpu_frac=0.8,
        mem_frac=0.8,
    )

    # Assertions
    assert result == ["/outdir/dynamic_spectra/mock_DS_scan_1.png"]
    assert mock_make_solar_DS.call_count == 2
    mock_os_system.assert_called_once_with("mv /workdir/dynamic_spectra /outdir")

    expected_msname = mock_msname.rstrip("/")
    expected_workdir = mock_workdir.rstrip("/")
    mock_drop_cache.assert_any_call(expected_msname)
    mock_drop_cache.assert_any_call(expected_workdir)

@pytest.mark.parametrize(
    "argv_args, expected_return",
    [
        (
            [
                "make_ds",
                "/data/mock.ms",
                "--workdir", "/tmp/mockwork",
                "--outdir", "/tmp/mockout",
                "--extension", "png",
                "--target_scans", "1", "2",
            ],
            0,
        ),
        (
            [
                "make_ds",
                "/invalid/path.ms",
                "--workdir", "/tmp/mockwork",
                "--outdir", "/tmp/mockout",
            ],
            1,
        ),
    ]
)
@patch("meersolar.pipeline.make_ds.save_pid")
@patch("meersolar.pipeline.make_ds.drop_cache")
@patch("meersolar.pipeline.make_ds.time.sleep", return_value=None)
@patch("meersolar.pipeline.make_ds.os.makedirs")
@patch("meersolar.pipeline.make_ds.os.path.exists", side_effect=lambda p: False if "invalid" in p else True)
@patch("meersolar.pipeline.make_ds.make_ds", return_value=["/tmp/mockout/dynamic_spectra/mock_DS_scan_1.png"])
@patch("meersolar.pipeline.make_ds.clean_shutdown")
@patch("meersolar.pipeline.make_ds.np.load", return_value=["mockjob", "mockpass"])
@patch("meersolar.pipeline.make_ds.init_logger")
def test_main(
    mock_init_logger,
    mock_np_load,
    mock_clean_shutdown,
    mock_make_ds,
    mock_exists,
    mock_makedirs,
    mock_sleep,
    mock_drop_cache,
    mock_save_pid,
    argv_args,
    expected_return,
):
    os.makedirs("/tmp/mockdata/pids", exist_ok=True)
    with patch("sys.argv", argv_args):
        from meersolar.pipeline import make_ds
        result = make_ds.main()
        assert result == expected_return

