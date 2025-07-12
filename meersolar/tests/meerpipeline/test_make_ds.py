import pytest
from unittest.mock import patch, MagicMock
from meersolar.meerpipeline.make_ds import *


@patch("meersolar.meerpipeline.make_ds.os.makedirs")
@patch("meersolar.meerpipeline.make_ds.get_cal_target_scans", return_value=([1], [], [], [], []))
@patch("meersolar.meerpipeline.make_ds.get_valid_scans", return_value=[1])
@patch("meersolar.meerpipeline.make_ds.msmetadata")
@patch("meersolar.meerpipeline.make_ds.casamstool")
@patch("meersolar.meerpipeline.make_ds.check_datacolumn_valid", return_value=True)
@patch("meersolar.meerpipeline.make_ds.get_dask_client")
@patch("meersolar.meerpipeline.make_ds.make_ds_file_per_scan", return_value="mockfile.npy")
@patch("meersolar.meerpipeline.make_ds.make_ds_plot", return_value="mockfile.png")
@patch("meersolar.meerpipeline.make_ds.glob.glob", return_value=["sci_mock.nc"])
@patch("meersolar.meerpipeline.make_ds.os.system")
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

    from meersolar.meerpipeline.make_ds import make_solar_DS

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


@patch("meersolar.meerpipeline.make_ds.drop_cache")
@patch("meersolar.meerpipeline.make_ds.os.system")
@patch("meersolar.meerpipeline.make_ds.os.path.samefile", return_value=False)
@patch("meersolar.meerpipeline.make_ds.glob.glob", return_value=["/outdir/dynamic_spectra/mock_DS_scan_1.png"])
@patch("meersolar.meerpipeline.make_ds.make_solar_DS")
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

    result = make_dsfiles(
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
    "ms_exists, ds_success, expected_code",
    [
        (True, True, 0),   # Valid MS, make_ds runs fine
        (True, False, 0),  # Valid MS, make_ds returns empty/None
        (False, False, 1), # Invalid MS path
    ],
)
@patch("meersolar.meerpipeline.make_ds.make_dsfiles")
@patch("meersolar.meerpipeline.make_ds.save_pid")
@patch("meersolar.meerpipeline.make_ds.get_cachedir", return_value="/mock/cache")
@patch("os.makedirs")
@patch("os.path.exists")
@patch("os.getpid", return_value=9999)
@patch("meersolar.meerpipeline.make_ds.clean_shutdown")
@patch("time.sleep", return_value=None)
@patch("traceback.print_exc", return_value=None)
def test_main(
    mock_trace,
    mock_sleep,
    mock_shutdown,
    mock_getpid,
    mock_exists,
    mock_makedirs,
    mock_cachedir,
    mock_save_pid,
    mock_make_ds,
    ms_exists,
    ds_success,
    expected_code,
):
    msname = "mock.ms"
    workdir = "/mock/work"
    outdir = "/mock/out"

    def exists_side_effect(path):
        return path == msname if ms_exists else False

    mock_exists.side_effect = exists_side_effect
    mock_make_ds.return_value = ["ds1.png", "ds2.png"] if ds_success else []

    result = main(
        msname=msname,
        workdir=workdir,
        outdir=outdir,
        extension="png",
        target_scans=["1", "3"],
        merge=True,
        seperate=True,
        cpu_frac=0.8,
        mem_frac=0.8,
        logfile=None,
        jobid="12",
        start_remote_log=False,
    )
    assert result == expected_code


@pytest.mark.parametrize(
    "argv, should_exit",
    [
        (["prog.py"], True),  # No args
        ([
            "prog.py", "mock.ms",
            "--workdir", "/mock/work",
            "--outdir", "/mock/out",
            "--extension", "pdf",
            "--no_merge", "--no_seperate"
        ], False),
    ],
)
@patch("meersolar.meerpipeline.make_ds.make_dsfiles", return_value=["merged.png"])
@patch("meersolar.meerpipeline.make_ds.save_pid")
@patch("meersolar.meerpipeline.make_ds.get_cachedir", return_value="/mock/cache")
@patch("os.makedirs")
@patch("os.path.exists", return_value=True)
@patch("os.getpid", return_value=5678)
@patch("meersolar.meerpipeline.make_ds.clean_shutdown")
@patch("time.sleep", return_value=None)
def test_cli(
    mock_sleep,
    mock_shutdown,
    mock_getpid,
    mock_exists,
    mock_makedirs,
    mock_cachedir,
    mock_save_pid,
    mock_make_ds,
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


