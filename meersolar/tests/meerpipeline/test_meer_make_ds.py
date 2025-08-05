import pytest
from unittest.mock import patch, MagicMock
from meersolar.meerpipeline.meer_make_ds import *


@patch(
    "meersolar.meerpipeline.meer_make_ds.glob.glob",
    return_value=["/mock/workdir/dynamic_spectra/sci_mock.nc"],
)
@patch("meersolar.meerpipeline.meer_make_ds.os.system")
@patch("meersolar.meerpipeline.meer_make_ds.make_ds_plot", return_value="mockfile.png")
@patch(
    "meersolar.meerpipeline.meer_make_ds.make_ds_file_per_scan",
    return_value="mockfile.npy",
)
@patch("meersolar.meerpipeline.meer_make_ds.delayed", side_effect=lambda f: f)
@patch("meersolar.meerpipeline.meer_make_ds.get_local_dask_cluster")
@patch("meersolar.meerpipeline.meer_make_ds.check_datacolumn_valid", return_value=True)
@patch("meersolar.meerpipeline.meer_make_ds.casamstool")
@patch("meersolar.meerpipeline.meer_make_ds.msmetadata")
@patch("meersolar.meerpipeline.meer_make_ds.get_valid_scans", return_value=[1])
@patch(
    "meersolar.meerpipeline.meer_make_ds.get_cal_target_scans",
    return_value=([1], [], [], [], []),
)
@patch("meersolar.meerpipeline.meer_make_ds.os.makedirs")
@patch("meersolar.meerpipeline.meer_make_ds.get_ms_scan_size", return_value=1.0)
def test_make_solar_DS(
    mock_get_scan_size,
    mock_makedirs,
    mock_get_scans,
    mock_valid_scans,
    mock_msmd_class,
    mock_mstool_class,
    mock_check_col,
    mock_get_dask,
    mock_delayed,
    mock_make_ds_file,
    mock_plot,
    mock_os_system,
    mock_glob,
    tmp_path,
):
    # === Setup fake MS and work directory ===
    msname = tmp_path / "mock.ms"
    msname.mkdir()
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    # === Mock msmetadata ===
    mock_msmd = MagicMock()
    mock_msmd.nchan.return_value = 10
    mock_msmd.nantennas.return_value = 64
    mock_msmd.open.return_value = None
    mock_msmd.close.return_value = None
    mock_msmd_class.return_value = mock_msmd

    # === Mock casamstool ===
    mock_mstool = MagicMock()
    mock_mstool.nrow.return_value = 10000
    mock_mstool.open.return_value = None
    mock_mstool.close.return_value = None
    mock_mstool.select.return_value = None
    mock_mstool_class.return_value = mock_mstool

    # === Dask mocks ===
    mock_client = MagicMock()
    mock_cluster = MagicMock()
    mock_client.compute.return_value = ["f1"]
    mock_client.gather.return_value = ["r1"]
    mock_get_dask.return_value = (mock_client, mock_cluster, "/mock/workdir/")

    # === Run ===
    dask_client = MagicMock()
    result = make_solar_DS(
        str(msname),
        dask_client,
        workdir=str(workdir),
        ds_file_name="mockDS",
        target_scans=[],
        showgui=False,
    )

    # === Assertions ===
    mock_make_ds_file.assert_called_once()
    mock_plot.assert_called_once()
    assert result is None or result == ["mockfile.png"]


@patch("meersolar.meerpipeline.meer_make_ds.drop_cache")
@patch("meersolar.meerpipeline.meer_make_ds.time.sleep", return_value=None)
@patch("meersolar.meerpipeline.meer_make_ds.os.system")
@patch("meersolar.meerpipeline.meer_make_ds.os.makedirs")
@patch(
    "meersolar.meerpipeline.meer_make_ds.os.path.samefile", side_effect=[False, False]
)
@patch("meersolar.meerpipeline.meer_make_ds.glob.glob")
@patch("meersolar.meerpipeline.meer_make_ds.make_solar_DS")
def test_make_dsfiles(
    mock_make_solar_DS,
    mock_glob,
    mock_samefile,
    mock_makedirs,
    mock_system,
    mock_sleep,
    mock_drop_cache,
    tmp_path,
):
    # ========== Setup ==========
    msname = tmp_path / "mock.ms"
    workdir = tmp_path / "work"
    outdir = tmp_path / "out"
    msname.mkdir()
    workdir.mkdir()
    outdir.mkdir()

    expected_files = [str(outdir / "dynamic_spectra/mock_DS_scan_1.png")]
    mock_glob.return_value = expected_files
    dask_client = MagicMock()
    result = make_dsfiles(
        str(msname),
        dask_client,
        workdir=str(workdir),
        outdir=str(outdir),
        extension="png",
        target_scans=[1],
        seperate_scans=True,
        cpu_frac=0.5,
        mem_frac=0.5,
    )

    assert result == expected_files
    assert mock_make_solar_DS.call_count == 1
    mock_makedirs.assert_called_once_with(f"{outdir}/dynamic_spectra", exist_ok=True)
    mock_system.assert_any_call(
        f"mv {workdir}/dynamic_spectra/* {outdir}/dynamic_spectra/"
    )
    mock_system.assert_any_call(f"rm -rf {workdir}/dynamic_spectra")
    mock_drop_cache.assert_any_call(str(msname))
    mock_drop_cache.assert_any_call(str(workdir))
    mock_make_solar_DS.reset_mock()
    result2 = make_dsfiles(
        str(msname),
        dask_client,
        workdir=str(workdir),
        outdir=str(outdir),
        seperate_scans=False,
    )
    assert result2 is None
    mock_make_solar_DS.assert_not_called()


@pytest.mark.parametrize(
    "ms_exists, ds_success, expected_code",
    [
        (True, True, 0),  # Valid MS, make_ds runs fine
        (True, False, 0),  # Valid MS, make_ds returns empty/None
        (False, False, 1),  # Invalid MS path
    ],
)
@patch("meersolar.meerpipeline.meer_make_ds.make_dsfiles")
@patch("meersolar.meerpipeline.meer_make_ds.save_pid")
@patch("meersolar.meerpipeline.meer_make_ds.get_cachedir", return_value="/mock/cache")
@patch("os.makedirs")
@patch("os.path.exists")
@patch("os.getpid", return_value=9999)
@patch("meersolar.meerpipeline.meer_make_ds.clean_shutdown")
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
    dask_client = MagicMock()
    result = main(
        msname=msname,
        workdir=workdir,
        outdir=outdir,
        extension="png",
        target_scans=["1", "3"],
        seperate=True,
        cpu_frac=0.8,
        mem_frac=0.8,
        logfile=None,
        jobid="12",
        start_remote_log=False,
        dask_client=dask_client,
    )
    assert result == expected_code


@pytest.mark.parametrize(
    "argv, should_exit",
    [
        (["prog.py"], True),  # No args
        (
            [
                "prog.py",
                "mock.ms",
                "--workdir",
                "/mock/work",
                "--outdir",
                "/mock/out",
                "--extension",
                "pdf",
                "--no_seperate",
            ],
            False,
        ),
    ],
)
@patch("meersolar.meerpipeline.meer_make_ds.main", return_value=0)
@patch("meersolar.meerpipeline.meer_make_ds.sys.exit")
@patch("meersolar.meerpipeline.meer_make_ds.argparse.ArgumentParser.print_help")
def test_cli(
    mock_print_help,
    mock_exit,
    mock_main,
    argv,
    should_exit,
):
    with patch("sys.argv", argv):
        from meersolar.meerpipeline import meer_make_ds

        result = meer_make_ds.cli()
        assert result == should_exit
