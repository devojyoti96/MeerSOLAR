import pytest
from unittest.mock import patch, MagicMock
from meersolar.meerpipeline.meer_pbcor import *


@pytest.mark.parametrize(
    "header_dict, expected_output, expect_warning",
    [
        ({"CTYPE3": "FREQ", "CRVAL3": 1.4e9}, 1.4e9, False),
        ({"CTYPE4": "FREQ", "CRVAL4": 1.5e9}, 1.5e9, False),
        ({"CTYPE3": "STOKES"}, None, True),
    ],
)
@patch("meersolar.utils.image_utils.fits.getheader")
def test_get_fits_freq(
    mock_getheader, header_dict, expected_output, expect_warning, capsys
):
    mock_hdr = MagicMock()
    mock_hdr.keys.return_value = header_dict.keys()
    mock_hdr.__getitem__.side_effect = header_dict.__getitem__
    mock_getheader.return_value = mock_hdr
    result = get_fits_freq("mock.fits")

    assert result == expected_output

    if expect_warning:
        captured = capsys.readouterr()
        assert "No frequency axis" in captured.out


@pytest.mark.parametrize(
    "apply_parang, returncode, expected_flag",
    [
        (True, 0, []),
        (False, 0, ["--no_apply_parang"]),
        (False, 1, ["--no_apply_parang"]),  # Simulate failure
    ],
)
def test_run_pbcor(apply_parang, returncode, expected_flag):
    imagename = "test.fits"
    pbdir = "/fake/pb"
    pbcor_dir = "/fake/pbcor"
    jobid = 42
    ncpu = 4
    fake_output = "Mocked primary beam correction output."

    with patch("meersolar.meerpipeline.meer_pbcor.subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = returncode
        mock_proc.stdout = fake_output
        mock_run.return_value = mock_proc

        result = run_pbcor(
            imagename,
            pbdir,
            pbcor_dir,
            apply_parang=apply_parang,
            jobid=jobid,
            ncpu=ncpu,
            verbose=True,
        )

        # Verify correct result code
        assert result == returncode

        # Ensure correct command construction
        called_args = mock_run.call_args[0][0]
        assert imagename in called_args
        assert "--pbdir" in called_args
        assert pbdir in called_args
        assert "--pbcor_dir" in called_args
        assert pbcor_dir in called_args
        assert "--ncpu" in called_args
        assert str(ncpu) in called_args
        assert "--jobid" in called_args
        assert str(jobid) in called_args

        for flag in expected_flag:
            assert flag in called_args


@pytest.mark.parametrize(
    "make_TB, make_plots, apply_parang",
    [(True, True, True), (False, True, True), (True, False, False)],
)
@patch("meersolar.meerpipeline.meer_pbcor.drop_cache")
@patch("meersolar.meerpipeline.meer_pbcor.os.system")
@patch("meersolar.meerpipeline.meer_pbcor.generate_tb_map")
@patch(
    "meersolar.meerpipeline.meer_pbcor.plot_in_hpc", return_value=["out.png", "out.pdf"]
)
@patch("meersolar.meerpipeline.meer_pbcor.save_in_hpc")
@patch("meersolar.meerpipeline.meer_pbcor.get_local_dask_cluster")
@patch("meersolar.meerpipeline.meer_pbcor.delayed")
@patch("meersolar.meerpipeline.meer_pbcor.get_fits_freq")
@patch("meersolar.meerpipeline.meer_pbcor.os.path.getsize")
@patch("meersolar.meerpipeline.meer_pbcor.glob.glob")
@patch("meersolar.meerpipeline.meer_pbcor.os.makedirs")
def test_pbcor_all_images(
    mock_makedirs,
    mock_glob,
    mock_getsize,
    mock_get_fits_freq,
    mock_delayed,
    mock_get_dask_client,
    mock_save_in_hpc,
    mock_plot_in_hpc,
    mock_generate_tb,
    mock_os_system,
    mock_drop_cache,
    make_TB,
    make_plots,
    apply_parang,
):
    # Create fake FITS image paths
    images = [f"/mock/imagedir/image_{i}.fits" for i in range(4)]
    # Group into 2 freq bins (simulate both first_set and remaining_set)
    mock_glob.side_effect = lambda path: (
        images if "pbcor" not in path and "tb_images" not in path else images
    )
    mock_get_fits_freq.side_effect = [100, 200, 100, 200]
    mock_getsize.return_value = 1024**3  # 1GB each
    mock_dask_client = MagicMock()
    dask_cluster = MagicMock()
    mock_get_dask_client.return_value = (
        mock_dask_client,
        dask_cluster,
        2,
        4,
        8.0,
        "/mock/dask_dir",
    )
    mock_delayed.side_effect = lambda f: f  # simulate delayed identity
    mock_dask_client.compute.return_value = [MagicMock(), MagicMock()]
    mock_dask_client.gather.return_value = [0, 0]
    result = pbcor_all_images(
        "/mock/imagedir",
        mock_dask_client,
        make_TB=make_TB,
        make_plots=make_plots,
        apply_parang=apply_parang,
        jobid=99,
        cpu_frac=0.5,
        mem_frac=0.5,
    )
    mock_dask_client.compute.assert_called()
    # Check expected cleanup was called
    assert result == 0
    assert mock_makedirs.call_count >= 1
    assert mock_save_in_hpc.called
    if make_TB:
        assert mock_generate_tb.called
    if make_plots:
        assert mock_plot_in_hpc.called


@pytest.mark.parametrize(
    "imagedir_exists, pbcor_success, expect_logger, expected_return",
    [
        (True, True, False, 0),  # Normal run without logger
        (True, False, False, 1),  # pbcor_all_images fails
        (False, False, False, 1),  # imagedir does not exist
    ],
)
@patch("meersolar.meerpipeline.meer_pbcor.pbcor_all_images")
@patch("meersolar.meerpipeline.meer_pbcor.save_pid")
@patch("meersolar.meerpipeline.meer_pbcor.get_cachedir", return_value="/mock/cache")
@patch("os.makedirs")
@patch("os.path.exists")
@patch("os.getpid", return_value=9999)
@patch("meersolar.meerpipeline.meer_pbcor.drop_cache")
@patch("meersolar.meerpipeline.meer_pbcor.clean_shutdown")
@patch("time.sleep", return_value=None)
@patch("traceback.print_exc", return_value=None)
def test_main_function(
    mock_traceback,
    mock_sleep,
    mock_shutdown,
    mock_drop,
    mock_getpid,
    mock_exists,
    mock_makedirs,
    mock_cachedir,
    mock_save_pid,
    mock_pbcor,
    imagedir_exists,
    pbcor_success,
    expect_logger,
    expected_return,
):
    imagedir = "mock/images"
    workdir = "/mock/work"

    def exists_side_effect(path):
        if path == imagedir:
            return imagedir_exists
        elif "jobname_password.npy" in path:
            return expect_logger
        return True

    mock_exists.side_effect = exists_side_effect
    mock_pbcor.return_value = 0 if pbcor_success else 1
    dask_client = MagicMock()
    msg = main(
        imagedir=imagedir,
        workdir=workdir,
        make_TB=True,
        make_plots=True,
        apply_parang=True,
        cpu_frac=0.7,
        mem_frac=0.6,
        logfile=None,
        jobid=1,
        dask_client=dask_client,
    )
    assert msg == expected_return


@pytest.mark.parametrize(
    "argv, expect_exit",
    [
        (["prog.py"], True),  # No arguments: help and exit
        (["prog.py", "mockdir", "--no_make_TB"], False),  # Normal run
    ],
)
@patch("meersolar.meerpipeline.meer_pbcor.main", return_value=0)
@patch("meersolar.meerpipeline.meer_pbcor.sys.exit")
@patch("meersolar.meerpipeline.meer_pbcor.argparse.ArgumentParser.print_help")
def test_cli_function(
    mock_print_help,
    mock_exit,
    mock_main,
    argv,
    expect_exit,
):
    with patch("sys.argv", argv):
        from meersolar.meerpipeline import meer_pbcor

        result = meer_pbcor.cli()
        assert result == expect_exit
