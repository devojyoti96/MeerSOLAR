import pytest
import sunpy
import os
import numpy as np
from astropy.io import fits
from unittest.mock import patch, MagicMock
from meersolar.utils.meer_ploting_utils import *


@patch("meersolar.utils.meer_ploting_utils.os.system")
@patch("meersolar.utils.meer_ploting_utils.drop_cache")
@patch("meersolar.utils.meer_ploting_utils.run_shadems")
@patch("meersolar.utils.meer_ploting_utils.glob.glob")
@patch("meersolar.utils.meer_ploting_utils.Image.open")
@patch("meersolar.utils.meer_ploting_utils.psutil")
@patch("meersolar.utils.meer_ploting_utils.get_ms_scan_size")
@patch("meersolar.utils.meer_ploting_utils.casamstool")
@patch("meersolar.utils.meer_ploting_utils.msmetadata")
def test_plot_ms_diagnostics(
    mock_msmetadata,
    mock_casamstool,
    mock_get_ms_scan_size,
    mock_psutil,
    mock_Image_open,
    mock_glob,
    mock_run_shadems,
    mock_drop_cache,
    mock_os_system,
    tmp_path,
):
    mock_psutil.cpu_count.return_value = 4
    vm = MagicMock()
    vm.available = 8 * 1024**3  
    mock_psutil.virtual_memory.return_value = vm
    mstool = MagicMock()
    mstool.nrow.return_value = 1000
    mock_casamstool.return_value = mstool
    msmd = MagicMock()
    msmd.ncorrforpol.return_value = [4]      
    msmd.scannumbers.return_value = [1, 2, 3]
    mock_msmetadata.return_value = msmd
    mock_get_ms_scan_size.side_effect = [300, 400, 300]
    def glob_side_effect(pattern):
        if pattern.endswith("_plots*.pdf"):
            return []
        if "amp" in pattern and pattern.endswith("*.png"):
            return ["amp1.png", "amp2.png"]
        if "phase" in pattern and pattern.endswith("*.png"):
            return ["phase1.png"]
        if "real" in pattern and pattern.endswith("*.png"):
            return []
        if "imag" in pattern and pattern.endswith("*.png"):
            return []
        return []
    mock_glob.side_effect = glob_side_effect
    mock_img_converted_primary = MagicMock()   
    mock_img_converted_extra = MagicMock()    
    def image_open_side_effect(path):
        m = MagicMock()
        if os.path.basename(path) in {"amp1.png", "phase1.png"}:
            m.convert.return_value = mock_img_converted_primary
        else:
            m.convert.return_value = mock_img_converted_extra
        return m
    mock_Image_open.side_effect = image_open_side_effect
    outdir = tmp_path / "out"
    outdir.mkdir()
    code, output_pdf_list = plot_ms_diagnostics("test.ms", str(outdir), dask_client=None)
    assert code == 0
    assert isinstance(output_pdf_list, list)
    assert len(output_pdf_list) >= 1  
    assert mock_img_converted_primary.save.call_count >= 1
    assert mock_run_shadems.call_count > 0
    rm_calls = [c for c in mock_os_system.call_args_list if "rm -rf" in c.args[0]]
    assert len(rm_calls) == 4
    mock_drop_cache.assert_called_once_with("test.ms")
    mstool.open.assert_called_once_with("test.ms")
    mstool.close.assert_called_once()
    msmd.open.assert_called_once_with("test.ms")
    msmd.close.assert_called_once()


@patch("meersolar.utils.meer_ploting_utils.Image.open")
@patch("meersolar.utils.meer_ploting_utils.table")
@patch("meersolar.utils.meer_ploting_utils.os")
@patch("meersolar.utils.meer_ploting_utils.plt")
def test_plot_caltable_diagnostics(
    mock_plt, mock_os, mock_table, mock_image_open, tmp_path
):
    # Mock CASA table structure
    mock_tb = MagicMock()
    mock_table.return_value = mock_tb
    mock_tb.getkeywords.return_value = {"VisCal": "K Jones"}
    mock_tb.getcol.side_effect = [
        np.array([[1e9, 1.1e9]]),  # CHAN_FREQ (GHz)
        np.random.rand(2, 10, 1),  # FPARAM
        np.zeros((2, 10, 1), dtype=bool),  # FLAG
        np.array([0]),  # ANTENNA1
        np.array([1.0]),  # TIME
    ]
    mock_tb.close = MagicMock()
    mock_img = MagicMock()
    mock_image_open.return_value.convert.return_value = mock_img
    mock_img.save = MagicMock()
    mock_plt.savefig = MagicMock()
    mock_plt.close = MagicMock()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    code, output_pdf = plot_caltable_diagnostics("mockcal.G", str(output_dir))
    assert code == 0
    assert output_pdf.endswith("_plots.pdf")


def test_get_meermap(dummy_image):
    result = get_meermap(dummy_image)
    assert isinstance(result, sunpy.map.GenericMap)


def test_save_in_hpc(dummy_image):
    outdir = os.path.dirname(os.path.abspath(dummy_image))
    outfile = f"{outdir}/{os.path.basename(dummy_image).split('.fits')[0]}_HPC.fits"
    result = save_in_hpc(dummy_image)
    assert result == outfile
    assert os.path.exists(outfile) == True
    header = fits.getheader(outfile)
    assert header["CTYPE1"] == "HPLN-TAN"
    assert header["CTYPE2"] == "HPLT-TAN"
    os.system(f"rm -rf {outfile}")
    assert os.path.exists(outfile) == False


def test_plot_in_hpc(dummy_image):
    imagelist, sunmap = plot_in_hpc(dummy_image, extensions=["png"])
    assert len(imagelist) == 1
    assert imagelist[0][-4:] == ".png"
    assert os.path.exists(imagelist[0]) == True
    assert isinstance(sunmap, sunpy.map.GenericMap)
    os.system(f"rm -rf {imagelist[0]}")
    assert os.path.exists(imagelist[0]) == False


def test_get_suvi_map():
    obs_date = "2024-06-10"
    obs_time = "09:30:00"
    result = get_suvi_map(obs_date, obs_time, os.getcwd(), wavelength=195)
    assert isinstance(result, sunpy.map.sources.suvi.SUVIMap)


def test_enhance_offlimb():
    obs_date = "2024-06-10"
    obs_time = "09:30:00"
    result = get_suvi_map(obs_date, obs_time, os.getcwd(), wavelength=195)
    assert isinstance(result, sunpy.map.sources.suvi.SUVIMap)
    scaled_map = enhance_offlimb(result, do_sharpen=True)
    assert isinstance(scaled_map, sunpy.map.sources.suvi.SUVIMap)


def test_make_meer_overlay(dummy_image):
    plot_file_prefix = "goes_overlay"
    workdir = os.path.dirname(os.path.abspath(dummy_image))
    result = make_meer_overlay(dummy_image, plot_file_prefix=plot_file_prefix)
    assert len(result) == 1
    assert str(result[0]) == f"{workdir}/{plot_file_prefix}.png"
    assert os.path.exists(result[0]) == True
    assert str(result[0][-4:]) == ".png"
    os.system(f"rm -rf {result[0]}")
    assert os.path.exists(result[0]) == False


def test_make_ds_file_per_scan(dummy_submsname):
    save_file = os.getcwd() + "/scan3_ds"
    result = make_ds_file_per_scan(
        f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms", save_file, 3, "DATA"
    )
    assert result == f"{save_file}.npy"
    assert os.path.exists(result) == True
    os.system(f"rm -rf {result}")
    assert os.path.exists(result) == False


def test_make_ds_plot(dummy_submsname):
    plot_file = f"{dummy_submsname}/SUBMSS/test_subms.ms.0000_ds.png"
    save_file = os.getcwd() + "/scan3_ds"
    dsfile = make_ds_file_per_scan(
        f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms", save_file, 3, "DATA"
    )
    assert os.path.exists(dsfile) == True
    result = make_ds_plot(dsfile, plot_file=plot_file)
    os.system(f"rm -rf {dsfile}")
    assert os.path.exists(dsfile) == False
    assert result == plot_file
    assert os.path.exists(result) == True
    os.system(f"rm -rf {result}")
    assert os.path.exists(result) == False


@patch("meersolar.utils.meer_ploting_utils.get_valid_scans", return_value=[3])
@patch(
    "meersolar.utils.meer_ploting_utils.get_cal_target_scans",
    return_value=([3], [3], [3], [3], [3]),
)
def test_plot_goes_full_timeseries(
    mock_get_calscans, mock_get_validscans, dummy_submsname
):
    workdir = os.getcwd()
    plot_file_prefix = "test_goes_tseries"
    result = plot_goes_full_timeseries(
        f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms",
        str(workdir),
        plot_file_prefix=plot_file_prefix,
    )
    mock_get_calscans.assert_called_once_with(
        f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms"
    )
    mock_get_validscans.assert_called_once_with(
        f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms"
    )
    assert str(result) == f"{workdir}/{plot_file_prefix}.png"
    assert os.path.exists(result) == True
    assert result[-4:] == ".png"
    os.system(f"rm -rf {result}")
    assert os.path.exists(result) == False


@pytest.mark.parametrize(
    "imagename, expected_suffix",
    [
        ("mock_MFS_image.fits", "_MFS.fits"),  # Should include _MFS
        ("mock_image.fits", ".fits"),  # No _MFS
    ],
)
@patch("meersolar.utils.meer_ploting_utils.make_meer_overlay")
@patch("meersolar.utils.meer_ploting_utils.plot_in_hpc")
@patch("meersolar.utils.meer_ploting_utils.save_in_hpc")
@patch("meersolar.utils.meer_ploting_utils.os.makedirs")
@patch("meersolar.utils.meer_ploting_utils.os.system")
@patch("meersolar.utils.meer_ploting_utils.fits.open")
@patch("meersolar.utils.meer_ploting_utils.fits.getheader")
@patch("meersolar.utils.meer_ploting_utils.Horizons")
@patch("meersolar.utils.meer_ploting_utils.SkyCoord")
@patch("meersolar.utils.meer_ploting_utils.Time")
@patch("meersolar.utils.meer_ploting_utils.cutout_image")
@patch("meersolar.utils.meer_ploting_utils.calc_solar_image_stat")
def test_rename_meersolar_image(
    mock_calc_stats,
    mock_cutout,
    mock_Time,
    mock_SkyCoord,
    mock_Horizons,
    mock_getheader,
    mock_fits_open,
    mock_os_system,
    mock_os_makedirs,
    mock_save_hpc,
    mock_plot_hpc,
    mock_overlay,
    imagename,
    expected_suffix,
):
    # Setup dummy image output name
    mock_cutout.return_value = imagename

    # Setup header
    mock_header = {"DATE-OBS": "2021-01-01T12:00:00", "CRVAL3": 1.4e9}
    mock_getheader.return_value = mock_header

    # Setup FITS open context manager
    mock_hdul = MagicMock()
    mock_fits_open.return_value.__enter__.return_value = mock_hdul
    mock_hdul.__getitem__.return_value.header = {}

    # Setup astropy Time and Horizons mocks
    mock_Time.return_value.jd = 2459215.0
    mock_Horizons.return_value.ephemerides.return_value = {"RA": [100.0], "DEC": [45.0]}
    mock_coords = MagicMock()
    mock_coords.ra.deg = 100.0
    mock_coords.dec.deg = 45.0
    mock_SkyCoord.return_value = mock_coords

    # Setup image stats
    mock_calc_stats.return_value = (10, -1, 1.0, 50, 5.0, 3.0, 2.0, 8.0)

    # Call rename_meersolar_image
    result = rename_meersolar_image(imagename, imagedir="/tmp/images")

    # Check suffix
    assert result.endswith(expected_suffix)
