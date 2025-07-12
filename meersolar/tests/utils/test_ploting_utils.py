import pytest
import sunpy
import os
from astropy.io import fits
from unittest.mock import patch, MagicMock
from meersolar.utils.ploting_utils import *

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


@patch("meersolar.utils.ploting_utils.get_valid_scans", return_value=[3])
@patch(
    "meersolar.utils.ploting_utils.get_cal_target_scans",
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
