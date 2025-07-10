import pytest
import numpy as np
import traceback
import warnings
import copy
import glob
import os
from astropy.io import fits
from astropy.wcs import FITSFixedWarning
from casatasks import casalog
from unittest.mock import patch, MagicMock
from meersolar.utils.image_utils import *


@pytest.mark.parametrize(
    "imagename, expected_suffix",
    [
        ("mock_MFS_image.fits", "_MFS.fits"),  # Should include _MFS
        ("mock_image.fits", ".fits"),          # No _MFS
    ]
)
@patch("meersolar.utils.image_utils.make_meer_overlay")
@patch("meersolar.utils.image_utils.plot_in_hpc")
@patch("meersolar.utils.image_utils.save_in_hpc")
@patch("meersolar.utils.image_utils.os.makedirs")
@patch("meersolar.utils.image_utils.os.system")
@patch("meersolar.utils.image_utils.fits.open")
@patch("meersolar.utils.image_utils.fits.getheader")
@patch("meersolar.utils.image_utils.Horizons")
@patch("meersolar.utils.image_utils.SkyCoord")
@patch("meersolar.utils.image_utils.Time")
@patch("meersolar.utils.image_utils.cutout_image")
@patch("meersolar.utils.image_utils.calc_solar_image_stat")
def test_rename_image(
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

    # Call rename_image
    result = rename_image(imagename, imagedir="/tmp/images")

    # Check suffix
    assert result.endswith(expected_suffix)

@patch("meersolar.utils.image_utils.fits.writeto")
@patch("meersolar.utils.image_utils.fits.getheader")
@patch("meersolar.utils.image_utils.fits.getdata")
@patch("meersolar.utils.image_utils.os.path.exists", return_value=True)
@patch("meersolar.utils.image_utils.os.system")
@patch("meersolar.utils.image_utils.run_wsclean", return_value=0)
def test_create_circular_mask_success(
    mock_run_wsclean,
    mock_system,
    mock_exists,
    mock_getdata,
    mock_getheader,
    mock_writeto,
    tmp_path,
):
    # Prepare mock FITS data
    imsize = 100
    dummy_data = MagicMock()
    dummy_data.__getitem__.return_value = np.zeros((imsize, imsize), dtype=float)
    mock_getdata.return_value = np.zeros((1, 1, imsize, imsize), dtype=float)
    mock_getheader.return_value = {"SIMPLE": True}

    # Create fake MS path
    ms_path = tmp_path / "dummy.ms"
    ms_path.mkdir()

    # Run the function
    mask_file = create_circular_mask(
        str(ms_path), cellsize=5.0, imsize=imsize, mask_radius=10
    )

    # Assertions
    assert mask_file.endswith("-mask.fits")
    mock_run_wsclean.assert_called_once()
    mock_writeto.assert_called_once()
    assert mock_exists.called


@pytest.mark.parametrize(
    "radius",
    [
        (10),
        (90),
        (5),
        (60),
    ],
)
def test_create_circular_mask_array(radius):
    mock_data = np.empty((200, 200))
    masked_array = create_circular_mask_array(mock_data, radius)
    max_pix = radius**2
    assert np.nansum(masked_array) > max_pix


def test_calc_solar_image_stat(dummy_image):
    maxval, minval, rms, total_val, mean_val, median_val, rms_dyn, minmax_dyn = (
        calc_solar_image_stat(dummy_image, disc_size=18)
    )
    assert maxval == 422.67
    assert minval == -24.9
    assert rms == 17.14
    assert total_val == 30912980.0
    assert mean_val == 92.04
    assert median_val == 81.02
    assert rms_dyn == 24.66
    assert minmax_dyn == 16.97


def test_calc_dyn_range(dummy_image):
    flux, dr, rms = calc_dyn_range(dummy_image, dummy_image, dummy_image)
    assert flux == 34900992.0
    assert dr == 11.98
    assert rms == 41.58


def test_generate_tb_map(dummy_image):
    outfile = dummy_image.split(".fits")[0] + "_TB.fits"
    assert generate_tb_map(dummy_image) == outfile
    header = fits.getheader(outfile)
    assert header["BUNIT"] == "K"
    os.system(f"rm -rf {outfile}")


@pytest.mark.parametrize(
    "cutout_size",
    [
        (0.1),
        (1.0),
        (0.2),
    ],
)
def test_cutout_image(dummy_image, cutout_size):
    output_file = dummy_image.split(".fits")[0] + "-cutout.fits"
    assert cutout_image(dummy_image, output_file, x_deg=cutout_size) == output_file
    header = fits.getheader(output_file)
    cdelt = abs(header["CDELT1"])
    npix = header["NAXIS1"]
    imsize = round(cdelt * npix, 1)
    assert imsize == cutout_size
    os.system(f"rm -rf {output_file}")


def test_make_timeavg_image(dummy_image):
    outfile_name = dummy_image.split(".fits")[0] + "-tavg.fits"
    assert (
        make_timeavg_image(
            [dummy_image, dummy_image, dummy_image],
            outfile_name,
            keep_wsclean_images=True,
        )
        == outfile_name
    )
    os.system(f"rm -rf {outfile_name}")


def test_make_freqavg_image(dummy_image):
    outfile_name = dummy_image.split(".fits")[0] + "-favg.fits"
    assert (
        make_freqavg_image(
            [dummy_image, dummy_image, dummy_image],
            outfile_name,
            keep_wsclean_images=True,
        )
        == outfile_name
    )
    os.system(f"rm -rf {outfile_name}")


def test_make_stokes_wsclean_imagecube(dummy_image):
    outfile_name = dummy_image.split(".fits")[0] + "-StokesI.fits"
    result = make_stokes_wsclean_imagecube([dummy_image], outfile_name)
    assert result == outfile_name
    os.system(f"rm -rf {outfile_name}")
