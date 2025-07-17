import pytest
import numpy as np
import os
from astropy.io import fits
from unittest.mock import patch, MagicMock
from meersolar.utils.image_utils import *


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
    assert rms_dyn == 24.67
    assert minmax_dyn == 16.98


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
