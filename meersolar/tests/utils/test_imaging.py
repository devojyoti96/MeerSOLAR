import pytest
import os
import traceback
from meersolar.utils.imaging import *


def test_calc_sun_dia():
    assert calc_sun_dia(1000.0) == 34.2


def test_calc_maxuv(dummy_submsname):
    maxuv, maxuv_l = calc_maxuv(f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms")
    assert maxuv == 7390.71
    assert maxuv_l == 18785.68


def test_calc_minuv(dummy_submsname):
    minuv, minuv_l = calc_minuv(f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms")
    assert minuv == 26.82
    assert minuv_l == 68.17


def test_calc_field_of_view(dummy_msname):
    assert calc_field_of_view(dummy_msname, FWHM=True) == 7384.97
    assert calc_field_of_view(dummy_msname, FWHM=False) == 12348.63


def test_get_optimal_image_interval(dummy_submsname):
    ntime, nchan = get_optimal_image_interval(
        f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms",
        temporal_tol_factor=0.1,
        spectral_tol_factor=0.1,
    )
    assert ntime == 1
    assert nchan == 11
    ntime, nchan = get_optimal_image_interval(
        f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms",
        temporal_tol_factor=1.0,
        spectral_tol_factor=0.001,
    )
    assert ntime == 1
    assert nchan == 11


def test_calc_psf(dummy_msname):
    assert calc_psf(dummy_msname) == 12.68


def test_calc_npix_in_psf():
    assert calc_npix_in_psf("natural") == 3.0
    assert calc_npix_in_psf("uniform") == 5.0
    assert calc_npix_in_psf("briggs", robust=0.0) == 4.0


def test_calc_cellsize(dummy_submsname):
    assert calc_cellsize(f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms", 3) == 4.4
    assert calc_cellsize(f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms", 5) == 2.6


def test_calc_multiscale_scales(dummy_submsname):
    scales = calc_multiscale_scales(
        f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms", 3, max_scale=16
    )
    assert scales == [0, 6, 12, 24, 48, 96, 192]
    scales = calc_multiscale_scales(
        f"{dummy_submsname}/SUBMSS/test_subms.ms.0000.ms", 3, max_scale=8
    )
    assert scales == [0, 6, 12, 24, 48, 96]


def test_get_multiscale_bias():
    assert get_multiscale_bias(700) == 0.6
    assert get_multiscale_bias(1700) == 0.9
    assert get_multiscale_bias(1400) == 0.794
