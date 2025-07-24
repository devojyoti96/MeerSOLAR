import types
import numpy as np
import glob
import os
from casatools import msmetadata, ms as casamstool, table
from .basic_utils import *

##################################
# Imaging related
##################################
def calc_sun_dia(freqMHz):
    """
    Function to calculate the diameter of the Sun at a given frequency (White 2016)

    Parameters
    ----------
    freq : float
        Frequency in MHz

    Returns
    -------
    float
        Diameter of the Sun in arcmin
    """
    freqGHz = freqMHz / 10**3  # Convert in GHz
    dia = 32 + (2.2 * (freqGHz) ** (-0.6))
    return round(dia, 2)


def calc_maxuv(msname, chan_number=-1):
    """
    Calculate maximum UV

    Parameters
    ----------
    msname : str
        Name of the measurement set
    chan_number : int, optional
        Channel number

    Returns
    -------
    float
        Maximum UV in meter
    float
        Maximum UV in wavelength
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.chanfreqs(0)[chan_number]
    wavelength = 299792458.0 / (freq)
    msmd.close()
    msmd.done()
    tb = table()
    tb.open(msname)
    uvw = tb.getcol("UVW")
    tb.close()
    u, v, w = [uvw[i, :] for i in range(3)]
    uv = np.sqrt(u**2 + v**2)
    uv[uv == 0] = np.nan
    maxuv = np.nanmax(uv)
    return round(float(maxuv), 2), round(float(maxuv / wavelength), 2)


def calc_minuv(msname, chan_number=-1):
    """
    Calculate minimum UV

    Parameters
    ----------
    msname : str
        Name of the measurement set
    chan_number : int, optional
        Channel number

    Returns
    -------
    float
        Minimum UV in meter
    float
        Minimum UV in wavelength
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.chanfreqs(0)[chan_number]
    wavelength = 299792458.0 / (freq)
    msmd.close()
    msmd.done()
    tb = table()
    tb.open(msname)
    uvw = tb.getcol("UVW")
    tb.close()
    u, v, w = [uvw[i, :] for i in range(3)]
    uv = np.sqrt(u**2 + v**2)
    uv[uv == 0] = np.nan
    minuv = np.nanmin(uv)
    return round(float(minuv), 2), round(float(minuv / wavelength), 2)


def calc_field_of_view(msname, FWHM=True):
    """
    Calculate optimum field of view in arcsec.

    Parameters
    ----------
    msname : str
        Measurement set name
    FWHM : bool, optional
        Upto FWHM, otherwise upto first null

    Returns
    -------
    float
        Field of view in arcsec
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.chanfreqs(0)[0]
    msmd.close()
    tb = table()
    tb.open(msname + "/ANTENNA")
    dish_dia = np.nanmin(tb.getcol("DISH_DIAMETER"))
    tb.close()
    wavelength = 299792458.0 / freq
    if FWHM:
        FOV = 1.22 * wavelength / dish_dia
    else:
        FOV = 2.04 * wavelength / dish_dia
    fov_arcsec = np.rad2deg(FOV) * 3600  # In arcsecs
    return round(float(fov_arcsec), 2)


def get_optimal_image_interval(
    msname,
    temporal_tol_factor=0.1,
    spectral_tol_factor=0.1,
    chan_range="",
    timestamp_range="",
    max_nchan=-1,
    max_ntime=-1,
):
    """
    Get optimal image spectral temporal interval such that total flux max-median in each chunk is within tolerance limit

    Parameters
    ----------
    msname : str
        Name of the measurement set
    temporal_tol_factor : float, optional
        Tolerance factor for temporal variation (default : 0.1, 10%)
    spectral_tol_factor : float, optional
        Tolerance factor for spectral variation (default : 0.1, 10%)
    chan_range : str, optional
        Channel range
    timestamp_range : str, optional
        Timestamp range
    max_nchan : int, optional
        Maxmium number of spectral chunk
    max_ntime : int, optional
        Maximum number of temporal chunk

    Returns
    -------
    int
        Number of time intervals to average
    int
        Number of channels to averages
    """

    def is_valid_chunk(chunk, tolerance):
        mean_flux = np.nanmedian(chunk)
        if mean_flux == 0:
            return False
        return (np.nanmax(chunk) - np.nanmin(chunk)) / mean_flux <= tolerance

    def find_max_valid_chunk_length(fluxes, tolerance):
        n = len(fluxes)
        for window in range(n, 1, -1):  # Try from largest to smallest
            valid = True
            for start in range(0, n, window):
                end = min(start + window, n)
                chunk = fluxes[start:end]
                if len(chunk) < window:  # Optionally require full window
                    valid = False
                    break
                if not is_valid_chunk(chunk, tolerance):
                    valid = False
                    break
            if valid:
                return window  # Return the largest valid window
        return 1  # Minimum chunk size is 1 if nothing else is valid

    tb = table()
    mstool = casamstool()
    msmd = msmetadata()
    msmd.open(msname)
    nchan = msmd.nchan(0)
    times = msmd.timesforspws(0)
    ntime = len(times)
    del times
    msmd.close()
    tb.open(msname)
    u, v, w = tb.getcol("UVW")
    tb.close()
    uvdist = np.sort(np.unique(np.sqrt(u**2 + v**2)))
    mstool.open(msname)
    if uvdist[0] == 0.0:
        mstool.select({"uvdist": [0.0, 0.0]})
    else:
        mstool.select({"antenna1": 0, "antenna2": 1})
    data_and_flag = mstool.getdata(["DATA", "FLAG"], ifraxis=True)
    data = data_and_flag["data"]
    flag = data_and_flag["flag"]
    data[flag] = np.nan
    mstool.close()
    if chan_range != "":
        start_chan = int(chan_range.split(",")[0])
        end_chan = int(chan_range.split(",")[-1])
        spectra = np.nanmedian(data[:, start_chan:end_chan, ...], axis=(0, 2, 3))
    else:
        spectra = np.nanmedian(data, axis=(0, 2, 3))
    if timestamp_range != "":
        t_start = int(timestamp_range.split(",")[0])
        t_end = int(timestamp_range.split(",")[-1])
        t_series = np.nanmedian(data[..., t_start:t_end], axis=(0, 1, 2))
    else:
        t_series = np.nanmedian(data, axis=(0, 1, 2))
    t_series = t_series[t_series != 0]
    spectra = spectra[spectra != 0]
    t_chunksize = find_max_valid_chunk_length(t_series, temporal_tol_factor)
    f_chunksize = find_max_valid_chunk_length(spectra, spectral_tol_factor)
    n_time_interval = int(len(t_series) / t_chunksize)
    n_spectral_interval = int(len(spectra) / f_chunksize)
    if max_nchan > 0 and n_spectral_interval > max_nchan:
        n_spectral_interval = max_nchan
    if max_ntime > 0 and n_time_interval > max_ntime:
        n_time_interval = max_ntime
    return n_time_interval, n_spectral_interval


def calc_psf(msname, chan_number=-1):
    """
    Function to calculate PSF size in arcsec

    Parameters
    ----------
    msname : str
        Name of the measurement set
    chan_number : int, optional
        Channel number

    Returns
    -------
    float
            PSF size in arcsec
    """
    maxuv_m, maxuv_l = calc_maxuv(msname, chan_number=chan_number)
    psf = np.rad2deg(1.2 / maxuv_l) * 3600.0  # In arcsec
    return round(float(psf), 2)


def calc_npix_in_psf(weight, robust=0.0):
    """
    Calculate number of pixels in a PSF (could be in fraction)

    Parameters
    ----------
    weight : str
        Image weighting scheme
    robust : float, optional
        Briggs weighting robust parameter (-1,1)

    Returns
    -------
    float
        Number of pixels in a PSF
    """
    if weight.upper() == "NATURAL":
        npix = 3.0
    elif weight.upper() == "UNIFORM":
        npix = 5.0
    else:  # -1 to +1, uniform to natural
        robust = np.clip(robust, -1.0, 1.0)
        npix = 5.0 - ((robust + 1.0) / 2.0) * (5.0 - 3.0)
    return round(npix, 1)


def calc_cellsize(msname, num_pixel_in_psf):
    """
    Calculate pixel size in arcsec

    Parameters
    ----------
    msname : str
        Name of the measurement set
    num_pixel_in_psf : float
        Number of pixels in one PSF

    Returns
    -------
    int
        Pixel size in arcsec
    """
    psf = calc_psf(msname)
    pixel = round(psf / num_pixel_in_psf, 1)
    return pixel


def calc_multiscale_scales(msname, num_pixel_in_psf, chan_number=-1, max_scale=16):
    """
    Calculate multiscale scales

    Parameters
    ----------
    msname : str
        Name of the measurement set
    num_pixel_in_psf : float
            Number of pixels in one PSF
    max_scale : float, optional
        Maximum scale in arcmin

    Returns
    -------
    list
            Multiscale scales in pixel units
    """
    psf = calc_psf(msname, chan_number=chan_number)
    minuv, minuv_l = calc_minuv(msname, chan_number=chan_number)
    max_interferometric_scale = (
        0.5 * np.rad2deg(1.0 / minuv_l) * 60.0
    )  # In arcmin, half of maximum scale
    max_interferometric_scale = min(max_scale, max_interferometric_scale)
    max_scale_pixel = int((max_interferometric_scale * 60.0) / (psf / num_pixel_in_psf))
    multiscale_scales = [0]
    current_scale = num_pixel_in_psf
    while True:
        current_scale = current_scale * 2
        if current_scale >= max_scale_pixel:
            break
        multiscale_scales.append(current_scale)
    return multiscale_scales


def get_multiscale_bias(freq, bias_min=0.6, bias_max=0.9, minfreq=1015, maxfreq=1670):
    """
    Get frequency dependent multiscale bias

    Parameters
    ----------
    freq : float
        Frequency in MHz
    bias_min : float, optional
        Minimum bias at minimum L-band frequency
    bias_max : float, optional
        Maximum bias at maximum L-band frequency
    minfreq : float, optional
        Minimum frequency range in MHz
    maxfreq : float, optional
        Maximum frequency range in MHz

    Returns
    -------
    float
        Multiscale bias patrameter
    """
    if freq <= minfreq:
        return bias_min
    elif freq >= maxfreq:
        return bias_max
    else:
        freq_min = 1015
        freq_max = 1670
        logf = np.log10(freq)
        logf_min = np.log10(freq_min)
        logf_max = np.log10(freq_max)
        frac = (logf - logf_min) / (logf_max - logf_min)
        return round(
            np.clip(bias_min + frac * (bias_max - bias_min), bias_min, bias_max), 3
        )


# Expose functions and classes
__all__ = [
    name
    for name, obj in globals().items()
    if (
        (isinstance(obj, types.FunctionType) or isinstance(obj, type))
        and obj.__module__ == __name__
    )
]
