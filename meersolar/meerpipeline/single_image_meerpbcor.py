import astropy.units as u
import logging
import dask
import numpy as np
import argparse
import traceback
import warnings
import copy
import time
import sys
import os
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.wcs import FITSFixedWarning
from casatasks import casalog
from numpy.linalg import inv
from astropy.wcs import WCS
from scipy.interpolate import RectBivariateSpline
from joblib import Parallel, delayed as joblid_delayed
from meersolar.utils import (
    get_datadir,
    get_cachedir,
    SmartDefaultsHelpFormatter,
    save_pid,
)

logging.getLogger("distributed").setLevel(logging.WARNING)

try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except BaseException:
    pass

datadir = get_datadir()

warnings.simplefilter("ignore", category=FITSFixedWarning)

# Define MeerKAT location
MEERLAT = -30.7133
MEERLON = 21.4429
MEERALT = 1086.6
datadir = get_datadir()


def get_IQUV(filename):
    """
    Get IQUV from a fits

    Parameters
    ----------
    filename : str
        Fits image name

    Returns
    -------
    dict
        Stokes
    """
    data = fits.getdata(filename).astype("float32")
    header = fits.getheader(filename)
    keys = header.keys()
    if "CTYPE3" in keys and header["CTYPE3"] == "STOKES":
        stokesaxis = 3
    elif "CTYPE4" in keys and header["CTYPE4"] == "STOKES":
        stokesaxis = 4
    else:
        stokesaxis = 1
    shape = data.shape
    stokes = {}
    if shape[0] > 1 or shape[1] > 1 and (stokesaxis == 3 or stokesaxis == 4):
        if stokesaxis == 3:
            stokes["I"] = data[0, 0, :, :]
            stokes["Q"] = data[0, 1, :, :]
            stokes["U"] = data[0, 2, :, :]
            stokes["V"] = data[0, 3, :, :]
        elif stokesaxis == 4:
            stokes["I"] = data[0, 0, :, :]
            stokes["Q"] = data[1, 0, :, :]
            stokes["U"] = data[2, 0, :, :]
            stokes["V"] = data[3, 0, :, :]
    else:
        stokes["I"] = data[0, 0, :, :]
        stokes["Q"] = copy.deepcopy(stokes["I"]) * 0
        stokes["U"] = copy.deepcopy(stokes["I"]) * 0
        stokes["V"] = copy.deepcopy(stokes["I"]) * 0
    return stokes


def put_IQUV(filename, stokes, header):
    """
    Put IQUV into a fits

    Parameters
    ----------
    filename : str
        Fits image name
    stokes : dict
        Stokes
    header : dict
        Image header

    Returns
    -------
    dict
        Stokes
    """
    keys = header.keys()
    if "CTYPE3" in keys and header["CTYPE3"] == "STOKES":
        stokesaxis = 3
    elif "CTYPE4" in keys and header["CTYPE4"] == "STOKES":
        stokesaxis = 4
    else:
        stokesaxis = 1
    naxis = header["NAXIS"]
    shape = tuple(header[f"NAXIS{axis}"] for axis in range(naxis, 0, -1))
    data = np.empty(shape, dtype=np.float32)
    if shape[0] > 1 or shape[1] > 1 and (stokesaxis == 3 or stokesaxis == 4):
        if stokesaxis == 3:
            data[0, 0, :, :] = stokes["I"]
            data[0, 1, :, :] = stokes["Q"]
            data[0, 2, :, :] = stokes["U"]
            data[0, 3, :, :] = stokes["V"]
        elif stokesaxis == 4:
            data[0, 0, :, :] = stokes["I"]
            data[1, 0, :, :] = stokes["Q"]
            data[2, 0, :, :] = stokes["U"]
            data[3, 0, :, :] = stokes["V"]
    else:
        data[0, 0, :, :] = stokes["I"]
    fits.writeto(filename, data=data, header=header, overwrite=True)
    return filename


def get_brightness(stokes):
    """

    Returns brightness matrix from stokes dictionary (X and Y are in opposite convention of IAU in MeerKAT)
    """
    I = stokes["I"].astype("float32")
    Q = stokes["Q"].astype("float32")
    U = stokes["U"].astype("float32")
    V = stokes["V"].astype("float32")
    XX = I - Q
    XY = U - 1j * V
    YX = U + 1j * V
    YY = I + Q
    B = np.array([XX, XY, YX, YY]).astype("complex64")
    B = B.T
    B = B.reshape(B.shape[0], B.shape[1], 2, 2)
    return B


def make_stokes(b):
    """
    Makes stokes images from brightness matrix
    """
    XX = b[0, 0, ...].astype("complex64")
    XY = b[0, 1, ...].astype("complex64")
    YX = b[1, 0, ...].astype("complex64")
    YY = b[1, 1, ...].astype("complex64")
    stokes = {}
    stokes["I"] = np.real(XX + YY) / 2.0
    stokes["Q"] = np.real(YY - XX) / 2.0
    stokes["U"] = np.real(XY + YX) / 2.0
    stokes["V"] = np.real(1j * (XY - YX)) / 2.0
    return stokes


def load_beam(image_file, band=""):
    """
    Load MeerKAT beam

    Parameters
    ----------
    image_file : str
        Image name (Assuming single spectral image)
    band : str, optional
        Band name (If not provided, check from header or frequency)

    Returns
    -------
    numpy.array
        l,m coordinates
    numpy.array
        Full Jones complex beam
    """
    hdr = fits.getheader(image_file)
    keys = hdr.keys()
    if "CTYPE3" in keys and hdr["CTYPE3"] == "FREQ":
        freq = hdr["CRVAL3"]
        delfreq = hdr["CDELT3"]
    elif "CTYPE4" in keys and hdr["CTYPE4"] == "FREQ":
        freq = hdr["CRVAL4"]
        delfreq = hdr["CDELT4"]
    else:
        print(f"No frequency axis in image {image_file}.")
        return
    freq1 = (freq - (delfreq / 2)) / 10**6  # In MHz
    freq2 = (freq + (delfreq / 2)) / 10**6  # In MHz
    if band == "":
        try:
            band = hdr["BAND"]
        except BaseException:
            if freq1 >= 544 and freq2 <= 1088:  # UHF band
                band = "U"
            elif freq1 >= 856 and freq2 <= 1712:  # L band
                band = "L"
            else:
                print(f"Image: {image_file} is not in UHF or L-band.")
                return
    if band == "U":
        beam_data = np.load(datadir + "/MeerKAT_antavg_Uband.npz", mmap_mode="r")
    elif band == "L":
        beam_data = np.load(datadir + "/MeerKAT_antavg_Lband.npz", mmap_mode="r")
    else:
        print(f"Image: {image_file} is not in UHF or L-band.")
        return
    freqs = beam_data["freqs"]
    coords = np.deg2rad(
        beam_data["coords"]
    )  # It is done as l,m values were converted into degree
    pos = np.where((freq >= freq1) & (freqs <= freq1))[0]
    beam = beam_data["beams"][:, pos, ...].mean(1)
    beam = beam.astype("complex64")
    del beam_data, freqs
    return coords, beam


def get_radec_grid(image_file):
    """
    Get RA and Dec arrays for all pixels in an image.

    Parameters
    ----------
    image_file : str
        FITS image file name

    Returns
    -------
    ra : 2D numpy.ndarray
        RA values in degrees for each pixel
    dec : 2D numpy.ndarray
        Dec values in degrees for each pixel
    """
    hdr = fits.getheader(image_file)
    wcs = WCS(hdr).celestial
    ny, nx = hdr["NAXIS2"], hdr["NAXIS1"]
    y, x = np.mgrid[0:ny, 0:nx]  # pixel coordinates
    world = wcs.pixel_to_world(x, y)
    ra = world.ra.deg
    dec = world.dec.deg
    return ra, dec


def get_pointingcenter_radec(image_file):
    """
    Get image pointing center RA DEC

    Parameters
    ----------
    image_file : str
        Image file name

    Returns
    -------
    float
        RA in degree
    float
        DEC in degree
    """
    hdr = fits.getheader(image_file)
    image_wcs = WCS(hdr)
    image_shape = (hdr["NAXIS2"], hdr["NAXIS1"])
    ra0 = float(hdr["CRVAL1"])
    dec0 = float(hdr["CRVAL2"])
    return ra0, dec0


def radec_to_lm(ra_deg, dec_deg, ra0_deg, dec0_deg):
    """
    Convert RA/Dec to l,m direction cosines relative to a phase center.

    Parameters
    ----------
    ra_deg, dec_deg : 2D arrays
        RA and Dec in degrees
    ra0_deg, dec0_deg : float
        Phase center RA and Dec in degrees

    Returns
    -------
    l, m : 2D arrays
        Direction cosines (dimensionless)
    """
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    ra0 = np.radians(ra0_deg)
    dec0 = np.radians(dec0_deg)
    delta_ra = ra - ra0
    l = np.cos(dec) * np.sin(delta_ra)
    m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(delta_ra)
    return l, m


def get_parallactic_angle(
    obs_time, ra_deg, dec_deg, LAT=MEERLAT, LON=MEERLON, ALT=MEERALT
):
    """
    Get parallactic angle

    Parameters
    ----------
    obs_time : str
        Observation time in YYY-MM-DDThh:mm:ss format
    ra : float
        RA in degree
    dec : float
        DEC in degree

    Returns
    -------
    float
        Parallactic angle in degree
    """
    sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    obstime = Time(obs_time)
    meerpos = EarthLocation(lat=LAT * u.deg, lon=LON * u.deg, height=ALT * u.m)
    altaz = sky.transform_to(AltAz(obstime=obstime, location=meerpos))
    az = altaz.az.rad
    alt = altaz.alt.rad
    lat = np.deg2rad(LAT)
    p = np.arctan2(
        np.sin(az) * np.cos(lat),
        np.cos(alt) * np.sin(lat) - np.sin(alt) * np.cos(lat) * np.cos(az),
    )
    return round(np.rad2deg(p), 2)


def get_beam_interpolator(jones, coords):
    """
    Get beam interpolator

    Parameters
    ----------
    jones : numpy.array
        Jones array (shape, npol, l_npix, m_npix)
    coords : numpy.array
        l,m coordinates

    Returns
    -------
    interpolator
        Interpolation functions
    """
    j00_r = RectBivariateSpline(
        x=coords, y=coords, z=np.nan_to_num(np.real(jones[0, ...]))
    )
    j00_i = RectBivariateSpline(
        x=coords, y=coords, z=np.nan_to_num(np.imag(jones[0, ...]))
    )
    j01_r = RectBivariateSpline(
        x=coords, y=coords, z=np.nan_to_num(np.real(jones[1, ...]))
    )
    j01_i = RectBivariateSpline(
        x=coords, y=coords, z=np.nan_to_num(np.imag(jones[1, ...]))
    )
    j10_r = RectBivariateSpline(
        x=coords, y=coords, z=np.nan_to_num(np.real(jones[2, ...]))
    )
    j10_i = RectBivariateSpline(
        x=coords, y=coords, z=np.nan_to_num(np.imag(jones[2, ...]))
    )
    j11_r = RectBivariateSpline(
        x=coords, y=coords, z=np.nan_to_num(np.real(jones[3, ...]))
    )
    j11_i = RectBivariateSpline(
        x=coords, y=coords, z=np.nan_to_num(np.imag(jones[3, ...]))
    )
    return j00_r, j00_i, j01_r, j01_i, j10_r, j10_i, j11_r, j11_i


def apply_parallactic_rotation(jones, p_angle):
    """
    Apply left-side parallactic rotation: J' = J.R(p_angle)
    as needed in the RIME context (sky-frame transformation).

    Parameters
    ----------
    jones : ndarray
        Jones matrix, shape (4, H, W), with components:
        [0] = J_00, [1] = J_01, [2] = J_10, [3] = J_11
    chi : float
        Parallactic angle in degree

    Returns
    -------
    jones_rot : ndarray
        Rotated Jones matrix, shape (4, H, W)
    """
    p_angle = np.deg2rad(p_angle)
    c = np.cos(p_angle)
    s = -np.sin(p_angle)
    j00, j01, j10, j11 = jones
    jj00 = j00 * c - j01 * s
    jj01 = j00 * s + j01 * c
    jj10 = j10 * c - j11 * s
    jj11 = j10 * s + j11 * c
    return np.stack([jj00, jj01, jj10, jj11], axis=0).astype("complex64")


def get_image_beam(
    image_file,
    pbdir,
    save_beam=True,
    band="",
    apply_parang=True,
    n_cpu=8,
    verbose=False,
):
    """
    Get image beam

    Parameters
    ----------
    image_file : str
        Image file name
    pbdir : str
        Primary beam directory
    save_beam : bool, optional
        Save beam of the image
    band : str, optional
        Band name
    apply_parang : bool, optional
        Apply parallactic angle correction
    n_cpu : int, optinal
        Number of CPU threads to use
    verbose : bool, optional
        Verbose output

    Returns
    -------
    numpy.array
        Jones array
    """
    if n_cpu > 8:
        n_cpu = 8
    start_time = time.time()
    ##################################
    header = fits.getheader(image_file)
    if header["CTYPE3"] == "FREQ":
        freq = header["CRVAL3"]
    elif header["CTYPE4"] == "FREQ":
        freq = header["CRVAL4"]
    else:
        print(f"No frequency axis in image: {image_file}.")
        return
    freq = round(freq / 10**6, 1)  # In MHz
    pbfile = f"{pbdir}/freq_{freq}_pb.npy"
    obs_time = header["DATE-OBS"]
    ra0, dec0 = get_pointingcenter_radec(image_file)  # Phase center
    p_angle = get_parallactic_angle(
        obs_time, ra0, dec0
    )  # Parallactic angle of the center
    #######################################
    # If beam file exists
    #######################################
    fresh_run = True
    if os.path.exists(pbfile):
        if verbose:
            print(f"Loading beam from: {pbfile}")
        try:
            jones_array = np.load(pbfile, allow_pickle=True)
            fresh_run = False
        except BaseException:
            fresh_run = True
            os.system(f"rm -rf {pbfile}")
    #################################
    # Fresh run
    #################################
    if fresh_run:
        ra_grid, dec_grid = get_radec_grid(image_file)  # RA DEC grid
        l_grid, m_grid = radec_to_lm(ra_grid, dec_grid, ra0, dec0)
        ############################
        # Load beam
        ############################
        beam_results = load_beam(image_file, band=band)
        if beam_results is None:
            return
        lm_coords, beam = beam_results
        j00_r, j00_i, j01_r, j01_i, j10_r, j10_i, j11_r, j11_i = get_beam_interpolator(
            beam, lm_coords
        )
        l_grid_flat = l_grid.flatten()
        m_grid_flat = m_grid.flatten()
        grid_shape = l_grid.shape
        del l_grid, m_grid
        with Parallel(n_jobs=n_cpu, backend="threading") as parallel:
            results = parallel(
                [
                    joblid_delayed(j00_r)(l_grid_flat, m_grid_flat, grid=False),
                    joblid_delayed(j00_i)(l_grid_flat, m_grid_flat, grid=False),
                    joblid_delayed(j01_r)(l_grid_flat, m_grid_flat, grid=False),
                    joblid_delayed(j01_i)(l_grid_flat, m_grid_flat, grid=False),
                    joblid_delayed(j10_r)(l_grid_flat, m_grid_flat, grid=False),
                    joblid_delayed(j10_i)(l_grid_flat, m_grid_flat, grid=False),
                    joblid_delayed(j11_r)(l_grid_flat, m_grid_flat, grid=False),
                    joblid_delayed(j11_i)(l_grid_flat, m_grid_flat, grid=False),
                ]
            )
        del parallel
        (
            j00_r_arr,
            j00_i_arr,
            j01_r_arr,
            j01_i_arr,
            j10_r_arr,
            j10_i_arr,
            j11_r_arr,
            j11_i_arr,
        ) = results
        j00_r_arr = j00_r_arr.reshape(grid_shape)
        j00_i_arr = j00_i_arr.reshape(grid_shape)
        j01_r_arr = j01_r_arr.reshape(grid_shape)
        j01_i_arr = j01_i_arr.reshape(grid_shape)
        j10_r_arr = j10_r_arr.reshape(grid_shape)
        j10_i_arr = j10_i_arr.reshape(grid_shape)
        j11_r_arr = j11_r_arr.reshape(grid_shape)
        j11_i_arr = j11_i_arr.reshape(grid_shape)
        jones_array = np.array(
            [
                j00_r_arr + 1j * j00_i_arr,
                j01_r_arr + 1j * j01_i_arr,
                j10_r_arr + 1j * j10_i_arr,
                j11_r_arr + 1j * j11_i_arr,
            ]
        ).astype("complex64")
        del (
            j00_r_arr,
            j00_i_arr,
            j01_r_arr,
            j01_i_arr,
            j10_r_arr,
            j10_i_arr,
            j11_r_arr,
            j11_i_arr,
        )
        if save_beam and os.path.exists(pbfile) == False:
            np.save(pbfile, np.array(jones_array, dtype="object"))
            if verbose:
                print(f"Beam saved in: {pbfile}")
    if apply_parang:
        # This is to account B'=P(X)BP(-X) parallactic trasnform on brightness
        # matrix
        jones_array = apply_parallactic_rotation(jones_array, p_angle).T
    jones_array = jones_array.reshape(jones_array.shape[0], jones_array.shape[1], 2, 2)
    if verbose:
        print(f"Beam calculated in : {round(time.time()-start_time,1)}s")
    return jones_array


def get_pbcor_image(
    image_file,
    pbdir,
    pbcor_dir,
    save_beam=True,
    band="",
    apply_parang=True,
    n_cpu=8,
    verbose=False,
):
    """
    Get primary beam corrected image

    Parameters
    ----------
    image_file : str
        Image file name
    pbdir : str
        Primary beam directory
    pbcor_dir : str
        Primary beam corrected image directory
    save_beam : bool, optional
        Save the beam for the image
    band : str, optional
        Band name
    apply_parang : bool, optional
        Apply parallactic correction
    n_cpu : int, optional
        Number of CPU threads to use
    verbose : bool, optional
        Verbose output

    Returns
    -------
    str
        Primary beam corrected image
    """
    try:
        image_file = image_file.rstrip("/")
        print(f"Correcting beam for image: {os.path.basename(image_file)}...")
        beam = get_image_beam(
            image_file,
            pbdir,
            save_beam=save_beam,
            band=band,
            apply_parang=apply_parang,
            n_cpu=int(n_cpu),
            verbose=verbose,
        )
        if not isinstance(beam, np.ndarray):
            print(f"Error in correct beam for image: {os.path.basename(image_file)}")
            return
        det = beam[..., 0, 0] * beam[..., 1, 1] - beam[..., 0, 1] * beam[..., 1, 0]
        inv_beam = np.empty_like(beam, dtype=np.complex64)
        inv_beam[..., 0, 0] = beam[..., 1, 1] / det
        inv_beam[..., 0, 1] = -beam[..., 0, 1] / det
        inv_beam[..., 1, 0] = -beam[..., 1, 0] / det
        inv_beam[..., 1, 1] = beam[..., 0, 0] / det
        beam_H = np.conj(np.swapaxes(beam, -1, -2))
        del beam
        det = (
            beam_H[..., 0, 0] * beam_H[..., 1, 1]
            - beam_H[..., 0, 1] * beam_H[..., 1, 0]
        )
        inv_beam_H = np.empty_like(beam_H, dtype=np.complex64)
        inv_beam_H[..., 0, 0] = beam_H[..., 1, 1] / det
        inv_beam_H[..., 0, 1] = -beam_H[..., 0, 1] / det
        inv_beam_H[..., 1, 0] = -beam_H[..., 1, 0] / det
        inv_beam_H[..., 1, 1] = beam_H[..., 0, 0] / det
        del beam_H
        image_stokes = get_IQUV(image_file)
        B_matrix = get_brightness(image_stokes)
        del image_stokes
        B_tmp = np.matmul(B_matrix, inv_beam_H)
        del inv_beam_H
        B_cor = np.matmul(inv_beam, B_tmp)
        del B_tmp, inv_beam
        B_cor = np.transpose(B_cor, (2, 3, 1, 0))
        pbcor_stokes = make_stokes(B_cor)
        del B_cor
        #################################
        pbcor_file = (
            pbcor_dir
            + "/"
            + os.path.basename(image_file).split(".fits")[0]
            + "_pbcor.fits"
        )
        header = fits.getheader(image_file)
        pbcor_file = put_IQUV(pbcor_file, pbcor_stokes, header)
        return pbcor_file
    except Exception as e:
        traceback.print_exc()
        return


def main(
    imagename,
    pbdir="",
    pbcor_dir="",
    save_beam=True,
    band="",
    apply_parang=True,
    verbose=False,
    ncpu=1,
    jobid=0,
):
    """
    Single image primary beam correction for MeerKAT

    Parameters
    ----------
    imagename : str
        Image name
    pbdir : str, optional
        Primary beam directory
    pbcor_dir : str, optional
        Primary beam corrected image directory
    save_beam : bool, optional
        Save primary beams for later use
    band : str, optional
        MeerKAT band name
    apply_parang : bool, optional
        Applt parallactic correction
    verbose : bool, optional
        Verbose output
    ncpu : int, optional
        Number of CPUs to use
    jobid : str, optional
        Job ID

    Returns
    -------
    int
        Success message
    """
    pid = os.getpid()
    cachedir = get_cachedir()
    save_pid(pid, f"{cachedir}/pids/pids_{jobid}.txt")

    try:
        if imagename and os.path.exists(imagename):
            if pbdir == "":
                print("Provide an existing directory name in pbdir.")
                msg = 1
            else:
                os.makedirs(pbdir, exist_ok=True)
                if pbcor_dir == "":
                    pbcor_dir = pbdir
                os.makedirs(pbcor_dir, exist_ok=True)
                pbcor_image = get_pbcor_image(
                    imagename,
                    pbdir,
                    pbcor_dir,
                    band=band,
                    apply_parang=apply_parang,
                    save_beam=save_beam,
                    n_cpu=int(ncpu),
                    verbose=verbose,
                )
                if pbcor_image is None or not os.path.exists(pbcor_image):
                    msg = 1
                    print(f"Primary beam correction is not successful")
                else:
                    msg = 0
                    print(f"Primary beam corrected image: {pbcor_image}")
        else:
            print("Please provide correct image name.")
            msg = 1
    except Exception as e:
        traceback.print_exc()
        msg = 1
    return msg


def cli():
    parser = argparse.ArgumentParser(
        description="Correct image for full-polar antenna averaged MeerKAT primary beam",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "imagename", type=str, help="Name of image (required positional argument)"
    )
    basic_args.add_argument(
        "--pbdir",
        type=str,
        default="",
        help="Name of primary beam directory",
    )
    basic_args.add_argument(
        "--pbcor_dir",
        type=str,
        default="",
        help="Name of primary beam corrected image directory",
    )

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--no_save_beam",
        action="store_false",
        dest="save_beam",
        help="Do not save beam to disk",
    )
    adv_args.add_argument("--band", type=str, default="", help="Band name")
    adv_args.add_argument(
        "--no_apply_parang",
        action="store_false",
        dest="apply_parang",
        help="Do not apply parallactic angle correction",
    )
    adv_args.add_argument("--verbose", action="store_true", help="Verbose output")

    # Resource management parameters
    hard_args = parser.add_argument_group(
        "###################\nHardware resource management parameters\n###################"
    )
    hard_args.add_argument(
        "--ncpu",
        type=int,
        default=1,
        help="Number of CPU threads to use",
        metavar="Integer",
    )
    hard_args.add_argument("--jobid", type=int, default=0, help="Job ID")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    msg = main(
        imagename=args.imagename,
        pbdir=args.pbdir,
        pbcor_dir=args.pbcor_dir,
        save_beam=args.save_beam,
        band=args.band,
        apply_parang=args.apply_parang,
        verbose=args.verbose,
        ncpu=args.ncpu,
        jobid=args.jobid,
    )
    return msg


if __name__ == "__main__":
    result = cli()
    os._exit(result)
