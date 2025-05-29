import numpy as np, copy, psutil, os, astropy.units as u, warnings, gc, traceback, time
from astropy.io import fits
from numpy.linalg import inv
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from scipy.interpolate import RectBivariateSpline
from dask import delayed, compute
from joblib import Parallel, delayed
from astropy.wcs import FITSFixedWarning
from meersolar.pipeline.basic_func import get_datadir
from optparse import OptionParser

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
        Stokes parameters
    """
    data = fits.getdata(filename).astype("float32")
    header = fits.getheader(filename)
    if header["CTYPE3"] == "STOKES":
        stokesaxis = 3
    elif header["CTYPE4"] == "STOKES":
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
        Stokes parameters
    Returns
    -------
    dict
        Stokes parameters
    """
    if header["CTYPE3"] == "STOKES":
        stokesaxis = 3
    elif header["CTYPE4"] == "STOKES":
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
    if hdr["CTYPE3"] == "FREQ":
        freq = hdr["CRVAL3"]
        delfreq = hdr["CDELT3"]
    elif hdr["CTYPE4"] == "FREQ":
        freq = hdr["CRVAL4"]
        delfreq = hdr["CDELT4"]
    else:
        print("No frequency axis in image.")
        return
    freq1 = (freq - (delfreq / 2)) / 10**6  # In MHz
    freq2 = (freq + (delfreq / 2)) / 10**6  # In MHz
    if band == "":
        try:
            band = hdr["BAND"]
        except:
            if freq1 >= 544 and freq2 <= 1088:  # UHF band
                band = "U"
            elif freq1 >= 856 and freq2 <= 1712:  # L band
                band = "L"
            else:
                print("Image is not in UHF or L-band.")
                return
    if band == "U":
        beam_data = np.load(datadir + "/MeerKAT_antavg_Uband.npz", mmap_mode="r")
    elif band == "L":
        beam_data = np.load(datadir + "/MeerKAT_antavg_Lband.npz", mmap_mode="r")
    else:
        print("Image is not in UHF or L-band.")
        return
    freqs = beam_data["freqs"]
    coords = np.deg2rad(
        beam_data["coords"]
    )  # It is done as l,m values were converted into degree
    pos = np.where((freq >= freq1) & (freqs <= freq1))[0]
    beam = beam_data["beams"][:, pos, ...].mean(1)
    beam = beam.astype("complex64")
    del beam_data, freqs
    gc.collect()
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
    return np.rad2deg(p)


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
    Apply left-side parallactic rotation: J' = R(-p_angle) · J
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
    s = np.sin(p_angle)
    j00, j01, j10, j11 = jones
    # R(-chi) · J
    jj00 = c * j00 + s * j10
    jj01 = c * j01 + s * j11
    jj10 = -s * j00 + c * j10
    jj11 = -s * j01 + c * j11
    return np.stack([jj00, jj01, jj10, jj11], axis=0).astype("complex64")


def get_image_beam(image_file, pbdir, save_beam=True, band="", n_cpu=8, verbose =False):
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
        print("No frequency axis in image.")
        return
    freq = round(freq / 10**6, 1)  # In MHz
    pbfile = f"{pbdir}/freq_{freq}_pb.npy"
    #######################################
    # If beam file exists
    #######################################
    if os.path.exists(pbfile):
        if verbose:
            print(f"Loading beam from: {pbfile}")
        jones_array = np.load(pbfile, allow_pickle=True)
        if verbose:
            print(f"Beam calculated in : {round(time.time()-start_time,1)}s")
        return jones_array
    obs_time = header["DATE-OBS"]
    ra0, dec0 = get_pointingcenter_radec(image_file)  # Phase center
    ra_grid, dec_grid = get_radec_grid(image_file)  # RA DEC grid
    l_grid, m_grid = radec_to_lm(ra_grid, dec_grid, ra0, dec0)
    p_angle = get_parallactic_angle(
        obs_time, ra0, dec0
    )  # Parallactic angle of the center
    ############################
    # Load beam
    ############################
    beam_results = load_beam(image_file, band=band)
    if beam_results == None:
        return
    lm_coords, beam = beam_results
    j00_r, j00_i, j01_r, j01_i, j10_r, j10_i, j11_r, j11_i = get_beam_interpolator(
        beam, lm_coords
    )
    l_grid_flat = l_grid.flatten()
    m_grid_flat = m_grid.flatten()
    grid_shape = l_grid.shape
    del l_grid, m_grid
    gc.collect()
    with Parallel(n_jobs=n_cpu, backend="threading") as parallel:
        results = parallel(
            [
                delayed(j00_r)(l_grid_flat, m_grid_flat, grid=False),
                delayed(j00_i)(l_grid_flat, m_grid_flat, grid=False),
                delayed(j01_r)(l_grid_flat, m_grid_flat, grid=False),
                delayed(j01_i)(l_grid_flat, m_grid_flat, grid=False),
                delayed(j10_r)(l_grid_flat, m_grid_flat, grid=False),
                delayed(j10_i)(l_grid_flat, m_grid_flat, grid=False),
                delayed(j11_r)(l_grid_flat, m_grid_flat, grid=False),
                delayed(j11_i)(l_grid_flat, m_grid_flat, grid=False),
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
    gc.collect()
    jones_array = apply_parallactic_rotation(
        jones_array, p_angle
    ).T  # This is to account B'=P(-X)BP(X) parallactic trasnform on brightness matrix
    jones_array = jones_array.reshape(jones_array.shape[0], jones_array.shape[1], 2, 2)
    gc.collect()
    if save_beam and os.path.exists(pbfile) == False:
        np.save(pbfile, jones_array)
        if verbose:
            print(f"Beam saved in: {pbfile}")
    if verbose:
        print(f"Beam calculated in : {round(time.time()-start_time,1)}s")
    return jones_array


def get_pbcor_image(image_file, pbdir, pbcor_dir, save_beam=True, band="", n_cpu=8, verbose=False):
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
            image_file, pbdir, save_beam=save_beam, band=band, n_cpu=n_cpu, verbose=verbose,
        )
        if type(beam) != np.ndarray:
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
        gc.collect()
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
        gc.collect()
        image_stokes = get_IQUV(image_file)
        B_matrix = get_brightness(image_stokes)
        del image_stokes
        gc.collect()
        B_tmp = np.matmul(B_matrix, inv_beam_H)
        del inv_beam_H
        gc.collect()
        B_cor = np.matmul(inv_beam, B_tmp)
        del B_tmp, inv_beam
        gc.collect()
        B_cor = np.transpose(B_cor, (2, 3, 1, 0))
        pbcor_stokes = make_stokes(B_cor)
        del B_cor
        gc.collect()
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
        gc.collect()
        return


def main():
    usage = "Correct image for full-polar antenna averaged MeerKAT primary beam"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--imagename",
        dest="imagename",
        default="",
        help="Name of image",
        metavar="String",
    )
    parser.add_option(
        "--pbdir",
        dest="pbdir",
        default="",
        help="Name of primary beam directory",
        metavar="String",
    )
    parser.add_option(
        "--pbcor_dir",
        dest="pbcor_dir",
        default="",
        help="Name of primary beam corrected image directory",
        metavar="String",
    )
    parser.add_option(
        "--save_beam",
        dest="save_beam",
        default=True,
        help="Save beam to disk or not",
        metavar="Boolean",
    )
    parser.add_option(
        "--band",
        dest="band",
        default="",
        help="Band name",
        metavar="String",
    )
    parser.add_option(
        "--verbose",
        dest="verbose",
        default=False,
        help="Verbose output",
        metavar="Boolean",
    )
    parser.add_option(
        "--ncpu",
        dest="ncpu",
        default=8,
        help="Number of CPU threads to use",
        metavar="Integer",
    )
    (options, args) = parser.parse_args()
    if options.imagename != "" and os.path.exists(options.imagename):
        try:
            if options.pbdir == "":
                print("Provide existing work directory name.")
                return 1
            os.makedirs(options.pbdir, exist_ok=True)
            if options.pbcor_dir == "":
                pbcor_dir = options.pbdir
            else:
                pbcor_dir = options.pbcor_dir
            os.makedirs(pbcor_dir, exist_ok=True)
            pbcor_image = get_pbcor_image(
                options.imagename,
                options.pbdir,
                pbcor_dir,
                band=options.band,
                save_beam=eval(str(options.save_beam)),
                n_cpu=int(options.ncpu),
                verbose=eval(str(options.verbose)),
            )
            if pbcor_image == None or os.path.exists(pbcor_image) == False:
                msg = 1
                print(f"Primary beam correction is not successful")
            else:
                msg = 0
                print(f"Primary beam corrected image: {pbcor_image}")
            return msg
        except Exception as e:
            traceback.print_exc()
            return 1
    else:
        print("Please provide correct image name.\n")
        return 1


if __name__ == "__main__":
    result = main()
    os._exit(result)
