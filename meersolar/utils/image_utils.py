import types
import numpy as np
import traceback
import warnings
import copy
import glob
import os
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
from .basic_utils import *
from .udocker_utils import *

warnings.simplefilter("ignore", category=FITSFixedWarning)


##########################
# Image analysis related
##########################
def create_circular_mask(msname, cellsize, imsize, mask_radius=20):
    """
    Create fits solar mask

    Parameters
    ----------
    msname : str
        Name of the measurement set
    cellsize : float
        Cell size in arcsec
    imsize : int
        Imsize in number of pixels
    mask_radius : float
        Mask radius in arcmin

    Returns
    -------
    str
        Fits mask file name
    """
    try:
        msname = msname.rstrip("/")
        imagename_prefix = (
            os.path.dirname(os.path.abspath(msname))
            + "/"
            + os.path.basename(msname).split(".ms")[0]
            + "_solar"
        )
        wsclean_args = [
            "-quiet",
            "-scale " + str(cellsize) + "asec",
            "-size " + str(imsize) + " " + str(imsize),
            "-nwlayers 1",
            "-niter 0 -name " + imagename_prefix,
            "-channel-range 0 1",
            "-interval 0 1",
        ]
        wsclean_cmd = "wsclean " + " ".join(wsclean_args) + " " + msname
        msg = run_wsclean(wsclean_cmd, "solarwsclean", verbose=False)
        if msg == 0:
            center = (int(imsize / 2), int(imsize / 2))
            radius = mask_radius * 60 / cellsize
            Y, X = np.ogrid[:imsize, :imsize]
            dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
            mask = dist_from_center <= radius
            os.system(
                "cp -r "
                + imagename_prefix
                + "-image.fits mask-"
                + os.path.basename(imagename_prefix)
                + ".fits"
            )
            os.system("rm -rf " + imagename_prefix + "*")
            data = fits.getdata("mask-" + os.path.basename(imagename_prefix) + ".fits")
            header = fits.getheader(
                "mask-" + os.path.basename(imagename_prefix) + ".fits"
            )
            data[0, 0, ...][mask] = 1.0
            data[0, 0, ...][~mask] = 0.0
            fits.writeto(
                imagename_prefix + "-mask.fits",
                data=data,
                header=header,
                overwrite=True,
            )
            os.system("rm -rf mask-" + os.path.basename(imagename_prefix) + ".fits")
            if os.path.exists(imagename_prefix + "-mask.fits"):
                return imagename_prefix + "-mask.fits"
            else:
                print("Circular mask could not be created.")
                return
        else:
            print("Circular mask could not be created.")
            return
    except Exception as e:
        traceback.print_exc()
        return


def create_circular_mask_array(data, radius):
    """
    Creating circular mask of a Numpy array

    Parameters
    ----------
    data : numpy.array
        2D numpy array
    radius : int
        Radius in pixels

    Returns
    -------
    numpy.array
        Mask array
    """
    shape = data.shape
    center = (shape[0] // 2, shape[1] // 2)
    Y, X = np.ogrid[: shape[0], : shape[1]]
    dist_from_center = (X - center[1]) ** 2 + (Y - center[0]) ** 2
    mask = dist_from_center <= radius**2
    return mask


def calc_solar_image_stat(imagename, disc_size=18):
    """
    Calculate solar image dynamic range

    Parameters
    ----------
    imagename : str
        Fits image name
    disc_size : float, optional
        Solar disc size in arcmin (default : 18)

    Returns
    -------
    float
        Maximum value
    float
        Minimum value
    float
        RMS values
    float
        Total value
    float
        Mean value
    float
        Median value
    float
        RMS dynamic range
    float
        Min-max dynamic range
    """
    data = fits.getdata(imagename)
    header = fits.getheader(imagename)
    pix_size = abs(header["CDELT1"]) * 3600.0  # In arcsec
    radius = int((disc_size * 60) / pix_size)
    if len(data.shape) > 2:
        data = data[0, 0, ...]
    mask = create_circular_mask_array(data, radius)
    masked_data = copy.deepcopy(data)
    masked_data[mask] = np.nan
    unmasked_data = copy.deepcopy(data)
    unmasked_data[~mask] = np.nan
    maxval = float(np.nanmax(unmasked_data))
    minval = float(np.nanmin(masked_data))
    rms = float(np.nanstd(masked_data))
    total_val = float(np.nansum(unmasked_data))
    rms_dyn = float(maxval / rms)
    minmax_dyn = float(maxval / abs(minval))
    mean_val = float(np.nanmean(unmasked_data))
    median_val = float(np.nanmedian(unmasked_data))
    del data, mask, unmasked_data, masked_data
    return (
        round(maxval, 2),
        round(minval, 2),
        round(rms, 2),
        round(total_val, 2),
        round(mean_val, 2),
        round(median_val, 2),
        round(rms_dyn, 2),
        round(minmax_dyn, 2),
    )


def calc_dyn_range(imagename, modelname, residualname, fits_mask=""):
    """
    Calculate dynamic ranges.


    Parameters
    ----------
    imagename : list or str
        Image FITS file(s)
    modelname : list or str
        Model FITS file(s)
    residualname : list ot str
        Residual FITS file(s)
    fits_mask : str, optional
        FITS file mask

    Returns
    -------
    model_flux : float
        Total model flux.
    dyn_range_rms : float
        Max/RMS dynamic range.
    rms : float
        RMS of the image
    """

    def load_data(name):
        return fits.getdata(name)

    def to_list(x):
        return [x] if isinstance(x, str) else x

    imagename = to_list(imagename)
    modelname = to_list(modelname)
    residualname = to_list(residualname)

    use_mask = bool(fits_mask and os.path.exists(fits_mask))
    mask_data = fits.getdata(fits_mask).astype(bool) if use_mask else None

    model_flux, dr1, rmsvalue = 0, 0, 0

    for i in range(len(imagename)):
        img = imagename[i]
        res = residualname[i]
        image = load_data(img)
        residual = load_data(res)
        rms = np.nanstd(residual)
        if use_mask:
            maxval = np.nanmax(image[mask_data])
        else:
            maxval = np.nanmax(image)
        dr1 += maxval / rms if rms else 0
        rmsvalue += rms

    for mod in modelname:
        model = load_data(mod)
        model_flux += np.nansum(model[mask_data] if use_mask else model)

    rmsvalue = rmsvalue / np.sqrt(len(residualname))
    return float(model_flux), round(float(dr1), 2), round(float(rmsvalue), 2)


def generate_tb_map(imagename, outfile=""):
    """
    Function to generate brightness temperature map

    Parameters
    ----------
    imagename : str
        Name of the flux calibrated image
    outfile : str, optional
        Output brightess temperature image name

    Returns
    -------
    str
        Output image name
    """
    if outfile == "":
        outfile = imagename.split(".fits")[0] + "_TB.fits"
    image_header = fits.getheader(imagename)
    image_data = fits.getdata(imagename)
    major = float(image_header["BMAJ"]) * 3600.0  # In arcsec
    minor = float(image_header["BMIN"]) * 3600.0  # In arcsec
    if image_header["CTYPE3"] == "FREQ":
        freq = image_header["CRVAL3"] / 10**9  # In GHz
    elif image_header["CTYPE4"] == "FREQ":
        freq = image_header["CRVAL4"] / 10**9  # In GHz
    else:
        print(f"No frequency information is present in header for {imagename}.")
        return
    TB_conv_factor = (1.222e6) / ((freq**2) * major * minor)
    TB_data = image_data * TB_conv_factor
    image_header["BUNIT"] = "K"
    fits.writeto(outfile, data=TB_data, header=image_header, overwrite=True)
    return outfile


def cutout_image(fits_file, output_file, x_deg=2):
    """
    Cutout central part of the image

    Parameters
    ----------
    fits_file : str
        Input fits file
    output_file : str
        Output fits file name (If same as input, input image will be overwritten)
    x_deg : float, optional
        Size of the output image in degree

    Returns
    -------
    str
        Output image name
    """
    hdu = fits.open(fits_file)[0]
    data = hdu.data  # shape: (nfreq, nstokes, ny, nx)
    header = hdu.header
    wcs = WCS(header)
    _, _, ny, nx = data.shape
    center_x, center_y = nx // 2, ny // 2
    # Get pixel scale (deg/pixel)
    pix_scale_deg = np.abs(header["CDELT1"])
    x_pix = int((x_deg / pix_scale_deg) / 2)
    # Adjust if cutout size exceeds image size
    max_half_x = nx // 2
    max_half_y = ny // 2
    x_pix = min(x_pix, max_half_x)
    y_pix = min(x_pix, max_half_y)  # Assume square pixels
    # Define slice indices
    x0 = center_x - x_pix
    x1 = center_x + x_pix
    y0 = center_y - y_pix
    y1 = center_y + y_pix
    # Slice data
    cutout_data = data[:, :, y0:y1, x0:x1]
    # Update header
    new_header = header.copy()
    new_header["NAXIS1"] = x1 - x0
    new_header["NAXIS2"] = y1 - y0
    new_header["CRPIX1"] -= x0
    new_header["CRPIX2"] -= y0
    # Save
    fits.writeto(output_file, cutout_data, header=new_header, overwrite=True)
    return output_file


def make_timeavg_image(wsclean_images, outfile_name, keep_wsclean_images=True):
    """
    Convert WSClean images into a time averaged image

    Parameters
    ----------
    wsclean_images : list
        List of WSClean images.
    outfile_name : str
        Name of the output file.
    keep_wsclean_images : bool, optional
        Whether to retain the original WSClean images (default: True).

    Returns
    -------
    str
        Output image name.
    """
    timestamps = []
    for i in range(len(wsclean_images)):
        image = wsclean_images[i]
        if i == 0:
            data = fits.getdata(image)
        else:
            data += fits.getdata(image)
        timestamps.append(fits.getheader(image)["DATE-OBS"])
    data /= len(wsclean_images)
    avg_timestamp = average_timestamp(timestamps)
    header = fits.getheader(wsclean_images[0])
    header["DATE-OBS"] = avg_timestamp
    fits.writeto(outfile_name, data=data, header=header, overwrite=True)
    if not keep_wsclean_images:
        for img in wsclean_images:
            os.system(f"rm -rf {img}")
    return outfile_name


def make_freqavg_image(wsclean_images, outfile_name, keep_wsclean_images=True):
    """
    Convert WSClean images into a frequency averaged image

    Parameters
    ----------
    wsclean_images : list
        List of WSClean images.
    outfile_name : str
        Name of the output file.
    keep_wsclean_images : bool, optional
        Whether to retain the original WSClean images (default: True).

    Returns
    -------
    str
        Output image name.
    """
    freqs = []
    for i in range(len(wsclean_images)):
        image = wsclean_images[i]
        if i == 0:
            data = fits.getdata(image)
        else:
            data += fits.getdata(image)
        header = fits.getheader(image)
        if header["CTYPE3"] == "FREQ":
            freqs.append(float(header["CRVAL3"]))
            freqaxis = 3
        elif header["CTYPE4"] == "FREQ":
            freqs.append(float(header["CRVAL4"]))
            freqaxis = 4
    data /= len(wsclean_images)
    if len(freqs) > 0:
        mean_freq = np.nanmean(freqs)
        width = max(freqs) - min(freqs)
        header = fits.getheader(wsclean_images[0])
        if freqaxis == 3:
            header["CRAVL3"] = mean_freq
            header["CDELT3"] = width
        elif freqaxis == 4:
            header["CRAVL4"] = mean_freq
            header["CDELT4"] = width
    fits.writeto(outfile_name, data=data, header=header, overwrite=True)
    if not keep_wsclean_images:
        for img in wsclean_images:
            os.system(f"rm -rf {img}")
    return outfile_name


def make_stokes_wsclean_imagecube(
    wsclean_images, outfile_name, keep_wsclean_images=True
):
    """
    Convert WSClean images into a Stokes cube image.

    Parameters
    ----------
    wsclean_images : list
        List of WSClean images.
    outfile_name : str
        Name of the output file.
    keep_wsclean_images : bool, optional
        Whether to retain the original WSClean images (default: True).

    Returns
    -------
    str
        Output image name.
    """
    stokes = sorted(
        set(
            (
                os.path.basename(i).split(".fits")[0].split(" - ")[-2]
                if " - " in i
                else "I"
            )
            for i in wsclean_images
        )
    )
    valid_stokes = [
        {"I"},
        {"I", "V"},
        {"I", "Q", "U", "V"},
        {"XX", "YY"},
        {"LL", "RR"},
        {"Q", "U"},
        {"I", "Q"},
    ]
    if set(stokes) not in valid_stokes:
        print("Invalid Stokes combination.")
        return
    imagename_prefix = "temp_" + os.path.basename(wsclean_images[0]).split(" - I")[0]
    imagename = imagename_prefix + ".image"
    data, header = fits.getdata(wsclean_images[0]), fits.getheader(wsclean_images[0])
    for img in wsclean_images[1:]:
        data = np.append(data, fits.getdata(img), axis=0)
    header.update(
        {"NAXIS4": len(stokes), "CRVAL4": 1 if "I" in stokes else -5, "CDELT4": 1}
    )
    temp_fits = imagename_prefix + ".fits"
    fits.writeto(outfile_name, data=data, header=header, overwrite=True)
    if not keep_wsclean_images:
        for img in wsclean_images:
            os.system(f"rm -rf {img}")
    return outfile_name


# Expose functions and classes
__all__ = [
    name
    for name, obj in globals().items()
    if (
        (isinstance(obj, types.FunctionType) or isinstance(obj, type))
        and obj.__module__ == __name__
    )
]
