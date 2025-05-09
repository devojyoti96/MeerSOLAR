import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.transform import resize
from scipy.ndimage import rotate
import glob
import multiprocessing as mp
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, FK5, SkyCoord
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# Define MeerKAT location
meerkat = EarthLocation(lat=-30.7133 * u.deg, lon=21.4429 * u.deg, height=1086.6 * u.m)

def create_stokes_from_image(files):
    """
    Only for fits images with seperate stokes files
    """
    stokes = {}
    for file in files:
        fname = file.split("-")
        data = fits.getdata(file)
        data = data[0, 0]
        stokes[f"{fname[-2]}"] = data
    return stokes


def get_brightness(stokes):
    """
    Returns brightness matrix from stokes dictionary
    """
    I = stokes["I"]
    Q = stokes["Q"]
    U = stokes["U"]
    V = stokes["V"]
    XX = I + Q
    XY = U + 1.0j * V
    YX = U - 1.0j * U
    YY = I - Q
    return ((XX, XY), (YX, YY))


def make_stokes(b):
    """
    Makes stokes images from brightness matrix
    """
    XX = b[:, :, 0, 0]
    XY = b[:, :, 0, 1]
    YX = b[:, :, 1, 0]
    YY = b[:, :, 1, 1]
    I = np.real(XX + YY) / 2
    Q = np.real(YY - XX) / 2
    U = np.real(XY + YX) / 2
    V = np.real(YX - XY) / 2
    return I, Q, U, V


def create_jones_inv(beam):
    """
    Creates inverse of jones matrix for a given jones from beam file
    """
    HH = beam[0]
    HV = beam[1]
    VH = beam[2]
    VV = beam[3]
    jones = ((HH, HV), (VH, VV))
    return np.linalg.inv(jones)


def extract_ra_dec_crvals(fits_header):
    ra_crval = None
    dec_crval = None
    for i in range(1, 10):
        ctype = fits_header.get(f"CTYPE{i}", None)
        if ctype is None:
            continue
        if "RA" in ctype:
            ra_crval = fits_header.get(f"CRVAL{i}", None)
            ra_arr = fits_header.get(f"NAXIS{i}", None)
            ra = [
                (
                    fits_header[f"CRVAL{i}"]
                    + fits_header[f"CDELT{i}"] * (0 - fits_header[f"CRPIX{i}"])
                ),
                (
                    fits_header[f"CRVAL{i}"]
                    + fits_header[f"CDELT{i}"]
                    * (fits_header[f"NAXIS{i}"] - fits_header[f"CRPIX{i}"])
                ),
            ]
        elif "DEC" in ctype:
            dec_crval = fits_header.get(f"CRVAL{i}", None)
            dec_arr = fits_header.get(f"NAXIS{i}", None)
            dec = [
                (
                    fits_header[f"CRVAL{i}"]
                    + fits_header[f"CDELT{i}"] * (0 - fits_header[f"CRPIX{i}"])
                ),
                (
                    fits_header[f"CRVAL{i}"]
                    + fits_header[f"CDELT{i}"]
                    * (fits_header[f"NAXIS{i}"] - fits_header[f"CRPIX{i}"])
                ),
            ]

    return (
        ra_crval * u.deg,
        dec_crval * u.deg,
        ra_arr,
        dec_arr,
        np.array(ra),
        np.array(dec),
    )


def get_image_boundary(fitsimage, jones):
    """
    Converts the Jones matrices to RA-DEC and crops it according to image size.
    """
    header = fits.getheader(fitsimage)
    obs_time = Time(header["DATE-OBS"])  # Replace with actual timestamps
    ra0, dec0, ra_arr, dec_arr, ra, dec = extract_ra_dec_crvals(header)
    RA_min = ra.min() * u.deg
    RA_max = ra.max() * u.deg
    DEC_min = dec.min() * u.deg
    DEC_max = dec.max() * u.deg
    # phase_center = FK5(ra=ra0, dec=dec0)
    # Compute parallactic angles for each frame
    # parallactic_angle = phase_center.transform_to(AltAz(obstime=obs_time, location=meerkat)).parallactic_angle().to(u.deg).value
    # Compute Local Sidereal Time (LST)
    lst = obs_time.sidereal_time("mean", longitude=meerkat.lon)
    lst_deg = lst.to(u.deg)

    # Compute Hour Angle (HA = LST - RA)
    hour_angle = (lst_deg - ra0).wrap_at(180 * u.deg)  # Wrap within [-180, 180] degrees

    # Compute parallactic angle using the correct formula
    parallactic_angle = (
        np.arctan2(
            np.sin(hour_angle),
            np.cos(dec0) * np.tan(meerkat.lat) - np.sin(dec0) * np.cos(hour_angle),
        )
        .to(u.deg)
        .value
    )
    # print('Parallactic angle: ',parallactic_angle)
    # Here we generate a similar RA/DEC grid for I_lm corresponding to its pixel positions
    # For simplicity, assuming I_lm is centered at the same RA/DEC as the phase center
    l_range = np.linspace(-6.0, 6.0, 128) * np.pi / 180  # Example range for l
    m_range = np.linspace(-6.0, 6.0, 128) * np.pi / 180  # Example range for m
    l_grid, m_grid = np.meshgrid(l_range, m_range)
    # Create empty array for rotated images
    jones_rotated = np.empty_like(jones)

    # Rotate each frame by parallactic angle
    for i in range(2):
        for j in range(2):
            jones_rotated[i, j] = rotate(
                jones[i, j, :, :], -parallactic_angle, reshape=False, order=3
            )
        # Step 1: Rotate the l,m grid by the parallactic angle
    rotation_matrix = np.array(
        [
            [
                np.cos(np.radians(parallactic_angle)),
                -np.sin(np.radians(parallactic_angle)),
            ],
            [
                np.sin(np.radians(parallactic_angle)),
                np.cos(np.radians(parallactic_angle)),
            ],
        ]
    )
    # Apply rotation to the l, m grid
    l_m_coords = np.vstack([l_grid.flatten(), m_grid.flatten()])
    rotated_l_m = np.dot(rotation_matrix, l_m_coords).reshape(2, 128, 128)

    rotated_l_grid = rotated_l_m[0, :, :]
    rotated_m_grid = rotated_l_m[1, :, :]

    # Create a grid of RA, DEC corresponding to I_radec pixel positions
    ra_grid, dec_grid = np.meshgrid(
        np.linspace(RA_min, RA_max, ra_arr), np.linspace(DEC_min, DEC_max, dec_arr)
    )

    ra_lm_grid = ra0 + np.rad2deg(
        np.arctan2(rotated_l_grid, np.cos(dec0) - rotated_m_grid * np.sin(dec0))
    )
    dec_lm_grid = np.rad2deg(np.arcsin(rotated_m_grid * np.cos(dec0) + np.sin(dec0)))
    # Now, we need to determine the overlap between the RA/DEC ranges of I_lm and I_radec
    # Find the indices in I_lm corresponding to the RA/DEC bounds of I_radec
    RA_min_lm, RA_max_lm = np.min(ra_lm_grid), np.max(ra_lm_grid)
    DEC_min_lm, DEC_max_lm = np.min(dec_lm_grid), np.max(dec_lm_grid)

    # Find the corresponding pixel indices for cropping
    indices = np.where(
        (ra_lm_grid >= RA_min)
        & (ra_lm_grid <= RA_max)
        & (dec_lm_grid >= DEC_min)
        & (dec_lm_grid <= DEC_max)
    )[0]
    l_cropped = ra_lm_grid[
        indices.min() : indices.max(), indices.min() : indices.max()
    ]  # Crop correctly
    m_cropped = dec_lm_grid[
        indices.min() : indices.max(), indices.min() : indices.max()
    ]
    jones_cropped = jones_rotated[
        :, :, indices.min() : indices.max(), indices.min() : indices.max()
    ]

    target_x = ra_arr
    target_y = dec_arr
    return (
        target_x,
        target_y,
        RA_min_lm,
        RA_max_lm,
        DEC_min_lm,
        DEC_max_lm,
        jones_cropped,
    )


def resampled(data, target_x, target_y, RA_min_lm, RA_max_lm, DEC_min_lm, DEC_max_lm):
    # Prepare coordinate axes
    x = np.linspace(RA_min_lm, RA_max_lm, data.shape[-2])
    y = np.linspace(DEC_min_lm, DEC_max_lm, data.shape[-1])
    x_new = np.linspace(RA_min_lm, RA_max_lm, target_y)
    y_new = np.linspace(DEC_min_lm, DEC_max_lm, target_x)
    Ynew, Xnew = np.meshgrid(y_new, x_new, indexing="ij")  # shape (target_x, target_y)
    points = np.stack([Ynew.ravel(), Xnew.ravel()], axis=-1)

    # Preallocate output
    resampled_jones = np.empty((2, 2, target_x, target_y), dtype="complex")

    # Interpolate each Jones component
    for i in range(2):
        for j in range(2):
            J = data[i, j]
            interp_real = RegularGridInterpolator(
                (y, x), J.real, bounds_error=False, fill_value=0
            )
            interp_imag = RegularGridInterpolator(
                (y, x), J.imag, bounds_error=False, fill_value=0
            )

            real_resampled = interp_real(points).reshape(target_x, target_y)
            imag_resampled = interp_imag(points).reshape(target_x, target_y)
            resampled_jones[i, j] = real_resampled + 1j * imag_resampled
    XX = resampled_jones[0, 0]
    YY = resampled_jones[1, 1]
    I_beam = 0.5 * (np.abs(XX) ** 2 + np.abs(YY) ** 2)
    return resampled_jones, I_beam


def interpolate_using_resize(data, target_x, target_y):
    """
    Interpolate complex jones matrices to image resolution
    """

    def complex_resample(data):
        resampled_real = resize(
            data.real,
            (target_x, target_y),
            preserve_range=True,
            anti_aliasing=True,
            order=3,
        )
        resampled_imag = resize(
            data.imag,
            (target_x, target_y),
            preserve_range=True,
            anti_aliasing=True,
            order=3,
        )
        resampled = resampled_real + 1.0j * resampled_imag
        return resampled

    resampled_jones = np.zeros((2, 2, target_x, target_y), dtype="complex")
    for i in range(2):
        for j in range(2):
            resampled_jones[i, j] = complex_resample(data[i, j])
    XX = resampled_jones[0, 0]
    YY = resampled_jones[1, 1]
    I_beam = 0.5 * (np.abs(XX) ** 2 + np.abs(YY) ** 2)
    return resampled_jones, I_beam


def find_chan(band, fitsimage):
    """
    Finds band and channel range. Written for MeerKAT
    """
    header = fits.getheader(fitsimage)
    if header["CTYPE3"] == "FREQ":
        freq = header["CRVAL3"]
        del_freq = header["CDELT3"]
    elif header["CTYPE4"] == "FREQ":
        freq = header["CRVAL4"]
        del_freq = header["CDELT4"]
    if band == "uhf":
        del_nu = 544e6 / 1024
        chan0 = (freq - 544e6) // del_nu
        del_chan = del_freq // del_nu
    elif band == "l":
        del_nu = 856e6 / 1024
        chan0 = (freq - 856e6) // del_nu
        del_chan = del_freq // del_nu
    print(chan0, del_chan)
    return chan0 - del_chan // 2, chan0 + del_chan // 2


def load_beam(band):
    if band == "uhf":
        beam = np.load(
            "/data/dpatra/meerkat/MeerKAT_primary_beam/ant_averaged_beam.npy",
            mmap_mode="r",
        )
    elif band == "l":
        beam = np.load(
            "/data/dpatra/meerkat/MeerKAT_primary_beam/ant_averaged_beam_L_band.npy",
            mmap_mode="r",
        )
    return beam


def pbcorr(fitsfiles, band, pb_threshold=0.1):
    beam = load_beam(band)
    header = fits.getheader(fitsfiles[0])
    data = fits.getdata(fitsfiles[0])[0, 0]
    chan1, chan2 = find_chan(band, fitsfiles[0])
    stokes = create_stokes_from_image(fitsfiles)
    b = np.array(get_brightness(stokes))
    b = b.transpose(2, 3, 0, 1)
    beam = beam[:, int(chan1) : int(chan2)].mean(1)
    target_x, target_y, RA_min_lm, RA_max_lm, DEC_min_lm, DEC_max_lm, jones = (
        get_image_boundary(fitsfiles[0], beam.reshape(2, 2, 128, 128))
    )

    inv_beam = np.zeros_like(jones, dtype="complex")
    for i in range(jones.shape[-2]):
        for j in range(jones.shape[-1]):
            inv_beam[:, :, i, j] = np.linalg.inv(jones[:, :, i, j])

    # resampled_jones=interpolate_using_resize(inv_beam,target_x,target_y)
    resampled_jones, resampled_I = resampled(
        inv_beam, target_x, target_y, RA_min_lm, RA_max_lm, DEC_min_lm, DEC_max_lm
    )
    # print(resampled_jones.shape)
    resampled_jones = resampled_jones.transpose(2, 3, 0, 1)
    new_b = np.matmul(
        np.matmul(resampled_jones, b), resampled_jones.conjugate().swapaxes(2, 3)
    )
    I = np.real(new_b[:, :, 0, 0] + new_b[:, :, 1, 1]) / 2
    Q = np.real(new_b[:, :, 1, 1] - new_b[:, :, 0, 0]) / 2
    U = np.real(new_b[:, :, 0, 1] + new_b[:, :, 1, 0]) / 2
    V = np.imag(new_b[:, :, 1, 0] - new_b[:, :, 0, 1]) / 2
    threshold = np.ones_like(I)
    threshold[resampled_I < pb_threshold] = np.nan
    I = threshold * I
    Q = threshold * Q
    U = threshold * U
    V = threshold * V
    return np.array((I, Q, U, V))


def pbcorr_I(file, band, pb_threshold=0.1):
    beam = load_beam(band)
    header = fits.getheader(file)
    data = fits.getdata(file)[0, 0]
    chan1, chan2 = find_chan(band, file)
    # stokes=create_stokes_from_image(fitsfiles)
    beam = beam[:, int(chan1) : int(chan2)].mean(1)
    _x, _y, jones = get_image_boundary(file, beam.reshape(2, 2, 128, 128))
    XX = jones[0, 0]
    YY = jones[1, 1]
    I = 0.5 * (np.abs(XX) ** 2 + np.abs(YY) ** 2)
    target_x = len(_x)
    target_y = len(_y)
    resampled_I = resize(I, (target_x, target_y))
    new_I = data / resampled_I
    new_I[resampled_I < pb_threshold] = np.nan  # Avoid division by very low values
    return new_I


# files=glob.glob('test*image.fits')
# corr_image=pbcorr(files,'uhf')
# for i in range(4):
#     fname=files[i].split('.')
#     fits.writeto(fname[0]+'_pb_corrected.fits',corr_image[i],header=fits.getheader(files[i]),overwrite=True)
"""filename=input('Filename: ')
outtfilename=input('Output filename: ')
band='l'
new_I=pbcorr_I(filename,band)
fits.writeto(outtfilename,new_I,header=fits.getheader(filename),overwrite='True')"""
# plt.imshow(new_I,aspect='auto');plt.show()
