import types
import astropy.units as u
import logging
import psutil
import numpy as np
import warnings
import glob
import dask
import os
import traceback
import matplotlib
import matplotlib.pyplot as plt
from dask import delayed, compute
from multiprocessing.pool import ThreadPool
from sunpy.net import Fido, attrs as a
from sunpy.map import Map
from astropy.visualization import ImageNormalize, PowerStretch, LogStretch
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.wcs import FITSFixedWarning
from casatools import msmetadata, ms as casamstool
from datetime import datetime as dt, timedelta
from .basic_utils import *
from .proc_manage_utils import *
from .ms_metadata import *
from .meer_utils import *

warnings.simplefilter("ignore", category=FITSFixedWarning)


#################################
# Plotting related functions
#################################


def get_meermap(fits_image, band="", do_sharpen=False):
    """
    Make MeerKAT sunpy map

    Parameters
    ----------
    fits_image : str
        MeerKAT fits image
    band : str, optional
        Band name
    do_sharpen : bool, optional
        Sharpen the image

    Returns
    -------
    sunpy.map
        Sunpy map
    """
    from scipy.ndimage import gaussian_filter
    from sunpy.map import make_fitswcs_header
    from sunpy.coordinates import frames, sun

    logging.getLogger("sunpy").setLevel(logging.ERROR)

    MEERLAT = -30.7133
    MEERLON = 21.4429
    MEERALT = 1086.6
    meer_hdu = fits.open(fits_image)  # Opening MeerKAT fits file
    meer_header = meer_hdu[0].header  # meer header
    meer_data = meer_hdu[0].data
    if len(meer_data.shape) > 2:
        meer_data = meer_data[0, 0, :, :]  # meer data
    if meer_header["CTYPE3"] == "FREQ":
        frequency = meer_header["CRVAL3"] * u.Hz
    elif meer_header["CTYPE4"] == "FREQ":
        frequency = meer_header["CRVAL4"] * u.Hz
    else:
        frequency = ""
    if band == "":
        try:
            band = meer_header["BAND"]
        except BaseException:
            band = ""
    try:
        pixel_unit = meer_header["BUNIT"]
    except BaseException:
        pixel_nuit = ""
    obstime = Time(meer_header["date-obs"])
    meerpos = EarthLocation(
        lat=MEERLAT * u.deg, lon=MEERLON * u.deg, height=MEERALT * u.m
    )
    # Converting into GCRS coordinate
    meer_gcrs = SkyCoord(meerpos.get_gcrs(obstime))
    reference_coord = SkyCoord(
        meer_header["crval1"] * u.Unit(meer_header["cunit1"]),
        meer_header["crval2"] * u.Unit(meer_header["cunit2"]),
        frame="gcrs",
        obstime=obstime,
        obsgeoloc=meer_gcrs.cartesian,
        obsgeovel=meer_gcrs.velocity.to_cartesian(),
        distance=meer_gcrs.hcrs.distance,
    )
    reference_coord_arcsec = reference_coord.transform_to(
        frames.Helioprojective(observer=meer_gcrs)
    )
    cdelt1 = (np.abs(meer_header["cdelt1"]) * u.deg).to(u.arcsec)
    cdelt2 = (np.abs(meer_header["cdelt2"]) * u.deg).to(u.arcsec)
    P1 = sun.P(obstime)  # Relative rotation angle
    new_meer_header = make_fitswcs_header(
        meer_data,
        reference_coord_arcsec,
        reference_pixel=u.Quantity(
            [meer_header["crpix1"] - 1, meer_header["crpix2"] - 1] * u.pixel
        ),
        scale=u.Quantity([cdelt1, cdelt2] * u.arcsec / u.pix),
        rotation_angle=-P1,
        wavelength=frequency.to(u.MHz).round(2),
        observatory="MeerKAT",
    )
    if do_sharpen:
        blurred = gaussian_filter(meer_data, sigma=10)
        meer_data = meer_data + (meer_data - blurred)
    meer_map = Map(meer_data, new_meer_header)
    meer_map_rotate = meer_map.rotate()
    return meer_map_rotate


def save_in_hpc(fits_image, outdir="", xlim=[-1600, 1600], ylim=[-1600, 1600]):
    """
    Save solar image in helioprojective coordinates

    Parameters
    ----------
    fits_image : str
        FITS image name
    outdir : str, optional
        Output directory
    xlim : list
        X axis limit in arcsecond
    ylim : list
        Y axis limit in arcsecond

    Returns
    -------
    str
        FITS image in helioprojective coordinate
    """
    logging.getLogger("sunpy").setLevel(logging.ERROR)
    fits_header = fits.getheader(fits_image)
    meermap = get_meermap(fits_image)
    if len(xlim) == 2 and len(ylim) == 2:
        top_right = SkyCoord(
            xlim[1] * u.arcsec, ylim[1] * u.arcsec, frame=meermap.coordinate_frame
        )
        bottom_left = SkyCoord(
            xlim[0] * u.arcsec, ylim[0] * u.arcsec, frame=meermap.coordinate_frame
        )
        meermap = meermap.submap(bottom_left, top_right=top_right)
    if outdir == "":
        outdir = os.path.dirname(os.path.abspath(fits_image))
    outfile = f"{outdir}/{os.path.basename(fits_image).split('.fits')[0]}_HPC.fits"
    if os.path.exists(outfile):
        os.system(f"rm -rf {outfile}")
    meermap.save(outfile, filetype="fits")
    data = fits.getdata(outfile)
    data = data[np.newaxis, np.newaxis, ...]
    hpc_header = fits.getheader(outfile)
    for key in [
        "NAXIS",
        "NAXIS3",
        "NAXIS4",
        "BUNIT",
        "CTYPE3",
        "CRPIX3",
        "CRVAL3",
        "CDELT3",
        "CUNIT3",
        "CTYPE4",
        "CRPIX4",
        "CRVAL4",
        "CDELT4",
        "CUNIT4",
        "AUTHOR",
        "PIPELINE",
        "BAND",
        "MAX",
        "MIN",
        "RMS",
        "SUM",
        "MEAN",
        "MEDIAN",
        "RMSDYN",
        "MIMADYN",
    ]:
        if key in fits_header:
            hpc_header[key] = fits_header[key]
    fits.writeto(outfile, data=data, header=hpc_header, overwrite=True)
    return outfile


def plot_in_hpc(
    fits_image,
    draw_limb=False,
    extensions=["png"],
    outdirs=[],
    plot_range=[],
    power=0.5,
    xlim=[-1600, 1600],
    ylim=[-1600, 1600],
    contour_levels=[],
    band="",
    showgui=False,
):
    """
    Function to convert MeerKAT image into Helioprojective co-ordinate

    Parameters
    ----------
    fits_image : str
        Name of the fits image
    draw_limb : bool, optional
        Draw solar limb or not
    extensions : list, optional
        Output file extensions
    outdirs : list, optional
        Output directories for each extensions
    plot_range : list, optional
        Plot range
    power : float, optional
        Power stretch
    xlim : list
        X axis limit in arcsecond
    ylim : list
        Y axis limit in arcsecond
    contour_levels : list, optional
        Contour levels in fraction of peak, both positive and negative values allowed
    band : str, optional
        Band name
    showgui : bool, optional
        Show GUI

    Returns
    -------
    outfiles
        Saved plot file names
    sunpy.Map
        MeerKAT image in helioprojective co-ordinate
    """
    import matplotlib.ticker as ticker
    from matplotlib.patches import Ellipse, Rectangle
    from matplotlib.colors import ListedColormap
    from matplotlib import cm
    from sunpy.coordinates import sun

    logging.getLogger("sunpy").setLevel(logging.ERROR)
    if not showgui:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")
    matplotlib.rcParams.update({"font.size": 12})
    fits_image = fits_image.rstrip("/")
    meer_header = fits.getheader(fits_image)  # Opening MeerKAT fits file
    if meer_header["CTYPE3"] == "FREQ":
        frequency = meer_header["CRVAL3"] * u.Hz
    elif meer_header["CTYPE4"] == "FREQ":
        frequency = meer_header["CRVAL4"] * u.Hz
    else:
        frequency = ""
    if band == "":
        try:
            band = meer_header["BAND"]
        except BaseException:
            band = ""
    try:
        pixel_unit = meer_header["BUNIT"]
    except BaseException:
        pixel_nuit = ""
    obstime = Time(meer_header["date-obs"])
    meer_map_rotate = get_meermap(fits_image, band=band)
    top_right = SkyCoord(
        xlim[1] * u.arcsec, ylim[1] * u.arcsec, frame=meer_map_rotate.coordinate_frame
    )
    bottom_left = SkyCoord(
        xlim[0] * u.arcsec, ylim[0] * u.arcsec, frame=meer_map_rotate.coordinate_frame
    )
    cropped_map = meer_map_rotate.submap(bottom_left, top_right=top_right)
    meer_data = cropped_map.data
    if len(plot_range) < 2:
        norm = ImageNormalize(
            meer_data,
            vmin=0.03 * np.nanmax(meer_data),
            vmax=0.99 * np.nanmax(meer_data),
            stretch=PowerStretch(power),
        )
    else:
        norm = ImageNormalize(
            meer_data,
            vmin=np.nanmin(plot_range),
            vmax=np.nanmax(plot_range),
            stretch=PowerStretch(power),
        )
    if band == "U":
        cmap = "inferno"
        pos_color = "white"
        neg_color = "cyan"
    elif band == "L":
        pos_color = "hotpink"
        neg_color = "yellow"
        if "YlGnBu_inferno" not in plt.colormaps():
            # Sample YlGnBu_r colormap with 256 colors
            cmap_ylgnbu = cm.get_cmap("YlGnBu_r", 256)
            colors = cmap_ylgnbu(np.linspace(0, 1, 256))
            # Create perceptually linear spacing using inferno luminance
            cmap_inferno = cm.get_cmap("inferno", 256)
            # Sort YlGnBu colors by the inferred brightness from inferno
            luminance_ranks = np.argsort(
                np.mean(cmap_inferno(np.linspace(0, 1, 256))[:, :3], axis=1)
            )
            colors_uniform = colors[luminance_ranks]
            # New perceptual-YlGnBu-inspired colormap
            YlGnBu_inferno = ListedColormap(colors_uniform, name="YlGnBu_inferno")
            plt.colormaps.register(name="YlGnBu_inferno", cmap=YlGnBu_inferno)
        cmap = "YlGnBu_inferno"
    else:
        cmap = "cubehelix"
        pos_color = "cyan"
        neg_color = "gold"
    try:
        fig = plt.figure()
        ax = plt.subplot(projection=cropped_map)
        cropped_map.plot(norm=norm, cmap=cmap, axes=ax)
        if len(contour_levels) > 0:
            contour_levels = np.array(contour_levels)
            pos_cont = contour_levels[contour_levels >= 0]
            neg_cont = contour_levels[contour_levels < 0]
            if len(pos_cont) > 0:
                cropped_map.draw_contours(
                    np.sort(pos_cont) * np.nanmax(meer_data), colors=pos_color
                )
            if len(neg_cont) > 0:
                cropped_map.draw_contours(
                    np.sort(neg_cont) * np.nanmax(meer_data), colors=neg_color
                )
        ax.coords.grid(False)
        rgba_vmin = plt.get_cmap(cmap)(norm(norm.vmin))
        ax.set_facecolor(rgba_vmin)
        # Read synthesized beam from header
        try:
            bmaj = meer_header["BMAJ"] * u.deg.to(u.arcsec)  # in arcsec
            bmin = meer_header["BMIN"] * u.deg.to(u.arcsec)
            bpa = meer_header["BPA"] - sun.P(obstime).deg  # in degrees
        except KeyError:
            bmaj = bmin = bpa = None
        # Plot PSF ellipse in bottom-left if all values are present
        if bmaj and bmin and bpa is not None:
            # Coordinates where to place the beam (e.g., 5% above bottom-left
            # corner)
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()

            beam_center = SkyCoord(
                x0 + 0.08 * (x1 - x0),
                y0 + 0.08 * (y1 - y0),
                unit=u.arcsec,
                frame=cropped_map.coordinate_frame,
            )

            # Add ellipse patch
            beam_ellipse = Ellipse(
                (beam_center.Tx.value, beam_center.Ty.value),  # center in arcsec
                width=bmin,
                height=bmaj,
                angle=bpa,
                edgecolor="white",
                facecolor="white",
                lw=1,
            )
            ax.add_patch(beam_ellipse)
            # Draw square box around the ellipse
            box_size = 100  # slightly bigger than beam
            rect = Rectangle(
                (
                    beam_center.Tx.value - box_size / 2,
                    beam_center.Ty.value - box_size / 2,
                ),
                width=box_size,
                height=box_size,
                edgecolor="white",
                facecolor="none",
                lw=1.2,
                linestyle="solid",
            )
            ax.add_patch(rect)
        if draw_limb:
            cropped_map.draw_limb()
        formatter = ticker.FuncFormatter(lambda x, _: f"{int(x):.0e}")
        cbar = plt.colorbar(format=formatter)
        # Optional: set max 5 ticks to prevent clutter
        cbar.locator = ticker.MaxNLocator(nbins=5)
        cbar.update_ticks()
        if pixel_unit == "K":
            cbar.set_label("Brightness temperature (K)")
        elif pixel_unit == "JY/BEAM":
            cbar.set_label("Flux density (Jy/beam)")
        fig.tight_layout()
        output_image_list = []
        for i in range(len(extensions)):
            ext = extensions[i]
            try:
                outdir = outdirs[i]
            except BaseException:
                outdir = os.path.dirname(os.path.abspath(fits_image))
            if len(contour_levels) > 0:
                output_image = (
                    outdir
                    + "/"
                    + os.path.basename(fits_image).split(".fits")[0]
                    + f"_contour.{ext}"
                )
            else:
                output_image = (
                    outdir
                    + "/"
                    + os.path.basename(fits_image).split(".fits")[0]
                    + f".{ext}"
                )
            output_image_list.append(output_image)
        for output_image in output_image_list:
            fig.savefig(output_image)
        if showgui:
            plt.show()
        plt.close(fig)
    except Exception:
        traceback.print_exc()
    finally:
        plt.close("all")
    return output_image_list, cropped_map


def get_suvi_map(obs_date, obs_time, workdir, wavelength=195):
    """
    Get GOES SUVI map

    Parameters
    ----------
    obs_date : str
        Observation date in yyyy-mm-dd format
    obs_time : str
        Observation time in hh:mm format
    workdir : str
        Work directory
    wavelength : float, optional
        Wavelength, options: 94, 131, 171, 195, 284, 304 Å

    Returns
    -------
    sunpy.map
        Sunpy SUVIMap
    """
    logging.getLogger("sunpy").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message="This download has been started in a thread which is not the main thread",
    )
    os.makedirs(workdir, exist_ok=True)
    start_time = dt.fromisoformat(f"{obs_date}T{obs_time}")
    t_start = (start_time - timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M")
    t_end = (start_time + timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M")
    time = a.Time(t_start, t_end)
    instrument = a.Instrument("suvi")
    wavelength = a.Wavelength(wavelength * u.angstrom)
    results = Fido.search(time, instrument, wavelength, a.Level(2))
    downloaded_files = Fido.fetch(results, path=workdir, progress=False)
    obs_times = []
    for image in downloaded_files:
        suvimap = Map(image)
        dateobs = suvimap.meta["date-obs"].split(".")[0]
        obs_times.append(dateobs)
    times_dt = [dt.strptime(t, "%Y-%m-%dT%H:%M:%S") for t in obs_times]
    closest_time = min(times_dt, key=lambda t: abs(t - start_time))
    pos = times_dt.index(closest_time)
    closest_time_str = closest_time.strftime("%Y-%m-%dT%H:%M")
    final_image = downloaded_files[pos]
    suvi_map = Map(final_image)
    for f in downloaded_files:
        os.system(f"rm -rf {f}")
    return suvi_map


def enhance_offlimb(sunpy_map, do_sharpen=True):
    """
    Enhance off-disk emission

    Parameters
    ----------
    sunpy_map : sunpy.map
        Sunpy map
    do_sharpen : bool, optional
        Sharpen images

    Returns
    -------
    sunpy.map
        Off-disk enhanced emission
    """
    from scipy.ndimage import gaussian_filter
    from sunpy.map.maputils import all_coordinates_from_map

    logging.getLogger("sunpy").setLevel(logging.ERROR)
    hpc_coords = all_coordinates_from_map(sunpy_map)
    r = np.sqrt(hpc_coords.Tx**2 + hpc_coords.Ty**2) / sunpy_map.rsun_obs
    rsun_step_size = 0.01
    rsun_array = np.arange(1, r.max(), rsun_step_size)
    y = np.array(
        [
            sunpy_map.data[(r > this_r) * (r < this_r + rsun_step_size)].mean()
            for this_r in rsun_array
        ]
    )
    pos = np.where(y < 10e-3)[0][0]
    r_lim = round(rsun_array[pos], 2)
    params = np.polyfit(
        rsun_array[rsun_array < r_lim], np.log(y[rsun_array < r_lim]), 1
    )
    scale_factor = np.exp((r - 1) * -params[0])
    scale_factor[r < 1] = 1
    if do_sharpen:
        blurred = gaussian_filter(sunpy_map.data, sigma=3)
        data = sunpy_map.data + (sunpy_map.data - blurred)
    else:
        data = sunpy_map.data
    scaled_map = Map(data * scale_factor, sunpy_map.meta)
    scaled_map.plot_settings["norm"] = ImageNormalize(stretch=LogStretch(10))
    return scaled_map


def make_meer_overlay(
    meerkat_image,
    suvi_wavelength=195,
    plot_file_prefix=None,
    plot_meer_colormap=True,
    enhance_offdisk=True,
    contour_levels=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
    do_sharpen_suvi=True,
    xlim=[-1600, 1600],
    ylim=[-1600, 1600],
    extensions=["png"],
    outdirs=[],
    ncpu=-1,
    showgui=False,
    verbose=False,
):
    """
    Make overlay of MeerKAT image on GOES SUVI image

    Parameters
    ----------
    meerkat_image : str
        MeerKAT image
    suvi_wavelength : float, optional
        GOES SUVI wavelength, options: 94, 131, 171, 195, 284, 304 Å
    plot_file_prefix : str, optional
        Plot file prefix name
    plot_meer_colormap : bool, optional
        Plot MeerKAT map colormap
    enhance_offdisk : bool, optional
        Enhance off-disk emission
    contour_levels : list, optional
        Contour levels in fraction of peak
    do_sharpen_suvi : bool, optional
        Do sharpen SUVI images
    xlim : list, optional
        X-axis limit in arcsec
    tlim : list, optional
        Y-axis limit in arcsec
    extensions : list, optional
        Image file extensions
    outdirs : list, optional
        Output directories for each extensions
    ncpu : int, optional
        Number of CPUs to use
    showgui : bool, optional
        Show GUI
    verbose: bool, optinal
        Verbose output

    Returns
    -------
    list
        Plot file names
    """
    import matplotlib
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt
    from sunpy.coordinates import SphericalScreen
    from matplotlib.colors import ListedColormap
    from matplotlib import cm
    from sunpy.map import make_fitswcs_header

    logging.getLogger("sunpy").setLevel(logging.ERROR)
    logging.getLogger("reproject.common").setLevel(logging.WARNING)

    @delayed
    def reproject_map(smap, target_header):
        with SphericalScreen(smap.observer_coordinate):
            return smap.reproject_to(target_header)

    if showgui:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")
    workdir = os.path.dirname(os.path.abspath(meerkat_image))
    meermap = get_meermap(meerkat_image)
    obs_datetime = fits.getheader(meerkat_image)["DATE-OBS"]
    obs_date = obs_datetime.split("T")[0]
    obs_time = ":".join(obs_datetime.split("T")[-1].split(":")[:2])
    suvi_map = get_suvi_map(obs_date, obs_time, workdir, wavelength=suvi_wavelength)
    if enhance_offdisk:
        suvi_map = enhance_offlimb(suvi_map, do_sharpen=do_sharpen_suvi)
    projected_coord = SkyCoord(
        0 * u.arcsec,
        0 * u.arcsec,
        obstime=suvi_map.observer_coordinate.obstime,
        frame="helioprojective",
        observer=suvi_map.observer_coordinate,
        rsun=suvi_map.coordinate_frame.rsun,
    )
    projected_header = make_fitswcs_header(
        suvi_map.data.shape,
        projected_coord,
        scale=u.Quantity(suvi_map.scale),
        instrument=suvi_map.instrument,
        wavelength=suvi_map.wavelength,
    )
    reprojected = [
        reproject_map(meermap, projected_header),
        reproject_map(suvi_map, projected_header),
    ]
    if ncpu < 1:
        ncpu = 1
    pool = ThreadPool(processes=ncpu)
    with dask.config.set(pool=pool):
        meer_reprojected, suvi_reprojected = compute(*reprojected, scheduler="threads")
    meertime = meermap.meta["date-obs"].split(".")[0]
    suvitime = suvi_map.meta["date-obs"].split(".")[0]
    try:
        if plot_meer_colormap and len(contour_levels) > 0:
            matplotlib.rcParams.update({"font.size": 18})
            fig = plt.figure(figsize=(16, 8))
            ax_colormap = fig.add_subplot(1, 2, 1, projection=suvi_reprojected)
            ax_contour = fig.add_subplot(1, 2, 2, projection=suvi_reprojected)
        elif plot_meer_colormap:
            matplotlib.rcParams.update({"font.size": 14})
            fig = plt.figure(figsize=(10, 8))
            ax_colormap = fig.add_subplot(projection=suvi_reprojected)
        elif len(contour_levels) > 0:
            matplotlib.rcParams.update({"font.size": 14})
            fig = plt.figure(figsize=(10, 8))
            ax_contour = fig.add_subplot(projection=suvi_reprojected)
        else:
            print("No overlay is plotting.")
            return

        title = f"SUVI time: {suvitime}\n MeerKAT time: {meertime}"
        if "transparent_inferno" not in plt.colormaps():
            cmap = cm.get_cmap("inferno", 256)
            colors = cmap(np.linspace(0, 1, 256))
            x = np.linspace(0, 1, 256)
            alpha = 0.8 * (1 - np.exp(-3 * x))
            colors[:, -1] = alpha  # Update the alpha channel
            transparent_inferno = ListedColormap(colors)
            plt.colormaps.register(name="transparent_inferno", cmap=transparent_inferno)
        if plot_meer_colormap and len(contour_levels) > 0:
            suptitle = title.replace("\n", ",")
            title = ""
            fig.suptitle(suptitle)
        if plot_meer_colormap:
            z = 0
            suvi_reprojected.plot(
                axes=ax_colormap,
                title=title,
                autoalign=True,
                clip_interval=(3, 99.9) * u.percent,
                zorder=z,
            )
            z += 1
            meer_reprojected.plot(
                axes=ax_colormap,
                title=title,
                clip_interval=(3, 99.9) * u.percent,
                cmap="transparent_inferno",
                zorder=z,
            )
        if len(contour_levels) > 0:
            z = 0
            suvi_reprojected.plot(
                axes=ax_contour,
                title=title,
                autoalign=True,
                clip_interval=(3, 99.9) * u.percent,
                zorder=z,
            )
            z += 1
            contour_levels = np.array(contour_levels) * np.nanmax(meer_reprojected.data)
            meer_reprojected.draw_contours(
                contour_levels, axes=ax_contour, cmap="YlGnBu", zorder=z
            )
            ax_contour.set_facecolor("black")

        if len(xlim) > 0:
            x_pix_limits = []
            for x in xlim:
                sky = SkyCoord(
                    x * u.arcsec, 0 * u.arcsec, frame=suvi_reprojected.coordinate_frame
                )
                x_pix = suvi_reprojected.world_to_pixel(sky)[0].value
                x_pix_limits.append(x_pix)
            if plot_meer_colormap and len(contour_levels) > 0:
                ax_colormap.set_xlim(x_pix_limits)
                ax_contour.set_xlim(x_pix_limits)
            elif plot_meer_colormap:
                ax_colormap.set_xlim(x_pix_limits)
            elif len(contour_levels) > 0:
                ax_contour.set_xlim(x_pix_limits)
        if len(ylim) > 0:
            y_pix_limits = []
            for y in ylim:
                sky = SkyCoord(
                    0 * u.arcsec, y * u.arcsec, frame=suvi_reprojected.coordinate_frame
                )
                y_pix = suvi_reprojected.world_to_pixel(sky)[1].value
                y_pix_limits.append(y_pix)
            if plot_meer_colormap and len(contour_levels) > 0:
                ax_colormap.set_ylim(y_pix_limits)
                ax_contour.set_ylim(y_pix_limits)
            elif plot_meer_colormap:
                ax_colormap.set_ylim(y_pix_limits)
            elif len(contour_levels) > 0:
                ax_contour.set_ylim(y_pix_limits)
        if plot_meer_colormap and len(contour_levels) > 0:
            ax_colormap.coords.grid(False)
            ax_contour.coords.grid(False)
        elif plot_meer_colormap:
            ax_colormap.coords.grid(False)
        elif len(contour_levels) > 0:
            ax_contour.coords.grid(False)
        fig.tight_layout()
        plot_file_list = []
        if verbose:
            print("#######################")
        if plot_file_prefix:
            for i in range(len(extensions)):
                ext = extensions[i]
                try:
                    savedir = outdirs[i]
                except BaseException:
                    savedir = workdir
                plot_file = f"{savedir}/{plot_file_prefix}.{ext}"
                plt.savefig(plot_file, bbox_inches="tight")
                if verbose:
                    print(f"Plot saved: {plot_file}")
                plot_file_list.append(plot_file)
            if verbose:
                print("#######################\n")
        else:
            plot_file = None
        if showgui:
            plt.show()
            plt.close(fig)
        else:
            plt.close(fig)
    except Exception:
        traceback.print_exc()
    finally:
        plt.close("all")
    return plot_file_list


##############################
# Extract dynamic spectrum
##############################
def make_ds_file_per_scan(msname, save_file, scan, datacolumn):
    """
    Extract dynamic spectrum from measurement set

    Parameters
    ----------
    msname : str
        Measurement set name
    save_file : str
        File name to save dynamic spectrum
    scan : int
        Scan number
    datacolumn : str
        Data column name

    Returns
    -------
    str
        Dynamic spectrum file
    """
    if os.path.exists(f"{save_file}.npy") == False:
        mstool = casamstool()
        try:
            all_data = []
            for ant in range(5):
                print(f"Extracting data for antenna :{ant}, scan: {scan}")
                mstool.open(msname)
                mstool.selectpolarization(["I"])
                mstool.select(
                    {"antenna1": ant, "antenna2": ant, "scan_number": int(scan)}
                )
                data_dic = mstool.getdata(datacolumn)
                mstool.close()
                if datacolumn == "CORRECTED_DATA":
                    data = np.abs(data_dic["corrected_data"][0, ...])
                else:
                    data = np.abs(data_dic["data"][0, ...])
                del data_dic
                m = np.nanmedian(data, axis=1)
                data = data / m[:, None]
                all_data.append(data)
                del data
        except Exception as e:
            print("Auto-corrrelations are not present. Using short baselines.")
            count = 0
            all_data = []
            while count <= 5:
                for i in range(5):
                    for j in range(5):
                        if i != j:
                            print(
                                f"Extracting data for antennas :{i} and {j}, scan: {scan}"
                            )
                            mstool.open(msname)
                            mstool.selectpolarization(["I"])
                            mstool.select(
                                {
                                    "antenna1": i,
                                    "antenna2": j,
                                    "scan_number": int(scan),
                                }
                            )
                            data_dic = mstool.getdata(datacolumn)
                            mstool.close()
                            if datacolumn == "CORRECTED_DATA":
                                data = np.abs(data_dic["corrected_data"][0, ...])
                            else:
                                data = np.abs(data_dic["data"][0, ...])
                            del data_dic
                            m = np.nanmedian(data, axis=1)
                            data = data / m[:, None]
                            all_data.append(data)
                            del data
                            count += 1
        finally:
            try:
                mstool.close()
            except:
                pass
        all_data = np.array(all_data)
        data = np.nanmedian(all_data, axis=0)
        bad_chans = get_bad_chans(msname)
        if bad_chans != "":
            bad_chans = bad_chans.replace("0:", "").split(";")
            for bad_chan in bad_chans:
                s = int(bad_chan.split("~")[0])
                e = int(bad_chan.split("~")[-1]) + 1
                data[s:e, :] = np.nan
        msmd = msmetadata()
        msmd.open(msname)
        freqs = msmd.chanfreqs(0, unit="MHz")
        times = msmd.timesforscans(int(scan))
        timestamps = [mjdsec_to_timestamp(mjdsec, str_format=0) for mjdsec in times]
        msmd.close()
        np.save(
            save_file,
            np.array([freqs, times, timestamps, data], dtype="object"),
        )
    if ".npy" in save_file:
        return save_file
    else:
        return f"{save_file}.npy"


def make_ds_plot(dsfiles, plot_file=None, showgui=False):
    """
    Make dynamic spectrum plot

    Parameters
    ----------
    dsfile : list
        DS files list
    plot_file : str, optional
        Plot file name to save the plot
    showgui : bool, optional
        Show GUI

    Returns
    -------
    str
        Plot name
    """
    from matplotlib.gridspec import GridSpec

    if showgui:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 18})
    if type(dsfiles) == str:
        dsfiles = [dsfiles]
    for i, dsfile in enumerate(dsfiles):
        freqs_i, times_i, timestamps_i, data_i = np.load(dsfile, allow_pickle=True)
        if i == 0:
            freqs = freqs_i
            times = times_i
            timestamps = timestamps_i
            data = data_i
        else:
            gapsize = int(
                (np.nanmin(times_i) - np.nanmax(times)) / (times[1] - times[0])
            )
            if gapsize < 10:
                last_time_median = np.nanmedian(data[:, -1], axis=0)
                new_time_median = np.nanmedian(data_i[:, 0], axis=0)
                data_i = (data_i / new_time_median) * last_time_median
            # Insert vertical NaN gap (1 column wide)
            gap = np.full((data.shape[0], gapsize), np.nan)
            data = np.concatenate([data, gap, data_i], axis=1)
            # Insert dummy time and timestamp
            times = np.append(times, np.nan)
            timestamps = np.append(timestamps, "GAP")
            # Append new values
            times = np.append(times, times_i)
            timestamps = np.append(timestamps, timestamps_i)
            # (Optional) Check or merge freqs if needed — assuming same across files
    # Normalize by median bandshape
    median_bandshape = np.nanmedian(data, axis=-1)
    pos = np.where(np.isnan(median_bandshape) == False)[0]
    data /= median_bandshape[:, None]
    data = data[min(pos) : max(pos), :]
    freqs = freqs[min(pos) : max(pos)]
    temp_times = times[np.isnan(times) == False]
    maxtimepos = np.argmax(temp_times)
    mintimepos = np.argmin(temp_times)
    datestamp = f"{timestamps[mintimepos].split('T')[0]}"
    tstart = f"{timestamps[mintimepos].split('T')[0]} {':'.join(timestamps[mintimepos].split('T')[-1].split(':')[:2])}"
    tend = f"{timestamps[maxtimepos].split('T')[0]} {':'.join(timestamps[maxtimepos].split('T')[-1].split(':')[:2])}"
    print(f"Time range : {tstart}~{tend}")
    results = Fido.search(
        a.Time(tstart, tend), a.Instrument("XRS"), a.Resolution("avg1m")
    )
    files = Fido.fetch(results, path=os.path.dirname(dsfiles[0]), overwrite=False)
    goes_tseries = TimeSeries(files, concatenate=True)
    for goes_f in files:
        os.system(f"rm -rf {goes_f}")
    goes_tseries = goes_tseries.truncate(tstart, tend)
    timeseries = np.nanmean(data, axis=0)
    # Normalization
    data_std = np.nanstd(data)
    data_median = np.nanmedian(data)
    norm = ImageNormalize(
        data,
        stretch=LogStretch(1),
        vmin=0.99 * np.nanmin(data),
        vmax=0.99 * np.nanmax(data),
    )
    try:
        # Create figure and GridSpec layout
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(
            nrows=3, ncols=2, width_ratios=[1, 0.03], height_ratios=[4, 1.5, 2]
        )
        # Axes
        ax_spec = fig.add_subplot(gs[0, 0])
        ax_ts = fig.add_subplot(gs[1, 0])
        ax_goes = fig.add_subplot(gs[2, 0])
        cax = fig.add_subplot(gs[:, 1])  # colorbar spans both rows
        # Plot dynamic spectrum
        im = ax_spec.imshow(
            data, aspect="auto", origin="lower", norm=norm, cmap="magma"
        )
        ax_spec.set_ylabel("Frequency (MHz)")
        ax_spec.set_xticklabels([])  # Remove x-axis labels from top plot
        # Y-ticks
        yticks = ax_spec.get_yticks()
        yticks = yticks[(yticks >= 0) & (yticks < len(freqs))]
        ax_spec.set_yticks(yticks)
        ax_spec.set_yticklabels([f"{freqs[int(i)]:.1f}" for i in yticks])
        # Plot time series
        ax_ts.plot(timeseries)
        ax_ts.set_xlim(0, len(timeseries) - 1)
        ax_ts.set_ylabel("Mean \n flux density")
        goes_tseries.plot(axes=ax_goes)
        goes_times = goes_tseries.time
        times_dt = goes_times.to_datetime()
        ax_goes.set_xlim(times_dt[0], times_dt[-1])
        ax_goes.set_ylabel(r"Flux ($\frac{W}{m^2}$)")
        ax_goes.legend(ncol=2, loc="upper right")
        ax_goes.set_title("GOES light curve", fontsize=14)
        ax_ts.set_title("MeerKAT light curve", fontsize=14)
        ax_spec.set_title("MeerKAT dynamic spectrum", fontsize=14)
        ax_goes.set_xlabel("Time (UTC)")
        # Format x-ticks
        ax_ts.set_xticks([])
        ax_ts.set_xticklabels([])
        # Colorbar
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Flux density (arb. unit)")
        plt.tight_layout()
        # Save or show
        if plot_file:
            plt.savefig(plot_file, bbox_inches="tight")
            print(f"Plot saved: {plot_file}")
        if showgui:
            plt.show()
            plt.close(fig)
        else:
            plt.close(fig)
    except Exception:
        traceback.print_exc()
    finally:
        plt.close("all")
    return plot_file


def plot_goes_full_timeseries(
    msname, workdir, plot_file_prefix=None, extension="png", showgui=False
):
    """
    Plot GOES full time series on the day of observation

    Parameters
    ----------
    msname : str
        Measurement set
    workdir : str
        Work directory
    plot_file_prefix : str, optional
        Plot file name prefix
    extension : str, optional
        Save file extension
    showgui : bool, optional
        Show GUI

    Returns
    -------
    str
        Plot file name
    """
    os.makedirs(workdir, exist_ok=True)
    if showgui:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 14})
    scans, cal_scans, f_scans, g_scans, p_scans = get_cal_target_scans(msname)
    valid_scans = get_valid_scans(msname)
    filtered_scans = []
    for scan in scans:
        if scan in valid_scans:
            filtered_scans.append(scan)
    msmd = msmetadata()
    msmd.open(msname)
    tstart_mjd = min(msmd.timesforscan(int(min(filtered_scans))))
    tend_mjd = max(msmd.timesforscan(int(max(filtered_scans))))
    msmd.close()
    tstart = mjdsec_to_timestamp(tstart_mjd, str_format=2)
    tend = mjdsec_to_timestamp(tend_mjd, str_format=2)
    print(f"Time range: {tstart}~{tend}")
    results = Fido.search(
        a.Time(tstart, tend), a.Instrument("XRS"), a.Resolution("avg1m")
    )
    files = Fido.fetch(results, path=workdir, overwrite=False)
    goes_tseries = TimeSeries(files, concatenate=True)
    for f in files:
        os.system(f"rm -rf {f}")
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    goes_tseries.plot(axes=ax)
    times = goes_tseries.time
    times_dt = times.to_datetime()
    ax.axvspan(tstart, tend, alpha=0.2)
    ax.set_xlim(times_dt[0], times_dt[-1])
    plt.tight_layout()
    # Save or show
    if plot_file_prefix:
        plot_file = f"{workdir}/{plot_file_prefix}.{extension}"
        plt.savefig(plot_file, bbox_inches="tight")
        print(f"Plot saved: {plot_file}")
    else:
        plot_file = None
    if showgui:
        plt.show()
        plt.close(fig)
        plt.close("all")
    else:
        plt.close(fig)
    return plot_file


def rename_meersolar_image(
    imagename,
    imagedir="",
    pol="",
    band="",
    attcal="NOINFO",
    cutout_rsun=2.5,
    make_overlay=True,
    make_plots=True,
):
    """
    Rename and move image to image directory

    Parameters
    ----------
    imagename : str
        Image name
    imagedir : str, optional
        Image directory (default given image directory)
    pol : str, optional
        Stokes parameters
    band : str, optional
        Observing band
    attcal : str, optional
        Solar attenuation calibrated or not
    cutout_rsun : float, optional
        Cutout in solar radii from center (default: 2.5 solar radii)
    make_overlay : bool, optional
        Make overlay on SUVI
    make_plots : bool, optional
        Make radio map plot in helioprojective coordinates

    Returns
    -------
    str
        New imagename with full path
    """
    imagename = imagename.rstrip("/")
    imagename = cutout_image(
        imagename, imagename, x_deg=(cutout_rsun * 2 * 16.0) / 60.0
    )
    header = fits.getheader(imagename)
    time = header["DATE-OBS"]
    astro_time = Time(time, scale="utc")
    sun_jpl = Horizons(id="10", location="500", epochs=astro_time.jd)
    eph = sun_jpl.ephemerides()
    sun_coords = SkyCoord(
        ra=eph["RA"][0] * u.deg, dec=eph["DEC"][0] * u.deg, frame="icrs"
    )
    maxval, minval, rms, total_val, mean_val, median_val, rms_dyn, minmax_dyn = (
        calc_solar_image_stat(imagename, disc_size=18)
    )
    with fits.open(imagename, mode="update") as hdul:
        hdr = hdul[0].header
        hdr["AUTHOR"] = "DevojyotiKansabanik,DeepanPatra"
        if band != "":
            hdr["BAND"] = band
        hdr["PIPELINE"] = "MeerSOLAR"
        hdr["CRVAL1"] = sun_coords.ra.deg
        hdr["CRVAL2"] = sun_coords.dec.deg
        hdr["MAX"] = maxval
        hdr["MIN"] = minval
        hdr["RMS"] = rms
        hdr["SUM"] = total_val
        hdr["MEAN"] = mean_val
        hdr["MEDIAN"] = median_val
        hdr["RMSDYN"] = rms_dyn
        hdr["MIMADYN"] = minmax_dyn
        hdr["ATTCAL"] = str(attcal)
    freq = round(header["CRVAL3"] / 10**6, 2)
    t_str = "".join(time.split("T")[0].split("-")) + (
        "".join(time.split("T")[-1].split(":"))
    )
    new_name = "time_" + t_str + "_freq_" + str(freq)
    if pol != "":
        new_name += "_pol_" + str(pol)
    if "MFS" in imagename:
        new_name += "_MFS"
    new_name = new_name + ".fits"
    if imagedir == "":
        imagedir = os.path.dirname(os.path.abspath(imagename))
    new_name = imagedir + "/" + new_name
    os.system("mv " + imagename + " " + new_name)
    hpcdir = f"{os.path.dirname(imagedir)}/images/hpcs"
    os.makedirs(hpcdir, exist_ok=True)
    save_in_hpc(new_name, outdir=hpcdir)
    if make_plots:
        try:
            pngdir = f"{os.path.dirname(imagedir)}/images/pngs"
            os.makedirs(pngdir, exist_ok=True)
            outimages, cropped_map = plot_in_hpc(
                new_name,
                draw_limb=True,
                extensions=["png"],
                outdirs=[pngdir],
            )
        except Exception:
            pass
    if make_overlay:
        try:
            overlay_pngdir = f"{os.path.dirname(imagedir)}/overlays_pngs"
            os.makedirs(overlay_pngdir, exist_ok=True)
            outimages = make_meer_overlay(
                new_name,
                plot_file_prefix=os.path.basename(new_name).split(".fits")[0]
                + "_suvi_meerkat_overlay",
                extensions=["png"],
                outdirs=[overlay_pngdir],
                verbose=False,
            )
        except Exception:
            pass
    return new_name


# Expose functions and classes
__all__ = [
    name
    for name, obj in globals().items()
    if (
        (isinstance(obj, types.FunctionType) or isinstance(obj, type))
        and obj.__module__ == __name__
    )
]
