import sunpy.map as smap
import numpy as np, os
import matplotlib.pyplot as plt
import astropy.units as u
import sys
import h5py,gc
from astropy.coordinates import SkyCoord
from optparse import OptionParser


def chrmo_emission(aia_304_file, dem_file, outfile):
    """
    Estimate chromorspheric contribution to the radio emission
    Parameters
    ----------
    aia_304_file : str
        AIA 304 angstorm image name
    dem_file : str
        DEM file name
    outfile : str
        Output file name (.h5')
    Returns
    -------
    str
        Output file name
    """
    print("########################")
    print(
        "Estimating chromopspheric brightness temperature for free-free emission ......"
    )
    print("########################")
    if os.path.dirname(outfile) == "":
        outfile = os.getcwd() + "/" + outfile
    arcsec2rad = 1 / 3600.0 * np.pi / 180
    solarrad = 16 * 60  ##arcsec
    hf = h5py.File(dem_file)
    x1 = hf.attrs["x1"]
    y1 = hf.attrs["y1"]
    x2 = hf.attrs["x2"]
    y2 = hf.attrs["y2"]
    avg_length = hf.attrs["avg_length"]
    hf.close()
    aiamap = smap.Map(aia_304_file)
    bottom_left = SkyCoord(x1 * u.arcsec, y1 * u.arcsec, frame=aiamap.coordinate_frame)
    top_right = SkyCoord(x2 * u.arcsec, y2 * u.arcsec, frame=aiamap.coordinate_frame)
    submap = aiamap.submap(bottom_left=bottom_left, top_right=top_right)
    data = submap.data
    shape = data.shape
    ny = int(shape[0] / avg_length)
    nx = int(shape[1] / avg_length)
    del data
    new_dimensions = [nx, ny] * u.pixel
    aiamap = submap.resample(new_dimensions)
    aiadata = aiamap.data
    aiadata[aiadata < -10] = np.nan
    tot_flux = np.nansum(aiadata)
    omega_sun = np.pi * (solarrad * arcsec2rad) ** 2
    omega_pix = (aiamap.meta["cdelt1"] * arcsec2rad) ** 2
    tb_data = 10880.0 * omega_sun / omega_pix * aiadata / tot_flux
    aiamap.meta["pixlunit"] = "Tb"
    tbmap = smap.Map(tb_data, aiamap.meta)
    tbmap.plot_settings["title"] = "Chromospheric Tb contribution"
    os.system("rm -rf " + outfile)
    hf = h5py.File(outfile, "w")
    keys = tbmap.meta.keys()
    for key in keys:
        if key != "keycomments" and tbmap.meta[key] != None:
            hf.attrs[key] = tbmap.meta[key]
    tbmap.data[np.isnan(tbmap.data)] = 0.0
    hf.create_dataset("tb", data=tbmap.data)
    hf.close()
    gc.collect()
    return outfile


def main():
    usage = "Estimate chromospheric radio emission based on AIA 304 angstorm emission"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--aia_304",
        dest="aia_304",
        default=None,
        help="AIA 304 angstorm file name",
        metavar="String",
    )
    parser.add_option(
        "--DEM_file",
        dest="DEM_file",
        default=None,
        help="DEM file name (.h5)",
        metavar="String",
    )
    parser.add_option(
        "--outfile",
        dest="outfile",
        default="Chromo.h5",
        help="Output file name (.h5)",
        metavar="String",
    )
    (options, args) = parser.parse_args()
    if options.aia_304 == None or os.path.exists(options.aia_304) == False:
        print("Please provide AIA 304 angstorm fits file.\n")
        return 1
    elif options.DEM_file == None or os.path.exists(options.DEM_file) == False:
        print("Please provide correct DEM file path.\n")
        return 1
    else:
        chromo_emission_file = chrmo_emission(
            options.aia_304, options.DEM_file, options.outfile
        )
    if chromo_emission_file != None:
        print("Output file name: ", chromo_emission_file)
        gc.collect()
        return 0
    else:
        gc.collect()
        return 1
