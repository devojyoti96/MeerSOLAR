import numpy as np, os, sys, glob
from matplotlib import pyplot as plt
from meersolar.pipeline.basic_func import get_datadir
from meersolar.solstar.dem_python.dn2dem_pos import dn2dem_pos
import astropy.units as u
from sunpy.map import Map
from scipy.signal import convolve2d
import h5py, gc
import scipy.io as io
from astropy.coordinates import SkyCoord
from optparse import OptionParser


def make_dem(fits_files, outfile, fov, resolution, ncpu=-1):
    """
    Make DEM map using AIA multi-wavelength images
    Parameters
    ----------
    fits_files : list
        List of fits files
    outfile : str
        Output DEM file name (.h5')
    fov : str
        Field of view to be used in pixels (xblc,xtrc,yblc,ytrc)
    resolution : float
        Spatial resolution in arcsecond
    ncpu : int, optional
        Number of CPU threads to use
    Returns
    -------
    str
        Output DEM file
    """
    print("########################")
    print("Producing DEM map ......")
    print("########################")
    if resolution<=0.6:
        resolution=0.6
    avg_length=int(resolution/0.6) # 0.6arcsecond in AIA image pixel resolution  
    if avg_length<1:
        avg_length=1  
    pwd = os.getcwd()
    if os.path.dirname(outfile) == "":
        outfile = pwd + "/" + outfile
    temp_bin_num = 14
    fov = fov.split(",")
    x1 = int(fov[0])
    x2 = int(fov[1])
    y1 = int(fov[2])
    y2 = int(fov[3])
    low_temp = 5.2  ## in log10
    high_temp = 7.2  ## in log10
    datadir=get_datadir()
    trin = io.readsav(datadir+"/aia_tresp_en.dat")
    wavenum = ["94", "131", "171", "193", "211", "335"]
    channels = []
    for i in np.arange(len(wavenum)):
        channels.append(float(wavenum[i]) * u.angstrom)
    tresp_logt = np.array(trin["logt"])
    tresp_calibration = np.array(trin["tr"])
    nt = len(tresp_logt)
    nf = len(trin["tr"][:])
    trmatrix = np.zeros((nt, nf))
    for i in range(0, nf):
        trmatrix[:, i] = trin["tr"][i]
    temperatures = 10 ** np.linspace(low_temp, high_temp, num=temp_bin_num + 1)
    logtemps = np.linspace(low_temp, high_temp, num=temp_bin_num + 1)
    # we only want optically thin coronal wavelengths
    # load the fits with sunpy
    aia = Map(fits_files)
    # read dimensions from the header
    nf = len(wavenum)
    # normalise to dn/s
    aia = [Map(m.data / m.exposure_time.value, m.meta) for m in aia]
    # convert from our list to an array of data
    for j in range(nf):
        bottom_left = SkyCoord(
            x1 * u.arcsec, y1 * u.arcsec, frame=aia[j].coordinate_frame
        )
        top_right = SkyCoord(
            x2 * u.arcsec, y2 * u.arcsec, frame=aia[j].coordinate_frame
        )
        submap = aia[j].submap(bottom_left=bottom_left, top_right=top_right)
        if j == 0:
            nx = int(submap.dimensions.x.value)
            ny = int(submap.dimensions.y.value)
            new_dimensions = (int(nx / avg_length), int(ny / avg_length)) * u.pixel
        aia_resampled = submap.resample(new_dimensions)
        if j == 0:
            nx = int(aia_resampled.dimensions.x.value)
            ny = int(aia_resampled.dimensions.y.value)
            # create data array
            data = np.zeros([ny, nx, nf])
        data[:, :, j] = aia_resampled.data
        pixel_size = round(aia_resampled.meta["cdelt1"], 1)
        del aia_resampled, submap, bottom_left, top_right
    data[data < 0] = 0
    shape = data.shape
    # calculate our dem_norm
    off = 0.412
    gauss_stdev = 12
    dem_norm0 = np.zeros((shape[0], shape[1], temp_bin_num))
    dem_norm_temp = np.convolve(
        np.exp(
            -((np.arange(temp_bin_num) + 1 - (temp_bin_num - 2) * (off + 0.1)) ** 2)
            / gauss_stdev
        ),
        np.ones(3) / 3,
    )[1:-1]
    dem_norm0[:, :, :] = dem_norm_temp
    serr_per = 10.0
    # errors in dn/px/s
    npix = avg_length * avg_length
    edata = np.zeros_like(data)
    gains = np.array([18.3, 17.6, 17.7, 18.3, 18.3, 17.6])
    dn2ph = gains * [94, 131, 171, 193, 211, 335] / 3397.0
    rdnse = (
        np.array([1.14, 1.18, 1.15, 1.20, 1.20, 1.18]) * np.sqrt(npix) / npix
    )  ## previously it was 1.15
    drknse = 0.17
    qntnse = 0.288819 * np.sqrt(npix) / npix
    for j in np.arange(nf):
        etemp = np.sqrt(
            rdnse[j] ** 2.0
            + drknse**2.0
            + qntnse**2.0
            + (dn2ph[j] * abs(data[:, :, j])) / (npix * dn2ph[j] ** 2)
        )
        esys = serr_per * data[:, :, j] / 100.0
        edata[:, :, j] = np.sqrt(etemp**2.0 + esys**2.0)
    dem, edem, elogt, chisq, dn_reg = dn2dem_pos(
        data,
        edata,
        trmatrix,
        tresp_logt,
        temperatures,
        dem_norm0=dem_norm0,
        max_iter=30,
        n_jobs=ncpu
    )
    logt_bin = np.zeros(temp_bin_num)
    for i in np.arange(temp_bin_num):
        logt_bin[i] = (logtemps[i] + logtemps[i + 1]) / 2
    output = {
        "logt_bin": logt_bin,
        "elogt": elogt,
        "dem": dem,
        "edem": edem,
        "chisq": chisq,
        "dn_reg": dn_reg,
    }
    if os.path.exists(outfile):
        os.system("rm -rf " + outfile)
    hf = h5py.File(outfile, "w")
    hf.attrs["files"] = len(fits_files)
    hf.attrs["x1"] = x1
    hf.attrs["y1"] = y1
    hf.attrs["x2"] = x2
    hf.attrs["y2"] = y2
    hf.attrs["pixel_size"] = pixel_size
    hf.attrs["coord unit"] = "arcsec"
    hf.attrs["avg_length"] = avg_length
    hf.create_dataset("logt_bin", data=logt_bin)
    hf.create_dataset("elogt", data=elogt)
    hf.create_dataset("dem", data=dem)
    hf.create_dataset("chisq", data=chisq)
    hf.create_dataset("dn_reg", data=dn_reg)
    hf.create_dataset("edem", data=edem)
    hf.close()
    gc.collect()
    return outfile


def main():
    usage = "Make DEM map using AIA multi-wavelength images"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--fits_dir",
        dest="fits_dir",
        default=None,
        help="AIA fits file directory",
        metavar="String",
    )
    parser.add_option(
        "--outfile",
        dest="outfile",
        default="DEM.h5",
        help="Output file name (.h5)",
        metavar="String",
    )
    parser.add_option(
        "--fov",
        dest="fov",
        default="-2000,2000,-2000,2000",
        help="Field of view to be used in pixels (xblc,xtrc,yblc,ytrc) ",
        metavar="String",
    )
    parser.add_option(
        "--resolution",
        dest="resolution",
        default=5.0,
        help="Spatial resolution in arcsecond (should be more than 0.6 arcsecond)",
        metavar="Float",
    )
    parser.add_option(
        "--ncpu",
        dest="ncpu",
        default=-1,
        help="Number of CPU threads to use",
        metavar="Integer",
    )
    (options, args) = parser.parse_args()
    if options.fits_dir == None:
        print("Please provide correct AIA fits file directory or prefix name.\n")
        return 1
    elif os.path.exists(options.fits_dir) == False:
        print("Please provide correct fits directory.\n")
        return 1
    else:
        fits_files = glob.glob(options.fits_dir + "/*")
        filtered_fits_files = []
        wavenum = ["94", "131", "171", "193", "211", "335"]
        for w in wavenum:
            for f in fits_files:
                if w in f:
                    filtered_fits_files.append(f)
        if len(filtered_fits_files) != 6:
            print("Please provide AIA images at all wavelengths.")
            return 1
    dem_file = make_dem(
        filtered_fits_files, options.outfile, options.fov, float(options.resolution), ncpu = int(options.ncpu)
    )
    if dem_file != None:
        print("DEM file name: ", dem_file)
        gc.collect()
        return 0
    else:
        gc.collect()
        return 1
