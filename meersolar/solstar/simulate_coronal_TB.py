import ctypes, sys, os
from numpy.ctypeslib import ndpointer
import numpy as np, gc
import matplotlib.pyplot as plt
import h5py
from optparse import OptionParser


def initGET_MW(libname):
    """
    Python wrapper for fast GRFF codes.
    https://github.com/kuznetsov-radio/GRFF
    This is for the single thread version
    @param libname: path for locating compiled shared library
    @return: An executable for calling the GS codes in single thread
    """
    _intp = ndpointer(dtype=ctypes.c_int32, flags="F")
    _doublep = ndpointer(dtype=ctypes.c_double, flags="F")

    libc_mw = ctypes.CDLL(libname)
    mwfunc = libc_mw.PyGET_MW  ### use PyGET_MW_USER for user defined zeta values
    mwfunc.argtypes = [
        _intp,
        _doublep,
        _doublep,
        _doublep,
        _doublep,
        _doublep,
        _doublep,
    ]
    mwfunc.restype = ctypes.c_int
    return mwfunc


def simulate_GRFF(dem_file, outfile, start_freq, end_freq):
    """
    Simulate free free emission from DEM map
    Parameters
    ----------
    dem_file : str
        DEM file name (.h5)
    outfile : str
        Output free free emission spectral cube
    start_freq : str
        Start frequency in MHz
    end_freq : str
        End frequency in MHz
    Returns
    -------
    str
        Output spectral cube file
    """
    print("########################")
    print("Estimating coronal brightness temperature for free-free emission ......")
    print("########################")
    libname = os.path.dirname(__file__) + "/GRFF/binaries/GRFF_DEM_Transfer.so"
    start_freq *= 10**6
    end_freq *= 10**6
    Nf = int(
        (np.log10(end_freq) - np.log10(start_freq)) / 0.01
    )  # number of frequencies
    if Nf<10:
        Nf=10
    NSteps = 1  # number of nodes along the line-of-sight
    # Conversion units
    sfu2cgs = 1e-19
    vc = 2.998e10  # speed of light
    kb = 1.38065e-16
    rad2asec = 180 / np.pi * 3600
    asec2Mm = 0.73
    hf = h5py.File(dem_file, "r")
    logt_bin = np.array(hf["logt_bin"])
    dem = np.array(hf["dem"])
    x1 = hf.attrs["x1"]
    y1 = hf.attrs["y1"]
    x2 = hf.attrs["x2"]
    y2 = hf.attrs["y2"]
    dx = hf.attrs["pixel_size"]  # Pixel size in arcseconds
    dy = dx
    hf.close()
    T = 10**logt_bin
    N_temp_DEM = np.size(T)
    dT_arr = np.zeros(N_temp_DEM)
    for i in range(N_temp_DEM - 1):
        dT_arr[i] = T[i] * np.log(10) * (logt_bin[i + 1] - logt_bin[i])
    dT_arr[N_temp_DEM - 1] = (
        T[N_temp_DEM - 1]
        * np.log(10)
        * (logt_bin[N_temp_DEM - 1] - logt_bin[N_temp_DEM - 2])
    )
    GET_MW = initGET_MW(libname)
    #####
    # GRFF params
    #####
    Lparms = np.zeros(5, dtype="int32")  # array of dimensions etc.
    Lparms[0] = NSteps
    Lparms[1] = Nf
    Lparms[2] = N_temp_DEM
    Lparms[3] = 0
    Lparms[4] = 1
    Rparms = np.zeros(
        3, dtype="double"
    )  # array of global floating-point parameters - for a single LOS
    Rparms[0] = (dx * asec2Mm * 1e8) * (dy * asec2Mm * 1e8)  # area, cm^2
    Rparms[1] = start_freq  # starting frequency to calculate spectrum, Hz
    Rparms[2] = 0.01  # logarithmic step in frequency
    depth_cm = 1e10
    ParmLocal = np.zeros(
        15, dtype="double"
    )  # array of voxel parameters - for a single voxel
    ParmLocal[0] = (
        depth_cm 
    )  # source depth, cm (total depth - the depths for individual voxels will be computed later)
    ParmLocal[1] = 1e6  # plasma temperature, K (not used in this example)
    ParmLocal[2] = (
        1e10  # electron/atomic concentration, cm^{-3} (not used in this example)
    )
    ParmLocal[3] = 100  # magnetic field, G (will be changed later)
    ParmLocal[4] = 90  # viewing angle, degrees
    ParmLocal[5] = 0  # azimuthal angle, degrees
    ParmLocal[6] = 1 + 4  # emission mechanism flag (all on)
    ParmLocal[7] = 30  # maximum harmonic number
    ParmLocal[8] = 0  # proton concentration, cm^{-3} (not used in this example)
    ParmLocal[9] = 0  # neutral hydrogen concentration, cm^{-3}
    ParmLocal[10] = 0  # neutral helium concentration, cm^{-3}
    ParmLocal[11] = 0  # local DEM on/off key (on)
    ParmLocal[12] = 1  # local DDM on/off key (on)
    ParmLocal[13] = 0  # element abundance code (coronal, following Feldman 1992)
    ParmLocal[14] = 0  # reserved

    Parms = np.zeros(
        (15, NSteps), dtype="double", order="F"
    )  # 2D array of input parameters - for multiple voxels
    for i in range(NSteps):
        Parms[:, i] = ParmLocal  # most of the parameters are the same in all voxels
    dem_shape = dem.shape
    Tbmap = np.zeros((dem_shape[0], dem_shape[1], Nf))
    for i in range(dem_shape[0]):
        for j in range(dem_shape[1]):
            RL = np.zeros((7, Nf), dtype="double", order="F")  # input/output array
            DEM_arr = np.reshape(dem[i, j, :] / (depth_cm), (N_temp_DEM, NSteps))
            DDM_arr = DEM_arr
            res = GET_MW(Lparms, Rparms, Parms, T, DEM_arr, DDM_arr, RL)
            RR = RL[5]
            LL = RL[6]
            freqs = RL[0]
            Intensity = RR + LL
            sr = (dx / rad2asec) * (dy / rad2asec)
            Tb = (
                Intensity
                * sfu2cgs
                * vc
                * vc
                / (2.0 * kb * (freqs * 1e9) * (freqs * 1e9) * sr)
            )
            Tbmap[i, j, :] = Tb
            del DEM_arr, DDM_arr, RL

    hf = h5py.File(outfile, "w")
    hf.attrs["unit"] = "Jy"
    hf.create_dataset("Tb", data=Tbmap)
    hf.create_dataset("freqs", data=freqs)
    hf.attrs["x1"] = x1
    hf.attrs["x2"] = x2
    hf.attrs["y1"] = y1
    hf.attrs["y2"] = y2
    hf.attrs["LOS depth"] = depth_cm
    hf.attrs["B"] = ParmLocal[3]
    hf.attrs["cdelt1"] = dx
    hf.attrs["emission mechanism"] = ParmLocal[6]
    hf.close()
    gc.collect()
    return outfile


def main():
    usage = "Make spectral image cube for free free emisison using DEM map"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--DEM_file",
        dest="DEM_file",
        default=None,
        help="DEM file name",
        metavar="String",
    )
    parser.add_option(
        "--outfile",
        dest="outfile",
        default="Coronal.h5",
        help="Output file name (.h5)",
        metavar="String",
    )
    parser.add_option(
        "--start_freq",
        dest="start_freq",
        default=850.0,
        help="Start frequency in MHz",
        metavar="Float",
    )
    parser.add_option(
        "--end_freq",
        dest="end_freq",
        default=1500.0,
        help="End frequency in MHz",
        metavar="Float",
    )
    (options, args) = parser.parse_args()
    if options.DEM_file == None or os.path.exists(options.DEM_file) == False:
        print("Please provide correct DEM file path.\n")
        return 1
    else:
        ff_file = simulate_GRFF(
            options.DEM_file,
            options.outfile,
            float(options.start_freq),
            float(options.end_freq),
        )
        print("Free free coronal emission spectral image cube: ", ff_file)
        gc.collect()
        return 0
