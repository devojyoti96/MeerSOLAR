from scipy.interpolate import interp1d
from sunpy.coordinates import frames, sun
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.wcs import WCS
import astropy.units as u, numpy as np, h5py, os, gc
from astropy.time import Time
from astropy.io import fits
from sunpy.map import Map
from joblib import Parallel, delayed
from optparse import OptionParser
import warnings,signal,psutil
from astropy.utils.exceptions import AstropyWarning

# Suppress all Astropy warnings
warnings.simplefilter("ignore", category=AstropyWarning)

def signal_handler(sig, frame):
    print("Interrupt received. Cleaning up...")
    raise SystemExit(0)
    
def convert_hpc_to_radec(
    data,
    header,
    obs_time,
    freq,
    freqres,
    output_fitsfile,
    obs_lat,
    obs_lon,
    obs_alt,
    datatype,
):
    """
    Function to convert a Helioprojective coordinate image into RA/Dec coordinates.
    Parameters
    ----------
    data : numpy.array
        Image data
    header : str
        Image header
    obs_time : str
        Observation time (yyyy-mm-ddThh:mm:ss)
    output_fitsfile : str
        Output FITS file name
    freq : float
        Frequency in Hz
    freqres : float
        Frequency resolution in Hz
    obs_lat : float
        Observatory latitude in degree
    obs_lon : float
        Observatory longitude in degree
    obs_alt : float
        Observatory altitude in meter
    datatype : str
        Data type (TB or flux)
    Returns
    -------
    str
        File name of the FITS file with RA/Dec coordinates.
    """
    hpc_map = Map(data, header)
    # Extract data and header from the input map
    if obs_time == None:
        obstime = header["date-obs"]
    else:
        obstime = obs_time
    P = sun.P(obstime)
    hpc_map = hpc_map.rotate(-P)
    hpc_data = hpc_map.data
    hpc_header = hpc_map.fits_header
    # Observer information
    longitude = obs_lon * u.deg
    latitude = obs_lat * u.deg
    height = obs_alt * u.m
    observer = EarthLocation.from_geodetic(lon=longitude, lat=latitude, height=height)
    gcrs = SkyCoord(observer.get_gcrs(Time(obstime)))
    # Reference coordinate in helioprojective
    ref_coord_arcsec = SkyCoord(
        hpc_header["crval1"] * u.arcsec,
        hpc_header["crval2"] * u.arcsec,
        frame=frames.Helioprojective(observer=gcrs, obstime=obstime),
    )
    hpc_data[np.isnan(hpc_data) == True] = 0.0
    data = hpc_data[np.newaxis, np.newaxis, ...]
    # Convert reference coordinate to GCRS
    ref_coord_gcrs = ref_coord_arcsec.transform_to("gcrs")
    wcs = WCS(hpc_header)
    new_header = wcs.to_header()
    new_header["CTYPE1"] = "RA---SIN"
    new_header["CTYPE2"] = "DEC--SIN"
    new_header["CTYPE3"] = "FREQ"
    new_header["CTYPE4"] = "STOKES"
    new_header["CUNIT1"] = "deg"
    new_header["CUNIT2"] = "deg"
    new_header["CUNIT3"] = "Hz"
    new_header["CUNIT4"] = ""
    new_header["CRVAL1"] = ref_coord_gcrs.ra.deg
    new_header["CRVAL2"] = ref_coord_gcrs.dec.deg
    new_header["CRVAL3"] = freq
    new_header["CRVAL4"] = 1.0000
    new_header["CRPIX1"] = hpc_header["crpix1"]
    new_header["CRPIX2"] = hpc_header["crpix2"]
    new_header["CRPIX3"] = 1.0000
    new_header["CRPIX4"] = 1.0000
    new_header["CDELT1"] = hpc_header["cdelt2"] / 3600.0
    new_header["CDELT2"] = hpc_header["cdelt1"] / 3600.0
    new_header["CDELT3"] = freqres
    new_header["CDELT4"] = 1.0000
    new_header["DATE-OBS"] = obstime
    new_header["EQUINOX"] = 2.000000000000e03
    new_header["RADESYS"] = "FK5"
    if datatype == "TB":
        new_header["BUNIT"] = "K"
    else:
        new_header["BUNIT"] = "Jy"
    new_header["VELREF"] = 257
    new_header["SPECSYS"] = "LSRK"
    # Write to a new FITS file
    if os.path.exists(output_fitsfile):
        os.system("rm -rf " + output_fitsfile)
    fits.writeto(output_fitsfile, data, new_header, overwrite=True)
    gc.collect()
    return output_fitsfile


def make_spectral_map(total_tb_file, start_freq, end_freq, freqres, output_unit="TB"):
    """
    Make spectral cube at user defined spectral resolution
    Parameters
    ----------
    total_tb_file : str
        Total TB file (.h5')
    start_freq : float
        Start frequency in MHz
    end_freq : float
        End frequency in MHz
    freqres : float
        Frequency resolution in MHz
    output_unit : str
        Output spectral cube data unit (TB or flux)
    Returns
    -------
    numpy.array
        Interpolated spectral cube array
    numpy.array
        Frequency array of the spectral cube array in Hz
    dict
        Observation metadata
    """
    print("########################")
    print("Making spectral cube at user given frequencies ......")
    print("########################")
    hf = h5py.File(total_tb_file)
    freqs = hf["frequency"][:] * 10**9  # In Hz
    tb = hf["tb"][:]
    flux = hf["flux_jy"][:]
    keys = hf.attrs.keys()
    metadata = {}
    for key in keys:
        metadata[key] = hf.attrs[key]
    new_freqs = np.arange(start_freq, end_freq, freqres) * 10**6  # In Hz
    # Initialize the output array with the new shape
    interpolated_array = np.empty(
        (tb.shape[0], tb.shape[1], len(new_freqs)), dtype="float32"
    )
    # Perform spline interpolation for each pixel across the third axis
    if start_freq * 10**6 < min(freqs) or end_freq * 10**6 > max(freqs):
        print(
            "WARNING! Frequency range is outside data cube range. Extrapolation will be done.\n"
        )
    if len(freqs)<5:
        interp_mode="linear"
    else:
        interp_mode="cubic" 
    for i in range(tb.shape[0]):
        for j in range(tb.shape[1]):
            if output_unit == "TB":
                f = interp1d(
                    freqs, tb[i, j, :], kind=interp_mode, fill_value="extrapolate"
                )  # Spline interpolation
            else:
                f = interp1d(
                    freqs, flux[i, j, :], kind=interp_mode, fill_value="extrapolate"
                )  # Spline interpolation
            interpolated_array[i, j, :] = f(new_freqs)
    print("Spectral image cube at user given frequency is ready.\n")
    gc.collect()
    return interpolated_array, new_freqs, metadata


def make_spectral_slices(
    total_tb_file,
    obs_time,
    start_freq,
    end_freq,
    freqres,
    output_fitsfile_prefix,
    obs_lat,
    obs_lon,
    obs_alt,
    output_unit="TB",
    make_cube=True,
):
    """
    Parameters
    ----------
    total_tb_file : str
        Total TB file anme (.h5)
    obs_time : str
        Observation time (yyyy-mm-ddThh:mm:ss)
    start_freq : float
        Start frequency in MHz
    end_freq: float
        End frequency in MHz
    freqres : float
        Frequency resolution in MHz
    output_fitsfile_prefix : str
        Output file prefix name
    obs_lat : float
        Observatory latitude in degree
    obs_lon : float
        Observatory longitude in degree
    obs_alt : float
        Observatory altitude in meter
    output_unit : str
       Output spectral cube data unit (TB or flux)
    make_cube : str
        Make spectral cube or keep the frequency slices seperately
    Returns
    -------
    str
        Either spectral image cube name or list of spectral slices
    """
    data_cube, freqs, metadata = make_spectral_map(
        total_tb_file, start_freq, end_freq, freqres, output_unit=output_unit
    )
    freqres_Hz = freqres * 10**6  # In Hz
    signal.signal(signal.SIGINT, signal_handler)
    try:
        n_jobs=psutil.cpu_count()
        results = Parallel(n_jobs=n_jobs,backend='threading')(
            delayed(convert_hpc_to_radec)(
                data_cube[:, :, i],
                metadata,
                obs_time,
                freqs[i],
                freqres_Hz,
                os.path.dirname(total_tb_file)+"/spectral_slice_freq_" + str(round(freqs[i] / 10**6, 1)) + "MHz.fits",
                obs_lat,
                obs_lon,
                obs_alt,
                output_unit,
            )
            for i in range(data_cube.shape[-1])
        )
    except SystemExit:
        print("Exiting cleanly.")    
    if make_cube:
        header = fits.getheader(results[0])
        for i in range(len(results)):
            if i == 0:
                data = fits.getdata(results[i])
                filename = 'spectral_cube.dat'
                shape = (1,len(results),data.shape[2], data.shape[3])  # Example large array shape
                dtype = 'float32'       # Use lower precision if possible to save memory
                # Create a memory-mapped file for the array
                spectral_array = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
                spectral_array[0,0,:,:]=data[0,0,...]
            else:
                data=fits.getdata(results[i])
                spectral_array[0,i,:,:]=data[0,0,...]
        header["CRPIX3"] = float(min(freqs))
        header["CRPIX3"] = float(1.0)
        header["CDELT3"] = freqres * 10**6
        data = np.zeros(shape, dtype=dtype)
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(output_fitsfile_prefix + "_cube.fits", overwrite=True)
        with fits.open(output_fitsfile_prefix + "_cube.fits", mode='update') as hdul:
            for i in range(len(results)):
                hdul[0].data[:,i,:,:] = spectral_array[:,i,:,:]
            hdul.flush()
        for r in results:
            os.system("rm -rf " + r)
        os.system("rm -rf "+filename)    
        gc.collect()
        return output_fitsfile_prefix + "_cube.fits"
    else:
        gc.collect()
        return ",".join(results)


def main():
    usage = "Make total brightness temperature radio map"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--total_tb_file",
        dest="total_tb_file",
        default=None,
        help="Total TB file name (.h5)",
        metavar="String",
    )
    parser.add_option(
        "--obs_time",
        dest="obs_time",
        default=None,
        help="Observation time (yyyy-mm-ddThh:mm:ss)",
        metavar="String",
    )
    parser.add_option(
        "--start_freq",
        dest="start_freq",
        default=-1,
        help="Start frequency in MHz",
        metavar="Float",
    )
    parser.add_option(
        "--end_freq",
        dest="end_freq",
        default=-1,
        help="End frequency in MHz",
        metavar="Float",
    )
    parser.add_option(
        "--freqres",
        dest="freqres",
        default=-1,
        help="Frequency resolution in MHz",
        metavar="Float",
    )
    parser.add_option(
        "--output_prefix",
        dest="output_fitsfile_prefix",
        default="spectral",
        help="Output fits image prefix name",
        metavar="String",
    )
    parser.add_option(
        "--obs_lat",
        dest="obs_lat",
        default=0.0,
        help="Observatory latitude in degree",
        metavar="Float",
    )
    parser.add_option(
        "--obs_lon",
        dest="obs_lon",
        default=0.0,
        help="Observatory longitude in degree",
        metavar="Float",
    )
    parser.add_option(
        "--obs_alt",
        dest="obs_alt",
        default=0.0,
        help="Observatory altitude in meter",
        metavar="Float",
    )
    parser.add_option(
        "--output_product",
        dest="output_unit",
        default="TB",
        help="Output product, TB: for brightness temperature map, flux: for flux density map",
        metavar="String",
    )
    parser.add_option(
        "--make_cube",
        dest="make_cube",
        default=True,
        help="Make spectral cube or keep spectral slices seperate",
        metavar="Boolean",
    )
    (options, args) = parser.parse_args()
    if options.total_tb_file == None or os.path.exists(options.total_tb_file) == False:
        print("Please provide correct coronal TB file.\n")
        return 1
    if (
        float(options.start_freq) < 0
        or float(options.end_freq) < 0
        or float(options.freqres) < 0
    ):
        print("Please provide valid frequency informations in MHz.\n")
        return 1
    spectral_cube = make_spectral_slices(
        options.total_tb_file,
        str(options.obs_time),
        float(options.start_freq),
        float(options.end_freq),
        float(options.freqres),
        options.output_fitsfile_prefix,
        float(options.obs_lat),
        float(options.obs_lon),
        float(options.obs_alt),
        output_unit=str(options.output_unit),
        make_cube=eval(str(options.make_cube)),
    )
    print("Spectral image(s) is(are) made: ", spectral_cube)
    gc.collect()
    return 0
