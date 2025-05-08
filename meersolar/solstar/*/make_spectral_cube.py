from scipy.interpolate import interp1d
from sunpy.coordinates import frames, sun
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.wcs import WCS
import astropy.units as u, numpy as np, h5py, os, gc, traceback, time, warnings, copy
from astropy.time import Time
from astropy.io import fits
from sunpy.map import Map
from dask import delayed, compute
from optparse import OptionParser
from astropy.utils.exceptions import AstropyWarning
from meersolar.solstar.basic_func import *
from casatools import image,simulator,measures,quanta, table
from casatasks import exportfits, importfits
from casatasks import casalog

logfile = casalog.logfile()
os.system("rm -rf " + logfile)

# Suppress all Astropy warnings
warnings.simplefilter("ignore", category=AstropyWarning)

def makeimage(ra, dec, freq, freqres, cell, flux, imagename, unit="Jy/pixel"):
    '''
    ra: float
        RA of center of the image in radian
    dec : float
        DEC of center of the image in radian 
    freq: float
        Frequency in Hz
    freqres: float
        Frequency resolution in Hz
    cell: float
        Cell size in arcseconds
    data: numpy.array
        Image data 
    imagename: str
        Output image name
    unit : str, optional
        Image pixel unit
    Returns
    -------
    str
        Output imagename
    '''
    shape = np.shape(flux)
    qa = quanta()
    ia = image()
    # Remove existing image if present
    ia.close()
    if os.path.exists(imagename):
        os.system("rm -rf " + imagename)
    # Create a new empty image
    ia.fromshape(imagename,[shape[0],shape[1],1,1],overwrite=True)
    # Create coordinate system
    cs = ia.coordsys()
    cs.setunits(['rad', 'rad', '', 'Hz'])  # direction1, direction2, stokes, frequency
    cell_rad = qa.convert(qa.quantity(str(cell) + "arcsec"), "rad")['value']
    cs.setincrement([-cell_rad, cell_rad], 'direction')
    cs.setreferencevalue([ra, dec], type="direction")
    cs.setreferencevalue(str(freq)+"Hz", 'spectral')
    cs.setreferencepixel([0], 'spectral')
    cs.setincrement(str(freqres)+"Hz", 'spectral')
    # Apply coordinate system
    ia.setcoordsys(cs.torecord())
    # Set image units and initialize
    ia.setbrightnessunit(unit)
    ia.set(0.0)
    ia.close()
    # Now reopen, put data
    ia.open(imagename)
    data = ia.getchunk()
    data = flux.T
    ia.putchunk(data)
    ia.close()
    return imagename
	
def convert_hpc_to_radec(
    data,
    header,
    obs_time,
    freq,
    freqres,
    output_image,
    obs_lat,
    obs_lon,
    obs_alt,
    datatype,
    imagetype="fits",
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
    output_image : str
        Output CASA image file name
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
    imagetype : str
        Final image type (fits or casa)
    Returns
    -------
    str
        Output image file name 
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
    # Convert reference coordinate to GCRS
    ref_coord_gcrs = ref_coord_arcsec.transform_to("gcrs")
    cell=hpc_header["cdelt2"] 
    if datatype=="TB":
        unit="K"
    else:
        unit="Jy/pixel"
    imagefile=makeimage(ref_coord_gcrs.ra.rad,ref_coord_gcrs.dec.rad,freq,freqres,cell,hpc_data,output_image+".image",unit=unit)
    if imagetype=='casa':
        return imagefile
    else:   
        if os.path.exists(output_image+".fits"):
            os.system("rm -rf "+output_image+".fits")
        exportfits(imagename=imagefile,fitsimage=output_image+".fits",overwrite=True)
        os.system("rm -rf "+imagefile)
        return output_image+".fits"

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
    shape = (tb.shape[0], tb.shape[1], len(new_freqs))
    dtype = "float32"
    # Create a memmap file on disk (choose a suitable path)
    memmap_file = f"{total_tb_file}.dat"  # or use tempfile for auto-cleanup
    # Create the memmap array
    if os.path.exists(memmap_file):
        os.system(f"rm -rf {memmap_file}")
    interpolated_array = np.memmap(memmap_file, dtype=dtype, mode='w+', shape=shape)
    #interpolated_array = np.empty(
    #    (tb.shape[0], tb.shape[1], len(new_freqs)), dtype="float32"
    #)
    # Perform spline interpolation for each pixel across the third axis
    if start_freq * 10**6 < min(freqs) or end_freq * 10**6 > max(freqs):
        print(
            "WARNING! Frequency range is outside data cube range. Extrapolation will be done.\n"
        )
    if len(freqs) < 5:
        interp_mode = "linear"
    else:
        interp_mode = "cubic"
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
        interpolated_array.flush()
    print("Spectral image cube at user given frequency is ready.\n")
    gc.collect()
    return memmap_file, shape, new_freqs, metadata


def make_spectral_slices(
    total_tb_file,
    workdir,
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
    cpu_frac=0.8,
    mem_frac=0.8,
    imagetype="casa",
):
    """
    Parameters
    ----------
    total_tb_file : str
        Total TB file anme (.h5)
    workdir : str
        Work directory
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
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to ise 
    Returns
    -------
    str
        Either spectral image cube name or list of spectral slices
    """
    data_cube_file, data_shape, freqs, metadata = make_spectral_map(
        total_tb_file, start_freq, end_freq, freqres, output_unit=output_unit
    )
    data_cube = np.memmap(data_cube_file, dtype="float32", mode="r", shape=data_shape)
    freqres_Hz = freqres * 10**6  # In Hz
    imagetype_bkp=copy.deepcopy(imagetype)
    if make_cube and imagetype=="casa":
        imagetype="fits"
    try:
        dask_client, dask_cluster, n_workers, threads_per_worker = get_dask_client(len(freqs), dask_dir=workdir, cpu_frac=cpu_frac, mem_frac=mem_frac)
        results = []
        for n in range(0, data_shape[-1], n_workers):
            scattered_slices = dask_client.scatter(
                [data_cube[:, :, i] for i in range(n, n + n_workers)],
                broadcast=False
            )

            tasks = [
                delayed(convert_hpc_to_radec)(
                    scattered_slices[i - n],  # i-n to index locally
                    metadata,
                    obs_time,
                    freqs[i],
                    freqres_Hz,
                    output_fitsfile_prefix+f"_freq_{round(freqs[i]/1e6, 1)}MHz",
                    obs_lat,
                    obs_lon,
                    obs_alt,
                    output_unit,
                    imagetype=imagetype,
                )
                for i in range(n, min(n + n_workers, data_shape[-1]))
            ]

            chunk_results = compute(*tasks)
            results.extend(chunk_results)

        dask_client.close()
        dask_cluster.close()
        os.system("rm -rf "+workdir+"/dask-scratch-space")
        os.system("rm -rf "+data_cube_file)
    except Exception as e:
        traceback.print_exc()
        return 
    if make_cube:
        header = fits.getheader(results[0])
        for i in range(len(results)):
            if i == 0:
                data = fits.getdata(results[i])
                filename = "spectral_cube.dat"
                shape = (
                    1,
                    len(results),
                    data.shape[2],
                    data.shape[3],
                )  # Example large array shape
                dtype = "float32"  # Use lower precision if possible to save memory
                # Create a memory-mapped file for the array
                spectral_array = np.memmap(
                    filename, dtype=dtype, mode="w+", shape=shape
                )
                spectral_array[0, 0, :, :] = data[0, 0, ...]
            else:
                data = fits.getdata(results[i])
                spectral_array[0, i, :, :] = data[0, 0, ...]
        header["CRPIX3"] = float(min(freqs))
        header["CRPIX3"] = float(1.0)
        header["CDELT3"] = freqres * 10**6
        data = np.zeros(shape, dtype=dtype)
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(output_fitsfile_prefix + "_cube.fits", overwrite=True)
        with fits.open(output_fitsfile_prefix + "_cube.fits", mode="update") as hdul:
            for i in range(len(results)):
                hdul[0].data[:, i, :, :] = spectral_array[:, i, :, :]
            hdul.flush()
        for r in results:
            os.system("rm -rf " + r)
        os.system("rm -rf " + filename)
        gc.collect()
        if imagetype_bkp=="casa":
            if os.path.exists(output_fitsfile_prefix + "_cube.image"):
                os.system("rm -rf "+output_fitsfile_prefix + "_cube.image")
            importfits(fitsimage=output_fitsfile_prefix + "_cube.fits",imagename=output_fitsfile_prefix + "_cube.image")
            os.system("rm -rf "+output_fitsfile_prefix + "_cube.fits")
            return output_fitsfile_prefix + "_cube.image"
        else:
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
        "--workdir",
        dest="workdir",
        default=None,
        help="Work directory",
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
    parser.add_option(
        "--imagetype",
        dest="imagetype",
        default="casa",
        help="Image type",
        metavar="String",
    )
    parser.add_option(
        "--cpu_frac",
        dest="cpu_frac",
        default=0.8,
        help="CPU fraction to ise",
        metavar="Float",
    )
    parser.add_option(
        "--mem_frac",
        dest="mem_frac",
        default=0.8,
        help="CPU fraction to ise",
        metavar="Float",
    )
    (options, args) = parser.parse_args()
    if options.total_tb_file == None or os.path.exists(options.total_tb_file) == False:
        print("Please provide correct coronal TB file.\n")
        return 1
    if options.workdir == None or os.path.exists(options.workdir) == False:
        print("Please provide correct work directory.\n")
        return 1
    if (
        float(options.start_freq) < 0
        or float(options.end_freq) < 0
        or float(options.freqres) < 0
    ):
        print("Please provide valid frequency informations in MHz.\n")
        return 1
    try:
        spectral_cube = make_spectral_slices(
            options.total_tb_file,
            options.workdir,
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
            cpu_frac=float(options.cpu_frac),
            mem_frac=float(options.mem_frac),
            imagetype=options.imagetype,
        )
    except Exception as e:
        traceback.print_exc()
        return 1
    if spectral_cube!=None:
        gc.collect()
        return 0
    else:
        return 1
    
if __name__ == "__main__":
    result = main()
    if result > 0:
        result = 1
    os._exit(result)
