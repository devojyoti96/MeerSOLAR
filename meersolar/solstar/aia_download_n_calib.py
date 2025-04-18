import astropy.units as u, os, glob, time
from sunpy.net import Fido, attrs as a
from sunpy.map import Map
from aiapy.calibrate import *
from sunpy import config
from pathlib import Path
from optparse import OptionParser
import warnings, logging, gc

logging.getLogger("sunpy").setLevel(logging.ERROR)
logging.getLogger("parfive").setLevel(logging.ERROR)
# Suppress all warnings
warnings.filterwarnings("ignore")

# Set maximum timeout globally
config.set("downloads", "timeout", "30")


def download_aia_data(
    obs_date="2021-01-01", obs_time="09:00:00", outdir_prefix="aia_data"
):
    """
    Parameters
    ----------
    obs_date : str
        Observation date (yyyy-mm-dd)
    obs_time : str
        Observation time in UTC (hh:mm:ss)
    outdir_prefix : str
        Output directory prefix path
    Returns
    -------
    msg : int
        Success message
    str
        Output directory name for AIA level 1.5 images
    """
    # Step 1: Define the time range and wavelength for 2023-12-04
    time_range = a.Time(
        obs_date + "T" + obs_time,
        obs_date + "T" + ":".join(obs_time.split(":")[:-1]) + ":15",
    )
    alt_time_range = a.Time(obs_date + "T00:00:00", obs_date + "T00:00:30")
    date_str = "_".join(obs_date.split("-"))
    time_str = "_".join(obs_time.split(":"))
    output_dir_name = outdir_prefix + "_" + date_str + "_" + time_str
    output_dir = Path(output_dir_name)
    output_dir.mkdir(exist_ok=True)
    level1_dir_name = output_dir_name + "/level_1"
    level15_dir_name = output_dir_name + "/level_15"
    level1_dir = Path(level1_dir_name)
    level15_dir = Path(level15_dir_name)
    level1_dir.mkdir(exist_ok=True)
    level15_dir.mkdir(exist_ok=True)
    os.system("rm -rf " + level1_dir_name + "/* " + level15_dir_name + "/*")
    # Step 2: Search for Level 1 AIA data using Fido
    print("############################")
    print("Searching for AIA Level 1 data for " + obs_date + "...")
    print("############################")
    wavelengths = [94, 131, 171, 193, 211, 304, 335]
    use_alt_timerange = False
    for w in wavelengths:
        if use_alt_timerange == False:
            search_result = Fido.search(
                time_range, a.Instrument.aia, a.Wavelength(w * u.angstrom)
            )
            # Check if results are found
            if len(search_result) == 0:
                print("No data found for the specified time range and wavelength.")
            else:
                # Step 3: Download the data
                print(
                    "Downloading AIA Level 1 data for wavelength "
                    + str(w)
                    + " angstorm..."
                )
                try:
                    downloaded_files = Fido.fetch(
                        search_result, path=str(level1_dir / "{file}"), progress=True
                    )
                except Exception as e:
                    print("Exception: ", e)
            w_files = glob.glob(level1_dir_name + "/*" + str(w) + "*")
            if len(w_files) == 0:
                use_alt_timerange = True
                print("#####################################")
            elif len(w_files) > 1:
                for i in range(1, len(w_files)):
                    os.system("rm -rf " + w_files[i])
            time.sleep(1)
        if use_alt_timerange == True:
            time_range = alt_time_range
            search_result = Fido.search(
                time_range, a.Instrument.aia, a.Wavelength(w * u.angstrom)
            )
            # Check if results are found
            if len(search_result) == 0:
                print("############################")
                print("Error in downloading AIA data : No data is found in the given timerange.")
                print("############################")
                return 1,None
            else:
                # Step 3: Download the data
                print(
                    "Downloading AIA Level 1 data using alternate time range for wavelength "
                    + str(w)
                    + " angstorm..."
                )
                for i in range(len(search_result[0])):
                    downloaded_files = Fido.fetch(
                        search_result[0, i],
                        path=str(level1_dir / "{file}"),
                        progress=True,
                    )
                    if len(downloaded_files) > 0:
                        break

    level1_files = glob.glob(level1_dir_name + "/*")
    print("Downloaded level 1 AIA fits: " + ",".join(level1_files) + "\n")

    if len(level1_files) >= 7:
        # Step 4: Calibrate the data to Level 1.5
        print("############################")
        print("Calibrating AIA data to Level 1.5...")
        print("############################")

        for lev1_file in level1_files:
            try: #TODO: pointing correction and degradation correction
                # Load the Level 1 data as a SunPy Map
                aia_map = Map(lev1_file)
                # Step 1: Pointing correction
                try:    
                    pointing_corrected_map=update_pointing(aia_map)
                except:
                    print ("WARNING! AIA pointing correction could not be done.")
                    pointing_corrected_map=aia_map
                # Step 2: register (we are skipping PSF deconvolution)
                registered_map = register(pointing_corrected_map)
                # Step 3: instrument degradation correction
                try:
                    corrected_map=correct_degradation(registered_map)
                except:
                    print ("WARNING! Instrument degradation could not be corrected.")
                    corrected_map=registered_map
                
                # Step 4: Normalize by exposure time
                normalized_data = (
                    corrected_map.data / corrected_map.exposure_time.to(u.s).value
                )
                normalized_map = Map(normalized_data, corrected_map.meta)
                # Step 5: Save the calibrated Level 1.5 map
                x = os.path.basename(lev1_file).split(".")
                output_file = str(
                    level15_dir_name
                    + "/aia.level15."
                    + x[2]
                    + x[3]
                    + ".image_lev15.fits"
                )
                normalized_map.save(output_file, overwrite=True)
                print(f"Calibrated Level 1.5 data saved to: {output_file}")
                print("############################")
            except Exception as e:
                print(f"Failed to process {lev1_file}: {e}")
        print("############################")
        print("Calibration complete. All files saved to:" + level15_dir_name + "\n")
        if use_alt_timerange:
            print(
                "Used alternative timestamp since requestec timestamp data could noe be downloaded.\n"
            )
        print("############################")
        gc.collect()
        return 0, level15_dir_name
    else:
        print("############################")
        print("Error in downloading AIA data : All channels did not download.")
        print("############################")
        gc.collect()
        return 1, None


def main():
    usage = "Download and calibrate AIA images at closest user-given time"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--obs_date",
        dest="obs_date",
        default=None,
        help="Observation date (yyyy-mm-dd)",
        metavar="String",
    )
    parser.add_option(
        "--obs_time",
        dest="obs_time",
        default=None,
        help="Observation time (hh:mm:ss)",
        metavar="String",
    )
    parser.add_option(
        "--outdir_prefix",
        dest="outdir_prefix",
        default="aia_data",
        help="Output directory prefix path",
        metavar="String",
    )
    (options, args) = parser.parse_args()
    if options.obs_date == None:
        print("Please provide an observation date.\n")
        return 1
    if options.obs_time == None:
        options.obs_time = "00:00:00"
    if os.path.dirname(options.outdir_prefix) == "":
        options.outdir_prefix = os.getcwd() + "/" + options.outdir_prefix
    msg, outdir = download_aia_data(
        obs_date=str(options.obs_date),
        obs_time=str(options.obs_time),
        outdir_prefix=options.outdir_prefix,
    )
    output_files = glob.glob(outdir + "/*")
    if len(output_files) >= 7:
        print("############################")
        print("Output directory name: ", outdir)
        print("############################")
        gc.collect()
        return 0
    else:
        print("Error in downloading all channels.")
        gc.collect()
        return 1
