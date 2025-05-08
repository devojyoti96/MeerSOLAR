# This script uses JSOC's DRMS API to download AIA Level 1 data and calibrate it to Level 1.5 using aiapy
import drms
import os
import time
import glob
import traceback
import astropy.units as u
from sunpy.map import Map
from aiapy.calibrate import *
from aiapy.calibrate.util import *
from pathlib import Path
from optparse import OptionParser
import warnings, logging, gc

logging.getLogger("sunpy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def download_aia_data(
    obs_date="2021-01-01", obs_time="09:00:00", outdir_prefix="aia_data", email="devojyoti96@gmail.com"
):
    start_time=time.time()
    date_str = "_".join(obs_date.split("-"))
    time_str = "_".join(obs_time.split(":"))
    output_dir = outdir_prefix + "_" + date_str + "_" + time_str
    level1_dir = output_dir +"/level_1"
    level15_dir = output_dir +"/level_15"
    os.makedirs(level1_dir,exist_ok=True)
    os.makedirs(level15_dir,exist_ok=True)
    os.system("rm -rf "+level1_dir+"/*")
    os.system("rm -rf "+level15_dir+"/*")
    wavelengths = [94, 131, 171, 193, 211, 304, 335]
    wavelength_str = "{" + ",".join([str(w) for w in wavelengths]) + "}"
    drms_time = obs_date.replace("-", ".") + "_" + obs_time + "_TAI/12s"
    query_string = f"aia.lev1_euv_12s[{drms_time}]"+"{image}"
    print(f"📡 Submitting JSOC Level 1 request: {query_string}")
    client = drms.Client(email=email)

    try:
        result = client.export(query_string, protocol="as-is")
        files = result.download(level1_dir)
    except Exception as e:
        print("Download failed:", e)
        traceback.print_exc()
        return 1, None

    level1_files = glob.glob(level1_dir +"/*.fits")
    if len(level1_files) < len(wavelengths):
        print("Error: not all wavelength channels downloaded.")
        return 1, None

    print("✅ Download complete. Calibrating to Level 1.5...")

    pointing_table = None
    correction_table = None

    for lev1_file in level1_files:
        try:
            aia_map = Map(lev1_file)
            if pointing_table is None:
                pointing_table = get_pointing_table(
                    "JSOC",
                    time_range=(aia_map.date - 12 * u.h, aia_map.date + 12 * u.h),
                )
            # Pointing correction
            try:
                aia_map = update_pointing(aia_map, pointing_table=pointing_table)
            except:
                pass
            # Register
            aia_map = register(aia_map)
            # Degradation correction
            if correction_table is None:
                correction_table = get_correction_table(source="jsoc")
            try:
                aia_map = correct_degradation(aia_map, correction_table=correction_table)
            except:
                pass
            # Normalize
            normalized_data = aia_map.data / aia_map.exposure_time.to(u.s).value
            aia_map = Map(normalized_data, aia_map.meta)
            # Save
            lev1_file_name=os.path.basename(lev1_file)
            out_file = level15_dir +"/"+lev1_file_name.replace('lev1','lev15')
            aia_map.save(str(out_file))
            print(f"Saved Level 1.5: {out_file}")
        except Exception as e:
            print(f"Failed to calibrate {lev1_file}: {e}")
            traceback.print_exc()
    print (f"Total time taken: {round(time.time()-start_time,2)}")
    print ("####################################################")
    return 0, str(level15_dir)


def main():
    usage = "Download and calibrate AIA Level 1 images to Level 1.5 using aiapy"
    parser = OptionParser(usage=usage)
    parser.add_option("--obs_date", dest="obs_date", default="2021-01-01")
    parser.add_option("--obs_time", dest="obs_time", default="00:00:00")
    parser.add_option("--outdir_prefix", dest="outdir_prefix", default="aia_data")
    parser.add_option("--email", dest="email", default="devojyoti96@gmail.com")
    (options, args) = parser.parse_args()

    if options.obs_date is None:
        print("Please provide an observation date.")
        return 1

    msg, outdir = download_and_calibrate_aia(
        obs_date=options.obs_date,
        obs_time=options.obs_time,
        outdir_prefix=options.outdir_prefix,
        email=options.email,
    )

    if msg == 0:
        print(f"All files saved to: {outdir}")
    else:
        print("Some or all files failed.")

    return msg


if __name__ == "__main__":
    result = main()
    os._exit(result)

