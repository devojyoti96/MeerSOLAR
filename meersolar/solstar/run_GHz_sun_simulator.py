import os, glob, gc, time, psutil
from optparse import OptionParser
from meersolar.solstar.aia_download_n_calib import download_aia_data
from meersolar.solstar.make_ms import *
from meersolar.pipeline.basic_func import get_datadir
from casatasks import casalog

logfile = casalog.logfile()
os.system("rm -rf " + logfile)

datadir=get_datadir()

def main():
    start_time = time.time()
    usage = "Simulate radio spectral cube at GHz frequencies at closest user-given time"
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
        "--workdir",
        dest="workdir",
        default="radio_simulate",
        help="Working directory path",
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
        "--spatial_res",
        dest="resolution",
        default=2.0,
        help="Spatial resolution in arcseconds",
        metavar="Float",
    )
    parser.add_option(
        "--observatory",
        dest="observatory_name",
        default=None,
        help="Observatory name (MeerKAT, uGMRT, eOVSA, ASKAP, FASR, SKAO-MID, SKAO-LOW, JVLA)",
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
        "--cpu_frac",
        dest="cpu_frac",
        default=0.8,
        help="CPU fraction to use",
        metavar="Float",
    )
    parser.add_option(
        "--mem_frac",
        dest="mem_frac",
        default=0.8,
        help="Memory fraction to use",
        metavar="Float",
    )
    parser.add_option(
        "--output_product",
        dest="output_unit",
        default="flux",
        help="Output product, TB: for brightness temperature map, flux: for flux density map",
        metavar="String",
    )
    parser.add_option(
        "--make_cube",
        dest="make_cube",
        default=False,
        help="Make spectral cube or keep spectral slices seperate",
        metavar="Boolean",
    )
    parser.add_option(
        "--make_ms",
        dest="make_ms",
        default=True,
        help="Make simulated measurement set",
        metavar="Boolean",
    )
    parser.add_option(
        "--simulated_ms",
        dest="simulated_ms",
        default="",
        help="Simulated measurement set name",
        metavar="String",
    )
    parser.add_option(
        "--integration_time",
        dest="integration_time",
        default=2.0,
        help="Integration time in seconds",
        metavar="Float",
    )
    parser.add_option(
        "--duration",
        dest="duration",
        default=60.0,
        help="Duration in seconds",
        metavar="Float",
    )
    parser.add_option(
        "--pols",
        dest="pols",
        default="XX YY",
        help="Stokes in measurement set",
        metavar="Float",
    )
    (options, args) = parser.parse_args()

    if options.obs_date == None:
        print("Please provide an observation date.")
        return 1
    if options.obs_time == None:
        print("Please provide an observation time.")
        return 1
    if (
        float(options.start_freq) < 0
        or float(options.end_freq) < 0
        or float(options.freqres) < 0
        or float(options.start_freq) >= float(options.end_freq)
        or float(options.end_freq) - float(options.start_freq) < float(options.freqres)
    ):
        print("Please provide a valid frequency range and resolution in MHz.")
        return 1
    pwd = os.getcwd()
    total_cpus = psutil.cpu_count(logical=True)
    count = 0
    while True:
        available_cpu_pct = 100 - psutil.cpu_percent(interval=1)  # Percent CPUs currently free
        available_cpus = int(total_cpus * available_cpu_pct / 100.0)  # Number of free CPU cores
        usable_cpus = max(1, int(total_cpus * float(options.cpu_frac)))  # Target number of CPU cores we want available based on cpu_frac

        if available_cpus >= usable_cpus:
            # Enough free CPUs, exit loop
            break
        else:
            if count == 0:
                print("Waiting for available free CPUs...")
            time.sleep(5)  # Wait a bit and retry
        count += 1
        
    ##########################################
    # Making workding directory if not present
    ##########################################
    if os.path.dirname(options.workdir) == "":
        options.workdir = os.getcwd() + "/" + options.workdir
    if os.path.exists(options.workdir) == False:
        os.makedirs(options.workdir)
    os.chdir(options.workdir)

    ###########################################
    # Download AIA images
    ###########################################
    msg, level15_dir = download_aia_data(
        obs_date=str(options.obs_date),
        obs_time=str(options.obs_time),
        outdir_prefix=options.workdir + "/aia_data",
    )
    if msg != 0:
        print("############################")
        print("Exiting solstar ...\n")
        print("Error in downloading AIA data : All channels did not download.")
        print("############################")
        return msg
    output_files = glob.glob(level15_dir + "/*")
    if len(output_files) < 7:
        print("############################")
        print("Exiting solstar ...\n")
        print("Error in downloading AIA data : All channels did not download.")
        print("############################")
        return 1
    aia_304 = glob.glob(level15_dir + "/*304*")
    dt_string="".join(options.obs_date.split("-"))+"_"+"".join(options.obs_time.split(":"))
    ############################################
    # Producing DEM Map
    ############################################
    dem_cmd = (
        "gen_dem --fits_dir "
        + level15_dir
        + " --fov -2000,2000,-2000,2000 --resolution "
        + str(options.resolution)
        + " --outfile "
        + str(options.workdir)
        + f"/DEM_{dt_string}.h5 --ncpu "
        + str(usable_cpus)
    )
    print(dem_cmd + "\n")
    msg=os.system(dem_cmd)
    if msg!=0:
        print ("Error in calculating DEM.")
        return 1
    print("#################\n")

    ############################################
    # Calculation of coronal TB
    ############################################
    coronal_tb_cmd = (
        "simulate_coronal_tb --DEM_file "
        + str(options.workdir)
        + f"/DEM_{dt_string}.h5 --start_freq "
        + str(options.start_freq)
        + " --end_freq "
        + str(options.end_freq)
        + " --outfile "
        + str(options.workdir)
        + f"/Coronal_{dt_string}.h5"
    )
    print(coronal_tb_cmd + "\n")
    os.system(coronal_tb_cmd)
    print("#################\n")

    ############################################
    # Calculation of chromospheric TB
    ############################################
    if len(aia_304) > 0:
        aia_304 = aia_304[0]
        chromo_tb_cmd = (
            "simulate_chromo_tb --aia_304 "
            + aia_304
            + " --DEM_file "
            + str(options.workdir)
            + f"/DEM_{dt_string}.h5 --outfile "
            + str(options.workdir)
            + f"/Chromo_{dt_string}.h5"
        )
        print(chromo_tb_cmd + "\n")
        os.system(chromo_tb_cmd)
        print("#################\n")
    else:
        print("AIA 304 angstorm image is not present.\n")
        return 1

    #############################################
    # Calculate spectral image cubes
    #############################################
    total_tb_cmd = (
        "get_total_tb --coronal_tbfile "
        + str(options.workdir)
        + f"/Coronal_{dt_string}.h5 --chromo_tbfile "
        + str(options.workdir)
        + f"/Chromo_{dt_string}.h5 --outfile "
        + str(options.workdir)
        + f"/TotalTB_{dt_string}.h5"
    )
    print(total_tb_cmd + "\n")
    os.system(total_tb_cmd)
    print("#################\n")

    ############################################
    # Making radio spectral cubes
    ############################################
    obslat = options.obs_lat
    obslon = options.obs_lon
    obs_alt = options.obs_alt
    if options.observatory_name != None:
        pos = get_observatory_info(options.observatory_name)
        if pos != None:
            obslat, obslon, obsalt = pos
    make_cube=eval(str(options.make_cube))
    output_unit=options.output_unit
    if eval(str(options.make_ms)):
        imagetype="casa"
        if make_cube:
            print ("Changing to not making spectral cube because simulated measurement set is requested.")
            make_cube=False
        if output_unit=="TB":
            print ("Changing to flux instead of brightness temperature because simulated measurement set is requested.")
            output_unit="flux"
    else:
        imagetype="fits"
            
    spectral_cube_cmd = (
        "simulate_solar_spectral_cube --total_tb_file "
        + str(options.workdir)
        + f"/TotalTB_{dt_string}.h5 --obs_time "
        + str(options.obs_date)
        + "T"
        + str(options.obs_time)
        + " --start_freq "
        + str(options.start_freq)
        + " --end_freq "
        + str(options.end_freq)
        + " --freqres "
        + str(options.freqres)
        + " --obs_lat "
        + str(obslat)
        + " --obs_lon "
        + str(obslon)
        + " --obs_alt "
        + str(obsalt)
        + " --output_product "
        + str(output_unit)
        + " --make_cube "
        + str(make_cube)
        + " --cpu_frac "
        + str(options.cpu_frac)
        + " --mem_frac "
        + str(options.mem_frac)
        + " --output_prefix "
        + str(options.workdir)
        + "/spectral --imagetype "
        + str(imagetype)
    )
    print(spectral_cube_cmd + "\n")
    os.system(spectral_cube_cmd)
    print("#################\n")
    gc.collect()
    
    ####################################
    # Making simulated ms
    ####################################
    config_files=glob.glob(datadir+"/*.config")
    telescope_name=options.observatory_name.lower()
    config_file=""
    for c in config_files:
        if telescope_name in c:
            config_file=c
            break
    if config_file=="":
        print (f"No array configuration file is present for : {telescope_name}.") 
        print ("Could not make the measurement set.")
    else:
        image_cubes=glob.glob(str(options.workdir)+ "/spectral*.image")
        if options.simulated_ms=="":
            simulated_ms=options.workdir+f"/simulated_{dt_string}.ms"
        else:
            simulated_ms=os.path.abspath(options.simulated_ms)
        msg, simulated_ms = generate_ms(
            image_cubes,
            config_file=config_file,
            msname=simulated_ms,
            telescope_name=telescope_name,
            reftime=options.obs_date+"T"+options.obs_time,
            integration_time=float(options.integration_time),
            duration=float(options.duration),
            start_freq=float(options.start_freq),
            end_freq=float(options.end_freq),
            freqres=float(options.freqres),
            pols=options.pols,
        )
        if msg==0:
            print (f"Simulated measurement set: {simulated_ms}")
        else:
            print ("Measurement set simulation is not successful.")
            end_time = time.time()
            print("Total run time: " + str(round(end_time - start_time, 1)) + "s\n")
            return 1
    end_time = time.time()
    print("Total run time: " + str(round(end_time - start_time, 1)) + "s\n")
    return 0
    
if __name__ == "__main__":
    result = main()
    if result > 0:
        result = 1
    print("\n###################\nSolar image simulation is done.\n###################\n")
    os._exit(result)
