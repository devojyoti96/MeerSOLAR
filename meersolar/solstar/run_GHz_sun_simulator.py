import os, glob, gc, time
from optparse import OptionParser
from solstar.aia_download_n_calib import download_aia_data

def get_observatory_info(observatory_name):
    """
    Parameters 
    ----------
    observatory_name : str
        Name of the observatory
    Returns
    -------
    float
        Observatory latitude
    float
        Observatory longitude
    float
        Observatory altitude
    """                
    observatories = {
        "MeerKAT": {"latitude": -30.713, "longitude": 21.443, "altitude": 1038},  # In meters
        "uGMRT": {"latitude": 19.096, "longitude": 74.050, "altitude": 650},  # In meters
        "eOVSA": {"latitude": 37.233, "longitude": -118.283, "altitude": 1222},  # In meters
        "ASKAP": {"latitude": -26.696, "longitude": 116.630, "altitude": 377},  # In meters
        "FASR": {"latitude": 38.430, "longitude": -79.839, "altitude": 820},  # Approximate value
        "SKAO-MID": {"latitude": -30.721, "longitude": 21.411, "altitude": 1060},  # Approximate location
        "JVLA": {"latitude": 34.0784, "longitude": -107.6184, "altitude": 2124},  # In meters
    }
    keys=list(observatories.keys())
    if observatory_name not in keys:
        print ("Observatory: "+observatory_name+" is not in the list.\n")
        print ("Available observatories: MeerKAT, uGMRT, eOVSA, ASKAP, FASR, SKAO-MID, JVLA.\n")
        return 
    else:    
        pos=observatories[observatory_name]
        lat=pos['latitude']
        lon=pos['longitude']
        alt=pos['altitude']
        return lat,lon,alt

def main():
    start_time=time.time()
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
        default=5.0,
        help="Spatial resolution in arcseconds",
        metavar="Float",
    )
    parser.add_option(
        "--observatory",
        dest="observatory_name",
        default=None,
        help="Observatory name (MeerKAT, uGMRT, eOVSA, ASKAP, FASR, SKAO-MID)",
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
        "--ncpu",
        dest="ncpu",
        default=-1,
        help="Number of CPU threads to use",
        metavar="Integer",
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

    ############################################
    # Producing DEM Map
    ############################################
    dem_cmd = (
        "gen_dem --fits_dir "
        + level15_dir
        + " --fov -2000,2000,-2000,2000 --resolution "+str(options.resolution)+" --outfile "
        + str(options.workdir)
        + "/DEM.h5 --ncpu "+str(ncpu)
    )
    print(dem_cmd + "\n")
    os.system(dem_cmd)
    print("#################\n")

    ############################################
    # Calculation of coronal TB
    ############################################
    coronal_tb_cmd = (
        "simulate_coronal_tb --DEM_file "
        + str(options.workdir)
        + "/DEM.h5 --start_freq "
        + str(options.start_freq)
        + " --end_freq "
        + str(options.end_freq)
        + " --outfile "
        + str(options.workdir)
        + "/Coronal.h5"
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
            + "/DEM.h5 --outfile "
            + str(options.workdir)
            + "/Chromo.h5"
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
        + "/Coronal.h5 --chromo_tbfile "
        + str(options.workdir)
        + "/Chromo.h5 --outfile "
        + str(options.workdir)
        + "/TotalTB.h5"
    )
    print(total_tb_cmd + "\n")
    os.system(total_tb_cmd)
    print("#################\n")

    ############################################
    # Making radio spectral cubes
    ############################################
    obslat=options.obs_lat
    obslon=options.obs_lon
    obs_alt=options.obs_alt   
    if options.observatory_name!=None:
        pos=get_observatory_info(options.observatory_name)
        if pos!=None:
            obslat,obslon,obsalt=pos
       
    spectral_cube_cmd = (
        "make_GHz_solar_spectral_cube --total_tb_file "
        + str(options.workdir)
        + "/TotalTB.h5 --obs_time "
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
        + str(options.output_unit)
        + " --make_cube "
        + str(options.make_cube)
        + " --output_prefix "
        + str(options.workdir)
        + "/spectral"
    )
    print(spectral_cube_cmd + "\n")
    os.system(spectral_cube_cmd)
    print("#################\n")
    gc.collect()
    end_time=time.time()
    print ("Total run time: "+str(round(end_time-start_time,1))+"s\n")
    return 0
