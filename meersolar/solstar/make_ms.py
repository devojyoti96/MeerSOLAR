import os, traceback, time, numpy as np
from astropy.coordinates import get_sun
from astropy.time import Time
from casatools import simulator, measures, quanta, table
from meersolar.solstar.make_spectral_cube import *
from optparse import OptionParser
from casatasks import casalog, ft

logfile = casalog.logfile()
os.system("rm -rf " + logfile)
casalog.showconsole(True)

"""
This code is written by Devojyoti Kansabanik, Apr 25, 2025
"""
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
        "meerkat": {
            "latitude": -30.713,
            "longitude": 21.443,
            "altitude": 1038,
        },  # In meters
        "ugmrt": {
            "latitude": 19.096,
            "longitude": 74.050,
            "altitude": 650,
        },  # In meters
        "eovsa": {
            "latitude": 37.233,
            "longitude": -118.283,
            "altitude": 1222,
        },  # In meters
        "askap": {
            "latitude": -26.696,
            "longitude": 116.630,
            "altitude": 377,
        },  # In meters
        "fasr": {
            "latitude": 38.430,
            "longitude": -79.839,
            "altitude": 820,
        },  # Approximate value
        "skao-mid": {
            "latitude": -30.721,
            "longitude": 21.411,
            "altitude": 1060,
        },  # Approximate location
        "skao-low": {
            "latitude": -26.7033,
            "longitude": 116.6319,
            "altitude": 377,
        },  # Approximate location
        "jvla": {
            "latitude": 34.0784,
            "longitude": -107.6184,
            "altitude": 2124,
        },  # In meters
    }
    keys = list(observatories.keys())
    if observatory_name.lower() not in keys:
        print("Observatory: " + observatory_name + " is not in the list.\n")
        print(
            "Available observatories: MeerKAT, uGMRT, eOVSA, ASKAP, FASR, SKAO-MID, JVLA.\n"
        )
        return
    else:
        pos = observatories[observatory_name.lower()]
        lat = pos["latitude"]
        lon = pos["longitude"]
        alt = pos["altitude"]
        return lat, lon, alt
        
def generate_ms(
    imagelist=[],
    config_file="meerkat.config",
    msname="simulated.ms",
    telescope_name="MeerKAT",
    obs_coord={},
    reftime="",
    integration_time=2,
    duration=60,
    start_freq=-1,
    end_freq=-1,
    freqres=-1,
    pols="XX YY",
):
    """
    Parameters
    ----------
    imagelist : list
        Spectral image list
    config_file : str
        Array configuration file
    msname : str
        Name of the simulated measurement set
    telescope_name : str, optional
        Telescope name. If not provided, use the obs_coord.
    obs_coord : dict, optional
        Observatory reference coordinate {'lat':XXdeg,'lon':YYdeg,'alt':ZZm}. Only used when telescope_name is not given.
    reftime : str, optional
        Reference time in UTC (format: YYYY-MM-DDThh:mm:ss). Default is present UTC time.
    integration_time : float
        Integration time in seconds
    duration : float
        Duration of observation in seconds
    start_freq : float
        Start frequency in MHz
    end_freq: float
        End frequency in MHz
    freqres : float
        Frequency resolution in MHz
    pols : str
        pols parameters to simulate
    Returns
    -------
    int
        Successs message
    str
        Simulated measurement set name
    """
    try:
        if len(imagelist)==0:
            print ("Please provide image list.")
            return 1, ""
        msname=msname.rstrip("/")
        if os.path.exists(msname):
            os.system("rm -rf "+msname)
        sm = simulator()
        me = measures()
        qa = quanta()
        if duration == None:
            duration = integration_time
        #############################
        # Observatory info and config
        #############################
        if telescope_name == "":
            if len(obs_coord) == 0:
                print("Please provide either telescope name or array coordinate.")
                return 1, ""
            else:
                obslat = obs_coord["lat"]
                obslon = obs_coord["lon"]
                obsalt = obs_coord["alt"]
        obs_info = get_observatory_info(telescope_name)
        if obs_info == None:
            print("Please provide valid telescope name.")
            return 1, ""
        else:
            obslat, obslon, obsalt = obs_info
        ref_location = [
            qa.quantity(obslat, "deg"),  # Latitude
            qa.quantity(obslon, "deg"),  # Longitude
            qa.quantity(obsalt, "m"),  # Altitude
        ]

        ###############################
        # Antenna parameters
        ###############################
        antenna_params = np.genfromtxt(config_file, usecols=(0, 1, 2))
        antenna_params = np.loadtxt(config_file, dtype="str", unpack=True)
        x = antenna_params[0, :].astype("float")
        y = antenna_params[1, :].astype("float")
        z = antenna_params[2, :].astype("float")
        dish_dia = antenna_params[3, :].astype("float")
        ant_names = antenna_params[4, :]
        if np.nanmean(np.abs(x)) > 10000 or np.nanmean(np.abs(y)) > 10000:
            coordsystem = "global"
        else:
            coordsystem = "local"
            ref_location = []

        ###############################
        # Get solar coordinate
        ###############################
        if reftime=="":
            reftime_astropy = Time.now()
            reftime = reftime_astropy.utc.iso.replace(' ', 'T')
        else:
            reftime_astropy = Time(reftime, format="isot")
        sun_coord = get_sun(reftime_astropy)
        sunra_rad = sun_coord.ra.rad
        sundec_rad = sun_coord.dec.rad
        sun_dir = me.direction("J2000", str(sunra_rad) + "rad", str(sundec_rad) + "rad")

        ###############################
        # Simulating measurement set
        ################################
        sm.open(msname)
        sm.setconfig(
            telescopename=telescope_name,
            x=x,
            y=y,
            z=z,
            dishdiameter=dish_dia,
            mount=["alt-az"],
            coordsystem=coordsystem,
            antname=ant_names,
            referencelocation=ref_location,
        )
        nchan = int((end_freq - start_freq) / freqres)
        if "RR" in pols:
            sm.setfeed("perfect R L")
        elif "XX" in pols:
            sm.setfeed("perfect X Y")
        sm.setspwindow(spwname=f"SPW0",
            freq=str(start_freq) + "MHz",
            deltafreq=str(freqres) + "MHz",
            freqresolution=str(freqres) + "MHz",
            nchannels=nchan,
            stokes=pols,
        )
        
        sm.setfield(sourcename="Sun", sourcedirection=sun_dir)
        sm.setauto(autocorrwt=0.0)
        sm.settimes(
            integrationtime=str(integration_time) + "s",
            referencetime=me.epoch("UTC", reftime),
            usehourangle=False,
        )
        starttime = "0s"
        endtime = str(duration) + "s"
        sm.observe(sourcename="Sun", spwname=f"SPW0", starttime=starttime, stoptime=endtime)
        sm.setdata(spwid=0)
        sm.close()
        if os.path.exists(msname) == False:
            print(f"Simulated measurement set: {msname} could not be made.")
            return 1, ""
        else:
            tb = table()
            tb.open(msname + "/ANTENNA", nomodify=False)
            ants = tb.getcol("NAME")
            ants = ants.astype("<U13")
            for i in range(len(ant_names)):
                ants[i] = ant_names[i]
            tb.putcol("NAME", ants)
            tb.flush()
            tb.close()
            tb.open(msname,nomodify=False)
            flag=tb.getcol("FLAG")
            flag*=False
            tb.putcol("FLAG",flag)
            tb.flush()
            tb.close()
            imagelist=sorted(imagelist)
            for i in range(nchan):
                imagename=imagelist[i]
                print (imagename)
                if i==0:
                    ft(vis=msname,model=imagename,spw=f"0:{i}",incremental=False,usescratch=True)
                else:
                    ft(vis=msname,model=imagename,spw=f"0:{i}",incremental=True,usescratch=False)
            return 0, msname
    except Exception as e:
        traceback.print_exc()
        return 1, ""
        
def main():
    start_time = time.time()
    usage = "Simulate measurement set"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--imagelist",
        dest="imagelist",
        default="",
        help="Spectral image list",
        metavar="String",
    )
    parser.add_option(
        "--config_file",
        dest="config_file",
        default="meerkat.config",
        help="Array configuration file",
        metavar="String",
    )
    parser.add_option(
        "--msname",
        dest="msname",
        default="simulated.ms",
        help="Simulated measurement set name",
        metavar="String",
    )
    parser.add_option(
        "--telescope_name",
        dest="telescope_name",
        default="MeerKAT",
        help="Telescope name",
        metavar="String",
    )
    parser.add_option(
        "--obslat",
        dest="obslat",
        default=None,
        help="Observatory latitude in degree",
        metavar="Float",
    )
    parser.add_option(
        "--obslon",
        dest="obslon",
        default=None,
        help="Observatory longitude in degree",
        metavar="Float",
    )
    parser.add_option(
        "--obsalt",
        dest="obsalt",
        default=None,
        help="Observatory altitude in meters",
        metavar="Float",
    )
    parser.add_option(
        "--reftime",
        dest="reftime",
        default="",
        help="Reference time in UTC (format YYYY-MM-DDThh:mm:ss)",
        metavar="String",
    )
    parser.add_option(
        "--integration_time",
        dest="int_time",
        default=2,
        help="Integration time in seconds",
        metavar="Float",
    )
    parser.add_option(
        "--duration",
        dest="duration",
        default=60,
        help="Observation duration in seconds",
        metavar="Float",
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
        "--pols",
        dest="pols",
        default="XX YY",
        help="pols parameters",
        metavar="String",
    )    
    (options, args) = parser.parse_args()
    try:
        if options.imagelist=="":
            print ("Please provide image list.")
            return 1
        if options.obslat!=None and options.obslon!=None and options.obsalt!=None:
            obs_coord={'lat':float(options.obslat),'lon':float(options.obslon),'alt':float(options.obsalt)}
        else:
            obs_coord={}
        if float(options.start_freq)==-1 or float(options.end_freq)==-1 or float(options.freqres)==-1:
            print ("Please provide correct frequency configuration.")
            return 1
        if os.path.exists(os.path.abspath(options.config_file))==False:
            print (f"{os.path.abspath(options.config_file)} does not exist. Please provide correct telescope configuration file.")
            return 1
        imagelist=options.imagelist.split(",")   
        msg, simulated_ms = generate_ms(
            imagelist,
            config_file=options.config_file,
            msname=options.msname,
            telescope_name=options.telescope_name,
            obs_coord=obs_coord,
            reftime=options.reftime,
            integration_time=float(options.int_time),
            duration=float(options.duration),
            start_freq=float(options.start_freq),
            end_freq=float(options.end_freq),
            freqres=float(options.freqres),
            pols=options.pols,
        )
        print (f"Total time: {round(time.time()-start_time,2)}s")
        return msg
    except Exception as e:
        traceback.print_exc()
        return 1
        
if __name__ == "__main__":
    result = main()
    if result > 0:
        result = 1
    print("\n###################\nMeasurement set simulation is done.\n###################\n")
    os._exit(result)        
        
        
        
        
        
        
