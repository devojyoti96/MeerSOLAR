import types
import astropy.units as u
import glob
import os
from astroquery.jplhorizons import Horizons
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from casatools import msmetadata
from .basic_utils import *
from .udocker_utils import *
from .ms_metadata import *

#####################################
# Sun position related
#####################################


def get_solar_elevation(lat, lon, elev, date_time):
    """
    Get solar elevation

    Parameters
    ----------
    lat : float
        Latitude in degrees
    lon : float
        Longitude in degrees
    elev : float
        Elevation in degrees
    date_time : str
        Date time in YYYY-MM-DDThh:mm:ss (ISOT) format, default : present time


    Returns
    -------
    float
        Solar elevation in degree
    """
    latitude = lat * u.deg  # In degree
    longitude = lon * u.deg  # In degree
    elevation = elev * u.m  # In meter
    if date_time == "":
        astro_time = Time.now()
    else:
        astro_time = Time(date_time)
    location = EarthLocation(lat=latitude, lon=longitude, height=elevation)
    sun_jpl = Horizons(id="10", location="500", epochs=astro_time.jd)
    eph = sun_jpl.ephemerides()
    sun_coords = SkyCoord(
        ra=eph["RA"][0] * u.deg, dec=eph["DEC"][0] * u.deg, frame="icrs"
    )
    altaz_frame = AltAz(obstime=astro_time, location=location)
    sun_altaz = sun_coords.transform_to(altaz_frame)
    solar_elevation = sun_altaz.alt.deg
    return round(solar_elevation, 3)


def radec_sun(msname):
    """
    RA DEC of the Sun at the start of the scan

    Parameters
    ----------
    msname : str
        Name of the measurement set

    Returns
    -------
    str
        RA DEC of the Sun in J2000
    str
        RA string
    str
        DEC string
    float
        RA in degree
    float
        DEC in degree
    """
    msmd = msmetadata()
    msmd.open(msname)
    times = msmd.timesforspws(0)
    msmd.close()
    msmd.done()
    mid_time = times[int(len(times) / 2)]
    mid_timestamp = mjdsec_to_timestamp(mid_time)
    astro_time = Time(mid_timestamp, scale="utc")
    sun_jpl = Horizons(id="10", location="500", epochs=astro_time.jd)
    eph = sun_jpl.ephemerides()
    sun_coord = SkyCoord(
        ra=eph["RA"][0] * u.deg, dec=eph["DEC"][0] * u.deg, frame="icrs"
    )
    sun_ra = (
        str(int(sun_coord.ra.hms.h))
        + "h"
        + str(int(sun_coord.ra.hms.m))
        + "m"
        + str(round(sun_coord.ra.hms.s, 2))
        + "s"
    )
    sun_dec = (
        str(int(sun_coord.dec.dms.d))
        + "d"
        + str(abs(int(sun_coord.dec.dms.m)))
        + "m"
        + str(abs(round(sun_coord.dec.dms.s, 2)))
        + "s"
    )
    sun_radec_string = "J2000 " + str(sun_ra) + " " + str(sun_dec)
    radeg = sun_coord.ra.deg
    radeg = radeg % 360
    decdeg = sun_coord.dec.deg
    decdeg = decdeg % 360
    return sun_radec_string, sun_ra, sun_dec, radeg, decdeg


def move_to_sun(msname, only_uvw=False):
    """
    Move the phasecenter of the measurement set at the center of the Sun (Assuming ms has one scan)

    Parameters
    ----------
    msname : str
        Name of the measurement set
    only_uvw : bool, optional
        Note: This is required when visibilities are properly phase rotated in correlator to track the Sun,
        but while creating the MS, UVW values are estimated using a wrong phase center at the start of solar center at the start.

    Returns
    -------
    int
        Success message
    """
    sun_radec_string, sunra, sundec, sunra_deg, sundec_deg = radec_sun(msname)
    msg = run_chgcenter(
        msname, sunra, sundec, only_uvw=only_uvw, container_name="solarwsclean"
    )
    if msg != 0:
        print("Phasecenter could not be shifted.")
    return msg


def correct_solar_sidereal_motion(msname="", verbose=False, dry_run=False):
    """
    Correct sodereal motion of the Sun

    Parameters
    ----------
    msname : str
        Name of the measurement set

    Returns
    -------
    int
        Success message
    """
    if dry_run:
        mem = run_solar_sidereal_cor(dry_run=True)
        return mem
    print(f"Correcting sidereal motion for ms: {msname}\n")
    if os.path.exists(msname + "/.sidereal_cor") == False:
        msg = run_solar_sidereal_cor(
            msname=msname, container_name="solarwsclean", verbose=verbose
        )
        if msg != 0:
            print("Sidereal motion correction is not successful.")
        else:
            os.system("touch " + msname + "/.sidereal_cor")
        return msg
    else:
        print(f"Sidereal motion correction is already done for ms: {msname}")
        return 0


# Expose functions and classes
__all__ = [
    name
    for name, obj in globals().items()
    if (
        (isinstance(obj, types.FunctionType) or isinstance(obj, type))
        and obj.__module__ == __name__
    )
]
