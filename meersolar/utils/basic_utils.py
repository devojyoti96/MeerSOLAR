import types
import julian
import resource
import numpy as np
import os
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import Angle
from datetime import datetime as dt
from contextlib import contextmanager

##########################
# Basic utility funactions
##########################
@contextmanager
def suppress_casa_output():
    """
    Supress CASA terminal output
    """
    with open(os.devnull, "w") as fnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(fnull.fileno(), 1)
        os.dup2(fnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)


def get_cachedir():
    """
    Get cache directory
    """
    homedir = os.environ.get("HOME")
    if homedir is None:
        homedir = os.path.expanduser("~")
    username = os.getlogin()
    cachedir = f"{homedir}/.solarpipe"
    os.makedirs(cachedir, exist_ok=True)
    os.makedirs(f"{cachedir}/pids", exist_ok=True)
    return cachedir


def create_datadir(datadir=""):
    """
    Create data directory

    Parameters
    ----------
    datadir : str, optional
        User provided custom data directory
    """
    cachedir = get_cachedir()
    if datadir == "":
        datadir = f"{cachedir}/solarpipe_data"
    os.makedirs(datadir, exist_ok=True)
    with open(f"{cachedir}/solarpipe_data_dir.txt", "w") as f:
        f.write(str(datadir) + "\n")
    return


def get_datadir():
    """
    Get package data directory

    Returns
    -------
    str
        Data directory
    """
    cachedir = get_cachedir()
    if os.path.exists(f"{cachedir}/solarpipe_data_dir.txt") == False:
        return None
    with open(f"{cachedir}/solarpipe_data_dir.txt", "r") as f:
        datadir = f.read().strip()
    os.makedirs(datadir, exist_ok=True)
    return datadir


def split_into_chunks(lst, target_chunk_size):
    """
    Split a list into equal number of elements

    Parameters
    ----------
    lst : list
        List of numbers
    target_chunk_size: int
        Number of elements per chunk

    Returns
    -------
    list
        Chunked list
    """
    n = len(lst)
    num_chunks = max(1, round(n / target_chunk_size))
    avg_chunk_size = n // num_chunks
    remainder = n % num_chunks

    chunks = []
    start = 0
    for i in range(num_chunks):
        extra = 1 if i < remainder else 0  # Distribute remainder
        end = start + avg_chunk_size + extra
        chunks.append(lst[start:end])
        start = end
    return chunks


def average_timestamp(timestamps):
    """
    Compute the average timestamp using astropy from a list of ISO 8601 strings.

    Parameters
    ----------
    timestamps : list
        timestamps (list of str): List of timestamp strings in 'YYYY-MM-DDTHH:MM:SS' format.

    Returns
    --------
    str
        Average timestamp in 'YYYY-MM-DDTHH:MM:SS' format.
    """
    if len(timestamps) == 0:
        return ""
    times = Time(timestamps, format="isot", scale="utc")
    avg_time = Time(np.mean(times.jd), format="jd", scale="utc")
    return avg_time.isot.split(".")[0]  # Strip milliseconds for clean output


def ceil_to_multiple(n, base):
    """
    Round up to the next multiple

    Parameters
    ----------
    n : float
        The number
    base : float
        Whose multiple will be

    Returns
    -------
    float
        The modified number
    """
    return ((n // base) + 1) * base


def angular_separation_equatorial(ra1, dec1, ra2, dec2):
    """
    Calculate angular seperation between two equatorial coordinates

    Parameters
    ----------
    ra1 : float
        RA of the first coordinate in degree
    dec1 : float
        DEC of the first coordinate in degree
    ra2 : float
        RA of the second coordinate in degree
    dec2 : float
        DEC of the second coordinate in degree

    Returns
    -------
    float
        Angular distance in degree
    """
    # Convert RA and Dec from degrees to radians
    ra1 = np.radians(ra1)
    ra2 = np.radians(ra2)
    dec1 = np.radians(dec1)
    dec2 = np.radians(dec2)
    # Apply the spherical distance formula using NumPy functions
    cos_theta = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(
        ra1 - ra2
    )
    # Calculate the angular separation in radians
    theta_rad = np.arccos(cos_theta)
    # Convert the angular separation from radians to degrees
    theta_deg = np.degrees(theta_rad)
    return round(theta_deg, 2)


def ra_dec_to_deg(ra_hms, dec_dms):
    """
    Convert RA and Dec from hms and dms format to degrees

    Parameters
    ----------
    ra_hms: str
        Right Ascension in 'hms' format
    dec_dms : str
        Declination in 'dms' format

    Returns
    -------
    tuple
        RA and Dec in degrees
    """
    ra = Angle(ra_hms, unit=u.hourangle)
    dec = Angle(dec_dms, unit=u.deg)
    return round(ra.deg, 4), round(dec.deg, 4)


def ra_dec_to_hms_dms(ra_deg, dec_deg):
    """
    Convert RA and Dec in degrees to hms and dms format

    Parameters
    ----------
    ra_deg : float
        Right Ascension in degrees.
    dec_deg : float
        Declination in degrees.

    Returns
    -------
    tuple
        RA in h:m:s format, Dec in d:m:s format (e.g., '1h5m0s', '1d5m0s').
    """
    # Convert RA to h:m:s
    if ra_deg < 0:
        ra_deg += 360
    ra = Angle(ra_deg, unit=u.deg)
    ra_hms = ra.to_string(unit=u.hourangle, sep=":").split(":")
    ra_hms = ra_hms[0] + "h" + ra_hms[1] + "m" + ra_hms[2] + "s"
    # Convert Dec to d:m:s
    dec = Angle(dec_deg, unit=u.deg)
    dec_dms = dec.to_string(unit=u.deg, sep=":").split(":")
    dec_dms = dec_dms[0] + "d" + dec_dms[1] + "m" + dec_dms[2] + "s"
    return ra_hms, dec_dms


def timestamp_to_mjdsec(timestamp, date_format=0):
    """
    Convert timestamp to mjd second.


    Parameters
    ----------
    timestamp : str
        Time stamp to convert
    date_format : int, optional
        Datetime string format
            0: 'YYYY/MM/DD/hh:mm:ss'

            1: 'YYYY-MM-DDThh:mm:ss'

            2: 'YYYY-MM-DD hh:mm:ss'

            3: 'YYYY_MM_DD_hh_mm_ss'

    Returns
    -------
    float
        Return correspondong MJD second of the day
    """
    if date_format == 0:
        try:
            timestamp_datetime = dt.strptime(timestamp, "%Y/%m/%d/%H:%M:%S.%f")
        except BaseException:
            timestamp_datetime = dt.strptime(timestamp, "%Y/%m/%d/%H:%M:%S")
    elif date_format == 1:
        try:
            timestamp_datetime = dt.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        except BaseException:
            timestamp_datetime = dt.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")
    elif date_format == 2:
        try:
            timestamp_datetime = dt.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
        except BaseException:
            timestamp_datetime = dt.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    elif date_format == 3:
        try:
            timestamp_datetime = dt.strptime(timestamp, "%Y_%m_%d_%H_%M_%S.%f")
        except BaseException:
            timestamp_datetime = dt.strptime(timestamp, "%Y_%m_%d_%H_%M_%S")
    else:
        print("No proper format of timestamp.\n")
        return
    mjd = float(
        "{: .2f}".format(
            (julian.to_jd(timestamp_datetime) - 2400000.5) * (24.0 * 3600.0)
        )
    )
    return mjd


def mjdsec_to_timestamp(mjdsec, str_format=0):
    """
    Convert CASA MJD seceonds to CASA timestamp

    Parameters
    ----------
    mjdsec : float
            CASA MJD seconds
    str_format : int
        Time stamp format (0: yyyy-mm-ddTHH:MM:SS.ff, 1: yyyy/mm/dd/HH:MM:SS.ff, 2: yyyy-mm-dd HH:MM:SS)

    Returns
    -------
    str
            CASA time stamp in UTC at ISOT format
    """
    from casatools import measures, quanta

    me = measures()
    qa = quanta()
    today = me.epoch("utc", "today")
    mjd = np.array(mjdsec) / 86400.0
    today["m0"]["value"] = mjd
    hhmmss = qa.time(today["m0"], prec=8)[0]
    date = qa.splitdate(today["m0"])
    qa.done()
    if str_format == 0:
        utcstring = "%s-%02d-%02dT%s" % (
            date["year"],
            date["month"],
            date["monthday"],
            hhmmss,
        )
    elif str_format == 1:
        utcstring = "%s/%02d/%02d/%s" % (
            date["year"],
            date["month"],
            date["monthday"],
            hhmmss,
        )
    else:
        utcstring = "%s-%02d-%02d %s" % (
            date["year"],
            date["month"],
            date["monthday"],
            hhmmss,
        )
    return utcstring


# Exposing only functions
__all__ = [
    name
    for name, obj in globals().items()
    if isinstance(obj, types.FunctionType) and obj.__module__ == __name__
]
