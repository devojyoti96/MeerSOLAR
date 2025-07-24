import types
import psutil
import numpy as np
import traceback
import glob
import os
from datetime import datetime as dt, timezone
from .basic_utils import *
from .resource_utils import *

###############################
# Flagging related functions
################################


def do_flag_backup(msname, flagtype="flagdata"):
    """
    Take a flag backup

    Parameters
    ----------
    msname : str
        Measurement set name
    flagtype : str, optional
        Flag type
    """
    from casatools import agentflagger

    af = agentflagger()
    af.open(msname)
    versionlist = af.getflagversionlist()
    if len(versionlist) != 0:
        for version_name in versionlist:
            if flagtype in version_name:
                try:
                    version_num = (
                        int(version_name.split(":")[0].split(" ")[0].split("_")[-1]) + 1
                    )
                except BaseException:
                    version_num = 1
            else:
                version_num = 1
    else:
        version_num = 1
    dt_string = dt.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    af.saveflagversion(
        flagtype + "_" + str(version_num), "Flags autosave on " + dt_string
    )
    af.done()


def get_unflagged_antennas(msname="", scan="", n_threads=-1, dry_run=False):
    """
    Get unflagged antennas of a scan

    Parameters
    ----------
    msname : str
        Name of the measurement set
    scan : str
        Scans

    Returns
    -------
    numpy.array
        Unflagged antenna names
    numpy.array
        Flag fraction list
    """
    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    with suppress_casa_output():
        flag_summary = flagdata(vis=msname, scan=str(scan), mode="summary")
    antenna_flags = flag_summary["antenna"]
    unflagged_antenna_names = []
    flag_frac_list = []
    for ant in antenna_flags.keys():
        flag_frac = antenna_flags[ant]["flagged"] / antenna_flags[ant]["total"]
        if flag_frac < 1.0:
            unflagged_antenna_names.append(ant)
            flag_frac_list.append(flag_frac)
    unflagged_antenna_names = np.array(unflagged_antenna_names)
    flag_frac_list = np.array(flag_frac_list)
    return unflagged_antenna_names, flag_frac_list


def get_chans_flag(msname="", field="", n_threads=-1, dry_run=False):
    """
    Get flag/unflag channel list

    Parameters
    ----------
    msname : str
        Measurement set name
    field : str, optional
        Field name or ID

    Returns
    -------
    list
        Unflag channel list
    list
        Flag channel list
    """
    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    with suppress_casa_output():
        summary = flagdata(vis=msname, field=field, mode="summary", spwchan=True)
    unflag_chans = []
    flag_chans = []
    for chan in summary["spw:channel"]:
        r = summary["spw:channel"][chan]
        chan_number = int(chan.split("0:")[-1])
        flag_frac = r["flagged"] / r["total"]
        if flag_frac == 1:
            flag_chans.append(chan_number)
        else:
            unflag_chans.append(chan_number)
    return unflag_chans, flag_chans


def calc_flag_fraction(msname="", field="", scan="", n_threads=-1, dry_run=False):
    """
    Function to calculate the fraction of total data flagged.


    Parameters
    ----------
    msname : str
        Name of the measurement set
    field : str, optional
        Field names
    scan : str, optional
        Scan names

    Returns
    -------
    float
        Fraction of the total data flagged
    """
    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    with suppress_casa_output():
        summary = flagdata(vis=msname, field=field, scan=scan, mode="summary")
    flagged_fraction = summary["flagged"] / summary["total"]
    return flagged_fraction


def flag_outside_uvrange(vis, uvrange, n_threads=-1, flagbackup=True):
    """
    Flag outside the given uv range

    Parameters
    ----------
    vis : str
        Measurement set name
    uvrange : str
        UV-range
    flagbackup : bool, optional
        Flag backup
    """
    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    try:
        if "lambda" in uvrange:
            islambda = True
            uvrange = uvrange.replace("lambda", "")
        else:
            islambda = False
        if "~" in uvrange:
            low, high = uvrange.split("~")
            if islambda:
                low = f"{low}lambda"
                high = f"{high}lambda"
            cmds = [
                {"mode": "manual", "uvrange": f"<{low}", "flagbackup": flagbackup},
                {"mode": "manual", "uvrange": f">{high}", "flagbackup": flagbackup},
            ]
        elif ">" in uvrange:
            low = uvrange.split(">")[-1]
            if islambda:
                low = f"{low}lambda"
            cmds = [
                {"mode": "manual", "uvrange": f"<{low}", "flagbackup": flagbackup},
            ]
        elif "<" in uvrange:
            high = uvrange.split("<")[-1]
            if islambda:
                high = f"{high}lambda"
            cmds = [
                {"mode": "manual", "uvrange": f">{high}", "flagbackup": flagbackup},
            ]
        else:
            cmds = []
        if len(cmds) > 0:
            for cmd in cmds:
                print(f"Flagging command: {cmd}")
                flagdata(vis=vis, **cmd)
        return 0
    except Exception as e:
        traceback.print_exc()
        return 1


# Expose functions and classes
__all__ = [
    name
    for name, obj in globals().items()
    if (
        (isinstance(obj, types.FunctionType) or isinstance(obj, type))
        and obj.__module__ == __name__
    )
]
