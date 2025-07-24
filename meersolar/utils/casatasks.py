import types
import psutil
import numpy as np
import glob
import os
import traceback
from casatools import msmetadata, ms as casamstool, table
from .basic_utils import *
from .resource_utils import *

#############################
# General CASA tasks
#############################


def check_scan_in_caltable(caltable, scan):
    """
    Check scan number available in caltable or not

    Parameters
    ----------
    caltable : str
        Name of the caltable
    scan : int
        Scan number

    Returns
    -------
    bool
        Whether scan is present in the caltable or not
    """
    tb = table()
    tb.open(caltable)
    scans = tb.getcol("SCAN_NUMBER")
    tb.close()
    if int(scan) in scans:
        return True
    else:
        return False


def reset_weights_and_flags(
    msname="", restore_flag=True, force_reset=False, n_threads=-1, dry_run=False
):
    """
    Reset weights and flags for the ms

    Parameters
    ----------
    msname : str
        Measurement set
    restore_flag : bool, optional
        Restore flags or not
    force_reset : bool, optional
        Force reset
    """
    limit_threads(n_threads=n_threads)
    from casatasks import flagdata

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    if os.path.exists(f"{msname}/.reset") == False or force_reset:
        mspath = os.path.dirname(os.path.abspath(msname))
        os.chdir(mspath)
        if restore_flag:
            print(f"Restoring flags of measurement set : {msname}")
            if os.path.exists(msname + ".flagversions"):
                os.system("rm -rf " + msname + ".flagversions")
            flagdata(vis=msname, mode="unflag", flagbackup=False)
        print(f"Resetting previous weights of the measurement set: {msname}")
        msmd = msmetadata()
        msmd.open(msname)
        npol = msmd.ncorrforpol()[0]
        msmd.close()
        tb = table()
        tb.open(msname, nomodify=False)
        colnames = tb.colnames()
        nrows = tb.nrows()
        if "WEIGHT" in colnames:
            print(f"Resetting weight column to ones of measurement set : {msname}.")
            weight = np.ones((npol, nrows))
            tb.putcol("WEIGHT", weight)
        if "SIGMA" in colnames:
            print(f"Resetting sigma column to ones of measurement set: {msname}.")
            sigma = np.ones((npol, nrows))
            tb.putcol("SIGMA", sigma)
        if "WEIGHT_SPECTRUM" in colnames:
            print(f"Removing weight spectrum of measurement set: {msname}.")
            tb.removecols("WEIGHT_SPECTRUM")
        if "SIGMA_SPECTRUM" in colnames:
            print(f"Removing sigma spectrum of measurement set: {msname}.")
            tb.removecols("SIGMA_SPECTRUM")
        tb.flush()
        tb.close()
        os.system(f"touch {msname}/.reset")
    return


def correct_missing_col_subms(msname):
    """
    Correct for missing colurmns in sub-MSs

    Parameters
    ----------
    msname : str
        Name of the measurement set
    """
    tb = table()
    colname_list = []
    sub_mslist = glob.glob(msname + "/SUBMSS/*.ms")
    for ms in sub_mslist:
        tb.open(ms)
        colname_list.append(tb.colnames())
        tb.close()
    sets = [set(sublist) for sublist in colname_list]
    if len(sets) > 0:
        common_elements = set.intersection(*sets)
        unique_elements = set.union(*sets) - common_elements
        for ms in sub_mslist:
            tb.open(ms, nomodify=False)
            colnames = tb.colnames()
            for colname in unique_elements:
                if colname in colnames:
                    print(f"Removing column: {colname} from sub-MS: {ms}")
                    tb.removecols(colname)
            tb.flush()
            tb.close()
    return


def single_mstransform(
    msname="",
    outputms="",
    field="",
    scan="",
    width=1,
    timebin="",
    datacolumn="DATA",
    spw="",
    corr="",
    timerange="",
    numsubms="auto",
    n_threads=-1,
    dry_run=False,
):
    """
    Perform mstransform of a single scan

    Parameters
    ----------
    msname : str
        Name of the measurement set
    outputms : str
        Output ms name
    scan : int
        Scan to split (a single scan)
    field : str, optional
        Field name
    width : int, optional
        Number of channels to average
    timebin : str, optional
        Time to average
    datacolumn : str, optional
        Data column to split
    spw : str, optional
        Spectral window
    corr : str, optional
        Correlation to split
    timerange : str, optional
        Time range
    numsubms : str, optional
        Number of subms
    n_threads : int, optional
        Number of CPU threads

    Returns
    -------
    str
        Output measurement set name
    """
    limit_threads(n_threads=n_threads)
    from casatasks import mstransform, initweights, flagdata

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    if timebin == "" or timebin is None:
        timeaverage = False
    else:
        timeaverage = True
    if width > 1:
        chanaverage = True
    else:
        chanaverage = False
    outputms = outputms.rstrip("/")
    if os.path.exists(outputms):
        os.system("rm -rf " + outputms)
    if os.path.exists(outputms + ".flagversions"):
        os.system("rm -rf " + outputms + ".flagversions")
    try:
        if n_threads < 1:
            n_threads = 2
        else:
            n_threads = min(n_threads, 2)
        if field == "":
            msmd = msmetadata()
            msmd.open(msname)
            field = str(msmd.fieldsforscan(int(scan))[0])
            msmd.close()
        with suppress_casa_output():
            mstransform(
                vis=msname,
                outputvis=outputms,
                spw=spw,
                timerange=timerange,
                field=field,
                scan=scan,
                datacolumn=datacolumn,
                createmms=True,
                correlation=corr,
                timeaverage=timeaverage,
                timebin=timebin,
                chanaverage=chanaverage,
                chanbin=int(width),
                nthreads=n_threads,
                separationaxis="scan",
                numsubms=numsubms,
            )
        with suppress_casa_output():
            initweights(vis=outputms, wtmode="ones", dowtsp=True)
            flagdata(
                vis=outputms,
                mode="clip",
                clipzeros=True,
                datacolumn="data",
                flagbackup=False,
            )
        os.system(f"touch {outputms}/.splited")
        return outputms
    except Exception as e:
        traceback.print_exc()
        if os.path.exists(outputms):
            os.system("rm -rf " + outputms)
        return


# Expose functions and classes
__all__ = [
    name
    for name, obj in globals().items()
    if (
        (isinstance(obj, types.FunctionType) or isinstance(obj, type))
        and obj.__module__ == __name__
    )
]
