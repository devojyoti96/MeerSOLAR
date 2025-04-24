import sys, glob, time, gc, tempfile, copy, traceback
import os, numpy as np, dask, psutil
from datetime import datetime as dt, timezone
from casatasks import (
    casalog,
    importfits,
    listpartition,
)
from casatools import msmetadata, ms as casamstool, table, agentflagger
from dask.distributed import Client, LocalCluster
from dask import delayed, compute, config
from optparse import OptionParser
from astropy.time import Time
from astropy.coordinates import get_sun
from astropy.io import fits
from casatasks import casalog

logfile = casalog.logfile()
os.system("rm -rf " + logfile)

"""
This code is written by Devojyoti Kansabanik, Jul 30, 2024
"""


def get_datadir():
    """
    Get package data directory
    """
    from importlib.resources import files

    datadir_path = str(files("meersolar").joinpath("data"))
    return datadir_path


datadir = get_datadir()
udocker_dir = datadir + "/udocker"
os.environ["UDOCKER_DIR"] = udocker_dir
os.environ["UDOCKER_TARBALL"] = datadir + "/udocker-englib-1.2.11.tar.gz"


def init_udocker():
    os.system("udocker install")


def limit_threads(n_threads=-1):
    """
    Limit number of threads usuage
    Parameters
    ----------
    n_threads : int, optional
        Number of threads
    """
    if n_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)


def split_noise_diode_scans(
    msname="",
    noise_on_ms="",
    noise_off_ms="",
    field="",
    scan="",
    datacolumn="data",
    n_threads=-1,
    dry_run=True,
):
    """
    Split noise diode on and off timestamps into two seperate measurement sets
    Parameters
    ----------
    msname : str
        Measurement set
    noise_on_ms : str, optional
        Noise diode on ms
    noise_off_ms : str, optional
        Noise diode off ms
    field : str, optional
        Field name or id
    scan : str, optional
        Scan number
    datacolumn : str, optional
        Data column to split
    n_threads : int, optional
        Number of OpenMP threads
    Returns
    -------
    tuple
        splited ms names
    """
    limit_threads(n_threads=n_threads)
    from casatasks import split

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    print(f"Spliting ms: {msname} into noise diode on and off measurement sets.")
    if noise_on_ms == "":
        noise_on_ms = msname.split(".ms")[0] + "_noise_on.ms"
    if noise_off_ms == "":
        noise_off_ms = msname.split(".ms")[0] + "_noise_off.ms"
    if os.path.exists(noise_on_ms):
        os.system("rm -rf " + noise_on_ms)
    if os.path.exists(noise_on_ms + ".flagversions"):
        os.system("rm -rf " + noise_on_ms + ".flagversions")
    if os.path.exists(noise_off_ms):
        os.system("rm -rf " + noise_off_ms)
    if os.path.exists(noise_off_ms + ".flagversions"):
        os.system("rm -rf " + noise_off_ms + ".flagversions")
    tb = table()
    tb.open(msname)
    times = tb.getcol("TIME")
    tb.close()
    unique_times = np.unique(times)
    even_times = unique_times[::2]  # Even-indexed timestamps
    odd_times = unique_times[1::2]  # Odd-indexed timestamps
    even_timerange = ",".join(
        [mjdsec_to_timestamp(t, str_format=1) for t in even_times]
    )
    odd_timerange = ",".join([mjdsec_to_timestamp(t, str_format=1) for t in odd_times])
    even_ms = msname.split(".ms")[0] + "_even.ms"
    odd_ms = msname.split(".ms")[0] + "_odd.ms"
    split(
        vis=msname,
        outputvis=even_ms,
        timerange=even_timerange,
        field=field,
        scan=scan,
        datacolumn=datacolumn,
    )
    split(
        vis=msname,
        outputvis=odd_ms,
        timerange=odd_timerange,
        field=field,
        scan=scan,
        datacolumn=datacolumn,
    )
    mstool = casamstool()
    mstool.open(even_ms)
    mstool.select({"antenna1": 1, "antenna2": 1})
    even_data = np.nanmean(np.abs(mstool.getdata("DATA")["data"]))
    mstool.close()
    mstool.open(odd_ms)
    mstool.select({"antenna1": 1, "antenna2": 1})
    odd_data = np.nanmean(np.abs(mstool.getdata("DATA")["data"]))
    mstool.close()
    if even_data > odd_data:
        os.system("mv " + even_ms + " " + noise_on_ms)
        os.system("mv " + odd_ms + " " + noise_off_ms)
    else:
        os.system("mv " + odd_ms + " " + noise_on_ms)
        os.system("mv " + even_ms + " " + noise_off_ms)
    return noise_on_ms, noise_off_ms


def get_band_name(msname):
    """
    Get band name
    Parameters
    ----------
    msname : str
        Name of the ms
    Returns
    -------
    str
        Band name ('U','L','S')
    """
    msmd = msmetadata()
    msmd.open(msname)
    meanfreq = msmd.meanfreq(0) / 10**6
    msmd.close()
    msmd.done()
    if meanfreq > 580 and meanfreq < 1000:
        return "U"
    elif meanfreq > 880 and meanfreq < 1700:
        return "L"
    else:
        return "S"


def get_bad_chans(msname):
    """
    Get bad channels to flag
    Parameters
    ----------
    msname : str
        Name of the ms
    Returns
    -------
    str
        SPW string of bad channels
    """
    msmd = msmetadata()
    msmd.open(msname)
    chanfreqs = msmd.chanfreqs(0) / 10**6
    msmd.close()
    msmd.done()
    bandname = get_band_name(msname)
    if bandname == "U":
        bad_freqs = [
            (-1, 580),
            (925, 960),
            (1040, -1),
        ]
    elif bandname == "L":
        bad_freqs = [
            (-1, 879),
            (925, 960),
            (1565, 1585),
            (1217, 1237),
            (1375, 1387),
            (1166, 1186),
            (1592, 1610),
            (1242, 1249),
            (1616, 1626),
            (1526, 1554),
            (1681, -1),
        ]
    else:
        bad_freqs = []  # TODO: fill it
    spw = "0:"
    for freq_range in bad_freqs:
        start_freq = freq_range[0]
        end_freq = freq_range[1]
        if start_freq == -1:
            start_chan = 0
        else:
            start_chan = np.argmin(np.abs(start_freq - chanfreqs))
        if end_freq == -1:
            end_chan = len(chanfreqs) - 1
        else:
            end_chan = np.argmin(np.abs(end_freq - chanfreqs))
        spw += str(start_chan) + "~" + str(end_chan) + ";"
    spw = spw[:-1]
    return spw


def get_good_chans(msname):
    """
    Get good channel range to perform gaincal
    Parameters
    ----------
    msname : str
        Name of the ms
    Returns
    -------
    str
        SPW string
    """
    msmd = msmetadata()
    msmd.open(msname)
    chanfreqs = msmd.chanfreqs(0) / 10**6
    meanfreq = msmd.meanfreq(0) / 10**6
    msmd.close()
    msmd.done()
    bandname = get_band_name(msname)
    if bandname == "U":
        good_freqs = [(650, 700)]  # For UHF band
    elif bandname == "L":
        good_freqs = [(1400, 1450)]  # For L band
    else:
        good_freqs = []  # For S band #TODO: fill it
    spw = "0:"
    for freq_range in good_freqs:
        start_freq = freq_range[0]
        end_freq = freq_range[1]
        start_chan = np.argmin(np.abs(start_freq - chanfreqs))
        end_chan = np.argmin(np.abs(end_freq - chanfreqs))
        spw += str(start_chan) + "~" + str(end_chan) + ";"
    spw = spw[:-1]
    return spw


def get_bad_ants(msname="", fieldnames=[], n_threads=-1, dry_run=False):
    """
    Get bad antennas
    Parameters
    ----------
    msname : str
        Name of the ms
    fieldnames : list, optional
        Fluxcal field names
    n_threads : int, optional
        Number of OpenMP threads
    Returns
    -------
    list
        Bad antenna list
    str
        Bad antenna string
    """
    limit_threads(n_threads=n_threads)
    from casatasks import visstat

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    msmd = msmetadata()
    good_chan = get_good_chans(msname)
    all_field_bad_ants = []
    msmd.open(msname)
    nant = msmd.nantennas()
    msmd.close()
    msmd.done()
    for field in fieldnames:
        ant_means = []
        bad_ants = []
        for ant in range(nant):
            stat_mean = visstat(
                vis=msname,
                field=str(field),
                uvrange="0lambda",
                spw=good_chan,
                antenna=str(ant) + "&&" + str(ant),
                useflags=False,
            )["DATA_DESC_ID=0"]["mean"]
            ant_means.append(stat_mean)
        ant_means = np.array(ant_means)
        all_ant_mean = np.nanmean(ant_means)
        all_ant_std = np.nanstd(ant_means)
        pos = np.where(ant_means < all_ant_mean - (5 * all_ant_std))[0]
        if len(pos) > 0:
            for b_ant in pos:
                bad_ants.append(b_ant)
        all_field_bad_ants.append(bad_ants)
    bad_ants = [set(sublist) for sublist in all_field_bad_ants]
    common_elements = set.intersection(*bad_ants)
    bad_ants = list(common_elements)
    if len(bad_ants) > 0:
        bad_ants_str = ",".join([str(i) for i in bad_ants])
    else:
        bad_ants_str = ""
    return bad_ants, bad_ants_str


def get_common_spw(spw1, spw2):
    """
    Return common spectral windows in merged CASA string format.
    Parameters
    ----------
    spw1 : str
        First spectral window
    spw2 : str
        Second spectral window
    Returns
    -------
    str
        Merged spectral window
    """
    from itertools import groupby
    from collections import defaultdict

    def to_set(s):
        out, cur = set(), None
        for part in s.split(";"):
            if ":" in part:
                cur, rng = part.split(":")
            else:
                rng = part
            cur = int(cur)
            a, *b = map(int, rng.split("~"))
            out.update((cur, i) for i in range(a, (b[0] if b else a) + 1))
        return out

    def to_str(pairs):
        spw_dict = defaultdict(list)
        for spw, ch in sorted(pairs):
            spw_dict[spw].append(ch)
        result = []
        for spw, chans in spw_dict.items():
            chans.sort()
            for _, g in groupby(enumerate(chans), lambda x: x[1] - x[0]):
                grp = list(g)
                a, b = grp[0][1], grp[-1][1]
                result.append(f"{a}" if a == b else f"{a}~{b}")
        return "0:" + ";".join(result)

    return to_str(to_set(spw1) & to_set(spw2))


def scans_in_timerange(msname="", timerange="", dry_run=False):
    """
    Get scans in the given timerange
    Parameters
    ----------
    msname : str
        Measurement set
    timerange : str
        Time range with date and time
    Returns
    -------
    dict
        Scan dict for timerange
    """
    from casatools import ms, quanta

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    qa = quanta()
    ms_tool = ms()
    ms_tool.open(msname)
    # Get scan summary
    scan_summary = ms_tool.getscansummary()
    # Convert input timerange to MJD seconds
    timerange_list = timerange.split(",")
    valid_scans = {}
    for timerange in timerange_list:
        tr_start_str, tr_end_str = timerange.split("~")
        tr_start = timestamp_to_mjdsec(tr_start_str)  # Try parsing as date string
        tr_end = timestamp_to_mjdsec(tr_end_str)
        for scan_id, scan_info in scan_summary.items():
            t0_str = scan_info["0"]["BeginTime"]
            t1_str = scan_info["0"]["EndTime"]
            scan_start = qa.convert(qa.quantity(t0_str, "d"), "s")["value"]
            scan_end = qa.convert(qa.quantity(t1_str, "d"), "s")["value"]
            # Check overlap
            if scan_end >= tr_start and scan_start <= tr_end:
                if tr_end >= scan_end:
                    e = scan_end
                else:
                    e = tr_end
                if tr_start <= scan_start:
                    s = scan_start
                else:
                    s = tr_start
                if scan_id in valid_scans.keys():
                    old_t = valid_scans[scan_id].split("~")
                    old_s = timestamp_to_mjdsec(old_t[0])
                    old_e = timestamp_to_mjdsec(old_t[-1])
                    if s > old_s:
                        s = old_s
                    if e < old_e:
                        e = old_e
                valid_scans[int(scan_id)] = (
                    mjdsec_to_timestamp(s, str_format=1)
                    + "~"
                    + mjdsec_to_timestamp(e, str_format=1)
                )
    ms_tool.close()
    return valid_scans


def get_refant(msname="", n_threads=-1, dry_run=False):
    """
    Get reference antenna
    Parameters
    ----------
    msname : str
        Name of the measurement set
    n_threads : int, optional
        Number of OpenMP threads
    Returns
    -------
    str
        Reference antenna
    """
    limit_threads(n_threads=n_threads)
    from casatasks import visstat

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    casalog.filter("SEVERE")
    fluxcal_fields, fluxcal_scans = get_fluxcals(msname)
    msmd = msmetadata()
    msmd.open(msname)
    nant = int(msmd.nantennas() / 2)
    msmd.close()
    msmd.done()
    antamp = []
    antrms = []
    for ant in range(nant):
        ant = str(ant)
        t = visstat(
            vis=msname,
            field=fluxcal_fields[0],
            antenna=ant,
            timeaverage=True,
            timebin="500min",
            timespan="state,scan",
            reportingaxes="field",
        )
        item = str(list(t.keys())[0])
        amp = float(t[item]["median"])
        rms = float(t[item]["rms"])
        antamp.append(amp)
        antrms.append(rms)
    antamp = np.array(antamp)
    antrms = np.array(antrms)
    medamp = np.median(antamp)
    medrms = np.median(antrms)
    goodrms = []
    goodamp = []
    goodant = []
    for i in range(len(antamp)):
        if antamp[i] > medamp:
            goodant.append(i)
            goodamp.append(antamp[i])
            goodrms.append(antrms[i])
    goodrms = np.array(goodrms)
    referenceant = np.argmin(goodrms)
    return str(referenceant)


def get_submsname_scans(msname):
    """
    Get sub-MS names for each scans of an multi-MS
    Parameters
    ----------
    msname : str
        Name of the measurement set
    Returns
    -------
    list
        msname list
    list
        scan list
    """
    if os.path.exists(msname + "/SUBMSS") == False:
        print("Input measurement set is not a multi-MS")
        return
    partitionlist = listpartition(vis=msname, createdict=True)
    scans = []
    mslist = []
    for i in range(len(partitionlist)):
        subms = partitionlist[i]
        mslist.append(msname + "/SUBMSS/" + subms["MS"])
        scan_number = list(subms["scanId"].keys())[0]
        scans.append(scan_number)
    return mslist, scans


def get_chans_flag(msname="", field="0", n_threads=-1, dry_run=False):
    """
    Get flag/unflag channel list
    Parameters
    ----------
    msname : str
        Measurement set name
    field : str, optional
        Field name or ID
    n_threads : int, optional
        Number of OpenMP threads
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
    casalog.filter("SEVERE")
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


def reset_weights_and_flags(msname="", restore_flag=True, n_threads=-1, dry_run=False):
    """
    Reset weights and flags for the ms
    Parameters
    ----------
    msname : str
        Measurement set
    restore_flag : bool, optional
        Restore flags or not
    n_threads : int, optional
        Number of OpenMP threads
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
        print("Resetting weight column to ones.")
        weight = np.ones((npol, nrows))
        tb.putcol("WEIGHT", weight)
    if "SIGMA" in colnames:
        print("Resetting sigma column to ones.")
        sigma = np.ones((npol, nrows))
        tb.putcol("SIGMA", sigma)
    if "WEIGHT_SPECTRUM" in colnames:
        print("Removing weight spectrum.")
        tb.removecols("WEIGHT_SPECTRUM")
    if "SIGMA_SPECTRUM" in colnames:
        print("Removing sigma spectrum.")
        tb.removecols("SIGMA_SPECTRUM")
    tb.flush()
    tb.close()
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


def get_unflagged_antennas(msname="", scan="", n_threads=-1, dry_run=False):
    """
    Get unflagged antennas of a scan
    Parameters
    ----------
    msname : str
        Name of the measurement set
    scan : str
        Scans
    n_threads : int, optional
        Number of OpenMP threads
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
    n_threads : int, optional
        Number of OpenMP threads
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
    summary = flagdata(vis=msname, field=field, scan=scan, mode="summary")
    flagged_fraction = summary["flagged"] / summary["total"]
    return flagged_fraction


def get_fluxcals(msname):
    """
    Get fluxcal field names and scans
    Parameters
    ----------
    msname : str
        Name of the ms
    Returns
    -------
    list
        Fluxcal field names
    dict
        Fluxcal scans
    """
    msmd = msmetadata()
    if os.path.exists(msname + "/SUBMSS"):
        mslist = glob.glob(msname + "/SUBMSS/*.ms")
    else:
        mslist = [msname]
    fluxcal_fields = []
    fluxcal_scans = {}
    for msname in mslist:
        msmd.open(msname)
        field_names = msmd.fieldnames()
        for field in field_names:
            if field == "J1939-6342" or field == "J0408-6545":
                if field not in fluxcal_fields:
                    fluxcal_fields.append(field)
                scans = msmd.scansforfield(field).tolist()
                if field in fluxcal_scans:
                    for scan in scans:
                        fluxcal_scans[field].append(scan)
                else:
                    fluxcal_scans[field] = scans
    msmd.close()
    msmd.done()
    return fluxcal_fields, fluxcal_scans


def get_polcals(msname):
    """
    Get polarization calibrator field names and scans
    Parameters
    ----------
    msname : str
        Name of the ms
    Returns
    -------
    list
        Polcal field names
    dict
        Polcal scans
    """
    msmd = msmetadata()
    if os.path.exists(msname + "/SUBMSS"):
        mslist = glob.glob(msname + "/SUBMSS/*.ms")
    else:
        mslist = [msname]
    polcal_fields = []
    polcal_scans = {}
    for msname in mslist:
        msmd.open(msname)
        field_names = msmd.fieldnames()
        for field in field_names:
            if field in ["3C286", "1328+307", "1331+305", "J1331+3030"] or field in [
                "3C138",
                "0518+165",
                "0521+166",
                "J0521+1638",
            ]:
                if field not in polal_fields:
                    polcal_fields.append(field)
                scans = msmd.scansforfield(field).tolist()
                if field in polcal_scans:
                    for scan in scans:
                        polcal_scans[field].append(scan)
                else:
                    polcal_scans[field] = scans
    msmd.close()
    msmd.done()
    return polcal_fields, polcal_scans


def get_phasecals(msname):
    """
    Get phasecal field names and scans
    Parameters
    ----------
    msname : str
        Name of the ms
    Returns
    -------
    list
        Phasecal field names
    dict
        Phasecal scans
    dict
        Phasecal flux
    """
    msmd = msmetadata()
    if os.path.exists(msname + "/SUBMSS"):
        mslist = glob.glob(msname + "/SUBMSS/*.ms")
    else:
        mslist = [msname]
    phasecal_fields = []
    phasecal_scans = {}
    phasecal_flux_list = {}
    datadir = get_datadir()
    for msname in mslist:
        msmd.open(msname)
        field_names = msmd.fieldnames()
        bandname = get_band_name(msname)
        if bandname == "U":
            phasecals, phasecal_flux = np.load(
                datadir + "/UHF_band_cal.npy", allow_pickle=True
            ).tolist()
        elif bandname == "L":
            phasecals, phasecal_flux = np.load(
                datadir + "/L_band_cal.npy", allow_pickle=True
            ).tolist()
        for field in field_names:
            if field in phasecals and (field != "J1939-6342" and field != "J0408-6545"):
                if field not in phasecal_fields:
                    phasecal_fields.append(field)
                scans = msmd.scansforfield(field).tolist()
                if field in phasecal_scans:
                    for scan in scans:
                        phasecal_scans[field].append(scan)
                else:
                    phasecal_scans[field] = scans
                flux = phasecal_flux[phasecals.index(field)]
                phasecal_flux_list[field] = flux
    msmd.close()
    msmd.done()
    return phasecal_fields, phasecal_scans, phasecal_flux_list


def get_target_fields(msname):
    """
    Get target fields
    Parameters
    ----------
    msname : str
        Name of the measurement set
    Returns
    -------
    list
        Target field names
    dict
        Target field scans
    """
    fluxcal_field, fluxcal_scans = get_fluxcals(msname)
    phasecal_field, phasecal_scans, phasecal_fluxs = get_phasecals(msname)
    calibrator_field = fluxcal_field + phasecal_field
    msmd = msmetadata()
    msmd.open(msname)
    field_names = msmd.fieldnames()
    field_names = np.unique(field_names)
    target_fields = []
    target_scans = {}
    for f in field_names:
        if f not in calibrator_field:
            target_fields.append(f)
    for field in target_fields:
        scans = msmd.scansforfield(field).tolist()
        target_scans[field] = scans
    msmd.close()
    msmd.done()
    return target_fields, target_scans


def get_caltable_fields(caltable):
    """
    Get caltable field names
    Parameters
    ----------
    caltable : str
        Caltable name
    Returns
    -------
    list
        Field names
    """
    tb = table()
    tb.open(caltable + "/FIELD")
    field_names = tb.getcol("NAME")
    field_ids = tb.getcol("SOURCE_ID")
    tb.close()
    tb.open(caltable)
    fields = np.unique(tb.getcol("FIELD_ID"))
    tb.close()
    field_name_list = []
    for f in fields:
        pos = np.where(field_ids == f)[0][0]
        field_name_list.append(str(field_names[pos]))
    return field_name_list


def get_cal_target_scans(msname):
    """
    Get calibrator and target scans
    Parameters
    ----------
    msname : str
        Name of the measurement set
    Returns
    -------
    list
        Target scan numbers
    list
        Calibrator scan numbers
    """
    f_scans = []
    p_scans = []
    g_scans = []
    fluxcal_fields, fluxcal_scans = get_fluxcals(msname)
    phasecal_fields, phasecal_scans, phasecal_flux_list = get_phasecals(msname)
    polcal_fields, polcal_scans = get_polcals(msname)
    for fluxcal_scan in fluxcal_scans.values():
        for s in fluxcal_scan:
            f_scans.append(s)
    for polcal_scan in polcal_scans.values():
        for s in polcal_scan:
            p_scans.append(s)
    for phasecal_scan in phasecal_scans.values():
        for s in phasecal_scan:
            g_scans.append(s)
    cal_scans = f_scans + p_scans + g_scans
    msmd = msmetadata()
    msmd.open(msname)
    all_scans = msmd.scannumbers()
    msmd.close()
    msmd.done()
    target_scans = []
    for scan in all_scans:
        if scan not in cal_scans:
            target_scans.append(scan)
    return target_scans, cal_scans, f_scans, g_scans, p_scans


def get_solar_elevation_MeerKAT(date_time=""):
    """
    Get solar elevation at MeerKAT at a time
    Parameters
    ----------
    date_time : str
        Date and time in 'yyyy-mm-ddTHH:MM:SS' format (default: current time)
    Returns
    -------
    float
        Solar elevation in degree
    """
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, get_sun, SkyCoord
    from astropy.time import Time

    lat = -30.7130
    lon = 21.4430
    elev = 1038
    latitude = lat * u.deg  # In degree
    longitude = lon * u.deg  # In degree
    elevation = elev * u.m  # In meter
    if date_time == "":
        time = Time.now()
    else:
        time = Time(date_time)
    location = EarthLocation(lat=latitude, lon=longitude, height=elevation)
    sun_coords = get_sun(time)
    altaz_frame = AltAz(obstime=time, location=location)
    sun_altaz = sun_coords.transform_to(altaz_frame)
    solar_elevation = sun_altaz.alt.deg
    return solar_elevation


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

            1: 'YYYY - MM - DDThh:mm:ss'

            2: 'YYYY - MM - DD hh:mm:ss'

            3: 'YYYY_MM_DD_hh_mm_ss'
    Returns
    -------
    float
        Return correspondong MJD second of the day
    """
    import julian

    if date_format == 0:
        try:
            timestamp_datetime = dt.strptime(timestamp, "%Y/%m/%d/%H:%M:%S.%f")
        except:
            timestamp_datetime = dt.strptime(timestamp, "%Y/%m/%d/%H:%M:%S")
    elif date_format == 1:
        try:
            timestamp_datetime = dt.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        except:
            timestamp_datetime = dt.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")
    elif date_format == 2:
        try:
            timestamp_datetime = dt.strptime(timestamp, "%Y-%m-%d%H:%M:%S.%f")
        except:
            timestamp_datetime = dt.strptime(timestamp, "%Y-%m-%d%H:%M:%S")
    elif date_format == 3:
        try:
            timestamp_datetime = dt.strptime(timestamp, "%Y_%m_%d_%H_%M_%S.%f")
        except:
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
        Time stamp format (0: yyyy-mm-ddTHH:MM:SS.ff, 1: yyyy/mm/dd/HH:MM:SS.ff)
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
    else:
        utcstring = "%s/%02d/%02d/%s" % (
            date["year"],
            date["month"],
            date["monthday"],
            hhmmss,
        )
    return utcstring


def get_timeranges_for_scan(msname, scan, time_interval, time_window):
    """
    Get time ranges for a scan with certain time intervals
    Parameters
    ----------
    msname : str
        Name of the measurement set
    scan : int
        Scan number
    time_interval : float
        Time interval in seconds
    time_window : float
        Time window in seconds
    Returns
    -------
    list
        List of time ranges
    """
    msmd = msmetadata()
    msmd.open(msname)
    times = msmd.timesforscan(int(scan))
    msmd.close()
    msmd.done()
    time_ranges = []
    start_time = times[0]
    end_time = times[-1]
    while start_time < end_time:
        if start_time + time_window < end_time:
            t = (
                mjdsec_to_timestamp(start_time, str_format=1)
                + "~"
                + mjdsec_to_timestamp(start_time + time_window, str_format=1)
            )
        else:
            t = (
                mjdsec_to_timestamp(start_time, str_format=1)
                + "~"
                + mjdsec_to_timestamp(end_time, str_format=1)
            )
        time_ranges.append(t)
        start_time += time_interval
    return time_ranges


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
    mid_time_mjd = Time(mid_timestamp, format="isot")
    sun_coord = get_sun(mid_time_mjd)
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
        + str(int(sun_coord.dec.dms.m))
        + "m"
        + str(round(sun_coord.dec.dms.s, 2))
        + "s"
    )
    sun_radec_string = "J2000 " + str(sun_ra) + " " + str(sun_dec)
    radeg = sun_coord.ra.deg
    radeg = radeg % 360
    decdeg = sun_coord.dec.deg
    decdeg = decdeg % 360
    return sun_radec_string, sun_ra, sun_dec, radeg, decdeg


def get_phasecenter(msname, field):
    """
    Get phasecenter of the measurement set
    Parameters
    ----------
    msname : str
        Name of the measurement set
    Returns
    -------
    float
            RA in degree
    float
            DEC in degree
    """
    msmd = msmetadata()
    msmd.open(msname)
    phasecenter = msmd.phasecenter()
    msmd.close()
    msmd.done()
    radeg = np.rad2deg(phasecenter["m0"]["value"])
    radeg = radeg % 360
    decdeg = np.rad2deg(phasecenter["m1"]["value"])
    decdeg = decdeg % 360
    return radeg, decdeg


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
    return theta_deg


def move_to_sun(msname):
    """
    Move the phasecenter of the measurement set at the center of the Sun (Assuming ms has one scan)
    Parameters
    ----------
    msname : str
        Name of the measurement set
    Returns
    -------
    int
        Success message
    """
    sun_radec_string, sunra, sundec, sunra_deg, sundec_deg = radec_sun(msname)
    msg = run_chgcenter(msname, sunra, sundec, "meerwsclean")
    if msg != 0:
        print("Phasecenter could not be shifted.")
    return msg


def split_and_move_to_solarcenter(
    msname="", outputms="", timestamp="", field="", n_threads=-1, dry_run=False
):
    """
    Split and move phasecenter to the Sun
    Parameters
    ----------
    msname : str
       Measurement set
    outputms : str
        Output msname
    timestamp : str
        Time stamp
    field : str, optional
        Field name or ID
    Returns
    -------
    int
        Success message
    str
        Output msname
    """
    limit_threads(n_threads=n_threads)
    from casatasks import split

    casalog.filter("SEVERE")
    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    if os.path.exists(outputms):
        os.system("rm -rf " + outputms)
    if os.path.exists(outputms + ".flagversions"):
        os.system("rm -rf " + outputms + ".flagversions")
    split(
        vis=msname,
        outputvis=outputms,
        field=str(field),
        spw="",
        scan="",
        datacolumn="all",
        timerange=timestamp,
    )
    if os.path.exists(outputms) == False:
        return 1, ""
    else:
        msg = move_to_sun(outputms)
        return 0, outputms


def correct_solar_sidereal_motion(
    msname="", cpu_frac=0.8, mem_frac=0.8, keep_original=False, dry_run=False
):
    """
    Correct the shift in RA DEC of the Sun due to sidereal motion
    Shift each timestamp visibilities phase center to the solar center
    Parameters
    ----------
    msname : str
        Measurement set name
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    keep_original : bool, optional
        Keep original msname or not
    Returns
    -------
    str
        Phase rotated measurement set
    """
    starttime = time.time()
    total_cpus = psutil.cpu_count(logical=True)
    available_cpu_pct = 100 - psutil.cpu_percent(interval=1)
    available_cpus = int(total_cpus * available_cpu_pct / 100.0)
    usable_cpus = max(1, int(total_cpus * cpu_frac))
    ncpu = min(1, int(usable_cpus * cpu_frac))
    limit_threads(n_threads=ncpu)
    from casatasks import concat

    casalog.filter("SEVERE")
    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    if os.path.exists(msname + "/.sidereal"):
        print(f"Sidereal motion is already corrected for ms: {msname}")
        return msname
    print(f"Correcting sidereal motion of the Sun for ms: {msname}")
    msname = os.path.abspath(msname)
    mspath = os.path.dirname(msname)
    os.chdir(mspath)
    msmd = msmetadata()
    msmd.open(msname)
    times = msmd.timesforspws(0)
    total_time = times[1] - times[0]
    scan = int(msmd.scannumbers()[0])
    field = int(msmd.fieldsforscan(scan)[0])
    msmd.close()
    timestamps = [mjdsec_to_timestamp(i, str_format=1) for i in times]
    task = delayed(split_and_move_to_solarcenter)(dry_run=True)
    mem_limit = run_limited_memory_task(task)
    dask_client, dask_cluster, n_jobs, n_threads = get_dask_client(
        len(timestamps), cpu_frac, mem_frac, min_mem_per_job=mem_limit / 0.8
    )
    tasks = []
    if len(timestamps) > 0:
        tasks = []
        for i in range(len(timestamps)):
            outputms = mspath + "/t_" + str(i) + ".ms"
            tasks.append(
                delayed(split_and_move_to_solarcenter)(
                    msname, outputms, timestamps[i], field=field, n_threads=n_threads
                )
            )
    results = compute(*tasks)
    dask_client.close()
    dask_cluster.close()
    splited_mslist_time_index = sorted(
        [
            int(os.path.basename(i).split(".ms")[0].split("t_")[-1])
            for i in glob.glob(mspath + "/t_*.ms")
        ]
    )
    min_time_index = min(splited_mslist_time_index)
    vis = [
        mspath + "/t_" + str(t_index) + ".ms" for t_index in splited_mslist_time_index
    ]
    outputms = msname.split(".ms")[0] + "_rotated.ms"
    if os.path.exists(outputms):
        os.system("rm -rf " + outputms)
    if os.path.exists(outputms + ".flagversions"):
        os.system("rm -rf " + outputms + ".flagversions")
    concat(vis=vis, concatvis=outputms, dirtol="3600arcsec", timesort=True)
    if keep_original == False:
        os.system("rm -rf " + msname)
        os.system("mv " + outputms + " " + msname)
        outputms = msname
    os.system("rm -rf " + mspath + "/t_*.ms")
    os.system("touch " + msname + "/.sidereal")
    print(f"Total time taken: {round(time.time()-starttime,2)}s")
    return outputms


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


def determine_noise_diode_cal_scan(msname, scan):
    """
    Determine whether a calibrator scan is a noise-diode cal scan or not
    Parameters
    ----------
    msname : str
        Name of the measurement set
    scan : int
        Scan number
    Returns
    -------
    bool
        Whether it is noise-diode cal scan or not
    """
    print(f"Check noise-diode cal for scan : {scan}")
    good_spw = get_good_chans(msname)
    chan = int(good_spw.split(";")[0].split(":")[-1].split("~")[0])
    mstool = casamstool()
    mstool.open(msname)
    mstool.select({"antenna1": 1, "antenna2": 1, "scan_number": scan})
    mstool.selectchannel(nchan=1, width=1, start=chan)
    data = mstool.getdata("DATA", ifraxis=True)["data"][:, 0, 0, :]
    mstool.close()
    xx = np.abs(data[0, ...])
    yy = np.abs(data[-1, ...])
    even_xx = xx[1::2]
    odd_xx = xx[::2]
    minlen = min(len(even_xx), len(odd_xx))
    d_xx = even_xx[:minlen] - odd_xx[:minlen]
    even_yy = yy[1::2]
    odd_yy = yy[::2]
    d_yy = even_yy[:minlen] - odd_yy[:minlen]
    d_xx = np.abs(d_xx)
    d_yy = np.abs(d_yy)
    mean_d_xx = np.nanmedian(d_xx)
    mean_d_yy = np.nanmedian(d_yy)
    if mean_d_xx > 10 and mean_d_yy > 10:
        return True
    else:
        return False


def get_valid_scans(msname, field="", min_scan_time=1):
    """
    Get valid list of scans
    Parameters
    ----------
    msname : str
        Measurement set name
    min_scan_time : float
        Minimum valid scan time in minute
    Returns
    -------
    list
        Valid scan list
    """
    mstool = casamstool()
    mstool.open(msname)
    scan_summary = mstool.getscansummary()
    mstool.close()
    scans = np.sort(np.array([int(i) for i in scan_summary.keys()]))
    target_scans, cal_scans, f_scans, g_scans, p_scans = get_cal_target_scans(msname)
    selected_field = []
    valid_scans = []
    if field != "":
        field = field.split(",")
        msmd = msmetadata()
        msmd.open(msname)
        for f in field:
            try:
                field_id = msmd.fieldsforname(f)[0]
            except:
                field_id = int(f)
            selected_field.append(field_id)
        msmd.close()
        msmd.done()
    for scan in scans:
        scan_field = scan_summary[str(scan)]["0"]["FieldId"]
        if len(selected_field) == 0 or scan_field in selected_field:
            duration = (
                scan_summary[str(scan)]["0"]["EndTime"]
                - scan_summary[str(scan)]["0"]["BeginTime"]
            ) * 86400.0
            duration = round(duration / 60.0, 1)
            if duration >= min_scan_time:
                valid_scans.append(scan)
    return valid_scans


def calc_maxuv(msname):
    """
    Calculate maximum UV
    Parameters
    ----------
    msname : str
        Name of the measurement set
    Returns
    -------
    float
        Maximum UV in meter
    float
        Maximum UV in wavelength
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.meanfreq(0)
    wavelength = 299792458.0 / (freq)
    msmd.close()
    msmd.done()
    tb = table()
    tb.open(msname)
    uvw = tb.getcol("UVW")
    tb.close()
    u, v, w = [uvw[i, :] for i in range(3)]
    maxu = float(np.nanmax(u))
    maxv = float(np.nanmax(v))
    maxuv = np.nanmax([maxu, maxv])
    return maxuv, maxuv / wavelength


def calc_field_of_view(msname, FWHM=True):
    """
    Calculate optimum field of view in arcsec.
    Parameters
    ----------
    msname : str
        Measurement set name
    FWHM : bool, optional
        Upto FWHM, otherwise upto first null
    Returns
    -------
    float
        Field of view in arcsec
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.chanfreqs(0)[0]
    msmd.close()
    tb = table()
    tb.open(msname + "/ANTENNA")
    dish_dia = np.nanmin(tb.getcol("DISH_DIAMETER"))
    tb.close()
    wavelength = 299792458.0 / freq
    if FWHM == True:
        FOV = 1.22 * wavelength / dish_dia
    else:
        FOV = 2.04 * wavelength / dish_dia
    fov_arcsec = np.rad2deg(FOV) * 3600  ### In arcsecs
    return fov_arcsec


def calc_bw_smearing_freqwidth(msname):
    """
    Function to calculate spectral width to procude bandwidth smearing
    Parameters
    ----------
    msname : str
        Name of the measurement set
    Returns
    -------
    float
        Spectral width in MHz
    """
    R = 0.9
    fov = 3600  # 2 times size of the Sun
    psf = calc_psf(msname)
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.meanfreq(0)
    msmd.close()
    msmd.done()
    delta_nu = np.sqrt((1 / R**2) - 1) * (psf / fov) * freq
    delta_nu /= 10**6
    return round(delta_nu, 2)


def calc_time_smearing_timewidth(msname):
    """
    Calculate maximum time averaging to avoid time smearing over full FoV.
    Parameters
    ----------
    msname : str
        Measurement set name
    Returns
    -------
    delta_t_max : float
        Maximum allowable time averaging in seconds.
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq_Hz = msmd.chanfreqs(0)[0]
    msmd.close()
    c = 299792458.0  # speed of light in m/s
    omega_E = 7.2921159e-5  # Earth rotation rate in rad/s
    lam = c / freq_Hz  # wavelength in meters
    fov_deg = calc_field_of_view(msname) / 3600.0
    fov_rad = np.deg2rad(fov_deg)
    uv, uvlambda = calc_maxuv(msname)
    # Approximate maximum allowable time to avoid >10% amplitude loss
    delta_t_max = lam / (2 * np.pi * uv * omega_E * fov_rad)
    return delta_t_max


def max_time_solar_smearing(msname):
    """
    Max allowable time averaging to avoid solar motion smearing.
    Parameters
    ----------
    msname : str
        Measurement set name
    Returns
    -------
    t_max : float
        Maximum time averaging in seconds.
    """
    omega_sun = 2.5 / (60.0)  # solar apparent motion (2.5 arcsec/min to arcsec/sec)
    psf = calc_psf(msname)
    t_max = 0.5 * (psf / omega_sun)  # seconds
    return t_max


def calc_psf(msname):
    """
    Function to calculate PSF size in arcsec
    Parameters
    ----------
    msname : str
        Name of the measurement set
    Returns
    -------
    float
            PSF size in arcsec
    """
    maxuv_m, maxuv_l = calc_maxuv(msname)
    psf = np.rad2deg(1.2 / maxuv_l) * 3600.0  # In arcsec
    return psf


def calc_cellsize(msname, num_pixel_in_psf):
    """
    Calculate pixel size in arcsec
    Parameters
    ----------
    msname : str
        Name of the measurement set
    num_pixel_in_psf : int
            Number of pixels in one PSF
    Returns
    -------
    int
            Pixel size in arcsec
    """
    psf = calc_psf(msname)
    pixel = round(psf / num_pixel_in_psf, 1)
    return pixel


def calc_multiscale_scales(msname, num_pixel_in_psf, max_scale=16, nmax=5):
    """
    Calculate multiscale scales
    Parameters
    ----------
    msname : str
        Name of the measurement set
    num_pixel_in_psf : int
            Number of pixels in one PSF
    max_scale : float, optional
        Maximum scale in arcmin
    nmax : int, optional
        Maximum number of scales
    Returns
    -------
    list
            Multiscale scales in pixel units
    """
    psf = calc_psf(msname)
    multiscale_scales = [0, num_pixel_in_psf]
    max_scale_pixel = int(max_scale * 60 / psf)
    if nmax > 2:
        other_scales = np.linspace(
            3 * num_pixel_in_psf, max_scale_pixel, nmax - 2, endpoint=True
        ).astype("int")
        for scale in other_scales:
            multiscale_scales.append(scale)
    else:
        multiscale_scales.append(max_scale_pixel)
    return multiscale_scales

def delaycal(msname="", caltable="", refant="", solint="inf", dry_run=False):
    """
    General delay calibration using CASA, not assuming any point source
    Parameters
    ----------
    msname : str, optional
        Measurement set
    caltable : str, optional
        Caltable name
    refant : str, optional
        Reference antenna
    solint : str, optional
        Solution interval
    Returns
    -------
    str
        Caltable name
    """
    from casatasks import bandpass, gaincal

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    import warnings

    warnings.filterwarnings("ignore")
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    os.system("rm -rf " + caltable + "*")
    gaincal(
        vis=msname,
        caltable=caltable,
        refant=refant,
        gaintype="K",
        solint=solint,
        minsnr=1,
    )
    bandpass(
        vis=msname,
        caltable=caltable + ".tempbcal",
        refant=refant,
        solint=solint,
        minsnr=1,
    )
    tb = table()
    tb.open(caltable + ".tempbcal/SPECTRAL_WINDOW")
    freq = tb.getcol("CHAN_FREQ").flatten()
    tb.close()
    tb.open(caltable + ".tempbcal")
    gain = tb.getcol("CPARAM")
    flag = tb.getcol("FLAG")
    gain[flag] = np.nan
    tb.close()
    tb.open(caltable, nomodify=False)
    delay_gain = tb.getcol("FPARAM") * 0.0
    delay_flag = tb.getcol("FLAG")
    gain = np.nanmean(gain, axis=0)
    phase = np.angle(gain)
    for i in range(delay_gain.shape[0]):
        for j in range(delay_gain.shape[2]):
            try:
                delay = np.polyfit(2 * np.pi * freq, phase[:, j], deg=1)[0] / (
                    10**-9
                )  # Delay in nanosecond
                if np.isnan(delay):
                    delay = 0.0
                delay_gain[i, :, j] = delay
            except:
                delay_gain[i, :, j] = 0.0
    tb.putcol("FPARAM", delay_gain)
    tb.putcol("FLAG", delay_flag)
    tb.flush()
    tb.close()
    os.system("rm -rf " + caltable + ".tempbcal")
    return caltable


def make_stokes_wsclean_imagecube(
    wsclean_images, outfile_name, keep_wsclean_images=True
):
    """
    Convert WSClean images into a Stokes cube image.

    Parameters
    ----------
    wsclean_images : list
        List of WSClean images.
    outfile_name : str
        Name of the output file.
    keep_wsclean_images : bool, optional
        Whether to retain the original WSClean images (default: True).
    Returns
    -------
    str
        Output image name.
    """
    stokes = sorted(
        set(
            (
                os.path.basename(i).split(".fits")[0].split(" - ")[-2]
                if " - " in i
                else "I"
            )
            for i in wsclean_images
        )
    )
    valid_stokes = [
        {"I"},
        {"I", "V"},
        {"I", "Q", "U", "V"},
        {"XX", "YY"},
        {"LL", "RR"},
        {"Q", "U"},
        {"I", "Q"},
    ]
    if set(stokes) not in valid_stokes:
        print("Invalid Stokes combination.")
        return
    imagename_prefix = "temp_" + os.path.basename(wsclean_images[0]).split(" - I")[0]
    imagename = imagename_prefix + ".image"
    data, header = fits.getdata(wsclean_images[0]), fits.getheader(wsclean_images[0])
    for img in wsclean_images[1:]:
        data = np.append(data, fits.getdata(img), axis=0)
    header.update(
        {"NAXIS4": len(stokes), "CRVAL4": 1 if "I" in stokes else -5, "CDELT4": 1}
    )
    temp_fits = imagename_prefix + ".fits"
    fits.writeto(outfile_name, data=data, header=header, overwrite=True)
    if not keep_wsclean_images:
        for img in wsclean_images:
            os.system(f"rm -rf {img}")
    return outfile_name


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
                except:
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


def merge_caltables(caltables, merged_caltable, append=False, keepcopy=False):
    """
    Merge multiple same type of caltables
    Parameters
    ----------
    caltables : list
        Caltable list
    merged_caltable : str
        Merged caltable name
    append : bool, optional
        Append with exisiting caltable
    keepcopy : bool, opitonal
        Keep input caltables or not
    Returns
    -------
    str
        Merged caltable
    """
    if type(caltables) != list or len(caltables) == 0:
        print("Please provide a list of caltable.")
        return
    if os.path.exists(merged_caltable) and append == True:
        pass
    else:
        if os.path.exists(merged_caltable):
            os.system("rm -rf " + merged_caltable)
        if keepcopy:
            os.system("cp -r " + caltables[0] + " " + merged_caltable)
        else:
            os.system("mv " + caltables[0] + " " + merged_caltable)
        caltables.remove(caltables[0])
    if len(caltables) > 0:
        tb = table()
        for caltable in caltables:
            if os.path.exists(caltable):
                tb.open(caltable)
                tb.copyrows(merged_caltable)
                tb.close()
                if keepcopy == False:
                    os.system("rm -rf " + caltable)
    return merged_caltable


def create_batch_script_nonhpc(cmd, workdir, basename):
    """
    Function to make a batch script not non-HPC environment
    Parameters
    ----------
    cmd : str
            Command to run
    workdir : str
            Work directory of the measurement set
    basename : str
            Base name of the batch files
    """
    batch_file = workdir + "/" + basename + ".batch"
    cmd_batch = workdir + "/" + basename + "_cmd.batch"
    if os.path.isdir(workdir + "/logs") == False:
        os.makedirs(workdir + "/logs")
    outputfile = workdir + "/logs/" + basename + ".log"
    pid_file = workdir + "/pids.txt"
    finished_touch_file = workdir + "/.Finished_" + basename
    os.system("rm -rf " + finished_touch_file + "*")
    finished_touch_file_error = finished_touch_file + "_1"
    finished_touch_file_success = finished_touch_file + "_0"
    cmd_file_content = f"{cmd}; exit_code=$?; echo $exit_code; if [ $exit_code -ne 0 ]; then touch {finished_touch_file_error}; else touch {finished_touch_file_success}; fi"
    batch_file_content = f"""export PYTHONUNBUFFERED=1\nnohup sh {cmd_batch}> {outputfile} 2>&1 &\necho $! >> {pid_file}\nsleep 2\n rm -rf {batch_file}\n rm -rf {cmd_batch}"""
    if os.path.exists(cmd_batch):
        os.system("rm -rf " + cmd_batch)
    if os.path.exists(batch_file):
        os.system("rm -rf " + batch_file)
    with open(cmd_batch, "w") as cmd_batch_file:
        cmd_batch_file.write(cmd_file_content)
    with open(batch_file, "w") as b_file:
        b_file.write(batch_file_content)
    os.system("chmod a+rwx " + batch_file)
    os.system("chmod a+rwx " + cmd_batch)
    del cmd
    return workdir + "/" + basename + ".batch"


def get_dask_client(n_jobs, cpu_frac=0.8, mem_frac=0.8, min_mem_per_job=-1):
    """
    Create a Dask client optimized for one-task-per-worker execution,
    where each worker is a separate process that can use multiple threads internally.

    Parameters
    ----------
    n_jobs : int
        Number of MS tasks (ideally = number of MS files)
    cpu_frac : float
        Fraction of total CPUs to use
    mem_frac : float
        Fraction of total memory to use
    min_mem_per_job : float, optional
        Minimum memory per job
    Returns
    -------
    client : dask.distributed.Client
        Dask clinet
    cluster : dask.distributed.LocalCluster
        Dask cluster
    n_workers : int
        Number of workers
    threads_per_worker : int
        Threads per worker to use
    """
    import os

    # Detect system resources
    total_cpus = psutil.cpu_count(logical=True)
    total_mem = psutil.virtual_memory().total

    # Wait for enough CPU
    count = 0
    while True:
        available_cpu_pct = 100 - psutil.cpu_percent(interval=1)
        available_cpus = int(total_cpus * available_cpu_pct / 100.0)
        usable_cpus = max(1, int(total_cpus * cpu_frac))
        if available_cpus >= usable_cpus:
            break
        else:
            if count == 0:
                print("Waiting for available free CPUs...")
            time.sleep(5)
        count += 1

    # Wait for enough memory
    count = 0
    while True:
        available_mem = psutil.virtual_memory().available
        usable_mem = total_mem * mem_frac
        if available_mem >= usable_mem:
            break
        else:
            if count == 0:
                print("Waiting for available free memory...")
            time.sleep(5)
        count += 1

    # Determine resources per worker
    mem_per_worker = usable_mem / n_jobs
    # Safety checks
    min_mem_per_job = round(min_mem_per_job, 2)
    if min_mem_per_job > 0 and mem_per_worker < (min_mem_per_job * 1024**3):
        print(
            f"Total memory per job is smaller than {min_mem_per_job} GB. Adjusting total number of workers to meet this."
        )
        mem_per_worker = min_mem_per_job * 1024**3
        n_workers = min(n_jobs, int(usable_mem / mem_per_worker))
    else:
        n_workers = n_jobs
        mem_per_worker = usable_mem / n_workers
    threads_per_worker = max(1, usable_cpus // max(1, n_workers))

    print("\n#################################")
    print(
        f"Dask workers: {n_workers}, Threads per worker: {threads_per_worker}, Mem/worker: {mem_per_worker/1e9:.2f} GB"
    )

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        memory_limit=mem_per_worker,
        processes=True,  # one process per worker
        dashboard_address=":0",
    )
    client = Client(cluster)
    # Memory control settings
    dask.config.set(
        {
            "distributed.worker.memory.target": 0.6,
            "distributed.worker.memory.spill": 0.7,
            "distributed.worker.memory.pause": 0.8,
            "distributed.worker.memory.terminate": 0.95,
        }
    )

    client.run_on_scheduler(gc.collect)

    print(f"Dask Dashboard: {client.dashboard_link}")
    print("#################################\n")

    return client, cluster, n_workers, threads_per_worker


def run_limited_memory_task(task, timeout=30):
    """
    Run a task for a limited time, then kill and return memory usage.
    Parameters
    ----------
    task : dask.delayed
        Dask delayed task object
    timeout : int
        Time in seconds to let the task run
    Returns
    -------
    float
        Memory used by task (in GB)
    """
    import warnings

    warnings.filterwarnings("ignore")
    client = Client()
    future = client.compute(task)
    start = time.time()

    def get_worker_memory_info():
        proc = psutil.Process()
        return {
            "rss_GB": proc.memory_info().rss / 1024**3,
            "vms_GB": proc.memory_info().vms / 1024**3,
        }

    while not future.done():
        if time.time() - start > timeout:
            try:
                mem_info = client.run(get_worker_memory_info)
                total_rss = sum(v["rss_GB"] for v in mem_info.values())
                per_worker_mem = total_rss
            except Exception as e:
                per_worker_mem = None
            future.cancel()
            client.close()
            return per_worker_mem
        time.sleep(1)
    mem_info = client.run(get_worker_memory_info)
    total_rss = sum(v["rss_GB"] for v in mem_info.values())
    per_worker_mem = total_rss
    client.close()
    return per_worker_mem


def baseline_names(msname):
    """
    Get baseline names
    Parameters
    ----------
    msname : str
        Measurement set name
    Returns
    -------
    list
        Baseline names list
    """
    mstool = casamstool()
    mstool.open(msname)
    ants = mstool.getdata(["antenna1", "antenna2"])
    mstool.close()
    baseline_ids = set(zip(ants["antenna1"], ants["antenna2"]))
    baseline_names = []
    for ant1, ant2 in sorted(baseline_ids):
        baseline_names.append(str(ant1) + "&&" + str(ant2))
    return baseline_names


def get_chunk_size(msname, memory_limit=-1, ncol=3):
    """
    Get time chunk size for a memory limit
    Parameters
    ----------
    msname : str
        Measurement set
    memory_limit : int, optional
        Memory limit
    ncol : int, optional
        Number of columns
    Returns
    -------
    int
        Number of time chunk
    int
        Number of baseline chunk
    """
    if memory_limit == -1:
        memory_limit = psutil.virtual_memory().available / 1024**3  # In GB
    memory_limit = memory_limit / ncol
    msmd = msmetadata()
    msmd.open(msname)
    nrow = int(msmd.nrows())
    nchan = msmd.nchan(0)
    npol = msmd.ncorrforpol(0)
    nant = msmd.nantennas()
    nbaselines = msmd.nbaselines()
    msmd.close()
    if nbaselines == 0 or nrow % nbaselines != 0:
        nbaselines += nant
    ntimes = int(nrow / nbaselines)
    per_time_memory = float(npol * nchan * nbaselines * 16) / 1024**3
    per_baseline_memory = float(npol * nchan * ntimes * 16) / 1024**3
    time_chunk = int(memory_limit / per_time_memory)
    baseline_chunk = int(memory_limit / per_baseline_memory)
    if time_chunk == 0 or baseline_chunk == 0:
        print("Too small memory limit.")
        return None, None
    if time_chunk > ntimes:
        time_chunk = ntimes
    if baseline_chunk > nbaselines:
        baseline_chunk = nbaselines
    return time_chunk, baseline_chunk


def get_column_size(msname):
    """
    Get time chunk size for a memory limit
    Parameters
    ----------
    msname : str
        Measurement set
    Returns
    -------
    float
        A single datacolumn data size in GB
    """
    msmd = msmetadata()
    msmd.open(msname)
    nrow = int(msmd.nrows())
    msmd.close()
    datasize = nrow * 16 / (1024.0**3)
    return datasize


def check_datacolumn_valid(msname, datacolumn="DATA"):
    """
    Check whether a data column exists and valid
    Parameters
    ----------
    msname : str
        Measurement set
    datacolumn : str, optional
        Data column string in table (e.g.,DATA, CORRECTED_DATA', MODEL_DATA, FLAG, WEIGHT, WEIGHT_SPECTRUM, SIGMA, SIGMA_SPECTRUM)
    Returns
    -------
    bool
        Whether valid data column is present or not
    """
    tb = table()
    tb.open(msname)
    colnames = tb.colnames()
    if datacolumn not in colnames:
        tb.close()
        return False
    try:
        model_data = tb.getcol(datacolumn, startrow=0, nrow=1)
        tb.close()
        if model_data is None or model_data.size == 0:
            return False
        elif (model_data == 0).all():
            return False
        else:
            return True
    except:
        tb.close()
        return False


def create_circular_mask(msname, cellsize, imsize, mask_radius=20):
    """
    Create fits solar mask
    Parameters
    ----------
    msname : str
        Name of the measurement set
    cellsize : float
        Cell size in arcsec
    imsize : int
        Imsize in number of pixels
    mask_radius : float
        Mask radius in arcmin
    Returns
    -------
    str
        Fits mask file name
    """
    imagename_prefix = msname.split(".ms")[0] + "_solar"
    wsclean_args = [
        "-quiet",
        "-scale " + str(cellsize) + "asec",
        "-size " + str(imsize) + " " + str(imsize),
        "-nwlayers 1",
        "-niter 0 -name " + imagename_prefix,
        "-channel-range 0 1",
        "-interval 0 1",
    ]
    wsclean_cmd = "wsclean " + " ".join(wsclean_args) + " " + msname
    msg = run_wsclean(wsclean_cmd, "meerwsclean", verbose=False)
    if msg == 0:
        center = (int(imsize / 2), int(imsize / 2))
        radius = mask_radius * 60 / cellsize
        Y, X = np.ogrid[:imsize, :imsize]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        mask = dist_from_center <= radius
        os.system("cp -r " + imagename_prefix + "-image.fits mask.fits")
        os.system("rm -rf " + imagename_prefix + "*")
        data = fits.getdata("mask.fits")
        header = fits.getheader("mask.fits")
        data[0, 0, ...][mask] = 1.0
        data[0, 0, ...][~mask] = 0.0
        fits.writeto(
            imagename_prefix + "-mask.fits", data=data, header=header, overwrite=True
        )
        os.system("rm -rf mask.fits")
        if os.path.exists(imagename_prefix + "-mask.fits"):
            return imagename_prefix + "-mask.fits"
        else:
            print("Circular mask could not be created.")
            return
    else:
        print("Circular mask could not be created.")
        return


def calc_fractional_bandwidth(msname):
    """
    Calculate fractional bandwidh
    Parameters
    ----------
    msname : str
        Name of measurement set
    Returns
    -------
    float
        Fraction bandwidth in percentage
    """
    msmd = msmetadata()
    msmd.open(msname)
    frac_bandwidth = msmd.bandwidths(0) / msmd.meanfreq(0)
    msmd.close()
    return round(frac_bandwidth * 100.0, 2)


def calc_dyn_range(imagename, modelname, fits_mask=""):
    """
    Calculate dynamic ranges.

    Parameters
    ----------
    imagename : list or str
        Image FITS file(s).
    modelname : list or str
        Model FITS file(s).
    fits_mask : str, optional
        FITS file mask.

    Returns
    -------
    model_flux : float
        Total model flux.
    dyn_range_rms : float
        Max/RMS dynamic range.
    rms : float
        RMS of the image
    dyn_range_min : float
        Max/|Min| dynamic range.
    """

    def load_data(name):
        return fits.getdata(name)

    def to_list(x):
        return [x] if isinstance(x, str) else x

    imagename = to_list(imagename)
    modelname = to_list(modelname)

    use_mask = bool(fits_mask and os.path.exists(fits_mask))
    mask_data = fits.getdata(fits_mask).astype(bool) if use_mask else None

    model_flux, dr1, dr2 = 0, 0, 0

    for img in imagename:
        image = load_data(img)
        if use_mask:
            maxval = np.nanmax(image[mask_data])
            rms = np.nanstd(image[~mask_data])
            minval = np.nanmin(image[mask_data])
        else:
            maxval = np.nanmax(image)
            rms = np.nanstd(image)
            minval = np.nanmin(image)
        dr1 += maxval / rms if rms else 0
        dr2 += abs(maxval / minval) if minval else 0

    for mod in modelname:
        model = load_data(mod)
        model_flux += np.nansum(model[mask_data] if use_mask else model)

    return model_flux, round(dr1, 2), round(rms, 2), round(dr2, 2)


####################
# uDOCKER related
####################
def check_udocker_container(name):
    """
    Check whether a docker container is present or not
    Parameters
    ----------
    name : str
        Container name
    Returns
    -------
    bool
        Whether present or not
    """
    b = os.system("udocker --insecure --quiet inspect " + name + " >> tmp2 >> tmp1")
    os.system("rm -rf tmp1 tmp2")
    if b != 0:
        return False
    else:
        return True


def initialize_wsclean_container(name):
    """
    Initialize WSClean container
    Parameters
    ----------
    name : str
        Name of the container
    Returns
    -------
    bool
        Whether initialized successfully or not
    """
    a = os.system("udocker pull devojyoti96/wsclean-full:22.04")
    if a == 0:
        a = os.system(f"udocker create --name={name} devojyoti96/wsclean-full:22.04")
        print(f"Container started with name : {name}")
        return name
    else:
        print(f"Container could not be created with name : {name}")
        return


def run_wsclean(wsclean_cmd, container_name, verbose=False):
    """
    Run WSClean inside a udocker container (no root permission required).
    Parameters
    ----------
    wsclean_cmd : str
        Full WSClean command as a string.
    container_name : str
        Container name
    Returns
    -------
    int
        Success message
    """
    container_present = check_udocker_container(container_name)
    if container_present == False:
        container_name = initialize_wsclean_container(container_name)
        if container_name == None:
            print(
                "Container {container_name} is not initiated. First initiate container and then run."
            )
            return 1
    msname = wsclean_cmd.split(" ")[-1]
    msname = os.path.abspath(msname)
    mspath = os.path.dirname(msname)
    temp_docker_path = tempfile.mkdtemp(prefix="wsclean_udocker_", dir=mspath)
    wsclean_cmd_args = wsclean_cmd.split(" ")[:-1]
    if "-fits-mask" in wsclean_cmd_args:
        index = wsclean_cmd_args.index("-fits-mask")
        name = wsclean_cmd_args[index + 1]
        namedir = os.path.dirname(os.path.abspath(name))
        basename = os.path.basename(os.path.abspath(name))
        wsclean_cmd_args.remove(name)
        wsclean_cmd_args.insert(index + 1, temp_docker_path + "/" + basename)
    if "-name" not in wsclean_cmd_args:
        wsclean_cmd_args.append(
            "-name " + temp_docker_path + "/" + os.path.basename(msname).split(".ms")[0]
        )
    else:
        index = wsclean_cmd_args.index("-name")
        name = wsclean_cmd_args[index + 1]
        namedir = os.path.dirname(os.path.abspath(name))
        basename = os.path.basename(os.path.abspath(name))
        wsclean_cmd_args.remove(name)
        wsclean_cmd_args.insert(index + 1, temp_docker_path + "/" + basename)
    if "-temp-dir" not in wsclean_cmd_args:
        wsclean_cmd_args.append("-temp-dir " + temp_docker_path)
    else:
        index = wsclean_cmd_args.index("-temp-dir")
        name = os.path.abspath(wsclean_cmd_args[index + 1])
        wsclean_cmd_args.remove(name)
        wsclean_cmd_args.insert(index + 1, temp_docker_path)
    wsclean_cmd = (
        " ".join(wsclean_cmd_args)
        + " "
        + temp_docker_path
        + "/"
        + os.path.basename(msname)
    )
    try:
        full_command = f"udocker run --nobanner --volume={mspath}:{temp_docker_path} --workdir {temp_docker_path} meerwsclean {wsclean_cmd}"
        if verbose:
            print(wsclean_cmd)
        exit_code = os.system(full_command)
        os.system(f"rm -rf {temp_docker_path}")
        return 0 if exit_code == 0 else 1
    except Exception as e:
        os.system(f"rm -rf {temp_docker_path}")
        traceback.print_exc()
        return 1


def run_chgcenter(msname, ra, dec, container_name):
    """
    Run chgcenter inside a udocker container (no root permission required).
    Parameters
    ----------
    msname : str
        Name of the measurement set
    ra : str
        RA can either be 00h00m00.0s or 00:00:00.0
    dec : str
        Dec can either be 00d00m00.0s or 00.00.00.0
    Returns
    -------
    int
        Success message
    """
    container_present = check_udocker_container(container_name)
    if container_present == False:
        container_name = initialize_wsclean_container(container_name)
        if container_name == None:
            print(
                "Container {container_name} is not initiated. First initiate container and then run."
            )
            return 1
    msname = os.path.abspath(msname)
    mspath = os.path.dirname(msname)
    temp_docker_path = tempfile.mkdtemp(prefix="chgcenter_udocker_", dir=mspath)
    cmd = (
        "chgcentre "
        + temp_docker_path
        + "/"
        + os.path.basename(msname)
        + " "
        + ra
        + " "
        + dec
    )
    try:
        full_command = f"udocker --quiet run --nobanner --volume={mspath}:{temp_docker_path} --workdir {temp_docker_path} meerwsclean {cmd}"
        exit_code = os.system(full_command + " >>tmp1 >> tmp2")
        os.system(f"rm -rf {temp_docker_path} tmp1 tmp2")
        return 0 if exit_code == 0 else 1
    except Exception as e:
        os.system(f"rm -rf {temp_docker_path} tmp1 tmp2")
        traceback.print_exc()
        return 1
