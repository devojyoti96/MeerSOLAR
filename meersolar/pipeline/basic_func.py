import sys, glob, time, gc, tempfile, copy, traceback
import os, numpy as np, dask, psutil, logging, sunpy
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from sunpy.map import Map
from sunpy.coordinates import frames, sun
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
    if meanfreq >= 544 and meanfreq <= 1088:
        return "U"
    elif meanfreq >= 856 and meanfreq <= 1712:
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
            (1010, -1),
        ]
    elif bandname == "L":
        bad_freqs = [
            (-1, 879),
            (925, 960),
            (1166, 1186),
            (1217, 1237),
            (1242, 1249),
            (1375, 1387),
            (1526, 1626),
            (1681, -1),
        ]
    else:
        print("Data is not in UHF or L-band.")
        bad_freqs = []
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
        good_freqs = [(580, 620)]  # For UHF band
    elif bandname == "L":
        good_freqs = [(890, 920)]  # For L band
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


def get_chans_flag(msname="", field="", n_threads=-1, dry_run=False):
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


def get_optimal_image_interval(
    msname,
    temporal_tol_factor=0.1,
    spectral_tol_factor=0.1,
    chan_range="",
    timestamp_range="",
    max_nchan=-1,
    max_ntime=-1,
):
    """
    Get optimal image spectral temporal interval such that total flux max-median in each chunk is within tolerance limit
    Parameters
    ----------
    msname : str
        Name of the measurement set
    temporal_tol_factor : float, optional
        Tolerance factor for temporal variation (default : 0.1, 10%)
    spectral_tol_factor : float, optional
        Tolerance factor for spectral variation (default : 0.1, 10%)
    chan_range : str, optional
        Channel range
    timestamp_range : str, optional
        Timestamp range
    max_nchan : int, optional
        Maxmium number of spectral chunk
    max_ntime : int, optional
        Maximum number of temporal chunk
    Returns
    -------
    int
        Number of time intervals to average
    int
        Number of channels to averages
    """

    def is_valid_chunk(chunk, tolerance):
        mean_flux = np.nanmedian(chunk)
        if mean_flux == 0:
            return False
        return (np.nanmax(chunk) - np.nanmin(chunk)) / mean_flux <= tolerance

    def find_max_valid_chunk_length(fluxes, tolerance):
        n = len(fluxes)
        for window in range(n, 1, -1):  # Try from largest to smallest
            valid = True
            for start in range(0, n, window):
                end = min(start + window, n)
                chunk = fluxes[start:end]
                if len(chunk) < window:  # Optionally require full window
                    valid = False
                    break
                if not is_valid_chunk(chunk, tolerance):
                    valid = False
                    break
            if valid:
                return window  # Return the largest valid window
        return 1  # Minimum chunk size is 1 if nothing else is valid

    tb = table()
    mstool = casamstool()
    msmd = msmetadata()
    msmd.open(msname)
    nchan = msmd.nchan(0)
    times = msmd.timesforspws(0)
    ntime = len(times)
    del times
    msmd.close()
    tb.open(msname)
    u, v, w = tb.getcol("UVW")
    tb.close()
    uvdist = np.sort(np.unique(np.sqrt(u**2 + v**2)))
    mstool.open(msname)
    if uvdist[0] == 0.0:
        mstool.select({"uvdist": [0.0, 0.0]})
    else:
        mstool.select({"antenna1": 0, "antenna2": 1})
    data_and_flag = mstool.getdata(["DATA", "FLAG"], ifraxis=True)
    data = data_and_flag["data"]
    flag = data_and_flag["flag"]
    data[flag] = np.nan
    mstool.close()
    if chan_range != "":
        start_chan = int(chan_range.split(",")[0])
        end_chan = int(chan_range.split(",")[-1])
        spectra = np.nanmedian(data[:, start_chan:end_chan, ...], axis=(0, 2, 3))
    else:
        spectra = np.nanmedian(data, axis=(0, 2, 3))
    if timestamp_range != "":
        t_start = int(timestamp_range.split(",")[0])
        t_end = int(timestamp_range.split(",")[-1])
        t_series = np.nanmedian(data[..., t_start:t_end], axis=(0, 1, 2))
    else:
        t_series = np.nanmedian(data, axis=(0, 1, 2))
    t_series = t_series[t_series != 0]
    spectra = spectra[spectra != 0]
    t_chunksize = find_max_valid_chunk_length(t_series, temporal_tol_factor)
    f_chunksize = find_max_valid_chunk_length(spectra, spectral_tol_factor)
    n_time_interval = int(len(t_series) / t_chunksize)
    n_spectral_interval = int(len(spectra) / f_chunksize)
    if max_nchan > 0 and n_spectral_interval > max_nchan:
        n_spectral_interval = max_nchan
    if max_ntime > 0 and n_time_interval > max_ntime:
        n_time_interval = max_ntime
    return n_time_interval, n_spectral_interval


def reset_weights_and_flags(
    msname="", restore_flag=True, n_threads=-1, force_reset=False, dry_run=False
):
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
    if os.path.exists(f"{msname}/.reset") == False or force_reset == True:
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
    try:
        times = msmd.timesforscan(int(scan))
    except:
        times = msmd.timesforspws(0)
    msmd.close()
    msmd.done()
    time_ranges = []
    start_time = times[0]
    end_time = times[-1]
    if time_interval < 0 or time_window < 0:
        t = (
            mjdsec_to_timestamp(start_time, str_format=1)
            + "~"
            + mjdsec_to_timestamp(end_time, str_format=1)
        )
        time_ranges.append(t)
        return time_ranges
    total_time=(end_time-start_time)
    timeres=total_time/len(times)
    ntime_chunk=int(total_time/time_interval)
    ntime=int(time_window/timeres)
    start_time=times[:-ntime]
    indices = np.linspace(
        0, len(start_time) - 1, num=ntime_chunk, dtype=int
    )
    timelist = [start_time[i] for i in indices]
    for t in timelist:
        time_ranges.append(f"{mjdsec_to_timestamp(t, str_format=1)}~{mjdsec_to_timestamp(t+time_window, str_format=1)}")
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
    msg = run_chgcenter(msname, sunra, sundec, container_name="meerwsclean")
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
            msname=msname, container_name="meerwsclean", verbose=verbose
        )
        if msg != 0:
            print("Sidereal motion correction is not successful.")
        else:
            os.system("touch " + msname + "/.sidereal_cor")
        return msg
    else:
        print(f"Sidereal motion correction is already done for ms: {msname}")
        return 0


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


def calc_maxuv(msname, chan_number=-1):
    """
    Calculate maximum UV
    Parameters
    ----------
    msname : str
        Name of the measurement set
    chan_number : int, optional
        Channel number
    Returns
    -------
    float
        Maximum UV in meter
    float
        Maximum UV in wavelength
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.chanfreqs(0)[chan_number]
    wavelength = 299792458.0 / (freq)
    msmd.close()
    msmd.done()
    tb = table()
    tb.open(msname)
    uvw = tb.getcol("UVW")
    tb.close()
    u, v, w = [uvw[i, :] for i in range(3)]
    uv = np.sqrt(u**2 + v**2)
    uv[uv == 0] = np.nan
    maxuv = np.nanmax(uv)
    return maxuv, maxuv / wavelength


def calc_minuv(msname, chan_number=-1):
    """
    Calculate minimum UV
    Parameters
    ----------
    msname : str
        Name of the measurement set
    chan_number : int, optional
        Channel number
    Returns
    -------
    float
        Minimum UV in meter
    float
        Minimum UV in wavelength
    """
    import matplotlib.pyplot as plt

    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.chanfreqs(0)[chan_number]
    wavelength = 299792458.0 / (freq)
    msmd.close()
    msmd.done()
    tb = table()
    tb.open(msname)
    uvw = tb.getcol("UVW")
    tb.close()
    u, v, w = [uvw[i, :] for i in range(3)]
    uv = np.sqrt(u**2 + v**2)
    uv[uv == 0] = np.nan
    minuv = np.nanmin(uv)
    return minuv, minuv / wavelength


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
    freqres = msmd.chanres(0)[0] / 10**6
    msmd.close()
    msmd.done()
    delta_nu = np.sqrt((1 / R**2) - 1) * (psf / fov) * freq
    delta_nu /= 10**6
    delta_nu = ceil_to_multiple(delta_nu, freqres)
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
    times = msmd.timesforspws(0)
    msmd.close()
    timeres = times[1] - times[0]
    c = 299792458.0  # speed of light in m/s
    omega_E = 7.2921159e-5  # Earth rotation rate in rad/s
    lam = c / freq_Hz  # wavelength in meters
    fov_deg = calc_field_of_view(msname) / 3600.0
    fov_rad = np.deg2rad(fov_deg)
    uv, uvlambda = calc_maxuv(msname)
    # Approximate maximum allowable time to avoid >10% amplitude loss
    delta_t_max = lam / (2 * np.pi * uv * omega_E * fov_rad)
    delta_t_max = ceil_to_multiple(delta_t_max, timeres)
    return round(delta_t_max, 2)


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


def calc_psf(msname, chan_number=-1):
    """
    Function to calculate PSF size in arcsec
    Parameters
    ----------
    msname : str
        Name of the measurement set
    chan_number : int, optional
        Channel number
    Returns
    -------
    float
            PSF size in arcsec
    """
    maxuv_m, maxuv_l = calc_maxuv(msname, chan_number=chan_number)
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


def calc_multiscale_scales(msname, num_pixel_in_psf, chan_number=-1, max_scale=16):
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
    Returns
    -------
    list
            Multiscale scales in pixel units
    """
    psf = calc_psf(msname, chan_number=chan_number)
    minuv, minuv_l = calc_minuv(msname, chan_number=chan_number)
    max_interferometric_scale = (
        0.5 * np.rad2deg(1.0 / minuv_l) * 60.0
    )  # In arcmin, half of maximum scale
    max_interferometric_scale = min(max_scale, max_interferometric_scale)
    max_scale_pixel = int((max_interferometric_scale * 60.0) / (psf / num_pixel_in_psf))
    multiscale_scales = [0]
    current_scale = num_pixel_in_psf
    while True:
        current_scale = 2 * current_scale
        if current_scale >= max_scale_pixel:
            break
        multiscale_scales.append(current_scale)
    return multiscale_scales


def get_multiscale_bias(freq, bias_min=0.6, bias_max=0.9):
    """
    Get frequency dependent multiscale bias
    Parameters
    ----------
    freq : float
        Frequency in MHz
    bias_min : float, optional
        Minimum bias at minimum L-band frequency
    bias_max : float, optional
        Maximum bias at maximum L-band frequency
    Returns
    -------
    float
        Multiscale bias patrameter
    """
    if freq <= 1015:
        return bias_min
    elif freq >= 1670:
        return bias_max
    else:
        freq_min = 1015
        freq_max = 1670
        logf = np.log10(freq)
        logf_min = np.log10(freq_min)
        logf_max = np.log10(freq_max)
        frac = (logf - logf_min) / (logf_max - logf_min)
        return round(
            np.clip(bias_min + frac * (bias_max - bias_min), bias_min, bias_max), 3
        )


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
    times = Time(timestamps, format="isot", scale="utc")
    avg_time = Time(np.mean(times.jd), format="jd", scale="utc")
    return avg_time.isot.split(".")[0]  # Strip milliseconds for clean output


def make_timeavg_image(wsclean_images, outfile_name, keep_wsclean_images=True):
    """
    Convert WSClean images into a time averaged image
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
    timestamps = []
    for i in range(len(wsclean_images)):
        image = wsclean_images[i]
        if i == 0:
            data = fits.getdata(image)
        else:
            data += fits.getdata(image)
        timestamps.append(fits.getheader(image)["DATE-OBS"])
    data /= len(wsclean_images)
    avg_timestamp = average_timestamp(timestamps)
    header = fits.getheader(wsclean_images[0])
    header["DATE-OBS"] = avg_timestamp
    fits.writeto(outfile_name, data=data, header=header, overwrite=True)
    if not keep_wsclean_images:
        for img in wsclean_images:
            os.system(f"rm -rf {img}")
    return outfile_name


def make_freqavg_image(wsclean_images, outfile_name, keep_wsclean_images=True):
    """
    Convert WSClean images into a frequency averaged image
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
    freqs = []
    for i in range(len(wsclean_images)):
        image = wsclean_images[i]
        if i == 0:
            data = fits.getdata(image)
        else:
            data += fits.getdata(image)
        header = fits.getheader(image)
        if header["CTYPE3"] == "FREQ":
            freqs.append(float(header["CRVAL3"]))
            freqaxis = 3
        elif header["CTYPE4"] == "FREQ":
            freqs.append(float(header["CRVAL4"]))
            freqaxis = 4
    data /= len(wsclean_images)
    if len(freqs) > 0:
        mean_freq = np.nanmean(freqs)
        width = max(freqs) - min(freqs)
        header = fits.getheader(wsclean_images[0])
        if freqaxis == 3:
            header["CRAVL3"] = mean_freq
            header["CDELT3"] = width
        elif freqaxis == 4:
            header["CRAVL4"] = mean_freq
            header["CDELT4"] = width
    fits.writeto(outfile_name, data=data, header=header, overwrite=True)
    if not keep_wsclean_images:
        for img in wsclean_images:
            os.system(f"rm -rf {img}")
    return outfile_name


def plot_in_hpc(
    fits_image, draw_limb=False, extension="pdf", xlim=[-2000, 2000], ylim=[-2000, 2000]
):
    """
    Function to convert MeerKAT image into Helioprojective co-ordinate
    Parameters
    ----------
    fits_image : str
        Name of the fits image
    draw_limb : bool, optional
        Draw solar limb or not
    extension : str, optional
        Output file extension
    xlim : list
        X axis limit in arcsecond
    ylim : list
        Y axis limit in arcsecond
    Returns
    -------
    sunpy.Map
        MeerKAT image in helioprojective co-ordinate
    """
    from astropy.visualization import ImageNormalize, PowerStretch, PercentileInterval
    from matplotlib.patches import Ellipse, Rectangle
    import matplotlib, matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.size": 12})
    MEERLAT = -30.7133
    MEERLON = 21.4429
    MEERALT = 1086.6
    fits_image = fits_image.rstrip("/")
    output_image = fits_image.split(".fits")[0] + f".{extension}"
    meer_hdu = fits.open(fits_image)  # Opening MeerKAT fits file
    meer_header = meer_hdu[0].header  # meer header
    meer_data = meer_hdu[0].data
    if len(meer_data.shape) > 2:
        meer_data = meer_data[0, 0, :, :]  # meer data
    if meer_header["CTYPE3"] == "FREQ":
        frequency = meer_header["CRVAL3"] * u.Hz
    elif meer_header["CTYPE4"] == "FREQ":
        frequency = meer_header["CRVAL4"] * u.Hz
    else:
        frequency = ""
    try:
        band = meer_header["BAND"]
    except:
        band = ""
    try:
        pixel_unit = meer_header["BUNIT"]
    except:
        pixel_nuit = ""
    obstime = Time(meer_header["date-obs"])
    meerpos = EarthLocation(
        lat=MEERLAT * u.deg, lon=MEERLON * u.deg, height=MEERALT * u.m
    )
    meer_gcrs = SkyCoord(meerpos.get_gcrs(obstime))  # Converting into GCRS coordinate
    reference_coord = SkyCoord(
        meer_header["crval1"] * u.Unit(meer_header["cunit1"]),
        meer_header["crval2"] * u.Unit(meer_header["cunit2"]),
        frame="gcrs",
        obstime=obstime,
        obsgeoloc=meer_gcrs.cartesian,
        obsgeovel=meer_gcrs.velocity.to_cartesian(),
        distance=meer_gcrs.hcrs.distance,
    )
    reference_coord_arcsec = reference_coord.transform_to(
        frames.Helioprojective(observer=meer_gcrs)
    )
    cdelt1 = (np.abs(meer_header["cdelt1"]) * u.deg).to(u.arcsec)
    cdelt2 = (np.abs(meer_header["cdelt2"]) * u.deg).to(u.arcsec)
    P1 = sun.P(obstime)  # Relative rotation angle
    new_meer_header = sunpy.map.make_fitswcs_header(
        meer_data,
        reference_coord_arcsec,
        reference_pixel=u.Quantity(
            [meer_header["crpix1"] - 1, meer_header["crpix2"] - 1] * u.pixel
        ),
        scale=u.Quantity([cdelt1, cdelt2] * u.arcsec / u.pix),
        rotation_angle=-P1,
        wavelength=frequency.to(u.MHz).round(2),
        observatory="MeerKAT",
    )
    meer_map = Map(meer_data, new_meer_header)
    meer_map_rotate = meer_map.rotate()
    top_right = SkyCoord(
        xlim[1] * u.arcsec, ylim[1] * u.arcsec, frame=meer_map_rotate.coordinate_frame
    )
    bottom_left = SkyCoord(
        xlim[0] * u.arcsec, ylim[0] * u.arcsec, frame=meer_map_rotate.coordinate_frame
    )
    cropped_map = meer_map_rotate.submap(bottom_left, top_right=top_right)
    norm = ImageNormalize(
        meer_data,
        vmin=max(np.nanmin(meer_data), -0.01 * np.nanmax(meer_data)),
        stretch=PowerStretch(0.5),
    )
    if band == "U":
        cmap = "inferno"
    elif band == "L":
        cmap = "YlGnBu_r"
    else:
        cmap = "cubehelix"
    fig = plt.figure()
    ax = plt.subplot(projection=cropped_map)
    cropped_map.plot(norm=norm, cmap=cmap, axes=ax)
    ax.coords.grid(False)
    # Read synthesized beam from header
    try:
        bmaj = meer_header["BMAJ"] * u.deg.to(u.arcsec)  # in arcsec
        bmin = meer_header["BMIN"] * u.deg.to(u.arcsec)
        bpa = meer_header["BPA"] - sun.P(obstime).deg  # in degrees
    except KeyError:
        bmaj = bmin = bpa = None

    # Plot PSF ellipse in bottom-left if all values are present
    if bmaj and bmin and bpa is not None:
        # Coordinates where to place the beam (e.g., 5% above bottom-left corner)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()

        beam_center = SkyCoord(
            x0 + 0.08 * (x1 - x0),
            y0 + 0.08 * (y1 - y0),
            unit=u.arcsec,
            frame=cropped_map.coordinate_frame,
        )

        # Add ellipse patch
        beam_ellipse = Ellipse(
            (beam_center.Tx.value, beam_center.Ty.value),  # center in arcsec
            width=bmin,
            height=bmaj,
            angle=bpa,
            edgecolor="white",
            facecolor="white",
            lw=1,
        )
        ax.add_patch(beam_ellipse)
        # Draw square box around the ellipse
        box_size = 100  # slightly bigger than beam
        rect = Rectangle(
            (beam_center.Tx.value - box_size / 2, beam_center.Ty.value - box_size / 2),
            width=box_size,
            height=box_size,
            edgecolor="white",
            facecolor="none",
            lw=1.2,
            linestyle="solid",
        )
        ax.add_patch(rect)

    if draw_limb:
        cropped_map.draw_limb()
    cbar = plt.colorbar()
    if pixel_unit == "K":
        cbar.set_label("Brightness temperature (K)")
    elif pixel_unit == "JY/BEAM":
        cbar.set_label("Flux density (Jy/beam)")
    fig.tight_layout()
    fig.savefig(output_image)
    plt.close(fig)
    return output_image


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


def get_nprocess_meersolar():
    """
    Get numbers of MeerSOLAR processes currently running
    Returns
    -------
    int
        Number of running processes
    """
    pid_file = datadir + "/pids.txt"
    pids = np.loadtxt(pid_file, unpack=True)
    n_process = 0
    for pid in pids:
        if psutil.pid_exists(int(pid)):
            n_process += 1
    return n_process


def save_main_process_info(pid, cpu_frac, mem_frac):
    """
    Save MeerSOLAR main processes info
    Parameters
    ----------
    pid : int
        Main job process id
    cpu_frac : float
        CPU fraction of the job
    mem_frac : float
        Mempry fraction of the job
    """
    main_job_file = datadir + "/main_pids.txt"


def create_batch_script_nonhpc(cmd, workdir, basename, write_logfile=True):
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
    write_logfile : bool, optional
        Write log file or not
    Returns
    -------
    str
        Batch file name
    """
    batch_file = workdir + "/" + basename + ".batch"
    cmd_batch = workdir + "/" + basename + "_cmd.batch"
    pid_file = datadir + "/pids.txt"
    finished_touch_file = workdir + "/.Finished_" + basename
    os.system("rm -rf " + finished_touch_file + "*")
    finished_touch_file_error = finished_touch_file + "_1"
    finished_touch_file_success = finished_touch_file + "_0"
    cmd_file_content = f"{cmd}; exit_code=$?; if [ $exit_code -ne 0 ]; then touch {finished_touch_file_error}; else touch {finished_touch_file_success}; fi"
    if write_logfile:
        if os.path.isdir(workdir + "/logs") == False:
            os.makedirs(workdir + "/logs")
        outputfile = workdir + "/logs/" + basename + ".log"
    else:
        outputfile = "/dev/null"
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


def get_dask_client(
    n_jobs,
    dask_dir,
    cpu_frac=0.8,
    mem_frac=0.8,
    spill_frac=0.6,
    min_mem_per_job=-1,
    min_cpu_per_job=1,
    only_cal=False,
):
    """
    Create a Dask client optimized for one-task-per-worker execution,
    where each worker is a separate process that can use multiple threads internally.

    Parameters
    ----------
    n_jobs : int
        Number of MS tasks (ideally = number of MS files)
    dask_dir : str
        Dask temporary directory
    cpu_frac : float
        Fraction of total CPUs to use
    mem_frac : float
        Fraction of total memory to use
    spill_frac : float, optional
        Spill to disk at this fraction
    min_mem_per_job : float, optional
        Minimum memory per job
    min_cpu_per_job : int, optional
        Minimum CPU threads per job
    only_cal : bool, optional
        Only calculate number of workers
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
    # Create the Dask temporary working directory if it does not already exist
    os.makedirs(dask_dir, exist_ok=True)
    dask_dir_tmp = dask_dir + "/tmp"
    os.makedirs(dask_dir_tmp, exist_ok=True)

    # Detect total system resources
    total_cpus = psutil.cpu_count(logical=True)  # Total logical CPU cores
    total_mem = psutil.virtual_memory().total  # Total system memory (bytes)
    if cpu_frac > 0.8:
        print(
            "Given CPU fraction is more than 80%. Resetting to 80% to avoid system crash."
        )
        cpu_frac = 0.8
    if mem_frac > 0.8:
        print(
            "Given memory fraction is more than 80%. Resetting to 80% to avoid system crash."
        )
        mem_frac = 0.8

    ############################################
    # Wait until enough free CPU is available
    ############################################
    count = 0
    while True:
        available_cpu_pct = 100 - psutil.cpu_percent(
            interval=1
        )  # Percent CPUs currently free
        available_cpus = int(
            total_cpus * available_cpu_pct / 100.0
        )  # Number of free CPU cores
        usable_cpus = max(
            1, int(total_cpus * cpu_frac)
        )  # Target number of CPU cores we want available based on cpu_frac
        if available_cpus >= int(
            0.5 * usable_cpus
        ):  # Enough free CPUs (at-least more than 50%), exit loop
            usable_cpus = min(usable_cpus, available_cpus)
            break
        else:
            if count == 0:
                print("Waiting for available free CPUs...")
            time.sleep(5)  # Wait a bit and retry
        count += 1
    ############################################
    # Wait until enough free memory is available
    ############################################
    count = 0
    while True:
        available_mem = (
            psutil.virtual_memory().available
        )  # Current available system memory (bytes)
        usable_mem = total_mem * mem_frac  # Target usable memory based on mem_frac
        if (
            available_mem >= 0.5 * usable_mem
        ):  # Enough free memory, (at-least more than 50%) exit loop
            usable_mem = min(usable_mem, available_mem)
            break
        else:
            if count == 0:
                print("Waiting for available free memory...")
            time.sleep(5)  # Wait and retry
        count += 1

    ############################################
    # Calculate memory per worker
    ############################################
    mem_per_worker = usable_mem / n_jobs  # Assume initially one job per worker
    # Apply minimum memory per worker constraint
    min_mem_per_job = round(
        min_mem_per_job, 2
    )  # Ensure min_mem_per_job is a clean float
    if min_mem_per_job > 0 and mem_per_worker < (min_mem_per_job * 1024**3):
        # If calculated memory per worker is smaller than user-requested minimum, adjust number of workers
        print(
            f"Total memory per job is smaller than {min_mem_per_job} GB. Adjusting total number of workers to meet this."
        )
        mem_per_worker = (
            min_mem_per_job * 1024**3
        )  # Reset memory per worker to minimum allowed
        n_workers = min(
            n_jobs, int(usable_mem / mem_per_worker)
        )  # Reduce number of workers accordingly
    else:
        # Otherwise, just keep n_jobs workers
        n_workers = n_jobs

    #########################################
    # Cap number of workers to available CPUs
    n_workers = max(
        1, min(n_workers, int(usable_cpus / min_cpu_per_job))
    )  # Prevent CPU oversubscription
    # Recalculate final memory per worker based on capped n_workers
    mem_per_worker = usable_mem / n_workers
    # Calculate threads per worker
    threads_per_worker = max(
        1, usable_cpus // max(1, n_workers)
    )  # Each worker gets min_cpu_per_job or more threads

    ##########################################
    if only_cal == False:
        print("\n#################################")
        print(
            f"Dask workers: {n_workers}, Threads per worker: {threads_per_worker}, Mem/worker: {round(mem_per_worker/(1024.0**3),2)} GB"
        )
    # Memory control settings
    swap = psutil.swap_memory()
    swap_gb = swap.total / 1024.0**3
    if swap_gb > 16:
        pass
    elif swap_gb > 4:
        spill_frac = 0.6
    else:
        spill_frac = 0.5

    if spill_frac > 0.7:
        spill_frac = 0.7
    if only_cal:
        final_mem_per_worker = round((mem_per_worker * spill_frac) / (1024.0**3), 2)
        return None, None, n_workers, threads_per_worker, final_mem_per_worker
    dask.config.set({"temporary-directory": dask_dir})
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,  # one python-thread per worker, in workers OpenMP threads can be used
        memory_limit=f"{round(mem_per_worker/(1024.0**3),2)}GB",
        local_directory=dask_dir,
        processes=True,  # one process per worker
        dashboard_address=":0",
        env={
            "TMPDIR": dask_dir_tmp,
            "TMP": dask_dir_tmp,
            "TEMP": dask_dir_tmp,
            "DASK_TEMPORARY_DIRECTORY": dask_dir,
            "MALLOC_TRIM_THRESHOLD_": "0",
        },  # Explicitly set for workers
    )
    client = Client(cluster)
    dask.config.set(
        {
            "distributed.worker.memory.target": spill_frac,
            "distributed.worker.memory.spill": spill_frac + 0.1,
            "distributed.worker.memory.pause": spill_frac + 0.2,
            "distributed.worker.memory.terminate": spill_frac + 0.25,
        }
    )

    client.run_on_scheduler(gc.collect)
    if only_cal == False:
        print(f"Dask Dashboard: {client.dashboard_link}")
        print("#################################\n")
    final_mem_per_worker = round((mem_per_worker * spill_frac) / (1024.0**3), 2)
    return client, cluster, n_workers, threads_per_worker, final_mem_per_worker


def run_limited_memory_task(task, dask_dir="/tmp", timeout=30):
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

    dask.config.set({"temporary-directory": dask_dir})
    warnings.filterwarnings("ignore")
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,  # one python-thread per worker, in workers OpenMP threads can be used
        local_directory=dask_dir,
        processes=True,  # one process per worker
        dashboard_address=":0",
    )
    client = Client(cluster)
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
            cluster.close()
            return per_worker_mem
        time.sleep(1)
    mem_info = client.run(get_worker_memory_info)
    total_rss = sum(v["rss_GB"] for v in mem_info.values())
    per_worker_mem = total_rss
    client.close()
    cluster.close()
    return round(per_worker_mem, 2)


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


def get_ms_size(msname):
    """
    Get measurement set total size
    Parameters
    ----------
    msname : str
    Returns
    -------
    float
        Size in GB
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(msname):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024**3)  # in GB


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
    nchan = msmd.nchan(0)
    npol = msmd.ncorrforpol()[0]
    msmd.close()
    datasize = nrow * nchan * npol * 16 / (1024.0**3)
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
    msname = msname.rstrip("/")
    msname = os.path.abspath(msname)
    try:
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
    except:
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
    try:
        msname = msname.rstrip("/")
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
            os.system(
                "cp -r "
                + imagename_prefix
                + "-image.fits mask-"
                + os.path.basename(imagename_prefix)
                + ".fits"
            )
            os.system("rm -rf " + imagename_prefix + "*")
            data = fits.getdata("mask-" + os.path.basename(imagename_prefix) + ".fits")
            header = fits.getheader(
                "mask-" + os.path.basename(imagename_prefix) + ".fits"
            )
            data[0, 0, ...][mask] = 1.0
            data[0, 0, ...][~mask] = 0.0
            fits.writeto(
                imagename_prefix + "-mask.fits",
                data=data,
                header=header,
                overwrite=True,
            )
            os.system("rm -rf mask-" + os.path.basename(imagename_prefix) + ".fits")
            if os.path.exists(imagename_prefix + "-mask.fits"):
                return imagename_prefix + "-mask.fits"
            else:
                print("Circular mask could not be created.")
                return
        else:
            print("Circular mask could not be created.")
            return
    except Exception as e:
        traceback.print_exc()
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
    freqs = msmd.chanfreqs(0)
    bw = max(freqs) - min(freqs)
    frac_bandwidth = bw / msmd.meanfreq(0)
    msmd.close()
    return round(frac_bandwidth * 100.0, 2)


def calc_dyn_range(imagename, modelname, residualname, fits_mask=""):
    """
    Calculate dynamic ranges.

    Parameters
    ----------
    imagename : list or str
        Image FITS file(s)
    modelname : list or str
        Model FITS file(s)
    residualname : list ot str
        Residual FITS file(s)
    fits_mask : str, optional
        FITS file mask
    Returns
    -------
    model_flux : float
        Total model flux.
    dyn_range_rms : float
        Max/RMS dynamic range.
    rms : float
        RMS of the image
    """

    def load_data(name):
        return fits.getdata(name)

    def to_list(x):
        return [x] if isinstance(x, str) else x

    imagename = to_list(imagename)
    modelname = to_list(modelname)
    residualname = to_list(residualname)

    use_mask = bool(fits_mask and os.path.exists(fits_mask))
    mask_data = fits.getdata(fits_mask).astype(bool) if use_mask else None

    model_flux, dr1, rmsvalue = 0, 0, 0

    for i in range(len(imagename)):
        img = imagename[i]
        res = residualname[i]
        image = load_data(img)
        residual = load_data(res)
        rms = np.nanstd(residual)
        if use_mask:
            maxval = np.nanmax(image[mask_data])
        else:
            maxval = np.nanmax(image)
        dr1 += maxval / rms if rms else 0
        rmsvalue += rms

    for mod in modelname:
        model = load_data(mod)
        model_flux += np.nansum(model[mask_data] if use_mask else model)

    rmsvalue = rmsvalue / np.sqrt(len(residualname))
    return model_flux, round(dr1, 2), round(rmsvalue, 2)


def init_logger_console(logname, logfile, verbose=False):
    """
    Initial logger.

    Parameters
    ----------
    logname : str
        Name of the log
    workdir : str, optional
        Name of the working directory
    verbose : bool, optional
        Verbose output or not
    logfile : str, optional
        Log file name
    Returns
    -------
    logger
        Python logging object
    str
        Log file name
    """
    if os.path.exists(logfile):
        os.system("rm -rf " + logfile)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    if verbose == True:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)
    filehandle = logging.FileHandler(logfile)
    filehandle.setFormatter(formatter)
    logger.addHandler(filehandle)
    logger.propagate = False
    logger.info("Log file : " + logfile + "\n")
    return logger, logfile


def generate_tb_map(imagename, outfile=""):
    """
    Function to generate brightness temperature map
    Parameters
    ----------
    imagename : str
        Name of the flux calibrated image
    outfile : str, optional
        Output brightess temperature image name
    Returns
    -------
    str
        Output image name
    """
    print(f"Generating brightness temperature map for image: {imagename}")
    if outfile == "":
        outfile = imagename.split(".fits")[0] + "_TB.fits"
    image_header = fits.getheader(imagename)
    image_data = fits.getdata(imagename)
    major = float(image_header["BMAJ"]) * 3600.0  # In arcsec
    minor = float(image_header["BMIN"]) * 3600.0  # In arcsec
    if image_header["CTYPE3"] == "FREQ":
        freq = image_header["CRVAL3"] / 10**9  # In GHz
    elif image_header["CTYPE4"] == "FREQ":
        freq = image_header["CRVAL4"] / 10**9  # In GHz
    else:
        print("No frequency information is present in header.")
        return
    TB_conv_factor = (1.222e6) / ((freq**2) * major * minor)
    TB_data = image_data * TB_conv_factor
    image_header["BUNIT"] = "K"
    fits.writeto(outfile, data=TB_data, header=image_header, overwrite=True)
    return outfile


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
    pid = os.getpid()
    timestamp = int(time.time() * 1000)
    tmp1 = f"tmp1_{pid}_{timestamp}.txt"
    tmp2 = f"tmp2_{pid}_{timestamp}.txt"
    b = os.system(
        f"udocker --insecure --quiet inspect " + name + f" >> {tmp1} >> {tmp2}"
    )
    os.system(f"rm -rf {tmp1} {tmp2}")
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
    a = os.system("udocker pull devojyoti96/wsclean-solar:latest")
    if a == 0:
        a = os.system(f"udocker create --name={name} devojyoti96/wsclean-solar:latest")
        print(f"Container started with name : {name}")
        return name
    else:
        print(f"Container could not be created with name : {name}")
        return


def run_wsclean(wsclean_cmd, container_name, verbose=False, dry_run=False):
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
    pid = os.getpid()
    timestamp = int(time.time() * 1000)
    tmp1 = f"tmp1_{pid}_{timestamp}.txt"
    tmp2 = f"tmp2_{pid}_{timestamp}.txt"

    def show_file(path):
        try:
            print(open(path).read())
        except Exception as e:
            print(f"Error: {e}")

    container_present = check_udocker_container(container_name)
    if container_present == False:
        container_name = initialize_wsclean_container(container_name)
        if container_name == None:
            print(
                "Container {container_name} is not initiated. First initiate container and then run."
            )
            return 1
    if dry_run:
        cmd = f"chgenter >> {tmp1} >> {tmp2}"
        cwd = os.getcwd()
        full_command = (
            f"udocker --quiet run --nobanner --volume={cwd}:{cwd} meerwsclean {cmd}"
        )
        os.system(full_command)
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        os.system(f"rm -rf {tmp1} {tmp2}")
        return mem
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
        if verbose == False:
            full_command += f" >> {mspath}/{tmp1} "
        else:
            print(wsclean_cmd + "\n")
        exit_code = os.system(full_command)
        if exit_code != 0:
            print("##########################")
            print(os.path.basename(msname))
            print("##########################")
            show_file(f"{mspath}/{tmp1}")
        os.system(f"rm -rf {temp_docker_path} {mspath}/{tmp1}")
        return 0 if exit_code == 0 else 1
    except Exception as e:
        os.system(f"rm -rf {temp_docker_path}")
        traceback.print_exc()
        return 1


def run_solar_sidereal_cor(
    msname="", container_name="meerwsclean", verbose=False, dry_run=False
):
    """
    Run chgcenter inside a udocker container to correct solar sidereal motion (no root permission required).
    Parameters
    ----------
    msname : str
        Name of the measurement set
    container_name : str, optional
        Container name
    verbose : bool, optional
        Verbose output or not
    Returns
    -------
    int
        Success message
    """
    pid = os.getpid()
    timestamp = int(time.time() * 1000)
    tmp1 = f"tmp1_{pid}_{timestamp}.txt"
    tmp2 = f"tmp2_{pid}_{timestamp}.txt"
    container_present = check_udocker_container(container_name)
    if container_present == False:
        container_name = initialize_wsclean_container(container_name)
        if container_name == None:
            print(
                "Container {container_name} is not initiated. First initiate container and then run."
            )
            return 1

    if dry_run:
        cmd = f"chgcentre >> {tmp1} >> {tmp2}"
        cwd = os.getcwd()
        full_command = (
            f"udocker --quiet run --nobanner --volume={cwd}:{cwd} meerwsclean {cmd}"
        )
        os.system(full_command)
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        os.system(f"rm -rf {tmp1} {tmp2}")
        return mem

    msname = os.path.abspath(msname)
    mspath = os.path.dirname(msname)
    temp_docker_path = tempfile.mkdtemp(prefix="chgcenter_udocker_", dir=mspath)
    cmd = "chgcentre -solarcenter " + temp_docker_path + "/" + os.path.basename(msname)
    try:
        full_command = f"udocker --quiet run --nobanner --volume={mspath}:{temp_docker_path} --workdir {temp_docker_path} meerwsclean {cmd}"
        if verbose == False:
            full_command += f" >> {tmp1} >> {tmp2}"
        else:
            print(cmd)
        exit_code = os.system(full_command)
        os.system(f"rm -rf {temp_docker_path} {tmp1} {tmp2}")
        return 0 if exit_code == 0 else 1
    except Exception as e:
        os.system(f"rm -rf {temp_docker_path} {tmp1} {tmp2}")
        traceback.print_exc()
        return 1


def run_chgcenter(
    msname, ra, dec, container_name="meerwsclean", verbose=False, dry_run=False
):
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
    container_name : str, optional
        Container name
    verbose : bool, optional
        Verbose output
    Returns
    -------
    int
        Success message
    """
    pid = os.getpid()
    timestamp = int(time.time() * 1000)
    tmp1 = f"tmp1_{pid}_{timestamp}.txt"
    tmp2 = f"tmp2_{pid}_{timestamp}.txt"
    container_present = check_udocker_container(container_name)
    if container_present == False:
        container_name = initialize_wsclean_container(container_name)
        if container_name == None:
            print(
                "Container {container_name} is not initiated. First initiate container and then run."
            )
            return 1
    if dry_run:
        cmd = f"chgenter >> {tmp1} >> {tmp2}"
        cwd = os.getcwd()
        full_command = (
            f"udocker --quiet run --nobanner --volume={cwd}:{cwd} meerwsclean {cmd}"
        )
        os.system(full_command)
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        os.system(f"rm -rf {tmp1} {tmp2}")
        return mem
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
        if verbose == False:
            full_command += f" >> {tmp1} >> {tmp2}"
        else:
            print(cmd)
        exit_code = os.system(full_command)
        os.system(f"rm -rf {temp_docker_path} {tmp1} {tmp2}")
        return 0 if exit_code == 0 else 1
    except Exception as e:
        os.system(f"rm -rf {temp_docker_path} {tmp1} {tmp2}")
        traceback.print_exc()
        return 1
    return
