import types
import psutil
import numpy as np
import glob
import os
from casatools import msmetadata, ms as casamstool, table, measures
from .basic_utils import *
from .resource_utils import *

##########################
# Measurement set metadata
##########################


def get_phasecenter(msname, field):
    """
    Get phasecenter of the measurement set

    Parameters
    ----------
    msname : str
        Name of the measurement set
    field : str
        Field name

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
    return round(radeg, 5), round(decdeg, 5)


def get_timeranges_for_scan(
    msname, scan, time_interval, time_window, quack_timestamps=-1
):
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
    quack_timestamps : int, optional
        Number of timestamps ignored at the start and end of each scan

    Returns
    -------
    list
        List of time ranges
    """
    msmd = msmetadata()
    msmd.open(msname)
    try:
        times = msmd.timesforscan(int(scan))
    except BaseException:
        times = msmd.timesforspws(0)
    msmd.close()
    msmd.done()
    time_ranges = []
    if quack_timestamps > 0:
        times = times[quack_timestamps:-quack_timestamps]
    else:
        times = times[1:-1]
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
    total_time = end_time - start_time
    timeres = total_time / len(times)
    ntime_chunk = int(total_time / time_interval)
    ntime = int(time_window / timeres)
    start_time = times[:-ntime]
    indices = np.linspace(0, len(start_time) - 1, num=ntime_chunk, dtype=int)
    timelist = []
    for i in indices:
        try:
            timelist.append(start_time[i])
        except:
            pass
    if len(timelist) == 0:
        time_range = [f"{mjdsec_to_timestamp(times[0], str_format=1)}"]
        return time_range
    for t in timelist:
        time_ranges.append(
            f"{mjdsec_to_timestamp(t, str_format=1)}~{mjdsec_to_timestamp(t+time_window, str_format=1)}"
        )
    return time_ranges


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


def get_ms_size(msname, only_autocorr=False):
    """
    Get measurement set total size on-disk
    (Note: it could be smaller than actual data size, because of data compression)

    Parameters
    ----------
    msname : str
        Measurement set name
    only_autocorr : bool, optional
        Only auto-correlation

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
    if only_autocorr:
        msmd = msmetadata()
        msmd.open(msname)
        nant = msmd.nantennas()
        msmd.close()
        all_baselines = (nant * nant) / 2
        total_size /= all_baselines
        total_size *= nant
    return round(total_size / (1024**3), 2)  # in GB


def get_column_size(msname, only_autocorr=False):
    """
    Get datacolumn size
    (Note: this is true datasize in memory)

    Parameters
    ----------
    msname : str
        Measurement set
    only_autocorr : bool, optional
        Only auto-correlations

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
    nant = msmd.nantennas()
    msmd.close()
    datasize = nrow * nchan * npol * 16 / (1024.0**3)
    if only_autocorr:
        all_baselines = (nant * nant) / 2
        datasize /= all_baselines
        datasize *= nant
    return round(datasize, 2)


def get_ms_scan_size(msname, scan, only_autocorr=False):
    """
    Get measurement set scan size

    Parameters
    ----------
    msname : str
        Measurement set
    scan : int
        Scan number
    only_autocorr : bool, optional
        Only for auto-correlations

    Returns
    -------
    float
        Size in GB
    """
    tb = table()
    tb.open(msname)
    nrow = tb.nrows()
    tb.close()
    mstool = casamstool()
    mstool.open(msname)
    mstool.select({"scan_number": int(scan)})
    scan_nrow = mstool.nrow(True)
    mstool.close()
    ms_size = get_column_size(msname, only_autocorr=only_autocorr)
    scan_size = scan_nrow * (ms_size / nrow)
    return round(scan_size, 2)


def get_chunk_size(msname, memory_limit=-1, only_autocorr=False):
    """
    Get time chunk size for a memory limit

    Parameters
    ----------
    msname : str
        Measurement set
    memory_limit : int, optional
        Memory limit
    only_autocorr : bool, optional
        Only aut-correlation

    Returns
    -------
    int
        Number of chunks
    """
    if memory_limit == -1:
        memory_limit = psutil.virtual_memory().available / 1024**3  # In GB
    col_size = get_column_size(msname, only_autocorr=only_autocorr)
    nchunk = int(col_size / memory_limit)
    if nchunk < 1:
        nchunk = 1
    return nchunk


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
        except BaseException:
            tb.close()
            return False
    except BaseException:
        return False


def get_bad_ants(msname="", fieldnames=[], n_threads=-1, dry_run=False):
    """
    Get bad antennas

    Parameters
    ----------
    msname : str
        Name of the ms
    fieldnames : list, optional
        Fluxcal field names

    Returns
    -------
    list
        Bad antenna list
    str
        Bad antenna string
    """
    limit_threads(n_threads=n_threads)
    from casatasks import visstat

    if len(fieldnames) == 0:
        print("Provide field names.")
        return [], ""
    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    msmd = msmetadata()
    all_field_bad_ants = []
    msmd.open(msname)
    nant = msmd.nantennas()
    msmd.close()
    msmd.done()
    for field in fieldnames:
        ant_medians = []
        bad_ants = []
        for ant in range(nant):
            stat_median = visstat(
                vis=msname,
                field=str(field),
                uvrange="0lambda",
                antenna=str(ant) + "&&" + str(ant),
                useflags=False,
            )["DATA_DESC_ID=0"]["median"]
            ant_medians.append(stat_median)
        ant_medians = np.array(ant_medians)
        all_ant_median = np.nanmean(ant_medians)
        all_ant_std = np.nanstd(ant_medians)
        pos = np.where(ant_medians < all_ant_median - (5 * all_ant_std))[0]
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
        First spectral window (0:xx~yy)
    spw2 : str
        Second spectral window (0:xx1~yy1)

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
        if len(result) > 0:
            return "0:" + ";".join(result)
        else:
            return ""

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
        # Try parsing as date string
        tr_start = timestamp_to_mjdsec(tr_start_str)
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


def get_refant(msname="", field="", n_threads=-1, dry_run=False):
    """
    Get reference antenna

    Parameters
    ----------
    msname : str
        Name of the measurement set
    field : str, optional
        Field name

    Returns
    -------
    str
        Reference antenna
    """
    limit_threads(n_threads=n_threads)
    from casatasks import visstat, casalog

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msname = msname.rstrip("/")
    mspath = os.path.dirname(os.path.abspath(msname))
    os.chdir(mspath)
    casalog.filter("SEVERE")
    msmd = msmetadata()
    msmd.open(msname)
    nant = msmd.nantennas()
    msmd.close()
    msmd.done()
    antamp = []
    antrms = []
    selected_nant = min(10, int(0.1 * nant))
    selected_nant = min(selected_nant, nant)
    for ant in range(selected_nant):
        ant = str(ant)
        t = visstat(
            vis=msname,
            field=field,
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


def get_ms_scans(msname):
    """
    Get scans of the measurement set

    Parameters
    ----------
    msname : str
        Measurement set

    Returns
    -------
    list
        Scan list
    """
    msmd = msmetadata()
    msmd.open(msname)
    scans = msmd.scannumbers().tolist()
    msmd.close()
    return scans


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
    from casatasks import listpartition

    if os.path.exists(msname + "/SUBMSS") == False:
        print("Input measurement set is not a multi-MS")
        return
    partitionlist = listpartition(vis=msname, createdict=True)
    scans = []
    mslist = []
    for i in range(len(partitionlist)):
        subms = partitionlist[i]
        subms_name = msname + "/SUBMSS/" + subms["MS"]
        mslist.append(subms_name)
        os.system(f"rm -rf {subms_name}/.flagversions")
        scan_number = list(subms["scanId"].keys())[0]
        scans.append(scan_number)
    return mslist, scans


def get_observatory_name(msname):
    """
    Get observatory name

    Parameters
    ----------
    msname : str
        Measurement set

    Returns
    -------
    str
        Observatory name in all upper case
    """
    observatory = ""
    try:
        msmd = msmetadata()
        msmd.open(msname)
        observatory = msmd.observatorynames()[0].upper()
        msmd.close()
    except Exception:
        pass
    return observatory


def get_observatory_coord(msname):
    """
    Get observatory coordinate

    Parameters
    ----------
    msname : str
        Measurement set

    Returns
    -------
    float
        Latitude in degrees
    float
        Longitude in degrees
    float
        Height in meters
    """
    msmd = msmetadata()
    msmd.open(msname)
    position = msmd.observatoryposition()
    me = measures()
    obs_pos = me.observatory(msmd.observatorynames()[0])
    lon = obs_pos["m0"]["value"] * (180.0 / 3.141592653589793)
    lat = obs_pos["m1"]["value"] * (180.0 / 3.141592653589793)
    height = obs_pos["m2"]["value"]
    msmd.close()
    return round(lat, 3), round(lon, 3), round(height, 3)


def get_pol_names(msname, fullpol=True):
    """
    Get correlation names

    Parameters
    ----------
    msname : str
        Measurement set
    fullpol : bool, optional
        Full polarization products or not

    Returns
    -------
    list
        List of cross correlation product names
    """
    CASA_POL_PRODUCTS = {
        1: "I",
        2: "Q",
        3: "U",
        4: "V",
        5: "RR",
        6: "RL",
        7: "LR",
        8: "LL",
        9: "XX",
        10: "XY",
        11: "YX",
        12: "YY",
    }
    msmd = msmetadata()
    msmd.open(msname)
    pols = msmd.corrtypesforpol(0)
    msmd.close()
    pol_names = []
    for p in pols:
        pol_name = CASA_POL_PRODUCTS[int(p)]
        if fullpol is True:
            pol_names.append(pol_name)
        else:
            if pol_name in ["XX", "YY", "RR", "LL", "I"]:
                pol_names.append(pol_name)
            else:
                pass
    return pol_names


# Expose functions and classes
__all__ = [
    name
    for name, obj in globals().items()
    if (
        (isinstance(obj, types.FunctionType) or isinstance(obj, type))
        and obj.__module__ == __name__
    )
]
