from .all_depend import *
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
    Get measurement set total size

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
    Get column size

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
    mssize = get_ms_size(msname, only_autocorr=only_autocorr)
    tb = table()
    tb.open(msname)
    colnames = tb.colnames()
    tb.close()
    ncol = 1
    if "CORRECTED_DATA" in colnames:
        ncol += 1
    if "MODEL_DATA" in colnames:
        ncol += 1
    datasize = mssize / ncol
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
    ms_size = get_ms_size(msname, only_autocorr=only_autocorr)
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
    if min(chanfreqs) <= bad_freqs[0][1] and max(chanfreqs) >= bad_freqs[-1][0]:
        spw = "0:"
        count = 0
        for freq_range in bad_freqs:
            start_freq = freq_range[0]
            end_freq = freq_range[1]
            if start_freq == -1:
                start_chan = 0
            else:
                start_chan = np.argmin(np.abs(start_freq - chanfreqs))
            if count > 0 and start_chan <= end_chan:
                break
            if end_freq == -1:
                end_chan = len(chanfreqs) - 1
            else:
                end_chan = np.argmin(np.abs(end_freq - chanfreqs))
            if end_chan > start_chan:
                spw += str(start_chan) + "~" + str(end_chan) + ";"
            else:
                spw += str(start_chan) + ";"
            count += 1
        spw = spw[:-1]
    else:
        spw = ""
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
    if min(chanfreqs) <= good_freqs[0][1] and max(chanfreqs) >= good_freqs[-1][0]:
        spw = "0:"
        for freq_range in good_freqs:
            start_freq = freq_range[0]
            end_freq = freq_range[1]
            start_chan = np.argmin(np.abs(start_freq - chanfreqs))
            end_chan = np.argmin(np.abs(end_freq - chanfreqs))
            spw += str(start_chan) + "~" + str(end_chan) + ";"
        spw = spw[:-1]
    else:
        spw = f"0:0~{len(chanfreqs)-1}"
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


def get_refant(msname="", n_threads=-1, dry_run=False):
    """
    Get reference antenna

    Parameters
    ----------
    msname : str
        Name of the measurement set

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
    if nant > 10:
        nant = max(10, int(0.1 * nant))
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


def get_fluxcals(msname):
    """
    Get fluxcal field names and scans (all scans, valids and invalids

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
    Get polarization calibrator field names and scans (all scans, valids and invalids

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
                if field not in polcal_fields:
                    polcal_fields.append(field)
                scans = msmd.scansforfield(field).tolist()
                if field in polcal_scans:
                    for scan in scans:
                        polcal_scans[field].append(scan)
                else:
                    polcal_scans[field] = scans
    msmd.close()
    msmd.done()
    del msmd
    return polcal_fields, polcal_scans


def get_phasecals(msname):
    """
    Get phasecal field names and scans (all scans, valids and invalids)

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
    del msmd
    return phasecal_fields, phasecal_scans, phasecal_flux_list


def get_valid_scans(msname, field="", min_scan_time=1, n_threads=-1):
    """
    Get valid list of scans

    Parameters
    ----------
    msname : str
        Measurement set name
    field : str
        Field names (comma seperated)
    min_scan_time : float
        Minimum valid scan time in minute

    Returns
    -------
    list
        Valid scan list
    """
    limit_threads(n_threads=n_threads)
    from casatools import ms as casamstool

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
            with suppress_casa_output():
                try:
                    field_id = msmd.fieldsforname(f)[0]
                except Exception as e:
                    field_id = int(f)
            selected_field.append(field_id)
        msmd.close()
        msmd.done()
        del msmd
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
    del msmd
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
    list
        Fluxcal scans
    list
        Phasecal scans
    list
        Polcal scans
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
