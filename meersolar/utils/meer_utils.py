import types
import psutil
import numpy as np
import glob
import os
import traceback
import warnings
import astropy.units as u
from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries
from astroquery.jplhorizons import Horizons
from astropy.visualization import ImageNormalize, LogStretch
from astropy.wcs import FITSFixedWarning
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from casatools import msmetadata, ms as casamstool, table
from datetime import datetime as dt, timedelta
from .basic_utils import *
from .resource_utils import *
from .udocker_utils import *
from .ms_metadata import *
from .image_utils import *

warnings.simplefilter("ignore", category=FITSFixedWarning)


#######################
# MS metadata related
#######################


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
            if field in ["J1939-6342", "J0408-6545"]:
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
    del msmd
    for field in fluxcal_scans:
        scans = np.unique(fluxcal_scans[field]).tolist()
        fluxcal_scans[field] = scans
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
    for field in polcal_scans:
        scans = np.unique(polcal_scans[field]).tolist()
        polcal_scans[field] = scans
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
    for field in phasecal_scans:
        scans = np.unique(phasecal_scans[field]).tolist()
        phasecal_scans[field] = scans
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
        print("MeerKAT data is not in UHF or L-band.")
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
        good_freqs = []  # For S band
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


###########################
# Spliting noise diode
###########################


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

    def is_noisescan(msname, chan, scan):
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
        mean_d_xx = np.abs(np.nanmedian(d_xx))
        mean_d_yy = np.abs(np.nanmedian(d_yy))
        if mean_d_xx > 10 and mean_d_yy > 10:
            return True
        else:
            return False

    print(f"Check noise-diode cal for scan : {scan}")
    good_spw = get_good_chans(msname)
    chan = int(good_spw.split(";")[0].split(":")[-1].split("~")[0])
    return is_noisescan(msname, chan, scan)
