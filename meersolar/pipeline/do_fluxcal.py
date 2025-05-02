import os, time, psutil, numpy as np, glob, traceback, gc, copy, tempfile
from meersolar.pipeline.basic_func import *
from meersolar.pipeline.flagging import single_ms_flag
from meersolar.pipeline.import_model import import_fluxcal_models
from casatools import table, ms as casamstool, msmetadata
from optparse import OptionParser
from dask import delayed, compute, config
from casatasks import casalog

logfile = casalog.logfile()
os.system("rm -rf " + logfile)


def split_casatask(
    msname="", outputvis="", scan="", time_range="", n_threads=-1, dry_run=False
):
    limit_threads(n_threads=n_threads)
    from casatasks import split

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    split(
        vis=msname,
        outputvis=outputvis,
        scan=scan,
        timerange=time_range,
        datacolumn="data",
        uvrange="0",
        correlation="XX,YY",
    )
    return outputvis


def applycal_casatask(
    msname="", caltable="", applymode="", n_threads=-1, dry_run=False
):
    limit_threads(n_threads=n_threads)
    from casatasks import applycal

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    applycal(vis=msname, gaintable=caltable, applymode=applymode)
    return


def split_autocorr(
    msname, workdir, scan_list, time_window=-1, cpu_frac=0.8, mem_frac=0.8
):
    """
    Split auto-correlations
    Parameters
    ----------
    msname : str
        Measurement set
    workdir : str
        Working directory
    scan_list : list
        Scan list
    time_window : float, optional
        Time window in seconds from start time of the scan
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    Returns
    -------
    list
        Splited ms list
    """
    msname = msname.rstrip("/")
    task = delayed(split_casatask)(dry_run=True)
    mem_limit = run_limited_memory_task(task)
    dask_client, dask_cluster, n_jobs, n_threads = get_dask_client(
        len(scan_list),
        dask_dir=workdir,
        cpu_frac=cpu_frac,
        mem_frac=mem_frac,
        min_mem_per_job=mem_limit / 0.6,
    )
    tasks = []
    for scan in scan_list:
        if time_window > 0:
            tb = table()
            tb.open(msname)
            tbsel = tb.query(f"SCAN_NUMBER=={scan}")
            times = tbsel.getcol("TIME")
            tbsel.close()
            tb.close()
            if len(times) == 0:
                continue
            start_time = times[0]
            end_time = start_time + time_window  # add in seconds
            if end_time > max(times):
                end_time = max(times)
            start_time = mjdsec_to_timestamp(start_time, str_format=1)
            end_time = mjdsec_to_timestamp(end_time, str_format=1)
            time_range = f"{start_time}~{end_time}"
        else:
            time_range = ""
        outputvis = workdir + "/autocorr_scan_" + str(scan) + ".ms"
        if os.path.exists(outputvis):
            os.system("rm -rf " + outputvis)
        if os.path.exists(outputvis + ".flagversions"):
            os.system("rm -rf " + outputvis + ".flagversions")
        tasks.append(
            delayed(split_casatask)(
                msname, outputvis, scan, time_range, n_threads=n_threads
            )
        )
    if len(tasks) > 0:
        autocorr_mslist = compute(*tasks)
    else:
        autocorr_mslist = []
    dask_client.close()
    dask_cluster.close()
    return autocorr_mslist


def get_on_off_power(msname="", ant_list=[], on_cal="", off_cal="", dry_run=False):
    """
    Get noise diode on and off power averaged over antennas
    Parameters
    ----------
    msname : str
        Measurement set name
    ant_list : list
        Antenna id list
    on_cal : str
        Noise on calibration table
    off_cal : str
        Noise off calibration table
    Returns
    -------
    numpy.array
        On average spectrum of power
    numpy.array
        Off average spectrum of power
    """
    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    ######################
    mstool = casamstool()
    mstool.open(msname)
    mstool.select({"antenna1": ant_list, "antenna2": ant_list, "uvdist": [0.0, 0.0]})
    mstool.selectpolarization(["XX", "YY"])
    data_dict = mstool.getdata(["DATA", "FLAG"], ifraxis=True)
    mstool.close()
    del mstool
    data = np.abs(data_dict["data"])
    data[data_dict["flag"]] = np.nan
    del data_dict
    n_tstamps = min(data[..., ::2].shape[-1], data[..., 1::2].shape[-1])
    antslice = slice(min(ant_list), max(ant_list) + 1)
    if data[..., ::2][0, 0, 0, 0] > data[..., 1::2][0, 0, 0, 0]:
        data_on = copy.deepcopy(data[..., ::2][..., :n_tstamps])
        data_off = copy.deepcopy(data[..., 1::2][..., :n_tstamps])
    else:
        data_on = copy.deepcopy(data[..., 1::2][..., :n_tstamps])
        data_off = copy.deepcopy(data[..., ::2][..., :n_tstamps])
    del data
    # Load caltables
    tb = table()
    tb.open(on_cal)
    gain_on_slice = np.abs(tb.getcol("CPARAM"))[..., antslice]
    gain_on_slice[tb.getcol("FLAG")[..., antslice]] = np.nan
    tb.close()
    tb.open(off_cal)
    gain_off_slice = np.abs(tb.getcol("CPARAM"))[..., antslice]
    gain_off_slice[tb.getcol("FLAG")[..., antslice]] = np.nan
    tb.close()
    del tb
    # Gain correction
    #gain_on_slice = gain_on[..., antslice]
    #gain_off_slice = gain_off[..., antslice]
    #del gain_on, gain_off
    gain_on_exp = np.repeat(gain_on_slice[..., np.newaxis], data_on.shape[-1], axis=-1)
    gain_off_exp = np.repeat(
        gain_off_slice[..., np.newaxis], data_off.shape[-1], axis=-1
    )
    del gain_on_slice, gain_off_slice
    data_on /= gain_on_exp**2
    data_off /= gain_off_exp**2
    avg_on = np.nanmean(data_on, axis=2)
    avg_off = np.nanmean(data_off, axis=2)
    # Cleanup per chunk
    del data_on, data_off, gain_on_exp, gain_off_exp
    return avg_on, avg_off


def get_power_diff(
    msname="", on_cal="", off_cal="", n_threads=-1, memory_limit=-1, dry_run=False
):
    """
    Estimate power level difference between alternative correlator dumps.

    Parameters
    ----------
    msname : str
        Measurement set
    on_cal : str
        Noise diode on caltable
    off_cal : str
        Noise diode off caltable
    n_threads : int, optional
        Number of OpenMP threads
    memory_limit : float, optional
        Memory limit in GB

    Returns
    -------
    numpy.array
        Power level difference spectra for both polarizations
    """
    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    import warnings

    warnings.filterwarnings("ignore")
    starttime = time.time()
    limit_threads(n_threads=n_threads)
    if memory_limit == -1:
        memory_limit = psutil.virtual_memory().available / 1024**3  # GB
    msname = msname.rstrip("/")
    print(f"Estimating power difference for {msname}....")
    # Get MS metadata
    msmd = msmetadata()
    msmd.open(msname)
    nrow = int(msmd.nrows())
    nchan = msmd.nchan(0)
    npol = msmd.ncorrforpol(0)
    nant = msmd.nantennas()
    nbaselines = msmd.nbaselines()
    if nbaselines == 0 or nrow % nbaselines != 0:
        nbaselines += nant
    ntime = int(nrow / nbaselines)
    msmd.close()
    #########################################
    # Estimate per-antenna memory requirement
    #########################################
    # 2 times is added because, data, data_on (half of data), data_off (half of data) in get_on_off_power()
    per_ant_data_memory = (npol * nchan * ntime * 16 * 2) / 1024.0**3  
    per_ant_flag_memory = (npol * nchan * ntime) / 1024.0**3  # One flag object
    per_ant_gain_memory = (npol * nchan * ntime * 16) / 1024.0**3  # gain_on_exp and gain_off_exp in get_on_off_power()
    per_ant_data_memory_alltime=(npol * nchan * 16) / 1024.0**3  # avg_on, avg_off
    per_ant_memory=per_ant_data_memory+per_ant_flag_memory+per_ant_gain_memory+per_ant_data_memory_alltime # Per antenna total memory
    ########################################
    # Determining chunk size
    ########################################
    nant_per_chunk = min(nant, max(2, int(memory_limit / per_ant_memory)))
    ant_blocks = [
        list(range(i, min(i + nant_per_chunk, nant)))
        for i in range(0, nant, nant_per_chunk)
    ]
    on_data_avg = None
    off_data_avg = None
    for i, ant_block in enumerate(ant_blocks):
        print (f"{os.path.basename(msname)} -- Antenna block: {ant_block}")
        avg_on, avg_off = get_on_off_power(msname, ant_block, on_cal, off_cal)
        if i == 0:
            on_data_avg = avg_on
            off_data_avg = avg_off
        else:
            on_data_avg = (on_data_avg + avg_on) / 2.0
            off_data_avg = (off_data_avg + avg_off) / 2.0
    diff = np.nanmean((on_data_avg - off_data_avg), axis=-1)
    del on_data_avg, off_data_avg
    return diff


def estimate_att(
    msname,
    workdir,
    noise_on_caltable,
    noise_off_caltable,
    noise_diode_flux_scan,
    valid_target_scans,
    time_window=900,
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Estimate attenaution scaling
    Parameters
    ----------
    msname : str
        Measurement set name
    workdir : str
        Working directory
    noise_on_caltable : int
        Caltable with noise diode on
    noise_off_caltable : str
        Caltable with noise diode off
    noise_diode_flux_scan : float
        Fluxcal scan with noise diode
    valid_target_scans : list
        Valid list of target scans
    time_window : float, optional
        Time window in second to use
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    Returns
    -------
    int
        Success message
    dict
        Dictionary with attenuation spectra for each solar scan
    list
        Attenuation spectra file list
    """
    try:
        print("Estimating attenuation ...")
        ##########################################
        # Determining time window equal to fluxcal
        ##########################################
        tb = table()
        tb.open(msname)
        tbsel = tb.query(f"SCAN_NUMBER=={noise_diode_flux_scan}")
        times = tbsel.getcol("TIME")
        tbsel.close()
        tb.close()
        msmd = msmetadata()
        msmd.open(msname)
        freqs = msmd.chanfreqs(0)
        msmd.close()
        msmd.done()
        ###########################################
        # All auto-corr scans spliting
        ###########################################
        all_scans = [noise_diode_flux_scan] + valid_target_scans
        print("Spliting auto-correlation in different scans ...")
        autocorr_mslist = split_autocorr(
            msname,
            workdir,
            all_scans,
            time_window=time_window,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
        )
        if len(autocorr_mslist) == 0:
            print("No scans splited.")

            return 1, None, None

        ########################################
        # Apply solutions
        ########################################
        print("Applying calibrartion solutions auto-correlation measurement sets ...")
        task = delayed(applycal_casatask)(dry_run=True)
        mem_limit = run_limited_memory_task(task)
        dask_client, dask_cluster, n_jobs, n_threads = get_dask_client(
            len(autocorr_mslist),
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem_per_job=mem_limit / 0.6,
        )
        tasks = []
        for autocorr_msname in autocorr_mslist:
            tasks.append(
                delayed(applycal_casatask)(
                    autocorr_msname,
                    [noise_off_caltable],
                    "calflag",
                    n_threads=n_threads,
                )
            )
        results = compute(*tasks)
        dask_client.close()
        dask_cluster.close()

        ##########################################
        # Flagging on corrected data
        ##########################################
        fluxcal_fields, fluxcal_scans = get_fluxcals(msname)
        badspw = get_bad_chans(msname)
        bad_ants, bad_ants_str = get_bad_ants(msname, fieldnames=fluxcal_fields)
        print("Flagging auto-correlation measurement sets ...")
        task = delayed(single_ms_flag)(dry_run=True)
        mem_limit = run_limited_memory_task(task)
        dask_client, dask_cluster, n_jobs, n_threads = get_dask_client(
            len(autocorr_mslist),
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem_per_job=mem_limit / 0.6,
        )
        tasks = []
        for autocorr_msname in autocorr_mslist:
            tasks.append(
                delayed(single_ms_flag)(
                    autocorr_msname,
                    badspw=badspw,
                    bad_ants_str=bad_ants_str,
                    datacolumn="corrected",
                    use_tfcrop=False,
                    use_rflag=True,
                    flagdimension="freqtime",
                    flag_autocorr=False,
                    n_threads=n_threads,
                    memory_limit=mem_limit,
                )
            )
        results = compute(*tasks)
        dask_client.close()
        dask_cluster.close()

        ###########################################
        # Calculating fluxcal power levels
        ###########################################
        total_cpus = psutil.cpu_count(logical=True)
        free_cpu_percentages = 100 - psutil.cpu_percent(interval=1)
        total_cpus = int(total_cpus * free_cpu_percentages / 100.0)
        usable_cpus = int(total_cpus * (1 - cpu_frac))
        n_threads = max(1, usable_cpus)
        total_mem = psutil.virtual_memory().available
        memory_limit = (total_mem * (1 - mem_frac)) / 1024**3
        d_fluxcal_spectra = get_power_diff(
            f"{workdir}/autocorr_scan_{noise_diode_flux_scan}.ms",
            noise_on_caltable,
            noise_off_caltable,
            n_threads=n_threads,
            memory_limit=memory_limit,
        )
        att_level = {}

        ########################################
        # Calculating per scan level
        ########################################
        print("Calculating noise-diode power difference ...")
        task = delayed(get_power_diff)(dry_run=True)
        mem_limit = run_limited_memory_task(task)
        dask_client, dask_cluster, n_jobs, n_threads = get_dask_client(
            len(valid_target_scans),
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem_per_job=mem_limit / 0.6,
        )
        all_scaling_files = []
        filtered_scans = []
        tasks = []
        for scan in valid_target_scans:
            autocorr_msname = f"{workdir}/autocorr_scan_{scan}.ms"
            if autocorr_msname not in autocorr_mslist:
                pass
            tasks.append(
                delayed(get_power_diff)(
                    f"{workdir}/autocorr_scan_{scan}.ms",
                    noise_off_caltable,
                    noise_off_caltable,
                    n_threads=n_threads,
                    memory_limit=mem_limit,
                )
            )
            filtered_scans.append(scan)
        results = compute(*tasks)
        dask_client.close()
        dask_cluster.close()
        for i in range(len(filtered_scans)):
            d_source_spectra = results[i]
            att_value = d_source_spectra / d_fluxcal_spectra
            if np.nanmean(att_value) < 0:
                att_value *= -1
            scan = filtered_scans[i]
            att_level[scan] = att_value
            filename = (
                workdir
                + "/"
                + os.path.basename(msname).split(".ms")[0]
                + "_attval_scan_"
                + str(scan)
            )
            np.save(filename, np.array([scan, freqs, att_value], dtype="object"))
            all_scaling_files.append(filename + ".npy")
        return 0, att_level, all_scaling_files
    except Exception as e:
        traceback.print_exc()
        return 1, None, None


def run_noise_cal(
    msname,
    workdir,
    keep_backup=False,
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Perform flux calibration using noise diode
    Parameters
    ----------
    msname : str
        Measurement set
    workdir : str
        Working directory
    keep_backup : bool, optional
        Keep backup
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    Returns
    -------
    int
        Success message
    dict
        Attenuation values for different scans
    list
        File list saved attenuation values for different scans
    """
    start_time = time.time()
    try:
        os.chdir(workdir)
        msname = msname.rstrip("/")
        workdir = workdir.rstrip("/")
        print("##############################################")
        print("Performing flux calibration using noise-diode.")
        print("##############################################")
        ###################################
        # Determining noise diode cal scans
        ###################################
        target_scans, cal_scans, f_scans, g_scans, p_scans = get_cal_target_scans(
            msname
        )
        valid_scans = get_valid_scans(msname)
        noise_diode_cal_scan = ""
        for scan in cal_scans:
            if scan in valid_scans:
                noise_cal = determine_noise_diode_cal_scan(msname, scan)
                if noise_cal:
                    noise_diode_cal_scan = scan
                    break
        valid_target_scans = []
        for scan in target_scans:
            if scan in valid_scans:
                valid_target_scans.append(scan)
        if noise_diode_cal_scan == "":
            print("No noise diode cal scan is present.")
            print("##################")
            print("Total time taken : ", time.time() - start_time)
            print("##################\n")

            return 1, None, None

        ##############################
        # Split noise cal scan
        ##############################
        noisecal_ms = workdir + "/noisecal.ms"
        if os.path.exists(noisecal_ms):
            os.system("rm -rf " + noisecal_ms)
        if os.path.exists(noisecal_ms + ".flagversions"):
            os.system("rm -rf " + noisecal_ms + ".flagversions")
        from casatasks import split

        split(
            vis=msname,
            outputvis=noisecal_ms,
            scan=noise_diode_cal_scan,
            datacolumn="data",
        )

        ###############################
        # Flagging
        ###############################
        print("Flagging noise cal ms....")
        fluxcal_fields, fluxcal_scans = get_fluxcals(msname)
        badspw = get_bad_chans(msname)
        bad_ants, bad_ants_str = get_bad_ants(msname, fieldnames=fluxcal_fields)
        single_ms_flag(
            noisecal_ms,
            badspw=badspw,
            bad_ants_str=bad_ants_str,
            datacolumn="data",
            use_tfcrop=True,
            use_rflag=False,
            flagdimension="freqtime",
            flag_autocorr=False,
        )

        ##################################
        # Import models
        ##################################
        print("Importing calibrator models ....")
        ncpus = int(psutil.cpu_count() * (1 - cpu_frac))
        import_fluxcal_models(noisecal_ms, ncpus=ncpus, mem_frac=1 - cpu_frac)

        ##################################
        # Bandpass calibration
        ##################################
        print("Peforming bandpass with noise diode on and off....")
        msmd = msmetadata()
        msmd.open(noisecal_ms)
        times = msmd.timesforscan(noise_diode_cal_scan)
        msmd.close()
        even_times = times[::2]  # Even-indexed timestamps
        odd_times = times[1::2]  # Odd-indexed timestamps
        even_timerange = ",".join(
            [mjdsec_to_timestamp(t, str_format=1) for t in even_times]
        )
        odd_timerange = ",".join(
            [mjdsec_to_timestamp(t, str_format=1) for t in odd_times]
        )
        mstool = casamstool()
        mstool.open(noisecal_ms)
        mstool.select({"antenna1": 1, "antenna2": 1, "time": [times[0], times[1]]})
        dataeven = np.abs(mstool.getdata("DATA", ifraxis=True)["data"])
        mstool.close()
        mstool.open(noisecal_ms)
        mstool.select({"antenna1": 1, "antenna2": 1, "time": [times[1], times[2]]})
        dataodd = np.abs(mstool.getdata("DATA", ifraxis=True)["data"])
        mstool.close()
        if np.nansum(dataeven) > np.nansum(dataodd):
            on_timerange = even_timerange
            off_timerange = odd_timerange
        else:
            on_timerange = odd_timerange
            off_timerange = even_timerange
        oncal = noisecal_ms.split(".ms")[0] + "_on.bcal"
        offcal = noisecal_ms.split(".ms")[0] + "_off.bcal"
        from casatasks import bandpass

        bandpass(
            vis=noisecal_ms,
            caltable=oncal,
            timerange=on_timerange,
            uvrange=">200lambda",
            minsnr=1,
        )
        bandpass(
            vis=noisecal_ms,
            caltable=offcal,
            timerange=off_timerange,
            uvrange=">200lambda",
            minsnr=1,
        )

        #####################################
        # Determine attenuation scaling
        #####################################
        msg, att_level, all_scaling_files = estimate_att(
            msname,
            workdir,
            oncal,
            offcal,
            noise_diode_cal_scan,
            valid_target_scans,
            time_window=900,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
        )
        if keep_backup:
            print("Backup directory: " + workdir + "/backup")
            if os.path.isdir(workdir + "/backup") == False:
                os.makedirs(workdir + "/backup")
            else:
                os.system("rm -rf " + workdir + "/backup/*")
            os.system(
                "mv "
                + noisecal_ms
                + " "
                + oncal
                + " "
                + offcal
                + "  "
                + workdir
                + "/autocorr_scan_*.ms* "
                + workdir
                + "/backup/"
            )
        else:
            os.system(
                "rm -rf "
                + noisecal_ms
                + " "
                + oncal
                + " "
                + offcal
                + " "
                + workdir
                + "/autocorr_scan_*.ms*"
            )
        print("##################")
        print("Total time taken : ", time.time() - start_time)
        print("##################\n")
        return msg, att_level, all_scaling_files
    except Exception as e:
        traceback.print_exc()
        """os.system(
            "rm -rf "
            + noisecal_ms
            + " "
            + oncal
            + " "
            + offcal
            + " "
            + workdir
            + "/autocorr_scan_*.ms*"
        )"""
        print("##################")
        print("Total time taken : ", time.time() - start_time)
        print("##################\n")
        return 1, None, None


def main():
    usage = "Basic calibration using calibrators"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--msname",
        dest="msname",
        default="",
        help="Name of measurement set",
        metavar="Measurement Set",
    )
    parser.add_option(
        "--workdir",
        dest="workdir",
        default="",
        help="Name of work directory",
        metavar="String",
    )
    parser.add_option(
        "--keep_backup",
        dest="keep_backup",
        default=False,
        help="Keep backup of measurement set after each round",
        metavar="Boolean",
    )
    parser.add_option(
        "--print_casalog",
        dest="print_casalog",
        default=False,
        help="Print CASA log",
        metavar="Boolean",
    )
    parser.add_option(
        "--cpu_frac",
        dest="cpu_frac",
        default=0.8,
        help="CPU fraction to use",
        metavar="Float",
    )
    parser.add_option(
        "--mem_frac",
        dest="mem_frac",
        default=0.8,
        help="Memory fraction to use",
        metavar="Float",
    )
    (options, args) = parser.parse_args()
    if eval(str(options.print_casalog)) == True:
        casalog.showconsole(True)
    if options.msname != "" and os.path.exists(options.msname):
        print("\n###################################")
        print("Starting flux calibration using noise-diode.")
        print("###################################\n")
        try:
            if options.workdir == "" or os.path.exists(options.workdir) == False:
                workdir = os.path.dirname(os.path.abspath(options.msname)) + "/workdir"
                if os.path.exists(workdir) == False:
                    os.makedirs(workdir)
            else:
                workdir = options.workdir
            caldir = workdir + "/caltables"
            if os.path.exists(caldir) == False:
                os.makedirs(caldir)
            msg, att_level, all_scaling_files = run_noise_cal(
                options.msname,
                workdir,
                keep_backup=eval(str(options.keep_backup)),
                cpu_frac=float(options.cpu_frac),
                mem_frac=float(options.mem_frac),
            )
            if msg == 0 and all_scaling_files != None:
                for att_file in all_scaling_files:
                    os.system("mv " + att_file + " " + caldir)
            return msg
        except Exception as e:
            traceback.print_exc()
            return 1
    else:
        print("Please provide correct measurement set.\n")
        return 1


if __name__ == "__main__":
    result = main()
    print(
        "\n###################\nNoise diode based flux calibration is finished.\n###################\n"
    )
    os._exit(result)
