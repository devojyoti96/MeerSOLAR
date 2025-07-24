import logging
import psutil
import dask
import numpy as np
import argparse
import traceback
import warnings
import time
import sys
import os
from casatasks import casalog
from casatools import msmetadata, ms as casamstool, table
from dask import delayed
from dask.distributed import Client
from meersolar.utils import *
from meersolar.meerpipeline.flagging import single_ms_flag
from meersolar.meerpipeline.import_model import import_fluxcal_models

logging.getLogger("distributed").setLevel(logging.WARNING)

try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except BaseException:
    pass

datadir = get_datadir()


def split_casatask(
    msname="", outputvis="", scan="", time_range="", n_threads=-1, dry_run=False
):
    limit_threads(n_threads=n_threads)
    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    with suppress_casa_output():
        from casatasks import split

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


def split_autocorr(
    msname, workdir, scan_list, time_window=-1, cpu_frac=0.8, mem_frac=0.8, dask_addr=None,
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
    dask_addr : str, optional
        Dask scheduler address

    Returns
    -------
    list
        Splited ms list
    """
    msname = msname.rstrip("/")
    scan_size_list=[get_ms_scan_size(msname,scan,only_autocorr=True) for scan in scan_list]
    mem_limit=max(scan_size_list)
    if dask_addr is None:
        dask_client, dask_cluster, n_jobs, n_threads, mem_limit, dask_dir = get_dask_client(
            len(scan_list),
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem_per_job=mem_limit,
        )
    else:
        _, _, n_jobs, n_threads, mem_limit, dask_dir = get_dask_client(
            len(scan_list),
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem_per_job=mem_limit,
            only_cal=True,
        )
        dask_client=Client(address=dask_addr)
        os.system(f"rm -rf {dask_dir}")
    wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
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
        futures = dask_client.compute(tasks)
        autocorr_mslist = list(dask_client.gather(futures))
    else:
        autocorr_mslist = []
    dask_client.close()
    if dask_addr is None:
        dask_cluster.close()
        os.system(f"rm -rf {dask_dir}")
    return autocorr_mslist


def get_on_off_power(msname="", scale_factor="", ant_list=[], dry_run=False):
    """
    Get noise diode on and off power averaged over antennas

    Parameters
    ----------
    msname : str
        Measurement set name
    scale_factor : numpy.array
        Scaling factor for on-off gain offset in fluxcal scan (shape: npol, nchan)
    ant_list : list
        Antenna id list

    Returns
    -------
    numpy.array
        Spectra of power difference
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
    data_source = np.abs(data_dict["data"]).astype(np.float32)
    data_source[data_dict["flag"]] = np.nan
    del data_dict
    n_total = data_source.shape[-1]
    n_tstamps = (n_total // 2) * 2
    antslice = slice(min(ant_list), max(ant_list) + 1)
    if data_source[0, 0, 0, 0] > data_source[0, 0, 0, 1]:
        on_idx = slice(0, n_tstamps, 2)
        off_idx = slice(1, n_tstamps, 2)
    else:
        on_idx = slice(1, n_tstamps, 2)
        off_idx = slice(0, n_tstamps, 2)
    ###################################################
    # Averaging along time axis in the antenna chunk
    ###################################################
    diff_source = np.nanmean(
        scale_factor[..., None, None] * data_source[..., on_idx]
        - data_source[..., off_idx],
        axis=-1,
    )
    del data_source
    return diff_source


def get_att_per_ant(cal_msname, source_msname, scale_factor, ant_list=[]):
    """
    Get per antenna attenuatioin array

    Parameters
    ----------
    cal_msname : str
        Fluxcal scan with noise diode
    source_msname : str
        Source scan with noise diode
    scale_factor : numpy.array
        Scaling factor for on-off gain offset in fluxcal scan (shape: npol, nchan)
    ant_list : list, optional
        Antenna list, default: all antennas

    Returns
    -------
    numpy.array
        Attenuation per antenna array. Shape : npol, nchan, nantenna
    """
    if len(ant_list) == 0:
        msmd = msmetadata()
        msmd.open(cal_msname)
        nant = msmd.nantennas()
        msmd.close()
        del msmd
        ant_list = [i for i in range(nant)]
    cal_diff = get_on_off_power(cal_msname, scale_factor, ant_list=ant_list)
    source_diff = get_on_off_power(
        source_msname, scale_factor * 0 + 1, ant_list=ant_list
    )
    att = source_diff / cal_diff
    del source_diff, cal_diff
    return att


def get_power_diff(
    cal_msname="",
    source_msname="",
    on_cal="",
    off_cal="",
    n_threads=-1,
    memory_limit=-1,
    dry_run=False,
):
    """
    Estimate power level difference between alternative correlator dumps.


    Parameters
    ----------
    cal_msname : str
        Fluxcal measurement set
    source_msname : str
        Source measurement set
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
        Attenuation spectra for both polarizations avergaed over all antennas
    numpy.array
        Attenuation spectra for both polarizations for all antennas
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
    cal_msname = cal_msname.rstrip("/")
    source_msname = source_msname.rstrip("/")
    print(
        f"Estimating power difference from {os.path.basename(cal_msname)} and {os.path.basename(source_msname)}....\n"
    )
    # Get MS metadata
    msmd = msmetadata()
    msmd.open(source_msname)
    nrow = int(msmd.nrows())
    nchan = msmd.nchan(0)
    npol = msmd.ncorrforpol(0)
    nant = msmd.nantennas()
    nbaselines = msmd.nbaselines()
    if nbaselines == 0 or nrow % nbaselines != 0:
        nbaselines += nant
    ntime = int(nrow / nbaselines)
    msmd.close()
    del msmd
    ####################################
    # Calculate mean on-off gain offset
    ####################################
    tb = table()
    tb.open(on_cal)
    ongain = np.abs(tb.getcol("CPARAM")) ** 2
    tb.close()
    tb.open(off_cal)
    offgain = np.abs(tb.getcol("CPARAM")) ** 2
    tb.close()
    del tb
    G = (ongain - offgain) / offgain  # Power offset
    del ongain, offgain
    G_mean = np.nanmean(G, axis=-1)  # Averaged over antennas
    del G
    scale_factor = 1 / (1 + G_mean)
    del G_mean
    ########################################
    # Determining chunk size
    ########################################
    cal_mssize = get_column_size(cal_msname, only_autocorr=True)
    source_mssize = get_column_size(source_msname, only_autocorr=True)
    total_mssize = cal_mssize + source_mssize
    scale_factor_size = nant * ntime * scale_factor.nbytes / (1024.0**3)
    att_ant_array_size = (npol * nchan * nant * 16) / (1024.0**3)
    func_mem = get_on_off_power(dry_run=True)
    per_ant_memory = (scale_factor_size + total_mssize) / nant
    per_ant_memory += att_ant_array_size + func_mem
    nant_per_chunk = min(nant, max(2, int(memory_limit / per_ant_memory)))
    ##############################################
    # Estimating per antenna attenuation in chunks
    ##############################################
    ant_blocks = [
        list(range(i, min(i + nant_per_chunk, nant)))
        for i in range(0, nant, nant_per_chunk)
    ]
    for i, ant_block in enumerate(ant_blocks):
        if i == 0:
            att_ant_array = get_att_per_ant(
                cal_msname, source_msname, scale_factor, ant_list=ant_block
            )
        else:
            att_ant_array = np.append(
                att_ant_array,
                get_att_per_ant(
                    cal_msname, source_msname, scale_factor, ant_list=ant_block
                ),
                axis=-1,
            )
    ######################################
    # Averaging over all antennas
    ######################################
    att = np.nanmedian(att_ant_array, axis=-1)
    return att, att_ant_array


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
    dask_addr=None,
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
    dask_addr : str, optional
        Dask scheduler address

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
        ###########################################
        # All auto-corr scans spliting
        ###########################################
        all_scans = [noise_diode_flux_scan] + valid_target_scans
        scan_sizes = []
        for scan in all_scans:
            scan_sizes.append(get_ms_scan_size(msname, int(scan), only_autocorr=True))
        print("Spliting auto-correlation in different scans ...")
        autocorr_mslist = split_autocorr(
            msname,
            workdir,
            all_scans,
            time_window=time_window,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            dask_addr=dask_addr,
        )
        if len(autocorr_mslist) == 0:
            print("No scans splited.")

            return 1, None, None
        ##########################################
        # Flagging on corrected data
        ##########################################
        fluxcal_fields, fluxcal_scans = get_fluxcals(msname)
        badspw = get_bad_chans(msname)
        bad_ants, bad_ants_str = get_bad_ants(msname, fieldnames=fluxcal_fields)
        print("Flagging auto-correlation measurement sets ...")
        mem_limit = max(scan_sizes)
        if dask_addr is None:
            dask_client, dask_cluster, n_jobs, n_threads, mem_limit, dask_dir = (
                get_dask_client(
                    len(autocorr_mslist),
                    dask_dir=workdir,
                    cpu_frac=cpu_frac,
                    mem_frac=mem_frac,
                    min_mem_per_job=mem_limit,
                )
            )
        else:
            _, _, n_jobs, n_threads, mem_limit, dask_dir = (
                get_dask_client(
                    len(autocorr_mslist),
                    dask_dir=workdir,
                    cpu_frac=cpu_frac,
                    mem_frac=mem_frac,
                    min_mem_per_job=mem_limit,
                    only_cal=True
                )
            )
            dask_client=Client(address=dask_addr)
            os.system(f"rm -rf {dask_dir}")
        wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
        tasks = []
        for autocorr_msname in autocorr_mslist:
            tasks.append(
                delayed(single_ms_flag)(
                    autocorr_msname,
                    badspw=badspw,
                    bad_ants_str=bad_ants_str,
                    datacolumn="data",
                    use_tfcrop=True,
                    use_rflag=False,
                    flagdimension="freq",
                    flag_autocorr=False,
                    n_threads=n_threads,
                    memory_limit=mem_limit,
                )
            )
        futures = dask_client.compute(tasks)
        results = list(dask_client.gather(futures))
        dask_client.close()
        if dask_addr is None:
            dask_cluster.close()
            os.system(f"rm -rf {dask_dir}")
        for autocorr_msname in autocorr_mslist:
            drop_cache(autocorr_msname)
        att_level = {}
        ########################################
        # Calculating per scan level
        ########################################
        print("Calculating noise-diode power difference ...")
        scan_sizes=[get_column_size(ms) for ms in autocorr_mslist]
        mem_limit=max(scan_sizes)
        if dask_addr is None:
            dask_client, dask_cluster, n_jobs, n_threads, mem_limit, dask_dir = (
                get_dask_client(
                    len(valid_target_scans),
                    dask_dir=workdir,
                    cpu_frac=cpu_frac,
                    mem_frac=mem_frac,
                    min_mem_per_job=mem_limit,
                )
            )
        else:
            _, _, n_jobs, n_threads, mem_limit, dask_dir = (
                get_dask_client(
                    len(valid_target_scans),
                    dask_dir=workdir,
                    cpu_frac=cpu_frac,
                    mem_frac=mem_frac,
                    min_mem_per_job=mem_limit,
                    only_cal=True,
                )
            )
            dask_client=Client(address=dask_addr)
            os.system(f"rm -rf {dask_dir}")
        wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
        all_scaling_files = []
        filtered_scans = []
        tasks = []
        for scan in valid_target_scans:
            autocorr_msname = f"{workdir}/autocorr_scan_{scan}.ms"
            if autocorr_msname not in autocorr_mslist:
                pass
            tasks.append(
                delayed(get_power_diff)(
                    f"{workdir}/autocorr_scan_{noise_diode_flux_scan}.ms",
                    f"{workdir}/autocorr_scan_{scan}.ms",
                    noise_on_caltable,
                    noise_off_caltable,
                    n_threads=n_threads,
                    memory_limit=mem_limit,
                )
            )
            filtered_scans.append(scan)
        futures = dask_client.compute(tasks)
        results = list(dask_client.gather(futures))
        dask_client.close()
        if dask_addr is None:
            dask_cluster.close()
            os.system(f"rm -rf {dask_dir}")
        ##########################################
        # Determining frequencies
        ##########################################
        msmd = msmetadata()
        msmd.open(msname)
        freqs = msmd.chanfreqs(0)
        msmd.close()
        msmd.done()
        del msmd
        ########################
        for i in range(len(filtered_scans)):
            att_value = results[i][0]
            att_ant_array = results[i][1]
            if np.nanmean(att_value) < 0:
                att_value *= -1
                att_ant_array *= -1
            scan = filtered_scans[i]
            att_level[scan] = att_value
            filename = (
                workdir
                + "/"
                + os.path.basename(msname).split(".ms")[0]
                + "_attval_scan_"
                + str(scan)
            )
            att_ant_array_percentage_change = (
                att_value[..., None] - att_ant_array
            ) / att_value[..., None]
            flag_ants = []
            for pol in range(2):
                mean_percentage_change = np.nanmean(
                    att_ant_array_percentage_change[pol, ...], axis=0
                )
                std_percentage_change = np.nanstd(
                    att_ant_array_percentage_change[pol, ...], axis=0
                )
                pos = np.where(
                    np.abs(mean_percentage_change) > 3 * std_percentage_change
                )[0]
                if len(pos) > 0:
                    for i in range(len(pos)):
                        if pos[i] not in flag_ants:
                            flag_ants.append(pos[i])
            np.save(
                filename,
                np.array(
                    [scan, freqs, att_value, flag_ants, att_ant_array], dtype="object"
                ),
            )
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
    dask_addr=None,
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
    dask_addr : str, optional
        Dask scheduler address

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
    ncpus = int(psutil.cpu_count() * (1 - cpu_frac))
    limit_threads(n_threads=ncpus)
    from casatasks import split, bandpass

    msname = msname.rstrip("/")
    workdir = workdir.rstrip("/")
    try:
        os.chdir(workdir)
        print("##############################################")
        print("Performing flux calibration using noise-diode.")
        print("##############################################")
        ###################################
        # Determining noise diode cal scans
        ###################################
        fluxcal_fields, fluxcal_scans = get_fluxcals(msname)
        target_scans, cal_scans, f_scans, g_scans, p_scans = get_cal_target_scans(
            msname
        )
        valid_scans = get_valid_scans(msname)
        noise_diode_cal_scan = ""
        for scan in f_scans:
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
        print("Spliting auto-correlation in different scans ...")
        noisecal_ms = workdir + "/noisecal.ms"
        if os.path.exists(noisecal_ms):
            os.system("rm -rf " + noisecal_ms)
        if os.path.exists(noisecal_ms + ".flagversions"):
            os.system("rm -rf " + noisecal_ms + ".flagversions")

        with suppress_casa_output():
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
            flagdimension="freq",
            flag_autocorr=False,
        )

        ##################################
        # Import models
        ##################################
        print("Importing calibrator models ....")
        fluxcal_result = import_fluxcal_models(
            [noisecal_ms],
            fluxcal_fields,
            fluxcal_scans,
            ncpus=ncpus,
            mem_frac=1 - cpu_frac,
        )

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

        with suppress_casa_output():
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
            dask_addr=dask_addr,
        )
        if keep_backup:
            print("Backup directory: " + workdir + "/backup")
            os.makedirs(workdir + "/backup", exist_ok=True)
            os.system(f"rm -rf {workdir}/backup/autocorr_scan_*.ms*")
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
        print("##################")
        print("Total time taken : ", time.time() - start_time)
        print("##################\n")
        return 1, None, None


def main(
    msname,
    workdir,
    caldir,
    keep_backup=False,
    start_remote_log=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    dask_addr=None,
):
    """
    Apply calibration solutions to a measurement set.

    Parameters
    ----------
    msname : str
        Path to the target measurement set (MS) to which calibration solutions will be applied.
    workdir : str
        Directory for storing logs, PID files, and intermediate outputs.
    caldir : str
        Directory containing calibration tables (e.g., G, K, B, D Jones terms).
    keep_backup : bool, optional
        If True, keeps a backup of the original MS before applying calibration. Default is False.
    start_remote_log : bool, optional
        Whether to enable remote logging using credentials stored in the `workdir`. Default is False.
    cpu_frac : float, optional
        Fraction of CPU resources to use. Default is 0.8.
    mem_frac : float, optional
        Fraction of system memory to use. Default is 0.8.
    logfile : str or None, optional
        Path to the log file. If None, disables file logging. Default is None.
    jobid : int, optional
        Identifier for tracking the job and saving PID. Default is 0.
    dask_addr : str, optional
        Dask scheduler address

    Returns
    -------
    int
        Success message
    """
    pid = os.getpid()
    cachedir = get_cachedir()
    save_pid(pid, f"{cachedir}/pids/pids_{jobid}.txt")

    # === Set up workdir ===
    if workdir == "":
        workdir = os.path.dirname(os.path.abspath(msname)) + "/workdir"
    os.makedirs(workdir, exist_ok=True)

    if caldir == "" or not os.path.exists(caldir):
        caldir = f"{workdir}/caltables"
    os.makedirs(caldir, exist_ok=True)

    ############
    # Logger
    ############
    observer = None
    if (
        start_remote_log
        and os.path.exists(f"{workdir}/jobname_password.npy")
        and logfile is not None
    ):
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            observer = init_logger(
                "apply_selfcal", logfile, jobname=jobname, password=password
            )
    if observer == None:
        print("Remote link or jobname is blank. Not transmiting to remote logger.")
    ###########
    try:
        if os.path.exists(msname):
            print("\n###################################")
            print("Starting flux calibration using noise-diode.")
            print("###################################\n")

            msg, att_level, all_scaling_files = run_noise_cal(
                msname,
                workdir,
                keep_backup=keep_backup,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
                dask_addr=dask_addr,
            )

            if msg == 0 and all_scaling_files is not None:
                for att_file in all_scaling_files:
                    os.system(f"rm -rf {caldir}/{os.path.basename(att_file)}")
                    os.system("mv " + att_file + " " + caldir)
        else:
            print("Please provide a valid measurement set.")
            msg = 1
    except Exception:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
        drop_cache(msname)
        drop_cache(workdir)
        drop_cache(caldir)
        clean_shutdown(observer)
    return msg


def cli():
    parser = argparse.ArgumentParser(
        description="Basic calibration using calibrators",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "msname",
        type=str,
        help="Name of measurement set (required positional argument)",
    )
    basic_args.add_argument(
        "--workdir",
        type=str,
        required=True,
        default="",
        help="Working directory (default: auto-created next to MS)",
    )
    basic_args.add_argument(
        "--caldir",
        type=str,
        required=True,
        default="",
        help="Directory for calibration products (default: auto-created in the workdir MS)",
    )

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--keep_backup",
        action="store_true",
        help="Keep backup of measurement set after each round",
    )
    adv_args.add_argument(
        "--start_remote_log", action="store_true", help="Start remote logging"
    )

    # Resource management parameters
    hard_args = parser.add_argument_group(
        "###################\nHardware resource management parameters\n###################"
    )
    hard_args.add_argument(
        "--cpu_frac", type=float, default=0.8, help="CPU fraction to use"
    )
    hard_args.add_argument(
        "--mem_frac", type=float, default=0.8, help="Memory fraction to use"
    )
    hard_args.add_argument("--logfile", type=str, default=None, help="Path to log file")
    hard_args.add_argument(
        "--jobid", type=str, default="0", help="Job ID for logging and tracking"
    )

    # === Show help if no arguments ===
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    msg = main(
        args.msname,
        args.workdir,
        args.caldir,
        keep_backup=args.keep_backup,
        start_remote_log=args.start_remote_log,
        cpu_frac=float(args.cpu_frac),
        mem_frac=float(args.mem_frac),
        logfile=args.logfile,
        jobid=args.jobid,
    )
    return msg


if __name__ == "__main__":
    result = cli()
    print(
        "\n###################\nNoise diode based flux calibration is finished.\n###################\n"
    )
    os._exit(result)
