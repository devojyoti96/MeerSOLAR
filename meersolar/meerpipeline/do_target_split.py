import logging
import psutil
import dask
import numpy as np
import argparse
import traceback
import time
import sys
import os
from casatasks import casalog
from casatools import msmetadata
from dask import delayed
from dask.distributed import Client
from meersolar.utils import *

logging.getLogger("distributed").setLevel(logging.WARNING)


try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except BaseException:
    pass

datadir = get_datadir()


def chanlist_to_str(lst):
    lst = sorted(lst)
    ranges = []
    start = lst[0]
    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1] + 1:
            if lst[i - 1] > start:
                ranges.append(f"{start}~{lst[i - 1]}")
            elif lst[i - 1] == start:
                ranges.append(f"{start}")
            start = lst[i]
    if lst[-1] > start:
        ranges.append(f"{start}~{lst[-1]}")
    elif lst[-1] == start:
        ranges.append(f"{start}")
    return ";".join(ranges)


def split_target_scans(
    msname,
    workdir,
    timeres,
    freqres,
    datacolumn,
    spw="",
    spectral_chunk=-1,
    n_spectral_chunk=-1,
    scans=[],
    prefix="targets",
    time_interval=-1,
    time_window=-1,
    quack_timestamps=-1,
    merge_spws=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    max_cpu_frac=0.8,
    max_mem_frac=0.8,
    dask_addr=None,
):
    """
    Split target scans

    Parameters
    ----------
    msname : str
        Measurement set
    workdir : str
        Work directory
    timeres : float
        Time resolution in seconds
    freqres : float
        Frequency resolution in MHz
    datacolumn : str
        Data column to split
    spw : str, optional
        Spectral window
    spectral_chunk : float, optional
        Spectral chunk in MHz
    n_spectral_chunk : int, optional
        Number of spectral chunks to split from the beginning
    scans : list
        Scan list to split
    prefix : str, optional
        Splited ms prefix
    time_interval : float
        Time interval in seconds
    time_window : float
        Time window in seconds
    quack_timestamps : int, optional
        Number of timestamps ignored at the start and end of each scan
    merge_spws : bool, optional
        Merge spectral window ranges
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    max_cpu_frac : float, optional
        Maximum CPU fraction to use
    max_mem_frac : float, optional
        Maximum memory fraction to use
    dask_addr : str, optional
        Dask scheduler address

    Returns
    -------
    list
        Splited ms list
    """
    start_time = time.time()
    try:
        os.chdir(workdir)
        print(f"Spliting ms : {msname}")
        target_scans, cal_scans, f_scans, g_scans, p_scans = get_cal_target_scans(
            msname
        )
        valid_scans = get_valid_scans(msname)
        filtered_scan_list = []
        for scan in target_scans:
            if scan in valid_scans:
                if len(scans) == 0 or (len(scans) > 0 and scan in scans):
                    filtered_scan_list.append(scan)
        filtered_scan_list = sorted(filtered_scan_list)

        #######################################
        # Extracting time frequency information
        #######################################
        msmd = msmetadata()
        msmd.open(msname)
        chanres = msmd.chanres(0, unit="MHz")[0]
        freqs = msmd.chanfreqs(0, unit="MHz")
        bw = max(freqs) - min(freqs)
        nchan = msmd.nchan(0)
        msmd.close()
        if freqres > 0:  # Image resolution is in MHz
            chanwidth = int(freqres / chanres)
            if chanwidth < 1:
                chanwidth = 1
        else:
            chanwidth = 1
        if timeres > 0:  # Image resolution is in seconds
            timebin = str(timeres) + "s"
        else:
            timebin = ""

        #############################
        # Making spectral chunks
        #############################
        bad_spws = get_bad_chans(msname).split("0:")[-1].split(";")
        # Derive good spectral windows by finding gaps between bad SPWs
        good_spws_list = []
        for i in range(len(bad_spws) - 1):
            start_chan = int(bad_spws[i].split("~")[-1]) + 1
            end_chan = int(bad_spws[i + 1].split("~")[0]) - 1
            good_spws_list.append(f"{start_chan}~{end_chan}")

        # If no gaps found, fall back to full good channels
        if good_spws_list:
            good_spws = "0:" + ";".join(good_spws_list)
        else:
            good_spws = get_good_chans(msname)

        # Intersect with user-specified SPW if provided
        spw_suffix = good_spws.split("0:")[-1].split(";")
        if spw:
            common_spws = get_common_spw(good_spws, spw)
            if common_spws != "":
                good_spws = common_spws.split("0:")[-1].split(";")
            else:
                good_spws = spw_suffix
        else:
            good_spws = spw_suffix
        if len(good_spws) == 0:
            print("No good spectral window is found.")
            return 1, []

        #############################
        chanlist = []
        if spectral_chunk > 0:
            if spectral_chunk > bw:
                print(
                    f"Given spectral chunk: {spectral_chunk} is more than total bandwidth: {bw} MHz."
                )
                spectral_chunk = bw
            nchan_per_chunk = max(1, int(spectral_chunk / chanres))
            good_channels = []
            for good_spw in good_spws:
                start_chan = int(good_spw.split("~")[0])
                end_chan = int(good_spw.split("~")[-1])
                for s in range(start_chan, end_chan):
                    good_channels.append(s)
            channel_chunks = split_into_chunks(good_channels, nchan_per_chunk)
            for chunk in channel_chunks:
                chan_str = chanlist_to_str(chunk)
                if chan_str not in chanlist:
                    chanlist.append(chan_str)
            if n_spectral_chunk > 0:
                indices = np.linspace(
                    0, len(chanlist) - 1, num=n_spectral_chunk, dtype=int
                )
                chanlist = [chanlist[i] for i in indices]
        else:
            chan_range = ""
            for good_spw in good_spws:
                s = int(good_spw.split("~")[0])
                e = int(good_spw.split("~")[-1])
                chan_range += f"{s}~{e};"
            chan_range = chan_range[:-1]
            if chan_range not in chanlist:
                chanlist.append(chan_range)

        if merge_spws:
            temp_spw = ";".join(chanlist)
            chanlist = [temp_spw]

        print(f"Spliting channel blocks : {chanlist}")

        ##################################
        # Parallel spliting
        ##################################
        if len(chanlist) > 0:
            total_chunks = len(chanlist) * len(filtered_scan_list)
        else:
            total_chunks = len(filtered_scan_list)

        #############################################
        # Memory limit
        #############################################
        mem_limit = single_mstransform(dry_run=True)
        #######################
        _, _, max_n_jobs, n_threads, mem_limit, dask_dir = (
            get_dask_client(
                total_chunks,
                dask_dir=workdir,
                cpu_frac=max_cpu_frac,
                mem_frac=max_mem_frac,
                min_mem_per_job=mem_limit,
                only_cal=True,
            )
        )
        os.system(f"rm -rf {dask_dir}")
        tasks = []
        splited_ms_list = []
        for scan in filtered_scan_list:
            timerange_list = get_timeranges_for_scan(
                msname,
                scan,
                time_interval,
                time_window,
                quack_timestamps=quack_timestamps,
            )
            timerange = ",".join(timerange_list)
            for chanrange in chanlist:
                chanrange_str = (
                    chanrange.split(";")[0].split("~")[0]
                    + "~"
                    + chanrange.split(";")[-1].split("~")[-1]
                )
                outputvis = f"{workdir}/{prefix}_scan_{scan}_spw_{chanrange_str}.ms"
                if os.path.exists(f"{outputvis}/.splited"):
                    print(f"{outputvis} is already splited successfully.")
                    splited_ms_list.append(outputvis)
                else:
                    task = delayed(single_mstransform)(
                        msname=msname,
                        outputms=outputvis,
                        field="",
                        scan=scan,
                        width=chanwidth,
                        timebin=timebin,
                        datacolumn="DATA",
                        spw="0:" + chanrange,
                        corr="",
                        timerange=timerange,
                        n_threads=n_threads,
                        dry_run=False,
                    )
                    tasks.append(task)
        #####################################
        # Adaptive dask client
        #####################################
        if cpu_frac == max_cpu_frac and mem_frac == max_mem_frac:
            total_chunks = len(tasks)
            if total_chunks > 0:
                if dask_addr is None:
                    dask_client, dask_cluster, n_jobs, n_threads, mem_limit, dask_dir = (
                        get_dask_client(
                            total_chunks,
                            dask_dir=workdir,
                            cpu_frac=cpu_frac,
                            mem_frac=mem_frac,
                            min_mem_per_job=mem_limit,
                        )
                    )
                else:
                    _, _, n_jobs, n_threads, mem_limit, dask_dir = (
                        get_dask_client(
                            total_chunks,
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
                futures = dask_client.compute(tasks)
                results = list(dask_client.gather(futures))
                dask_client.close()
                if dask_addr is None:
                    dask_cluster.close()
                    os.system(f"rm -rf {dask_dir}")
                for r in results:
                    splited_ms_list.append(r)
        else:
            while True:
                total_chunks = len(tasks)
                if total_chunks == 0:
                    break
                else:
                    if dask_addr is None:
                        (
                            dask_client,
                            dask_cluster,
                            n_jobs,
                            n_threads,
                            mem_limit,
                            dask_dir,
                        ) = get_dask_client(
                            total_chunks,
                            dask_dir=workdir,
                            cpu_frac=cpu_frac,
                            mem_frac=mem_frac,
                            min_mem_per_job=mem_limit,
                        )
                    else:
                        (
                            _,
                            _,
                            n_jobs,
                            n_threads,
                            mem_limit,
                            dask_dir,
                        ) = get_dask_client(
                            total_chunks,
                            dask_dir=workdir,
                            cpu_frac=cpu_frac,
                            mem_frac=mem_frac,
                            min_mem_per_job=mem_limit,
                            only_cal=True,
                        )
                        dask_client=Client(address=dask_addr)
                        os.system(f"rm -rf {dask_dir}")
                    wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
                    chunk_tasks = tasks[0 : min(n_jobs, max_n_jobs)]
                    for ctask in chunk_tasks:
                        tasks.remove(ctask)
                    futures = dask_client.compute(tasks)
                    results = list(dask_client.gather(futures))
                    dask_client.close()
                    if dask_addr is None:
                        dask_cluster.close()
                        os.system(f"rm -rf {dask_dir}")
                    for r in results:
                        splited_ms_list.append(r)
                    n_current_process = (
                        get_nprocess_solarpipe(workdir) - 1
                    )  # One is subtracted for the current process
                    if len(tasks) == 0:
                        break
                    elif n_current_process == 0:
                        available_cpu_frac = round(
                            (100 - psutil.cpu_percent(interval=1)) / 100.0, 2
                        )
                        available_mem_frac = round(
                            psutil.virtual_memory().available
                            / psutil.virtual_memory().total,
                            2,
                        )
                        cpu_frac = min(max_cpu_frac, max(cpu_frac, available_cpu_frac))
                        mem_frac = min(max_mem_frac, max(mem_frac, available_mem_frac))
                        print(
                            f"Updated CPU fraction: {cpu_frac}, memory fraction: {mem_frac}."
                        )
        print("##################")
        print("Spliting of target scans are done successfully.")
        print("Total time taken : ", time.time() - start_time)
        print("##################\n")
        return 0, splited_ms_list
    except Exception as e:
        traceback.print_exc()
        print("##################")
        print("Spliting of target scans are unsuccessful.")
        print("Total time taken : ", time.time() - start_time)
        print("##################\n")
        return 1, []


def main(
    msname,
    workdir="",
    datacolumn="data",
    spw="",
    scans="",
    time_window=-1,
    time_interval=-1,
    quack_timestamps=-1,
    spectral_chunk=-1,
    n_spectral_chunk=-1,
    freqres=-1,
    timeres=-1,
    prefix="targets",
    merge_spws=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    max_cpu_frac=0.8,
    max_mem_frac=0.8,
    logfile=None,
    jobid=0,
    start_remote_log=False,
    dask_addr=None,
):
    """
    Split target scans from a measurement set into smaller chunks for parallel processing.

    Parameters
    ----------
    msname : str
        Path to the input measurement set (MS).
    workdir : str, optional
        Working directory for intermediate and output products. If empty, defaults to `<msname>/workdir`.
    datacolumn : str, optional
        Column of the MS to use for splitting (e.g., "DATA", "CORRECTED"). Default is "data".
    spw : str, optional
        Spectral windows to include, in CASA syntax (e.g., "0~3"). Default is "" (all).
    scans : str, optional
        Scan numbers to include, comma-separated (e.g., "1,3,5"). Default is "" (all).
    time_window : float, optional
        Time window in seconds to group scans. Set -1 to disable. Default is -1.
    time_interval : float, optional
        Integration time interval in seconds for time averaging. Set -1 to disable. Default is -1.
    quack_timestamps : float, optional
        Time in seconds to flag at the beginning of each scan ("quack"). -1 to disable. Default is -1.
    spectral_chunk : float, optional
        Width of spectral chunks in MHz. If set > 0, overrides `n_spectral_chunk`. Default is -1.
    n_spectral_chunk : int, optional
        Number of spectral chunks to split each SPW into. Ignored if `spectral_chunk` > 0. Default is -1.
    freqres : float, optional
        Frequency resolution in MHz for spectral averaging. Set -1 to disable. Default is -1.
    timeres : float, optional
        Time resolution in seconds for time averaging. Set -1 to disable. Default is -1.
    prefix : str, optional
        Prefix for the output split MS files. Default is "targets".
    merge_spws : bool, optional
        If True, merge all SPWs before splitting. Default is False.
    cpu_frac : float, optional
        Fraction of available CPUs to allocate per task. Default is 0.8.
    mem_frac : float, optional
        Fraction of available memory to allocate per task. Default is 0.8.
    max_cpu_frac : float, optional
        Maximum total CPU usage across all tasks. Default is 0.8.
    max_mem_frac : float, optional
        Maximum total memory usage across all tasks. Default is 0.8.
    logfile : str or None, optional
        Path to log file. If None, logging to file is disabled. Default is None.
    jobid : int, optional
        Job identifier for tracking and PID storage. Default is 0.
    start_remote_log : bool, optional
        If True, enables remote logging using credentials stored in workdir. Default is False.
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

    if workdir == "":
        workdir = os.path.dirname(os.path.abspath(msname)) + "/workdir"
    os.makedirs(workdir, exist_ok=True)

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
                "do_target_split", logfile, jobname=jobname, password=password
            )
    if observer == None:
        print("Remote link or jobname is blank. Not transmiting to remote logger.")
    ###########

    try:
        if msname and os.path.exists(msname):
            print("\n###################################")
            print("Start spliting target scans.")
            print("###################################\n")
            scans = [int(i) for i in scans.split(",")] if scans else []
            msg, final_target_mslist = split_target_scans(
                msname,
                workdir,
                float(timeres),
                float(freqres),
                datacolumn,
                spw=str(spw),
                time_window=float(time_window),
                time_interval=float(time_interval),
                quack_timestamps=int(quack_timestamps),
                scans=scans,
                n_spectral_chunk=int(n_spectral_chunk),
                prefix=prefix,
                merge_spws=merge_spws,
                spectral_chunk=float(spectral_chunk),
                cpu_frac=float(cpu_frac),
                mem_frac=float(mem_frac),
                max_cpu_frac=float(max_cpu_frac),
                max_mem_frac=float(max_mem_frac),
                dask_addr=dask_addr,
            )
        else:
            print("Please provide correct measurement set.\n")
            msg = 1
    except Exception as e:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
        drop_cache(msname)
        drop_cache(workdir)
        clean_shutdown(observer)
    return msg


def cli():
    parser = argparse.ArgumentParser(
        description="Split target scans", formatter_class=SmartDefaultsHelpFormatter
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
        default="",
        help="Name of work directory",
    )

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--datacolumn",
        type=str,
        default="data",
        help="Data column to split",
    )
    adv_args.add_argument(
        "--spw", type=str, default="", help="Spectral window to split"
    )
    adv_args.add_argument(
        "--scans",
        type=str,
        default="",
        help="Target scan list (default: all)",
        metavar="SCANS (Comma seperated)",
    )
    adv_args.add_argument(
        "--time_window",
        type=float,
        default=-1,
        help="Time window in seconds",
        metavar="Float",
    )
    adv_args.add_argument(
        "--time_interval",
        type=float,
        default=-1,
        help="Time interval in seconds",
        metavar="Float",
    )
    adv_args.add_argument(
        "--quack_timestamps",
        type=int,
        default=-1,
        help="Time stamps to ignore at the start and end of the each scan",
        metavar="Integer",
    )
    adv_args.add_argument(
        "--spectral_chunk",
        type=float,
        default=-1,
        help="Spectral chunk in MHz",
        metavar="Float",
    )
    adv_args.add_argument(
        "--n_spectral_chunk",
        type=int,
        default=-1,
        help="Numbers of spectral chunks to split",
        metavar="Integer",
    )
    adv_args.add_argument(
        "--freqres",
        type=float,
        default=-1,
        help="Frequency to average in MHz",
        metavar="Float",
    )
    adv_args.add_argument(
        "--timeres",
        type=float,
        default=-1,
        help="Time bin to average in seconds",
        metavar="Float",
    )
    adv_args.add_argument(
        "--prefix",
        type=str,
        default="targets",
        help="Splited ms prefix name",
    )
    adv_args.add_argument(
        "--merge_spws", action="store_true", help="Merge spectral windows"
    )
    adv_args.add_argument(
        "--start_remote_log", action="store_true", help="Start remote logging"
    )

    # Resource management parameters
    hard_args = parser.add_argument_group(
        "###################\nHardware resource management parameters\n###################"
    )
    hard_args.add_argument(
        "--cpu_frac",
        type=float,
        default=0.8,
        help="CPU fraction to use",
        metavar="Float",
    )
    hard_args.add_argument(
        "--mem_frac",
        type=float,
        default=0.8,
        help="Memory fraction to use",
        metavar="Float",
    )
    hard_args.add_argument(
        "--max_cpu_frac",
        type=float,
        default=0.8,
        help="Maximum CPU fraction to use",
        metavar="Float",
    )
    hard_args.add_argument(
        "--max_mem_frac",
        type=float,
        default=0.8,
        help="Maximum memory fraction to use",
        metavar="Float",
    )
    hard_args.add_argument("--logfile", type=str, default=None, help="Log file")
    hard_args.add_argument("--jobid", type=int, default=0, help="Job ID")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    msg = main(
        msname=args.msname,
        workdir=args.workdir,
        datacolumn=args.datacolumn,
        spw=args.spw,
        scans=args.scans,
        time_window=args.time_window,
        time_interval=args.time_interval,
        quack_timestamps=args.quack_timestamps,
        spectral_chunk=args.spectral_chunk,
        n_spectral_chunk=args.n_spectral_chunk,
        freqres=args.freqres,
        timeres=args.timeres,
        prefix=args.prefix,
        merge_spws=args.merge_spws,
        cpu_frac=args.cpu_frac,
        mem_frac=args.mem_frac,
        max_cpu_frac=args.max_cpu_frac,
        max_mem_frac=args.max_mem_frac,
        logfile=args.logfile,
        jobid=args.jobid,
        start_remote_log=args.start_remote_log,
    )
    return msg


if __name__ == "__main__":
    result = cli()
    print(
        "\n###################\nSpliting target scans are done.\n###################\n"
    )
    os._exit(result)
