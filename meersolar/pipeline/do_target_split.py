import numpy as np, os, time, traceback, gc, argparse
from meersolar.pipeline.basic_func import *
from casatasks import casalog
from dask import delayed, compute

try:
    logfile = casalog.logfile()
    os.system("rm -rf " + logfile)
except:
    pass


def chanlist_to_str(lst):
    lst = sorted(lst)
    ranges = []
    start = lst[0]

    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1] + 1:
            ranges.append(f"{start}~{lst[i - 1]}")
            start = lst[i]
    ranges.append(f"{start}~{lst[-1]}")
    return ";".join(ranges)


def split_scan(
    msname="",
    outputvis="",
    scan="",
    width="",
    timebin="",
    datacolumn="",
    spw="",
    corr="",
    timerange="",
    n_threads=-1,
    dry_run=False,
):
    """
    Split a single target scan

    Parameters
    ----------
    msname : str
        Measurement set
    outputvis : str
        Output measurement set
    scan : int
        Scan number
    width : int
        Channel width
    timebin : str
        Timebin width
    datacolumn : str
        Datacolumn to split
    spw : str, optional
        Spectral window to split
    corr : str, optional
        Correlation to split
    timerange : str, optional
        Time range to split
    n_threads : int, optional
        Number of OpenMP threads

    Returns
    -------
    str
        Splited measurement set
    """
    limit_threads(n_threads=n_threads)
    from casatasks import split, flagdata, initweights
    from casatools import msmetadata

    msmd = msmetadata()
    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    msmd.open(msname)
    fields = msmd.fieldsforscan(int(scan))
    msmd.close()
    del msmd
    fields_str = ""
    for f in fields:
        fields_str += str(f) + ","
    fields_str = fields_str[:-1]
    if os.path.exists(f"{outputvis}/.splited") == False:
        if os.path.exists(outputvis):
            os.system("rm -rf " + outputvis)
        if os.path.exists(outputvis + ".flagversions"):
            os.system("rm -rf " + outputvis + ".flagversions")
        print(f"Spliting scan : {scan} of ms: {msname}\n")
        print(
            f"split(vis='{msname}',outputvis='{outputvis}',field='{fields_str}',scan='{scan}',spw='{spw}',correlation='{corr}',timerange='{timerange}',width={width},timebin='{timebin}',datacolumn='{datacolumn}')\n"
        )
        with suppress_casa_output():
            split(
                vis=msname,
                outputvis=outputvis,
                field=fields_str,
                correlation=corr,
                scan=scan,
                spw=spw,
                timerange=timerange,
                width=width,
                timebin=timebin,
                datacolumn=datacolumn,
            )
        ##########################################
        # Initiate proper weighting
        ##########################################
        print("Initiating weights ....")
        with suppress_casa_output():
            initweights(vis=outputvis, wtmode="ones", dowtsp=True)
            flagdata(
                vis=outputvis,
                mode="clip",
                clipzeros=True,
                datacolumn="data",
                flagbackup=False,
            )
        os.system(f"touch {outputvis}/.splited")
    return outputvis


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
    fullpol=False,
    time_interval=-1,
    time_window=-1,
    quack_timestamps=-1,
    merge_spws=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    max_cpu_frac=0.8,
    max_mem_frac=0.8,
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
    fullpol : bool, optional
        Full polar split
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
        from casatools import msmetadata

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
        if fullpol == False:
            corr = "XX,YY"
        else:
            corr = ""

        #############################
        # Making spectral chunks
        #############################
        bad_spws = get_bad_chans(msname).split("0:")[-1].split(";")
        good_spws = []
        for i in range(len(bad_spws) - 1):
            start_chan = int(bad_spws[i].split("~")[-1]) + 1
            end_chan = int(bad_spws[i + 1].split("~")[0]) - 1
            good_spws.append(f"{start_chan}~{end_chan}")
        if spw != "":
            good_spws = "0:" + ";".join(good_spws)
            common_spws = get_common_spw(good_spws, spw)
            good_spws = common_spws.split("0:")[-1].split(";")
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
        task = delayed(split_scan)(dry_run=True)
        mem_limit = run_limited_memory_task(task, dask_dir=workdir)
        #######################
        dask_client, dask_cluster, max_n_jobs, n_threads, mem_limit = get_dask_client(
            total_chunks,
            dask_dir=workdir,
            cpu_frac=max_cpu_frac,
            mem_frac=max_mem_frac,
            min_mem_per_job=mem_limit / 0.6,
            only_cal=True,
        )
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
                    task = delayed(split_scan)(
                        msname,
                        outputvis,
                        scan,
                        chanwidth,
                        timebin,
                        datacolumn,
                        corr=corr,
                        timerange=timerange,
                        spw="0:" + chanrange,
                        n_threads=n_threads,
                    )
                    tasks.append(task)
        #####################################
        # Adaptive dask client
        #####################################
        if cpu_frac == max_cpu_frac and mem_frac == max_mem_frac:
            total_chunks = len(tasks)
            if total_chunks > 0:
                dask_client, dask_cluster, n_jobs, n_threads, mem_limit = (
                    get_dask_client(
                        total_chunks,
                        dask_dir=workdir,
                        cpu_frac=cpu_frac,
                        mem_frac=mem_frac,
                        min_mem_per_job=mem_limit / 0.6,
                    )
                )
                results = compute(*tasks)
                dask_client.close()
                dask_cluster.close()
                for r in results:
                    splited_ms_list.append(r)
        else:
            while True:
                total_chunks = len(tasks)
                if total_chunks == 0:
                    break
                else:
                    dask_client, dask_cluster, n_jobs, n_threads, mem_limit = (
                        get_dask_client(
                            total_chunks,
                            dask_dir=workdir,
                            cpu_frac=cpu_frac,
                            mem_frac=mem_frac,
                            min_mem_per_job=mem_limit / 0.6,
                        )
                    )
                    chunk_tasks = tasks[0 : min(n_jobs, max_n_jobs)]
                    for ctask in chunk_tasks:
                        tasks.remove(ctask)
                    results = compute(*chunk_tasks)
                    dask_client.close()
                    dask_cluster.close()
                    for r in results:
                        splited_ms_list.append(r)
                    n_current_process = (
                        get_nprocess_meersolar(workdir) - 1
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
    finally:
        time.sleep(5)
        drop_cache(msname)
        drop_cache(workdir)


def main():
    parser = argparse.ArgumentParser(
        description="Split target scans", formatter_class=SmartDefaultsHelpFormatter
    )

    ## Essential parameters
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

    ## Advanced parameters
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
        "--split_fullpol", action="store_true", help="Split full polar data"
    )
    adv_args.add_argument(
        "--merge_spws", action="store_true", help="Merge spectral windows"
    )

    ## Resource management parameters
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
    hard_args.add_argument(
        "--logfile", type=str, default=None, help="Log file"
    )
    hard_args.add_argument(
        "--jobid", type=int, default=0, help="Job ID"
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    pid = os.getpid()
    save_pid(pid, datadir + f"/pids/pids_{args.jobid}.txt")

    workdir = (
        args.workdir
        if args.workdir and os.path.exists(args.workdir)
        else os.path.dirname(os.path.abspath(args.msname)) + "/workdir"
    )
    os.makedirs(workdir,exist_ok=True)

    logfile = args.logfile
    observer = None
    if os.path.exists(f"{workdir}/jobname_password.npy") and logfile is not None:
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            observer = init_logger(
                "do_target_split", logfile, jobname=jobname, password=password
            )

    try:
        if args.msname and os.path.exists(args.msname):
            print("\n###################################")
            print("Start spliting target scans.")
            print("###################################\n")
            scans = [int(i) for i in args.scans.split(",")] if args.scans else []
            msg, final_target_mslist = split_target_scans(
                args.msname,
                workdir,
                float(args.timeres),
                float(args.freqres),
                args.datacolumn,
                spw=str(args.spw),
                time_window=float(args.time_window),
                time_interval=float(args.time_interval),
                quack_timestamps=int(args.quack_timestamps),
                scans=scans,
                fullpol=args.split_fullpol,
                n_spectral_chunk=int(args.n_spectral_chunk),
                prefix=args.prefix,
                merge_spws=args.merge_spws,
                spectral_chunk=float(args.spectral_chunk),
                cpu_frac=float(args.cpu_frac),
                mem_frac=float(args.mem_frac),
                max_cpu_frac=float(args.max_cpu_frac),
                max_mem_frac=float(args.max_mem_frac),
            )
        else:
            print("Please provide correct measurement set.\n")
            msg = 1
    except Exception as e:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
        drop_cache(args.msname)
        drop_cache(args.workdir)
        clean_shutdown(observer)

    return msg


if __name__ == "__main__":
    result = main()
    print(
        "\n###################\nSpliting target scans are done.\n###################\n"
    )
    os._exit(result)
