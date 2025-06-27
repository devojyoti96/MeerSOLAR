import os, time, copy, traceback, gc, argparse
from meersolar.pipeline.basic_func import *
from dask import delayed, compute
from casatasks import casalog

try:
    logfile = casalog.logfile()
    os.system("rm -rf " + logfile)
except:
    pass


def single_mstransform(
    msname="",
    outputms="",
    field="",
    scan="",
    width=1,
    timebin="",
    corr="",
    datacolumn="DATA",
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
    field : str, optional
        Field name
    scan : str, optional
        Scans to split
    width : int, optional
        Number of channels to average
    timebin : str, optional
        Time to average
    corr : str, optional
        Correlation to split
    datacolumn : str, optional
        Data column to split
    n_threads : int, optional
        Number of CPU threads

    Returns
    -------
    str
        Output measurement set name
    """
    limit_threads(n_threads=n_threads)
    from casatasks import mstransform

    if dry_run:
        process = psutil.Process(os.getpid())
        mem = round(process.memory_info().rss / 1024**3, 2)  # in GB
        return mem
    print(
        f"Transforming scan : {scan}, channel averaging: {width}, time averaging: '{timebin}'\n"
    )
    if timebin == "" or timebin == None:
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
        with suppress_casa_output():
            mstransform(
                vis=msname,
                outputvis=outputms,
                field=field,
                scan=scan,
                datacolumn=datacolumn,
                createmms=True,
                correlation=corr,
                timeaverage=timeaverage,
                timebin=timebin,
                chanaverage=chanaverage,
                chanbin=int(width),
                nthreads=2,
                separationaxis="scan",
                numsubms=1,
            )
        gc.collect()
        return outputms
    except Exception as e:
        if os.path.exists(outputms):
            os.system("rm -rf " + outputms)
        return


def partion_ms(
    msname,
    outputms,
    workdir,
    fields="",
    scans="",
    width=1,
    timebin="",
    fullpol=False,
    datacolumn="DATA",
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Perform mstransform of a single scan

    Parameters
    ----------
    msname : str
        Name of the measurement set
    outputms : str
        Output ms name
    workdir : str
        Work directory
    field : str, optional
        Fields to be splited
    scans : str, optional
        Scans to split
    width : int, optional
        Number of channels to average
    timebin : str, optional
        Time to average
    fullpol : bool, optional
        Split full polar
    datacolumn : str, optional
        Data column to split
    ncpu : int, optional
        Number of CPU threads to use

    Returns
    -------
    str
        Output multi-measurement set name
    """
    print("##################")
    print("Paritioning measurement set: " + msname)
    print("##################\n")
    print("Determining valid scan list ....")
    from casatools import msmetadata

    start_time = time.time()
    valid_scans = get_valid_scans(msname, min_scan_time=1)
    msmd = msmetadata()
    msname = os.path.abspath(msname.rstrip("/"))
    msmd.open(msname)
    if scans != "":
        scan_list = scans.split(",")
    else:
        scan_list = msmd.scannumbers()
    scan_list = [int(i) for i in scan_list]
    if fields != "":  # Filtering scans only in the given fields
        scan_list_field = []
        field_list = []
        for i in fields.split(","):
            try:
                i = int(i)
            except:
                pass
            field_list.append(i)
        for field in field_list:
            a = msmd.scansforfield(field).tolist()
            scan_list_field = scan_list_field + a
        backup_scan_list = copy.deepcopy(scan_list)
        for s in scan_list:
            if s not in scan_list_field or s not in valid_scans:
                backup_scan_list.remove(s)
        scan_list = copy.deepcopy(backup_scan_list)
    else:
        backup_scan_list = copy.deepcopy(scan_list)
        for s in scan_list:
            if s not in valid_scans:
                backup_scan_list.remove(s)
        scan_list = copy.deepcopy(backup_scan_list)
    msmd.close()
    if len(scan_list) == 0:
        print("Please provide at-least one valid scan to split.")
        return

    field_list = []
    msmd = msmetadata()
    msmd.open(msname)
    field_names = msmd.fieldnames()
    for scan in scan_list:
        field = msmd.fieldsforscan(scan)[0]
        field_list.append(str(field_names[field]))
    msmd.close()
    msmd.done()
    field = ",".join(field_list)
    if fullpol == False:
        corr = "XX,YY"
    else:
        corr = ""
    ###########################
    # Dask local cluster setup
    ###########################
    scan_sizes = []
    for scan in scan_list:
        scan_sizes.append(get_ms_scan_size(msname, int(scan)))
    total_required_size = round(2*np.nansum(scan_sizes), 2)
    task = delayed(single_mstransform)(dry_run=True)
    mem_limit = run_limited_memory_task(task, dask_dir=workdir)
    os.environ["TMPDIR"] = workdir
    with tmp_with_cache_rel(
        total_required_size, workdir, prefix="tmp_meersolar_partition_"
    ) as temp_workdir:
        dask_client, dask_cluster, n_jobs, n_threads, mem_limit = get_dask_client(
            len(scan_list),
            dask_dir=temp_workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem_per_job=mem_limit / 0.6,
        )
        tasks = []
        for i in range(len(scan_list)):
            scan = scan_list[i]
            outputvis = os.path.join(temp_workdir, f"scan_{scan}.ms")
            task = delayed(single_mstransform)(
                msname,
                outputvis,
                scan=str(scan),
                field="",
                corr=corr,
                width=width,
                timebin=timebin,
                n_threads=n_threads,
            )
            tasks.append(task)
        splited_ms_list = compute(*tasks)
        dask_client.close()
        dask_cluster.close()
        splited_ms_list_copy = copy.deepcopy(splited_ms_list)
        for ms in splited_ms_list:
            if ms == None:
                splited_ms_list_copy.remove(ms)
        splited_ms_list = copy.deepcopy(splited_ms_list_copy)
        outputms = outputms.rstrip("/")
        if os.path.exists(outputms):
            os.system("rm -rf " + outputms)
        if os.path.exists(outputms + ".flagversions"):
            os.system("rm -rf " + outputms + ".flagversions")
        if len(splited_ms_list) == 0:
            print("No splited ms to concat.")
        elif len(splited_ms_list) == 1:
            os.system(f"mv {splited_ms_list[0]} {outputms}")
        else:
            print("Making multi-MS ....")
            from casatasks import virtualconcat

            with suppress_casa_output():
                virtualconcat(vis=splited_ms_list, concatvis=outputms)
    print("##################")
    print("Total time taken : " + str(time.time() - start_time) + "s")
    print("##################\n")
    gc.collect()
    time.sleep(5)
    drop_cache(msname)
    drop_cache(workdir)
    return outputms


def main():
    parser = argparse.ArgumentParser(
        description="Partition measurement set in multi-MS format",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    ## Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "msname",
        type=str,
        help="Name of input measurement set (required positional argument)",
    )
    basic_args.add_argument(
        "--outputms",
        type=str,
        default="multi.ms",
        help="Name of output multi-MS",
    )
    basic_args.add_argument("--workdir", type=str, required=True, help="Work directory")

    ## Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--fields",
        type=str,
        default="",
        help="Comma-separated list of field IDs to split",
    )
    adv_args.add_argument(
        "--scans",
        type=str,
        default="",
        help="Comma-separated list of scans to split",
    )
    adv_args.add_argument(
        "--width",
        type=int,
        default=1,
        help="Number of spectral channels to average",
    )
    adv_args.add_argument(
        "--timebin",
        type=str,
        default="",
        help="Time averaging bin (e.g., '10s', '1min')",
    )
    adv_args.add_argument(
        "--datacolumn",
        type=str,
        default="data",
        help="Datacolumn to split",
    )
    adv_args.add_argument(
        "--split_fullpol",
        dest="fullpol",
        action="store_true",
        default=False,
        help="Split all polarizations (default: False)",
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
    )
    hard_args.add_argument(
        "--mem_frac",
        type=float,
        default=0.8,
        help="Memory fraction to use",
    )
    hard_args.add_argument("--logfile", type=str, default=None, help="Path to log file")
    hard_args.add_argument(
        "--jobid",
        type=str,
        default="0",
        help="Job ID for process tracking",
    )

    # Show help if nothing is passed
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    pid = os.getpid()
    save_pid(pid, datadir + f"/pids/pids_{args.jobid}.txt")

    if args.workdir == "" or not os.path.exists(args.workdir):
        workdir = os.path.dirname(os.path.abspath(args.msname)) + "/workdir"
    else:
        workdir = args.workdir
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
                "partition_cal", logfile, jobname=jobname, password=password
            )

    try:
        if os.path.exists(args.msname):
            outputms = partion_ms(
                args.msname,
                args.outputms,
                args.workdir,
                fields=args.fields,
                scans=args.scans,
                width=args.width,
                timebin=args.timebin,
                fullpol=args.fullpol,
                datacolumn=args.datacolumn,
                cpu_frac=args.cpu_frac,
                mem_frac=args.mem_frac,
            )
            if outputms is None or not os.path.exists(outputms):
                print("Error in partitioning measurement set.")
                msg = 0
            else:
                print("Partitioned multi-MS is created at:", outputms)
                msg = 1
        else:
            print("Please provide a valid measurement set.\n")
            msg = 1
    except Exception:
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
        "\n###################\nMeasurement set partitioning is finished.\n###################\n"
    )
    sys.exit(result)
