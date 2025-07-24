import logging
import dask
import numpy as np
import argparse
import traceback
import copy
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


def partion_ms(
    msname,
    outputms,
    workdir,
    fields="",
    scans="",
    width=1,
    timebin="",
    datacolumn="DATA",
    cpu_frac=0.8,
    mem_frac=0.8,
    dask_addr=None,
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
    datacolumn : str, optional
        Data column to split
    ncpu : int, optional
        Number of CPU threads to use
    dask_addr : str, optional
        Dask scheduler address

    Returns
    -------
    str
        Output multi-measurement set name
    """
    print("##################")
    print("Paritioning measurement set: " + msname)
    print("##################\n")
    print("Determining valid scan list ....")
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
            except BaseException:
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

    ###########################
    # Dask local cluster setup
    ###########################
    scan_sizes = []
    for scan in scan_list:
        scan_sizes.append(get_ms_scan_size(msname, int(scan)))
    total_required_size = round(2 * np.nansum(scan_sizes), 2)
    mem_limit = total_required_size / len(scan_list)
    if dask_addr is None:
        dask_client, dask_cluster, n_jobs, n_threads, mem_limit, dask_dir = (
            get_dask_client(
                len(scan_list),
                dask_dir=workdir,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
                min_mem_per_job=mem_limit,
            )
        )
        dask_cluster.adapt(minimum=0, maximum=n_jobs)
    else:
        _, _, n_jobs, n_threads, mem_limit, dask_dir = get_dask_client(
            len(scan_list),
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem_per_job=mem_limit,
            only_cal=True,
        )
        os.system(f"rm -rf {dask_dir}")
        dask_client = Client(address=dask_addr)
    wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
    tasks = []
    for i in range(len(scan_list)):
        scan = scan_list[i]
        outputvis = f"{workdir}/scan_{scan}.ms"
        task = delayed(single_mstransform)(
            msname=msname,
            outputms=outputvis,
            scan=str(scan),
            field="",
            width=width,
            timebin=timebin,
            n_threads=n_threads,
            numsubms=1,
        )
        tasks.append(task)
    futures = dask_client.compute(tasks)
    splited_ms_list = list(dask_client.gather(futures))
    dask_client.close()
    if dask_addr is None:
        dask_cluster.close()
        os.system(f"rm -rf {dask_dir}")
    splited_ms_list_copy = copy.deepcopy(splited_ms_list)
    for ms in splited_ms_list:
        if ms is None:
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
    return outputms


def main(
    msname,
    outputms="multi.ms",
    workdir="",
    fields="",
    scans="",
    width=1,
    timebin="",
    datacolumn="data",
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid="0",
    start_remote_log=False,
    dask_addr=None,
):
    """
    Partition a measurement set using field, scan, channel, and time selection in subms.

    Parameters
    ----------
    msname : str
        Path to the input measurement set (MS).
    outputms : str, optional
        Name of the output measurement set. Default is "multi.ms".
    workdir : str, optional
        Directory for logs and intermediate files. If empty, defaults to `<msname>/workdir`.
    fields : str, optional
        Field selection string, in CASA syntax (e.g., "0~2"). Default is "" (all fields).
    scans : str, optional
        Scan selection string, comma-separated (e.g., "1,3,5"). Default is "" (all scans).
    width : int, optional
        Number of channels to average together spectrally. Default is 1 (no averaging).
    timebin : str, optional
        Time averaging interval (e.g., "10s", "1min"). Empty string disables time averaging. Default is "".
    datacolumn : str, optional
        Name of the data column to operate on (e.g., "data", "corrected"). Default is "data".
    cpu_frac : float, optional
        Fraction of available CPUs to use per task. Default is 0.8.
    mem_frac : float, optional
        Fraction of available memory to use per task. Default is 0.8.
    logfile : str or None, optional
        Path to log file. If None, logging to file is disabled. Default is None.
    jobid : str, optional
        Unique job identifier used for PID tracking. Default is "0".
    start_remote_log : bool, optional
        If True, enables remote logging using credentials in `workdir`. Default is False.
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
                "partition_cal", logfile, jobname=jobname, password=password
            )
    if observer == None:
        print("Remote link or jobname is blank. Not transmiting to remote logger.")
    ###########

    try:
        if os.path.exists(msname):
            outputms = partion_ms(
                msname,
                outputms,
                workdir,
                fields=fields,
                scans=scans,
                width=width,
                timebin=timebin,
                datacolumn=datacolumn,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
                dask_addr=dask_addr,
            )
            if outputms is None or not os.path.exists(outputms):
                print("Error in partitioning measurement set.")
                msg = 1
            else:
                print("Partitioned multi-MS is created at:", outputms)
                msg = 0
        else:
            print("Please provide a valid measurement set.\n")
            msg = 1
    except Exception:
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
        description="Partition measurement set in multi-MS format",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    # Essential parameters
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

    # Advanced parameters
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

    msg = main(
        msname=args.msname,
        outputms=args.outputms,
        workdir=args.workdir,
        fields=args.fields,
        scans=args.scans,
        width=args.width,
        timebin=args.timebin,
        datacolumn=args.datacolumn,
        cpu_frac=float(args.cpu_frac),
        mem_frac=float(args.mem_frac),
        logfile=args.logfile,
        jobid=args.jobid,
        start_remote_log=args.start_remote_log,
    )
    return msg


if __name__ == "__main__":
    result = cli()
    print(
        "\n###################\nMeasurement set partitioning is finished.\n###################\n"
    )
    sys.exit(result)
