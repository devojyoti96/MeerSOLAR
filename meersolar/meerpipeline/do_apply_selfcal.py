import logging
import dask
import numpy as np
import argparse
import traceback
import time
import glob
import sys
import os
from casatasks import casalog
from casatools import msmetadata
from dask import delayed
from dask.distributed import Client
from meersolar.utils import *
from meersolar.meerpipeline.do_apply_basiccal import applysol

logging.getLogger("distributed").setLevel(logging.WARNING)

try:
    casalogfile = casalog.logfile()
    os.system("rm -rf " + casalogfile)
except BaseException:
    pass

datadir = get_datadir()


def run_all_applysol(
    mslist,
    workdir,
    caldir,
    overwrite_datacolumn=False,
    applymode="calonly",
    force_apply=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    dask_addr=None,
):
    """
    Apply self-calibrator solutions on all target scans

    Parameters
    ----------
    mslist : str
        Measurement set list
    workdir : str
        Working directory
    caldir : str
        Calibration directory
    overwrite_datacolumn : bool, optional
        Overwrite data column or not
    applymode : str, optional
        Apply mode
    force_apply : bool, optional
        Force to apply solutions even already applied
    cpu_frac : float, optional
        CPU fraction to use
    mem_frac : float, optional
        Memory fraction to use
    dask_addr : str, optional
        Dask scheduler address

    Returns
    --------
    list
        Calibrated target scans
    """
    start_time = time.time()
    try:
        os.chdir(workdir)
        mslist = np.unique(mslist).tolist()
        parang = False
        selfcal_tables = glob.glob(caldir + "/selfcal_scan*.gcal")
        print(f"Selfcal caltables: {selfcal_tables}\n")
        if len(selfcal_tables) == 0:
            print(f"No self-cal caltable is present in {caldir}.")
            return 1
        selfcal_tables_scans = np.array(
            [
                int(os.path.basename(i).split(".gcal")[0].split("scan_")[-1])
                for i in selfcal_tables
            ]
        )
        ####################################
        # Filtering any corrupted ms
        #####################################
        filtered_mslist = []  # Filtering in case any ms is corrupted
        for ms in mslist:
            checkcol = check_datacolumn_valid(ms)
            if checkcol:
                filtered_mslist.append(ms)
            else:
                print(f"Issue in : {ms}")
                os.system("rm -rf {ms}")
        mslist = filtered_mslist
        if len(mslist) == 0:
            print("No valid measurement set.")
            print(f"Total time taken: {round(time.time()-start_time,2)}s")
            return 1

        ####################################
        # Applycal jobs
        ####################################
        print(f"Total ms list: {len(mslist)}")
        ms_size_list = [get_column_size(ms) for ms in mslist]
        mem_limit = max(ms_size_list)
        if dask_addr is None:
            dask_client, dask_cluster, n_jobs, n_threads, mem_limit, dask_dir = (
                get_dask_client(
                    len(mslist),
                    dask_dir=workdir,
                    cpu_frac=cpu_frac,
                    mem_frac=mem_frac,
                    min_mem_per_job=mem_limit,
                )
            )
        else:
            _, _, n_jobs, n_threads, mem_limit, dask_dir = (
                get_dask_client(
                    len(mslist),
                    dask_dir=workdir,
                    cpu_frac=cpu_frac,
                    mem_frac=mem_frac,
                    min_mem_per_job=mem_limit,
                    only_cal=True,
                )
            )
            os.system(f"rm -rf {dask_dir}")
            dask_client = Client(address=dask_addr)
        wait_for_dask_workers(dask_client,min_worker=1,timeout=60)
        tasks = []
        msmd = msmetadata()
        for ms in mslist:
            msmd.open(ms)
            ms_scan = msmd.scannumbers()[0]
            msmd.close()
            if ms_scan not in selfcal_tables_scans:
                print(
                    f"Target scan: {ms_scan}. Corresponding self-calibration table is not present. Using the closet one."
                )
            caltable_pos = np.argmin(np.abs(selfcal_tables_scans - ms_scan))
            gaintable = [selfcal_tables[caltable_pos]]
            tasks.append(
                delayed(applysol)(
                    msname=ms,
                    gaintable=gaintable,
                    overwrite_datacolumn=overwrite_datacolumn,
                    applymode=applymode,
                    interp=["linear,linearflag"],
                    n_threads=n_threads,
                    parang=parang,
                    memory_limit=mem_limit,
                    force_apply=force_apply,
                    soltype="selfcal",
                )
            )
        futures = dask_client.compute(tasks)
        results = list(dask_client.gather(futures))
        dask_client.close()
        if dask_addr is None:
            dask_cluster.close()
            os.system(f"rm -rf {dask_dir}")
        if np.nansum(results) == 0:
            print("##################")
            print(
                "Applying self-calibration solutions for target scans are done successfully."
            )
            print("Total time taken : ", time.time() - start_time)
            print("##################\n")
            return 0
        else:
            print("##################")
            print(
                "Applying self-calibration solutions for target scans are not done successfully."
            )
            print("Total time taken : ", time.time() - start_time)
            print("##################\n")
            return 1
    except Exception as e:
        traceback.print_exc()
        os.system("rm -rf casa*log")
        print("##################")
        print(
            "Applying self-calibration solutions for target scans are not done successfully."
        )
        print("Total time taken : ", time.time() - start_time)
        print("##################\n")
        return 1


def main(
    mslist,
    workdir,
    caldir,
    applymode="calonly",
    overwrite_datacolumn=False,
    force_apply=False,
    start_remote_log=False,
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    dask_addr=None,
):
    """
    Apply calibration solutions to a list of measurement sets.

    Parameters
    ----------
    mslist : str
        Comma-separated list of measurement set paths to apply calibration to.
    workdir : str
        Directory for logs, intermediate files, and PID tracking.
    caldir : str
        Directory containing calibration tables (e.g., gain, bandpass, polarization).
    applymode : str, optional
        Mode for applying calibration (e.g., "calonly", "calflag", "flagonly"). Default is "calonly".
    overwrite_datacolumn : bool, optional
        If True, overwrites the existing corrected data column. Default is False.
    force_apply : bool, optional
        If True, applies calibration even if it appears to have been applied already. Default is False.
    start_remote_log : bool, optional
        Whether to enable remote logging using credentials found in `workdir`. Default is False.
    cpu_frac : float, optional
        Fraction of available CPU resources to allocate per task. Default is 0.8.
    mem_frac : float, optional
        Fraction of system memory to allocate per task. Default is 0.8.
    logfile : str or None, optional
        Path to the logfile for saving logs. If None, logging to file is disabled. Default is None.
    jobid : int, optional
        Job ID for PID tracking and logging. Default is 0.
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

    # Get first MS from mslist for fallback directory creation
    mslist = mslist.split(",")
    if workdir == "":
        workdir = os.path.dirname(os.path.abspath(mslist[0])) + "/workdir"
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
                "apply_selfcal", logfile, jobname=jobname, password=password
            )
    if observer == None:
        print("Remote link or jobname is blank. Not transmiting to remote logger.")
    ###########
    try:
        print("\n###################################")
        print("Starting applying solutions...")
        print("###################################\n")

        if caldir == "" or not os.path.exists(caldir):
            print("Provide existing caltable directory.")
            msg = 1
        else:
            msg = run_all_applysol(
                mslist,
                workdir,
                caldir,
                overwrite_datacolumn=overwrite_datacolumn,
                applymode=applymode,
                force_apply=force_apply,
                cpu_frac=cpu_frac,
                mem_frac=mem_frac,
                dask_addr=dask_addr,
            )
    except Exception:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
        for ms in mslist:
            drop_cache(ms)
        drop_cache(workdir)
        clean_shutdown(observer)
    return msg


def cli():
    parser = argparse.ArgumentParser(
        description="Apply self-calibration solutions to target scans",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "mslist",
        type=str,
        help="Comma-separated list of measurement sets (required)",
    )
    basic_args.add_argument(
        "--workdir",
        type=str,
        default="",
        required=True,
        help="Working directory for intermediate files",
    )
    basic_args.add_argument(
        "--caldir",
        type=str,
        default="",
        required=True,
        help="Directory containing self-calibration tables",
    )

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced parameters\n###################"
    )
    adv_args.add_argument(
        "--applymode",
        type=str,
        default="calonly",
        help="Applycal mode (e.g. 'calonly', 'calflag')",
    )
    adv_args.add_argument(
        "--overwrite_datacolumn",
        action="store_true",
        help="Overwrite corrected data column in MS",
    )
    adv_args.add_argument(
        "--force_apply",
        action="store_true",
        help="Force apply calibration even if already applied",
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
    hard_args.add_argument(
        "--logfile", type=str, default=None, help="Optional path to log file"
    )
    hard_args.add_argument(
        "--jobid", type=str, default="0", help="Job ID for logging and PID tracking"
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    msg = main(
        args.mslist,
        args.workdir,
        args.caldir,
        applymode=args.applymode,
        overwrite_datacolumn=args.overwrite_datacolumn,
        force_apply=args.force_apply,
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
        "\n###################\nApplying self-calibration solutions are done.\n###################\n"
    )
    os._exit(result)
