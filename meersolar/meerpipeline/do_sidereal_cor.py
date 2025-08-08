import logging
import dask
import numpy as np
import argparse
import traceback
import time
import sys
import os
from dask import delayed
from meersolar.utils import *

logging.getLogger("distributed").setLevel(logging.ERROR)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
datadir = get_datadir()


def cor_sidereal_motion(
    mslist,
    dask_client,
    workdir,
):
    """
    Perform sidereal motion correction

    Parameters
    ----------
    mslist : list
        Measurement set list
    dask_client : dask.client
        Dask client
    workdir : str
        Work directory

    Returns
    -------
    int
        Success message
    list
        List of sidereal motion corrected measurement sets
    """
    try:
        container_name = "meerwsclean"
        container_present = check_udocker_container(container_name)
        if not container_present:
            container_name = initialize_wsclean_container(name=container_name)
            if container_name is None:
                print(
                    "Container {container_name} is not initiated. First initiate container and then run."
                )
                return 1, []

        tasks = []
        for ms in mslist:
            tasks.append(delayed(correct_solar_sidereal_motion)(ms))
        print("Starting sidereal motion correction jobs...")
        results = list(dask_client.gather(dask_client.compute(tasks)))

        splited_ms_list_phaserotated = []
        for i in range(len(results)):
            msg = results[i]
            ms = mslist[i]
            if msg == 0:
                if os.path.exists(ms + "/.sidereal_cor"):
                    splited_ms_list_phaserotated.append(ms)
        if len(splited_ms_list_phaserotated) == 0:
            print("##################")
            print(
                "Sidereal motion correction is not successful for any measurement set."
            )
            print("##################")
            return 1, []
        else:
            print("##################")
            print("Sidereal motion corrections are done successfully.")
            print("##################")
            return 0, splited_ms_list_phaserotated
    except Exception as e:
        traceback.print_exc()
        print("##################")
        print("Sidereal motion correction is not successful for any measurement set.")
        print("##################")
        return 1, []


def main(
    mslist,
    workdir="",
    cpu_frac=0.8,
    mem_frac=0.8,
    logfile=None,
    jobid=0,
    start_remote_log=False,
    dask_client=None,
):
    """
    Run a parallel processing pipeline for solar sidereal motion correction

    Parameters
    ----------
    mslist : str
        Comma-separated list of paths to measurement sets to be processed.
    workdir : str, optional
        Directory for logs, intermediate files, and other outputs.
        If empty, defaults to the directory of the first MS with `/workdir` appended.
        Default is "".
    cpu_frac : float, optional
        Fraction of total CPU cores to allocate per task. Default is 0.8.
    mem_frac : float, optional
        Fraction of total system memory to allocate per task. Default is 0.8.
    logfile : str or None, optional
        Path to the log file for capturing logs. If None, logging to file is disabled. Default is None.
    jobid : int, optional
        Unique job identifier used for PID tracking and task differentiation. Default is 0.
    start_remote_log : bool, optional
        Whether to enable remote logging based on credentials stored in the workdir. Default is False.
    dask_client : dask.client, optional
        Dask client

    Returns
    -------
    int
        Success message
    """
    pid = os.getpid()
    cachedir = get_cachedir()
    save_pid(pid, f"{cachedir}/pids/pids_{jobid}.txt")

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
                "do_sidereal_cor", logfile, jobname=jobname, password=password
            )
    if observer == None:
        print("Remote link or jobname is blank. Not transmiting to remote logger.")

    dask_cluster = None
    if dask_client is None:
        dask_client, dask_cluster, dask_dir = get_local_dask_cluster(
            2,
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
        )
        nworker = max(2, int(psutil.cpu_count() * cpu_frac))
        scale_worker_and_wait(dask_cluster, nworker)

    try:
        if len(mslist) == 0:
            print("Please provide a list of measurement sets.")
            msg = 1
        else:
            msg, final_target_mslist = cor_sidereal_motion(
                mslist,
                dask_client,
                workdir,
            )
    except Exception as e:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
        for ms in mslist:
            drop_cache(ms)
        drop_cache(workdir)
        clean_shutdown(observer)
        if dask_cluster is not None:
            dask_client.close()
            dask_cluster.close()
            os.system(f"rm -rf {dask_dir}")
    return msg


def cli():
    parser = argparse.ArgumentParser(
        description="Correct measurement sets for sidereal motion",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    # Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "mslist",
        type=str,
        help="Comma-separated list of measurement sets (required positional argument)",
    )
    basic_args.add_argument("--workdir", type=str, default="", help="Working directory")

    # Advanced parameters
    adv_args = parser.add_argument_group(
        "###################\nAdvanced calibration and imaging parameters\n###################"
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
    hard_args.add_argument("--logfile", type=str, default=None, help="Log file")
    hard_args.add_argument("--jobid", type=int, default=0, help="Job ID")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args()

    msg = main(
        mslist=args.mslist,
        workdir=args.workdir,
        cpu_frac=args.cpu_frac,
        mem_frac=args.mem_frac,
        logfile=args.logfile,
        jobid=args.jobid,
        start_remote_log=args.start_remote_log,
    )
    return msg


if __name__ == "__main__":
    result = cli()
    print(
        "\n###################\nSidereal motion corrections are done.\n###################\n"
    )
    os._exit(result)
