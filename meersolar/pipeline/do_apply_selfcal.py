import numpy as np, glob, os, copy, warnings, traceback, gc, argparse
from casatools import table
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from meersolar.pipeline.basic_func import *
from meersolar.pipeline.do_apply_basiccal import applysol
from dask import delayed, compute
from casatasks import casalog

try:
    logfile = casalog.logfile()
    os.system("rm -rf " + logfile)
except:
    pass


def run_all_applysol(
    mslist,
    workdir,
    caldir,
    overwrite_datacolumn=False,
    applymode="calonly",
    force_apply=False,
    cpu_frac=0.8,
    mem_frac=0.8,
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
        task = delayed(applysol)(dry_run=True)
        mem_limit = run_limited_memory_task(task, dask_dir=workdir)
        ms_size_list = [get_ms_size(ms) + mem_limit for ms in mslist]
        mem_limit = max(ms_size_list)
        dask_client, dask_cluster, n_jobs, n_threads, mem_limit = get_dask_client(
            len(mslist),
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem_per_job=mem_limit / 0.6,
        )
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
        results = compute(*tasks)
        dask_client.close()
        dask_cluster.close()
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
    finally:
        time.sleep(5)
        for ms in mslist:
            drop_cache(ms)
        drop_cache(workdir)


def main():
    parser = argparse.ArgumentParser(
        description="Apply self-calibration solutions to target scans",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    ## Essential parameters
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
        help="Working directory for intermediate files",
    )
    basic_args.add_argument(
        "--caldir",
        type=str,
        default="",
        help="Directory containing self-calibration tables",
    )

    ## Advanced parameters
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

    ## Resource management parameters
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

    pid = os.getpid()
    save_pid(pid, datadir + f"/pids/pids_{args.jobid}.txt")

    # Get first MS from mslist for fallback directory creation
    mslist = args.mslist.split(",")
    if args.workdir == "" or not os.path.exists(args.workdir):
        workdir = os.path.dirname(os.path.abspath(mslist[0])) + "/workdir"
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
                "apply_selfcal", logfile, jobname=jobname, password=password
            )

    try:
        print("\n###################################")
        print("Starting applying solutions...")
        print("###################################\n")

        if args.workdir == "" or not os.path.exists(args.workdir):
            print("Provide existing work directory name.")
            msg = 1
        elif args.caldir == "" or not os.path.exists(args.caldir):
            print("Provide existing caltable directory.")
            msg = 1
        else:
            msg = run_all_applysol(
                mslist,
                args.workdir,
                args.caldir,
                overwrite_datacolumn=args.overwrite_datacolumn,
                applymode=args.applymode,
                force_apply=args.force_apply,
                cpu_frac=args.cpu_frac,
                mem_frac=args.mem_frac,
            )
    except Exception:
        traceback.print_exc()
        msg = 1
    finally:
        time.sleep(5)
        for ms in mslist:
            drop_cache(ms)
        drop_cache(args.workdir)
        clean_shutdown(observer)
    return msg


if __name__ == "__main__":
    result = main()
    print(
        "\n###################\nApplying self-calibration solutions are done.\n###################\n"
    )
    os._exit(result)
