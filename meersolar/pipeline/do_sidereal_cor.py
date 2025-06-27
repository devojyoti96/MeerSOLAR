import numpy as np, os, time, traceback, gc, argparse
from meersolar.pipeline.basic_func import *
from casatasks import casalog
from dask import delayed, compute

try:
    logfile = casalog.logfile()
    os.system("rm -rf " + logfile)
except:
    pass


def cor_sidereal_motion(
    mslist, workdir, cpu_frac=0.8, mem_frac=0.8, max_cpu_frac=0.8, max_mem_frac=0.8
):
    """
    Perform sidereal motion correction

    Parameters
    ----------
    mslist : list
        Measurement set list
    workdir : str
        Work directory
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
    int
        Success message
    list
        List of sidereal motion corrected measurement sets
    """
    start_time = time.time()
    try:
        #############################################
        # Memory limit
        #############################################
        task = delayed(correct_solar_sidereal_motion)(dry_run=True)
        mem_limit = run_limited_memory_task(task, dask_dir=workdir)
        #############################################
        tasks = []
        for ms in mslist:
            tasks.append(delayed(correct_solar_sidereal_motion)(ms))
        total_chunks = len(tasks)
        dask_client, dask_cluster, n_jobs, n_threads, mem_limit = get_dask_client(
            total_chunks,
            dask_dir=workdir,
            cpu_frac=cpu_frac,
            mem_frac=mem_frac,
            min_mem_per_job=mem_limit / 0.6,
        )
        results = compute(*tasks)
        dask_client.close()
        dask_cluster.close()
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
            print("Total time taken : ", time.time() - start_time)
            print("##################\n")
            return 1, []
        else:
            print("##################")
            print("Sidereal motion corrections are done successfully.")
            print("Total time taken : ", time.time() - start_time)
            print("##################\n")
            return 0, splited_ms_list_phaserotated
    except Exception as e:
        traceback.print_exc()
        print("##################")
        print("Sidereal motion correction is not successful for any measurement set.")
        print("Total time taken : ", time.time() - start_time)
        print("##################\n")
        return 1, []
    finally:
        time.sleep(5)
        for ms in mslist:
            drop_cache(ms)
        drop_cache(workdir)


def main():
    parser = argparse.ArgumentParser(
        description="Correct measurement sets for sidereal motion",
        formatter_class=SmartDefaultsHelpFormatter,
    )

    ## Essential parameters
    basic_args = parser.add_argument_group(
        "###################\nEssential parameters\n###################"
    )
    basic_args.add_argument(
        "mslist",
        type=str,
        help="Comma-separated list of measurement sets (required positional argument)",
    )
    basic_args.add_argument(
        "--workdir", type=str, default="", help="Working directory"
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

    if args.workdir == "" or os.path.exists(args.workdir) == False:
        workdir = (
            os.path.dirname(os.path.abspath(args.mslist.split(",")[0])) + "/workdir"
        )
    else:
        workdir = args.workdir
    os.makedirs(workdir,exist_ok=True)

    logfile = args.logfile
    observer = None
    if os.path.exists(f"{workdir}/jobname_password.npy") and logfile != None:
        time.sleep(5)
        jobname, password = np.load(
            f"{workdir}/jobname_password.npy", allow_pickle=True
        )
        if os.path.exists(logfile):
            observer = init_logger(
                "do_sidereal_cor", logfile, jobname=jobname, password=password
            )
    mslist = args.mslist.split(",")
    try:
        if len(mslist) == 0:
            print("Please provide a list of measurement sets.")
            msg = 1
        elif args.workdir == "" or os.path.exists(args.workdir) == False:
            print("Please provide a valid work directory.")
            msg = 1
        else:
            msg, final_target_mslist = cor_sidereal_motion(
                mslist,
                args.workdir,
                cpu_frac=float(args.cpu_frac),
                mem_frac=float(args.mem_frac),
                max_cpu_frac=float(args.max_cpu_frac),
                max_mem_frac=float(args.max_mem_frac),
            )
    except Exception as e:
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
        "\n###################\Sidereal motion corrections are done.\n###################\n"
    )
    os._exit(result)
