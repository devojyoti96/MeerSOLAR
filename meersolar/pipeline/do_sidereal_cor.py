import numpy as np, os, time, traceback, gc
from meersolar.pipeline.basic_func import *
from optparse import OptionParser
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
    start_time=time.time()
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
        print("Sidereal motion correction is not successful for any measurement set.")
        print("Total time taken : ", time.time() - start_time)
        print("##################\n")
        return 1, []
    else:
        print("##################")
        print("Sidereal motion corrections are done successfully.")
        print("Total time taken : ", time.time() - start_time)
        print("##################\n")
        return 0, splited_ms_list_phaserotated


def main():
    usage = "Correct measurement sets for sidereal motion"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--mslist",
        dest="mslist",
        default="",
        help="Name of measurement sets",
        metavar="List",
    )
    parser.add_option(
        "--workdir",
        dest="workdir",
        default="",
        help="Name of work directory",
        metavar="String",
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
    parser.add_option(
        "--max_cpu_frac",
        dest="max_cpu_frac",
        default=0.8,
        help="Maximum CPU fraction to use",
        metavar="Float",
    )
    parser.add_option(
        "--max_mem_frac",
        dest="max_mem_frac",
        default=0.8,
        help="Maximum memory fraction to use",
        metavar="Float",
    )
    parser.add_option(
        "--logfile",
        dest="logfile",
        default=None,
        help="Log file",
        metavar="String",
    )
    parser.add_option(
        "--jobid",
        dest="jobid",
        default=0,
        help="Job ID",
        metavar="Integer",
    )
    (options, args) = parser.parse_args()
    pid=os.getpid()
    save_pid(pid,datadir + f"/pids/pids_{options.jobid}.txt")
    if options.workdir == "" or os.path.exists(options.workdir) == False:
        workdir = os.path.dirname(os.path.abspath(options.msname)) + "/workdir"
        if os.path.exists(workdir) == False:
            os.makedirs(workdir)
    else:
        workdir = options.workdir
    logfile=options.logfile
    observer=None
    if os.path.exists(f"{workdir}/jobname_password.npy") and logfile!=None: 
        time.sleep(5)
        jobname,password=np.load(f"{workdir}/jobname_password.npy",allow_pickle=True)
        if os.path.exists(logfile):
            observer=init_logger("do_sidereal_cor",logfile,jobname=jobname,password=password)
    try:
        if options.mslist == "":
            print("Please provide a list of measurement sets.")
            msg=1
        elif options.workdir == "" or os.path.exists(options.workdir) == False:
            print("Please provide a valid work directory.")
            msg=1
        else:
            mslist = options.mslist.split(",")
            msg, final_target_mslist = cor_sidereal_motion(
                mslist,
                options.workdir,
                cpu_frac=float(options.cpu_frac),
                mem_frac=float(options.mem_frac),
                max_cpu_frac=float(options.max_cpu_frac),
                max_mem_frac=float(options.max_mem_frac),
            )
    except Exception as e:
        traceback.print_exc()
        msg=1
    finally:
        time.sleep(5)
        clean_shutdown(observer)
    return msg  

if __name__ == "__main__":
    result = main()
    print(
        "\n###################\Sidereal motion corrections are done.\n###################\n"
    )
    os._exit(result)
