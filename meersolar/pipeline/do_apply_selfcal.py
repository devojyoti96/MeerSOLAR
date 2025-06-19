import numpy as np, glob, os, copy, warnings, traceback, gc
from casatools import table
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from meersolar.pipeline.basic_func import *
from meersolar.pipeline.do_apply_basiccal import applysol
from dask import delayed, compute
from optparse import OptionParser
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


def main():
    usage = "Apply self-calibration solutions of target scans"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--mslist",
        dest="mslist",
        default="",
        help="Comma seperated list of measurement sets",
        metavar="String",
    )
    parser.add_option(
        "--workdir",
        dest="workdir",
        default="",
        help="Name of work directory",
        metavar="String",
    )
    parser.add_option(
        "--caldir",
        dest="caldir",
        default="",
        help="Caltable directory",
        metavar="String",
    )
    parser.add_option(
        "--applymode",
        dest="applymode",
        default="calonly",
        help="Applycal mode",
        metavar="String",
    )
    parser.add_option(
        "--overwrite_datacolumn",
        dest="overwrite_datacolumn",
        default=False,
        help="Overwrite data column or not",
        metavar="Boolean",
    )
    parser.add_option(
        "--force_apply",
        dest="force_apply",
        default=False,
        help="Force to apply solutions even it is already applied",
        metavar="Boolean",
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
            observer=init_logger("apply_selfcal",logfile,jobname=jobname,password=password)
    try:
        if options.mslist != "":
            print("\n###################################")
            print("Starting applying solutions...")
            print("###################################\n")
            if options.workdir == "" or os.path.exists(options.workdir) == False:
                print("Provide existing work directory name.")
                msg=1
            elif options.caldir == "" or os.path.exists(options.caldir) == False:
                print("Provide existing caltable directory.")
                msg=1
            else:
                msg = run_all_applysol(
                    options.mslist.split(","),
                    options.workdir,
                    options.caldir,
                    overwrite_datacolumn=eval(str(options.overwrite_datacolumn)),
                    applymode=options.applymode,
                    force_apply=eval(str(options.force_apply)),
                    cpu_frac=float(options.cpu_frac),
                    mem_frac=float(options.mem_frac),
                )
        else:
            print("Please provide valid measurement set list.\n")
            msg=1
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
        "\n###################\nApplying self-calibration solutions are done.\n###################\n"
    )
    os._exit(result)
