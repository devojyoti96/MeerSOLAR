import numpy as np, glob, os, copy, warnings, traceback, gc
from casatools import table
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from meersolar.pipeline.basic_func import *
from meersolar.pipeline.do_applycal import applysol
from dask import delayed, compute
from optparse import OptionParser
from casatasks import casalog

logfile = casalog.logfile()
os.system("rm -rf " + logfile)


def run_apply_selfcal(
    mslist,
    workdir,
    caldir,
    overwrite_datacolumn=False,
    cpu_frac=0.8,
    mem_frac=0.8,
):
    """
    Apply self-calibration solutions on all target scans
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
        gaintable = glob.glob(caldir + "/*cal")
        if len(gaintable) == 0:
            print(f"No self-calibration table is present in: {caldir}.")
            return []
        ####################################
        # Applycal jobs
        ####################################
        print(f"Total ms list: {len(mslist)}")
        task = delayed(applysol)(dry_run=True)
        mem_limit = run_limited_memory_task(task)
        dask_client, dask_cluster, n_jobs, n_threads = get_dask_client(
            len(mslist), cpu_frac, mem_frac, min_mem_per_job=mem_limit / 0.8
        )
        workers = list(dask_client.scheduler_info()["workers"].items())
        addr, stats = workers[0]
        memory_limit = stats["memory_limit"] / 1024**3
        target_frac = config.get("distributed.worker.memory.target")
        memory_limit *= target_frac
        tasks = []
        msmd = msmetadata()
        for ms in mslist:
            interp = ["nearest"] * len(gaintable)
            tasks.append(
                delayed(applysol)(
                    msname=ms,
                    gaintable=gaintable,
                    interp=interp,
                    overwrite_datacolumn=overwrite_datacolumn,
                    n_threads=n_threads,
                    memory_limit=memory_limit,
                )
            )
        results = compute(*tasks)
        dask_client.close()
        dask_cluster.close()
        os.system("rm -rf casa*log")
        print("##################")
        print(
            "Applying self-calibration solutions for target scans are done successfully."
        )
        print("Total time taken : ", time.time() - start_time)
        print("##################\n")
        return 0
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
        "--print_casalog",
        dest="print_casalog",
        default=False,
        help="Print CASA log",
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
        "--overwrite_datacolumn",
        dest="overwrite_datacolumn",
        default=False,
        help="Overwrite data column or not",
        metavar="Boolean",
    )
    (options, args) = parser.parse_args()
    if eval(str(options.print_casalog)) == True:
        casalog.showconsole(True)
    if options.mslist != "":
        print("\n###################################")
        print("Starting applying self-calibration solutions...")
        print("###################################\n")
        try:
            if options.workdir == "" or os.path.exists(options.workdir) == False:
                print("Provide existing work directory name.")
                return 1
            if options.caldir == "" or os.path.exists(options.caldir) == False:
                print("Provide existing caltable directory.")
                return 1
            msg = run_apply_selfcal(
                options.mslist.split(","),
                options.workdir,
                options.caldir,
                overwrite_datacolumn=eval(str(options, overwrite_datacolumn)),
                cpu_frac=float(options.cpu_frac),
                mem_frac=float(options.mem_frac),
            )
            return msg
        except Exception as e:
            traceback.print_exc()
            return 1
    else:
        print("Please provide valid measurement set list.\n")
        return 1


if __name__ == "__main__":
    result = main()
    print(
        "\n###################\nApplying self-calibration solutions are done.\n###################\n"
    )
    os._exit(result)
